#!/bin/bash
# Stage 2: LLM extraction + mapping + validation + export. Needs a GPU,
# needs NO internet access -- every network-fetchable thing this stage
# might otherwise need (full text, supplement files) was already fetched
# and disk-cached by run_discovery.sbatch (data/cache/<source>/*, keyed by
# request URL: sources/base.py's RateLimitedClient). Run
# run_discovery.sbatch first, against the SAME database.
#
# Partition/gres below are copied from multimodal_granite's own GPU jobs
# (e.g. fish/run_apply_granite_edna_ecology.sbatch) -- double check this
# is the right partition name for your account before submitting.
#SBATCH --job-name=faire_extraction
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extraction_%j.out
#SBATCH --error=logs/extraction_%j.err

set -euo pipefail
# $(dirname "$0") is NOT reliable here -- sbatch commonly stages/copies the
# submitted script before running it, so $0 can point at a spool directory
# you don't have write access to, not this repo. SLURM_SUBMIT_DIR is the
# standard, reliable way to get back to wherever `sbatch` was actually run
# from (confirmed live: this exact mkdir failed with "Permission denied"
# using the dirname approach).
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"   # repo root (fair_ocean_agent/), i.e. wherever you ran `sbatch cluster/run_extraction.sbatch` from
mkdir -p logs

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
echo "Database: ${FAIR_OCEAN_DATABASE_URL}"

if [ ! -f config/local.yaml ]; then
  echo "config/local.yaml is missing -- copy cluster/local.yaml.cluster-example to" >&2
  echo "config/local.yaml and edit it first (see cluster/README.md)." >&2
  exit 2
fi

# --- Start a local LLM server for this job. ---
# LLM_BACKEND=ollama (default), vllm, or external (config/local.yaml
# already points at a persistent, separately-managed endpoint you're
# starting/managing some other way -- this script starts nothing).
LLM_BACKEND="${LLM_BACKEND:-ollama}"
SERVER_PID=""

if [ "${LLM_BACKEND}" = "ollama" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found on PATH. Either install it (see cluster/README.md's" >&2
    echo "'One-time model download' section), set LLM_BACKEND=vllm, or set" >&2
    echo "LLM_BACKEND=external if you're using a different endpoint already running." >&2
    exit 2
  fi
  echo "Starting ollama serve..."
  ollama serve > logs/ollama_${SLURM_JOB_ID:-local}.log 2>&1 &
  SERVER_PID=$!
  trap '[ -n "${SERVER_PID}" ] && kill "${SERVER_PID}" 2>/dev/null || true' EXIT

  echo "Waiting for ollama to respond..."
  for _ in $(seq 1 30); do
    if curl -s -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
      echo "ollama is up."
      break
    fi
    sleep 2
  done
  if ! curl -s -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "ollama did not become responsive in time -- check logs/ollama_${SLURM_JOB_ID:-local}.log" >&2
    exit 1
  fi

  MODEL_NAME="$(python -c "import yaml; print(yaml.safe_load(open('config/local.yaml'))['llm']['model'])")"
  if ! ollama list | awk '{print $1}' | grep -qx "${MODEL_NAME}"; then
    echo "Model '${MODEL_NAME}' is not pulled on this node. GPU nodes have no" >&2
    echo "internet access -- pull it once from a login node first:" >&2
    echo "  ollama pull ${MODEL_NAME}" >&2
    echo "(see cluster/README.md's 'One-time model download' section)" >&2
    exit 2
  fi
  echo "Model '${MODEL_NAME}' is present."

elif [ "${LLM_BACKEND}" = "vllm" ]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm not found on PATH. Install it in this venv first:" >&2
    echo "  pip install vllm" >&2
    echo "(see cluster/README.md's 'vLLM instead of Ollama' section)" >&2
    exit 2
  fi
  MODEL_NAME="$(python -c "import yaml; print(yaml.safe_load(open('config/local.yaml'))['llm']['model'])")"
  # vLLM pulls from the Hugging Face Hub on first use and caches under
  # HF_HOME -- GPU nodes have no internet, so this must already be cached
  # from a prior download-only run on a node that does (see README).
  # HF_HUB_OFFLINE=1 makes a missing/uncached model fail fast and clearly
  # instead of hanging trying to reach huggingface.co.
  export HF_HOME="${HF_HOME:-$(pwd)/cluster/hf_cache}"
  export HF_HUB_OFFLINE=1
  # flashinfer's top-k/top-p sampling kernel JIT-compiles a CUDA extension
  # on first use via nvcc -- confirmed live: GPU compute nodes commonly
  # ship the CUDA driver/runtime but not the full toolkit, so this failed
  # with "Could not find nvcc and default cuda_home='/usr/local/cuda'
  # doesn't exist" partway through the very first inference request.
  # vLLM's PyTorch-native sampler needs nothing extra and is what it uses
  # automatically whenever flashinfer itself isn't installed, so this just
  # forces the same fallback rather than chasing down the cluster's CUDA
  # toolkit module/path.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # A cluster's module-injected LD_LIBRARY_PATH (e.g. a spack-managed
  # toolchain) can shadow this conda env's own bundled libstdc++ with an
  # older system one (/usr/lib64/libstdc++.so.6) -- confirmed live: vllm's
  # own sqlite3 import chain, via a bundled libicui18n.so, failed with
  # "version CXXABI_1.3.15 not found" the moment anything pulled in a C++
  # library, even though the newer libstdc++ needed was sitting right
  # there in the same conda env. Scoped to just this vllm process (not a
  # blanket export in cluster/env_activate.sh) -- confirmed live that
  # exporting it shell-wide instead broke plain `git` in an interactive
  # session afterward: `git`'s libldap dependency picked up this conda
  # env's own openssl build over the system one and crashed with
  # "undefined symbol: EVP_md2" the moment LD_LIBRARY_PATH leaked into a
  # shell running anything besides this pipeline's own python/vllm.
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "Starting vllm serve (model=${MODEL_NAME}, HF_HOME=${HF_HOME})..."
  vllm serve "${MODEL_NAME}" --host 0.0.0.0 --port 8000 > logs/vllm_${SLURM_JOB_ID:-local}.log 2>&1 &
  SERVER_PID=$!
  trap '[ -n "${SERVER_PID}" ] && kill "${SERVER_PID}" 2>/dev/null || true' EXIT

  echo "Waiting for vllm to respond (model loading can take a few minutes)..."
  for _ in $(seq 1 60); do
    if curl -s -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "vllm is up."
      break
    fi
    sleep 5
  done
  if ! curl -s -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "vllm did not become responsive in time -- check logs/vllm_${SLURM_JOB_ID:-local}.log" >&2
    echo "(a missing/uncached model under HF_HUB_OFFLINE=1 will show up there as a clear error)" >&2
    exit 1
  fi

elif [ "${LLM_BACKEND}" = "external" ]; then
  echo "LLM_BACKEND=external: not starting a server, trusting config/local.yaml's base_url."

else
  echo "Unknown LLM_BACKEND='${LLM_BACKEND}'. Use ollama, vllm, or external." >&2
  exit 2
fi

echo "=== LLM text extraction (main papers) ==="
python -m fair_ocean_agent.cli enqueue-text-extraction-backfill
python -m fair_ocean_agent.cli worker --until-empty

echo "=== LLM text extraction (supplements) ==="
python -m fair_ocean_agent.cli enqueue-supplement-text-extraction-backfill
python -m fair_ocean_agent.cli worker --until-empty

echo "=== mapping ==="
python -m fair_ocean_agent.cli enqueue-mapping-backfill
python -m fair_ocean_agent.cli worker --until-empty

# Belt-and-suspenders re-mapping sweep: a study whose connected component
# (shared entities across studies, e.g. citation-linked papers) is still
# growing when its own MAP_FAIRE task first runs can end up with a stale
# mapping that never gets a second, complete pass automatically -- caught
# live during this pipeline's own development (sample-alias reconciliation
# silently stayed stale after a batch run with heavy citation fan-out).
# Re-running map_study_to_faire per study now, after every other stage has
# settled, is cheap (no network, no LLM) and makes this self-healing
# instead of timing-dependent.
echo "=== explicit re-mapping sweep (self-healing pass) ==="
python3 - <<'PYEOF'
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.database.models import Study
from fair_ocean_agent.mapping.faire import map_study_to_faire

with session_scope() as session:
    study_ids = [s.study_id for s in session.query(Study).all()]
    for study_id in study_ids:
        try:
            created = map_study_to_faire(session, study_id)
            print(f"  {study_id}: {created} standardized values")
        except Exception as exc:
            print(f"  {study_id}: ERROR {exc}")
    session.commit()
PYEOF

echo "=== validation ==="
python -m fair_ocean_agent.cli enqueue-validation-backfill
python -m fair_ocean_agent.cli worker --until-empty
python -m fair_ocean_agent.cli enqueue-faire-completeness-backfill
python -m fair_ocean_agent.cli worker --until-empty

OUTPUT_DIR="${OUTPUT_DIR:-data/exports/audits/cluster_run_${SLURM_JOB_ID:-local}}"
echo "=== export ==="
python -m fair_ocean_agent.cli export-faire --output "${OUTPUT_DIR}"
python -m fair_ocean_agent.cli status

echo "Job finished: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "scp that directory back, e.g.:"
echo "  scp -r <cluster>:$(pwd)/${OUTPUT_DIR} ./"
