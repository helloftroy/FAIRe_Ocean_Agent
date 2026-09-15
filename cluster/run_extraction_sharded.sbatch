#!/bin/bash
# Genuinely parallel EXTRACT_TEXT_FACTS across N array tasks, EACH working
# against its OWN local, isolated SQLite copy of the database instead of
# the shared file -- for clusters where run_extraction_parallel.sbatch's
# shared-file array mode fails with "sqlite3.OperationalError: locking
# protocol" (confirmed live: this cluster's Lustre-backed scratch mount
# can't coordinate SQLite file locks across DIFFERENT COMPUTE NODES, even
# on a plain read, so no --array size of the shared-file version is safe
# there). This script only ever touches its OWN shard file -- no
# cross-node contention at all during the whole (slow, ~9 min/paper) LLM
# phase, at the cost of a "touch the database at the start, touch it
# again at the end" two-step workflow around it. See cluster/README.md's
# "Speeding up extraction" section for the full story and the tradeoffs
# against run_extraction_parallel.sbatch / submit_extraction_sequential.sh.
#
# Three-step workflow -- run these IN ORDER:
#   1. python scripts/shard_extraction_prep.py --shards 5
#      (single process, touches the shared DB once: enqueues the backlog,
#      partitions it, writes data/shard_dbs/shard_<i>.db + manifest.json)
#   2. sbatch --account=<acct> --array=1-5 cluster/run_extraction_sharded.sbatch
#      (THIS script -- --array size MUST match --shards above exactly;
#      each array task works entirely against its own shard_<i>.db)
#   3. python scripts/merge_extraction_shards.py
#      (single process, run once every array task has finished: stitches
#      every shard's new/changed rows back into the shared DB, then
#      re-runs FAIRe mapping for every touched study)
#
# Size --shards so each shard's slice fits inside the 48h time limit below
# (~9 min/paper -- e.g. a ~1500-paper backlog needs at least ~5 shards to
# fit in one round; a shard that's still running when SLURM kills it at
# 48h just leaves its unfinished tasks 'pending'/'claimed' in that shard's
# own copy, safely picked up by the NEXT shard_extraction_prep.py round).
#
# Partition/gres copied from run_extraction.sbatch -- double check this is
# the right partition name for your account before submitting.
#SBATCH --job-name=faire_extraction_sharded
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extraction_sharded_%A_%a.out
#SBATCH --error=logs/extraction_sharded_%A_%a.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

: "${SLURM_ARRAY_TASK_ID:?This script must be run as a job array (--array=1-N, matching shard_extraction_prep.py --shards N) -- it has no single-job mode.}"
ARRAY_LABEL="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-?}"
echo "Job started: $(date)"
echo "Array task: ${ARRAY_LABEL} (job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID})"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

MANIFEST="${SHARD_MANIFEST:-data/shard_dbs/manifest.json}"
if [ ! -f "${MANIFEST}" ]; then
  echo "No manifest at ${MANIFEST} -- run 'python scripts/shard_extraction_prep.py --shards N' first." >&2
  exit 2
fi
export FAIRE_SHARD_MANIFEST_PATH="${MANIFEST}"
SHARD_DB_PATH="$(python3 - <<'PYEOF'
import json
import os
import sys

manifest = json.load(open(os.environ["FAIRE_SHARD_MANIFEST_PATH"]))
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
match = next((s for s in manifest["shards"] if s["shard_index"] == idx), None)
if match is None:
    sys.exit(f"No shard with shard_index={idx} in {os.environ['FAIRE_SHARD_MANIFEST_PATH']} -- does --array match --shards?")
print(match["db_path"])
PYEOF
)"
echo "This array task's isolated shard database: ${SHARD_DB_PATH}"
export FAIR_OCEAN_DATABASE_URL="sqlite:///${SHARD_DB_PATH}"

if [ ! -f config/local.yaml ]; then
  echo "config/local.yaml is missing -- copy cluster/local.yaml.cluster-example to" >&2
  echo "config/local.yaml and edit it first (see cluster/README.md)." >&2
  exit 2
fi

# --- Start a local LLM server for this array task (own GPU, own node). ---
# Identical logic to run_extraction_parallel.sbatch's own server-start
# block -- see that script's comments for why each piece is there.
LLM_BACKEND="${LLM_BACKEND:-vllm}"
SERVER_PID=""

if [ "${LLM_BACKEND}" = "ollama" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found on PATH. Either install it, set LLM_BACKEND=vllm, or set" >&2
    echo "LLM_BACKEND=external if you're using a different endpoint already running." >&2
    exit 2
  fi
  echo "Starting ollama serve..."
  ollama serve > "logs/ollama_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
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
    echo "ollama did not become responsive in time -- check the log above." >&2
    exit 1
  fi

  MODEL_NAME="$(python -c "import yaml; print(yaml.safe_load(open('config/local.yaml'))['llm']['model'])")"
  if ! ollama list | awk '{print $1}' | grep -qx "${MODEL_NAME}"; then
    echo "Model '${MODEL_NAME}' is not pulled on this node. GPU nodes have no" >&2
    echo "internet access -- pull it once from a login node first: ollama pull ${MODEL_NAME}" >&2
    exit 2
  fi
  echo "Model '${MODEL_NAME}' is present."

elif [ "${LLM_BACKEND}" = "vllm" ]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm not found on PATH. Install it in this venv first: pip install vllm" >&2
    exit 2
  fi
  MODEL_NAME="$(python -c "import yaml; print(yaml.safe_load(open('config/local.yaml'))['llm']['model'])")"
  export HF_HOME="${HF_HOME:-$(pwd)/cluster/hf_cache}"
  export HF_HUB_OFFLINE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "Starting vllm serve (model=${MODEL_NAME}, HF_HOME=${HF_HOME})..."
  vllm serve "${MODEL_NAME}" --host 0.0.0.0 --port 8000 > "logs/vllm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
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
    echo "vllm did not become responsive in time -- check the log above." >&2
    exit 1
  fi

elif [ "${LLM_BACKEND}" = "external" ]; then
  echo "LLM_BACKEND=external: not starting a server, trusting config/local.yaml's base_url."
else
  echo "Unknown LLM_BACKEND='${LLM_BACKEND}'. Use ollama, vllm, or external." >&2
  exit 2
fi

# No enqueue-text-extraction-backfill call here -- shard_extraction_prep.py
# already enqueued and partitioned the backlog once, single-process,
# before this array was submitted. This shard's own database already
# contains exactly (and only) its own slice, pre-filtered.
echo "=== claiming this shard's EXTRACT_TEXT_FACTS tasks until its own queue is empty (array task ${ARRAY_LABEL}) ==="
python -m fair_ocean_agent.cli worker --task-type EXTRACT_TEXT_FACTS --until-empty

echo "Job finished: $(date)"
echo "Array task ${ARRAY_LABEL} done. Once every array task finishes, run:"
echo "  python scripts/merge_extraction_shards.py"
echo "to stitch every shard's results back into the shared database."
