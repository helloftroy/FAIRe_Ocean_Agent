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
cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)
mkdir -p logs

# If your cluster requires modules/conda, uncomment and edit:
# module load anaconda
source .venv/bin/activate

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

# --- Start a local Ollama server for this job, if requested (default). ---
# Set USE_OLLAMA=0 if config/local.yaml already points at a persistent,
# separately-managed endpoint (e.g. a shared vLLM service).
USE_OLLAMA="${USE_OLLAMA:-1}"
OLLAMA_PID=""
if [ "${USE_OLLAMA}" = "1" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found on PATH. Either install it (see cluster/README.md's" >&2
    echo "'One-time model download' section) or set USE_OLLAMA=0 if you're using" >&2
    echo "a different endpoint already running." >&2
    exit 2
  fi
  echo "Starting ollama serve..."
  ollama serve > logs/ollama_${SLURM_JOB_ID:-local}.log 2>&1 &
  OLLAMA_PID=$!
  trap '[ -n "${OLLAMA_PID}" ] && kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT

  echo "Waiting for ollama to respond..."
  for _ in $(seq 1 30); do
    if curl -s -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
      echo "ollama is up."
      break
    fi
    sleep 2
  done
  if ! curl -s -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "ollama did not become responsive in time." >&2
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
