#!/bin/bash
# Small, isolated LLM extraction test batch -- runs EXTRACT_TEXT_FACTS +
# MAP_FAIRE for a handful of real candidate studies against a SEPARATE
# COPY of the main database, so you can iterate on the extraction/mapping
# code (tweak, re-run, inspect, repeat) without touching the real
# pipeline's task queue at all. See scripts/llm_troubleshooting_batch.py's
# own docstring for what each phase does.
#
# First run: creates data/fair_ocean_llm_test.db as a live-safe SQLite
# .backup copy of the real database (never a plain `cp` against a database
# another job might be actively writing to -- see cluster/README.md's own
# established precedent on this) and picks LLM_TEST_COUNT (default 10)
# fresh candidate studies into data/llm_troubleshooting_manifest.json.
#
# Iterating after a code change (same studies, re-run the LLM step):
#   sbatch --account=<acct> --export=ALL,LLM_TEST_PHASES="reset enqueue worker remap report" cluster/run_llm_troubleshoot.sbatch
#
# Picking a fresh batch of studies from the current test DB (no re-copy):
#   sbatch --account=<acct> --export=ALL,LLM_TEST_PHASES="select enqueue worker remap report" cluster/run_llm_troubleshoot.sbatch
#
# Starting over completely (re-copy from the real, current main database):
#   sbatch --account=<acct> --export=ALL,LLM_TEST_DB_REFRESH=1 cluster/run_llm_troubleshoot.sbatch
#
# Other knobs:
#   sbatch --account=<acct> --export=ALL,LLM_TEST_COUNT=5 cluster/run_llm_troubleshoot.sbatch
#   sbatch --account=<acct> --export=ALL,LLM_BACKEND=vllm cluster/run_llm_troubleshoot.sbatch
#SBATCH --job-name=llm_troubleshoot
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/llm_troubleshoot_%j.out
#SBATCH --error=logs/llm_troubleshoot_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ ! -f config/local.yaml ]; then
  echo "config/local.yaml is missing -- copy cluster/local.yaml.cluster-example to" >&2
  echo "config/local.yaml and edit it first (see cluster/README.md)." >&2
  exit 2
fi

MAIN_DB="${MAIN_DB:-data/fair_ocean.db}"
LLM_TEST_DB="${LLM_TEST_DB:-data/fair_ocean_llm_test.db}"
LLM_TEST_COUNT="${LLM_TEST_COUNT:-10}"
LLM_TEST_MANIFEST="${LLM_TEST_MANIFEST:-data/llm_troubleshooting_manifest.json}"
LLM_TEST_PHASES="${LLM_TEST_PHASES:-select enqueue worker remap report}"

if [ "${LLM_TEST_DB_REFRESH:-0}" = "1" ] && [ -f "${LLM_TEST_DB}" ]; then
  echo "LLM_TEST_DB_REFRESH=1: removing existing ${LLM_TEST_DB} to re-copy fresh."
  rm -f "${LLM_TEST_DB}" "${LLM_TEST_DB}-wal" "${LLM_TEST_DB}-shm"
fi

if [ ! -f "${LLM_TEST_DB}" ]; then
  if [ ! -f "${MAIN_DB}" ]; then
    echo "${MAIN_DB} does not exist -- nothing to copy. Set MAIN_DB if it's elsewhere." >&2
    exit 2
  fi
  echo "Creating isolated test database ${LLM_TEST_DB} from ${MAIN_DB} (sqlite3 .backup, safe against a live writer)..."
  sqlite3 "${MAIN_DB}" ".backup '${LLM_TEST_DB}'"
else
  echo "Reusing existing isolated test database ${LLM_TEST_DB} (set LLM_TEST_DB_REFRESH=1 to re-copy from ${MAIN_DB})."
fi

export FAIR_OCEAN_DATABASE_URL="sqlite:///$(pwd)/${LLM_TEST_DB}"
echo "Database (isolated copy, NOT the main pipeline database): ${FAIR_OCEAN_DATABASE_URL}"
python -m alembic -c alembic.ini upgrade head

# --- Start a local LLM server for this job (same logic as run_extraction.sbatch). ---
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
    exit 1
  fi

elif [ "${LLM_BACKEND}" = "external" ]; then
  echo "LLM_BACKEND=external: not starting a server, trusting config/local.yaml's base_url."
else
  echo "Unknown LLM_BACKEND='${LLM_BACKEND}'. Use ollama, vllm, or external." >&2
  exit 2
fi

echo "Phases to run: ${LLM_TEST_PHASES}"
for phase in ${LLM_TEST_PHASES}; do
  echo
  echo "=== phase: ${phase} ==="
  case "${phase}" in
    select)
      python scripts/llm_troubleshooting_batch.py select --count "${LLM_TEST_COUNT}" --manifest "${LLM_TEST_MANIFEST}"
      ;;
    reset)
      python scripts/llm_troubleshooting_batch.py reset --manifest "${LLM_TEST_MANIFEST}"
      ;;
    enqueue)
      python scripts/llm_troubleshooting_batch.py enqueue --manifest "${LLM_TEST_MANIFEST}"
      ;;
    worker)
      python -m fair_ocean_agent.cli worker --task-type EXTRACT_TEXT_FACTS --until-empty
      ;;
    remap)
      python scripts/llm_troubleshooting_batch.py remap --manifest "${LLM_TEST_MANIFEST}"
      ;;
    report)
      python scripts/llm_troubleshooting_batch.py report --manifest "${LLM_TEST_MANIFEST}"
      ;;
    *)
      echo "Unknown phase '${phase}' in LLM_TEST_PHASES -- valid: select reset enqueue worker remap report" >&2
      exit 2
      ;;
  esac
done

echo
echo "Job finished: $(date)"
echo "Isolated test database: ${LLM_TEST_DB} (main pipeline database untouched: ${MAIN_DB})"
echo "Manifest: ${LLM_TEST_MANIFEST}"
