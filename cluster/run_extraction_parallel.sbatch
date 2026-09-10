#!/bin/bash
# Splits the EXTRACT_TEXT_FACTS backlog (the slow, GPU-bound stage -- at
# ~9 min/paper, a large backlog can take far longer than one 48h job) into
# several parallel array tasks, each with its own GPU and its own vLLM
# server, all safely claiming from the SAME shared data/fair_ocean.db.
#
# Safe to do now specifically because of a real fix already in place:
# workflow/task_queue.py's SQLite claim_next_task used to be single-
# worker-only (its own docstring admitted this) -- two concurrent
# processes could both claim the same task, confirmed live producing 416
# duplicate claims out of 500 under real 8-process contention. Fixed with
# a single atomic UPDATE ... RETURNING claim, combined with WAL mode +
# a real busy_timeout (database/session.py). Do not run this against a
# checkout that predates that fix.
#
# Each array task runs ONLY EXTRACT_TEXT_FACTS, bounded by
# EXTRACT_MAX_TASKS (default 300; at ~9 min/paper that's ~45h, leaving
# headroom under the 48h limit for server startup and per-task overhead)
# -- deliberately NOT --until-empty, since with N array tasks all
# draining the same queue, one array task could otherwise claim far more
# than its share while others sit idle near the end. Supplement
# extraction, mapping, validation, and export are deliberately NOT run
# here -- once every array task finishes (or the backlog's fully
# drained), run the ordinary run_extraction.sbatch once (no array) to
# pick up whatever's left of EXTRACT_TEXT_FACTS (a quick no-op if this
# already finished it) and continue through the rest of the pipeline.
#
# Usage:
#   sbatch --account=<acct> --array=1-5 cluster/run_extraction_parallel.sbatch
#   sbatch --account=<acct> --array=1-10 --export=ALL,EXTRACT_MAX_TASKS=300 cluster/run_extraction_parallel.sbatch
#
# If the backlog isn't fully drained after one round (check with
# `fair-ocean status` or the per-array-task logs), just resubmit the same
# command -- enqueue-text-extraction-backfill is idempotent and
# claim_next_task naturally picks up wherever the queue left off.
#
# Partition/gres copied from run_extraction.sbatch -- double check this is
# the right partition name for your account before submitting.
#SBATCH --job-name=faire_extraction_parallel
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/extraction_parallel_%A_%a.out
#SBATCH --error=logs/extraction_parallel_%A_%a.err

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

ARRAY_LABEL="${SLURM_ARRAY_TASK_ID:-single}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Job started: $(date)"
echo "Array task: ${ARRAY_LABEL} (job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID})"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
echo "Database (shared across every array task): ${FAIR_OCEAN_DATABASE_URL}"

if [ ! -f config/local.yaml ]; then
  echo "config/local.yaml is missing -- copy cluster/local.yaml.cluster-example to" >&2
  echo "config/local.yaml and edit it first (see cluster/README.md)." >&2
  exit 2
fi

EXTRACT_MAX_TASKS="${EXTRACT_MAX_TASKS:-300}"
echo "Max EXTRACT_TEXT_FACTS tasks this array task will claim: ${EXTRACT_MAX_TASKS}"

# --- Start a local LLM server for this array task (own GPU, own node). ---
# Identical logic to run_extraction.sbatch's own server-start block --
# see that script's comments for why each piece (HF_HUB_OFFLINE,
# VLLM_USE_FLASHINFER_SAMPLER, the LD_LIBRARY_PATH override) is there.
# Each array task runs on its own node with its own GPU, so binding to
# localhost:8000 per task never collides across array tasks.
LLM_BACKEND="${LLM_BACKEND:-vllm}"
SERVER_PID=""

if [ "${LLM_BACKEND}" = "ollama" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found on PATH. Either install it, set LLM_BACKEND=vllm, or set" >&2
    echo "LLM_BACKEND=external if you're using a different endpoint already running." >&2
    exit 2
  fi
  echo "Starting ollama serve..."
  ollama serve > "logs/ollama_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-local}.log" 2>&1 &
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
  vllm serve "${MODEL_NAME}" --host 0.0.0.0 --port 8000 > "logs/vllm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-local}.log" 2>&1 &
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

# enqueue-text-extraction-backfill is idempotent (Task.idempotency_key has
# a real DB-level UNIQUE constraint, and enqueue_task already catches the
# IntegrityError from a concurrent duplicate insert) -- safe for every
# array task to call this at roughly the same time.
echo "=== enqueueing EXTRACT_TEXT_FACTS backlog (safe to call from every array task) ==="
python -m fair_ocean_agent.cli enqueue-text-extraction-backfill

echo "=== claiming up to ${EXTRACT_MAX_TASKS} EXTRACT_TEXT_FACTS tasks (array task ${ARRAY_LABEL}) ==="
python -m fair_ocean_agent.cli worker --task-type EXTRACT_TEXT_FACTS --max-tasks "${EXTRACT_MAX_TASKS}"

echo "Job finished: $(date)"
echo "Array task ${ARRAY_LABEL} done. Once every array task finishes, run:"
echo "  sbatch --account=<acct> cluster/run_extraction.sbatch"
echo "to pick up any remainder plus supplements/mapping/validation/export."
