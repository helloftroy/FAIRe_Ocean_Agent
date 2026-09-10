#!/bin/bash
# Companion to run_llm_troubleshoot.sbatch, for its network-only phases
# (select/rediscover/rediscover_worker/reset/enqueue/remap/report/export)
# -- everything except `worker` (EXTRACT_TEXT_FACTS, the only phase that
# actually calls an LLM). Real gap found live: submitting those phases
# through run_llm_troubleshoot.sbatch itself (partition=gpu-a100, per
# this cluster's own documented "GPU nodes have no internet access")
# sent `rediscover_worker`'s real, live NCBI/ENA/etc HTTPS calls out from
# a node with no route to the internet at all -- every single task
# failed with "timed out", repeatedly, for hours, never once succeeding.
# Same split this codebase already uses for the main pipeline
# (run_discovery.sbatch, network/CPU-only vs run_extraction.sbatch,
# GPU/no-network) -- this is that same split applied to the
# troubleshooting-batch harness specifically. See
# scripts/llm_troubleshooting_batch.py's own docstring for what each
# phase does, and run_llm_troubleshoot.sbatch's own header for the
# `worker`-inclusive phase combos this script deliberately refuses.
#
# Forcing a real DISCOVER_IDENTIFIERS re-run after a discovery-side code
# change, then remapping/reporting/exporting once it's done:
#   sbatch --account=<acct> --export=ALL,LLM_TEST_PHASES="rediscover rediscover_worker remap report export" cluster/run_llm_troubleshoot_rediscover.sbatch
#
# Picking a fresh batch of studies from the current test DB (no re-copy),
# network-only phases only:
#   sbatch --account=<acct> --export=ALL,LLM_TEST_PHASES="select rediscover rediscover_worker" cluster/run_llm_troubleshoot_rediscover.sbatch
#
# Other knobs: same MAIN_DB/LLM_TEST_DB/LLM_TEST_COUNT/LLM_TEST_MANIFEST/
# LLM_TEST_EXPORT_DIR/LLM_TEST_DB_REFRESH env vars as
# run_llm_troubleshoot.sbatch (both scripts share the same isolated test
# database and manifest file, so switching between them mid-troubleshoot
# is safe).
#SBATCH --job-name=llm_troubleshoot_rediscover
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/llm_troubleshoot_rediscover_%j.out
#SBATCH --error=logs/llm_troubleshoot_rediscover_%j.err

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

JOB_START_EPOCH=$(date +%s)
echo "Job started: $(date)"
echo "Host: $(hostname)"

if [ ! -f config/local.yaml ]; then
  echo "config/local.yaml is missing -- copy cluster/local.yaml.cluster-example to" >&2
  echo "config/local.yaml and edit it first (see cluster/README.md)." >&2
  exit 2
fi

MAIN_DB="${MAIN_DB:-data/fair_ocean.db}"
LLM_TEST_DB="${LLM_TEST_DB:-data/fair_ocean_llm_test.db}"
LLM_TEST_COUNT="${LLM_TEST_COUNT:-10}"
LLM_TEST_MANIFEST="${LLM_TEST_MANIFEST:-data/llm_troubleshooting_manifest.json}"
LLM_TEST_EXPORT_DIR="${LLM_TEST_EXPORT_DIR:-data/exports/llm_troubleshooting_batch}"
LLM_TEST_PHASES="${LLM_TEST_PHASES:-rediscover rediscover_worker remap report export}"

# This script never starts an LLM server (no GPU on this partition) --
# `worker` (EXTRACT_TEXT_FACTS) would just fail every task trying to
# reach a model backend that was never started. Use
# run_llm_troubleshoot.sbatch for any phase combo that includes it.
for phase in ${LLM_TEST_PHASES}; do
  if [ "${phase}" = "worker" ]; then
    echo "LLM_TEST_PHASES includes 'worker', which needs a real LLM backend -- this" >&2
    echo "script runs on the network-capable 'service' partition (no GPU, no LLM" >&2
    echo "server started). Use run_llm_troubleshoot.sbatch for 'worker'." >&2
    exit 2
  fi
done

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
  python3 -c "
import sqlite3
src = sqlite3.connect('${MAIN_DB}')
dst = sqlite3.connect('${LLM_TEST_DB}')
with dst:
    src.backup(dst)
src.close()
dst.close()
"
else
  echo "Reusing existing isolated test database ${LLM_TEST_DB} (set LLM_TEST_DB_REFRESH=1 to re-copy from ${MAIN_DB})."
fi

export FAIR_OCEAN_DATABASE_URL="sqlite:///$(pwd)/${LLM_TEST_DB}"
echo "Database (isolated copy, NOT the main pipeline database): ${FAIR_OCEAN_DATABASE_URL}"
python -m alembic -c alembic.ini upgrade head

echo "Phases to run: ${LLM_TEST_PHASES}"
for phase in ${LLM_TEST_PHASES}; do
  echo
  echo "=== phase: ${phase} ==="
  case "${phase}" in
    select)
      python scripts/llm_troubleshooting_batch.py select --count "${LLM_TEST_COUNT}" --manifest "${LLM_TEST_MANIFEST}"
      ;;
    rediscover)
      python scripts/llm_troubleshooting_batch.py rediscover --manifest "${LLM_TEST_MANIFEST}"
      ;;
    rediscover_worker)
      python -m fair_ocean_agent.cli worker --task-type DISCOVER_IDENTIFIERS --until-empty
      ;;
    reset)
      python scripts/llm_troubleshooting_batch.py reset --manifest "${LLM_TEST_MANIFEST}"
      ;;
    enqueue)
      python scripts/llm_troubleshooting_batch.py enqueue --manifest "${LLM_TEST_MANIFEST}"
      ;;
    remap)
      python scripts/llm_troubleshooting_batch.py remap --manifest "${LLM_TEST_MANIFEST}"
      ;;
    report)
      python scripts/llm_troubleshooting_batch.py report --manifest "${LLM_TEST_MANIFEST}"
      ;;
    export)
      python scripts/llm_troubleshooting_batch.py export --manifest "${LLM_TEST_MANIFEST}" --output-dir "${LLM_TEST_EXPORT_DIR}"
      ;;
    *)
      echo "Unknown phase '${phase}' in LLM_TEST_PHASES -- valid: select rediscover rediscover_worker reset enqueue remap report export" >&2
      exit 2
      ;;
  esac
done

echo
echo "Job finished: $(date)"
echo "FAIRe CSVs (if the export phase ran): ${LLM_TEST_EXPORT_DIR} -- scp that directory back, e.g.:"
echo "  scp -r <cluster>:$(pwd)/${LLM_TEST_EXPORT_DIR} ./"
echo "Isolated test database: ${LLM_TEST_DB} (main pipeline database untouched: ${MAIN_DB})"
echo "Manifest: ${LLM_TEST_MANIFEST}"
JOB_ELAPSED_SECONDS=$(($(date +%s) - JOB_START_EPOCH))
echo "Total job time: $((JOB_ELAPSED_SECONDS / 3600))h $(((JOB_ELAPSED_SECONDS % 3600) / 60))m $((JOB_ELAPSED_SECONDS % 60))s (${JOB_ELAPSED_SECONDS}s)"
