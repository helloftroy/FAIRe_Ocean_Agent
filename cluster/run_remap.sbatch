#!/bin/bash
# Re-map every study to FAIRe (gap 2, 2026-11-11), after curation and with
# the gap-2 mapping rules. scripts/remap_all.py settles every component
# (discovery must be stopped), cancels stale settle checks and queues one
# MAP_FAIRE per study; the worker then maps them. Resumable: re-submit after
# a timeout and only the remaining MAP_FAIRE tasks run (set SKIP_QUEUE=1 to
# go straight to the worker).
#
#   sbatch --account=191001-364393 cluster/run_remap.sbatch
#SBATCH --job-name=faire_remap
# Partition: morrill (CPU, no internet). service is the DTN, for downloads only.
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/remap_%j.out
#SBATCH --error=logs/remap_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export FAIR_OCEAN_SKIP_SQLITE_WAL=1
# shellcheck source=/dev/null
source cluster/env_activate.sh
export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
echo "Job started: $(date)  db: ${FAIR_OCEAN_DATABASE_URL}"

if squeue -u "$USER" -h -o "%j" | grep -qE "^faire_(discovery|discovery_parallel)$"; then
  echo "A discovery job is running: stop it first (components cannot be settled while it runs)." >&2
  exit 3
fi
python -m alembic -c alembic.ini upgrade head
if [ "${SKIP_QUEUE:-0}" != "1" ]; then
  python scripts/remap_all.py
fi
python -m fair_ocean_agent.cli worker --task-type MAP_FAIRE --until-empty --worker-id "remap-${SLURM_JOB_ID}"
python -m fair_ocean_agent.cli status
echo "Job finished: $(date)"
echo "Next: export-faire, then scripts/faire_strict/correct_to_faire.py (see correct_to_faire.md)."
