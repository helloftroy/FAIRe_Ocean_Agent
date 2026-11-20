#!/bin/bash
# Profile MAP_FAIRE on one study (2026-11-20): maps it under cProfile and
# ROLLS BACK, nothing is saved. Run only while no remap job is running (it
# holds the database write lock while mapping).
#
#   sbatch --account=191001-364393 --export=ALL,STUDY=STUDY-xxxxxxxxxxxx cluster/run_profile_map.sbatch
#SBATCH --job-name=faire_profile_map
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/profile_map_%j.out
#SBATCH --error=logs/profile_map_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?run via sbatch}"
: "${STUDY:?set STUDY=STUDY-...}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export FAIR_OCEAN_SKIP_SQLITE_WAL=1
# shellcheck source=/dev/null
source cluster/env_activate.sh
export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
if squeue -u "$USER" -h -o "%j" | grep -qE "^faire_remap$"; then
  echo "A remap job is running: profile after it ends." >&2
  exit 3
fi
echo "Job started: $(date)  study: ${STUDY}"
python scripts/profile_map_faire.py --skip-report --profile "${STUDY}"
echo "Job finished: $(date)"
