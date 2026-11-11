#!/bin/bash
# Read-only spot-check of a curation dry-run plan (2026-11-11):
# scripts/curation_spotcheck.py writes spotcheck_*.tsv into the plan folder.
#   sbatch --account=191001-364393 --export=ALL,PLAN_DIR=data/audits/curation-2026-11-11/plan cluster/run_curation_spotcheck.sbatch
#SBATCH --job-name=faire_spotcheck
# Partition: morrill (CPU, no internet).
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/spotcheck_%j.out
#SBATCH --error=logs/spotcheck_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
# shellcheck source=/dev/null
source cluster/env_activate.sh
DB_FILE="${DB_FILE:-$(pwd)/data/fair_ocean.db}"
: "${PLAN_DIR:?set PLAN_DIR to the dry-run plan folder, e.g. data/audits/curation-2026-11-11/plan}"
echo "Job started: $(date)  db: ${DB_FILE}  plan: ${PLAN_DIR}"
python scripts/curation_spotcheck.py "${DB_FILE}" "${PLAN_DIR}"
echo "Job finished: $(date)"
