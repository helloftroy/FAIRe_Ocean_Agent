#!/bin/bash
# Read-only data-integrity audits on the live database (2026-11-10).
# Run after run_fetch_missing_biosamples.sbatch has finished, so the marine
# classification sees the newly fetched organism / host data.
# Changes nothing in the database. See data_integrity_audit.md.
#
# Writes data/audits/<date>/:
#   marine_audit.csv, marine_audit_summary.csv   (scripts/marine_audit.py)
#   bioproject_audit.csv                         (scripts/bioproject_audit.py)
#   inventory.csv, candidates.csv                (scripts/faire_strict/mapping_*.py)
#
# Submit from the repo root:
#   sbatch --account=191001-364393 cluster/run_audits.sbatch
#SBATCH --job-name=faire_audits
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/audits_%j.out
#SBATCH --error=logs/audits_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export FAIR_OCEAN_SKIP_SQLITE_WAL=1

# shellcheck source=/dev/null
source cluster/env_activate.sh
DB="${AUDIT_DB:-$(pwd)/data/fair_ocean.db}"
OUT="data/audits/$(date +%Y-%m-%d)"
mkdir -p "${OUT}"
echo "Job started: $(date)  host: $(hostname)  db: ${DB}  out: ${OUT}"

echo "=== marine audit (per sample, rolled up per study) ==="
python scripts/marine_audit.py "${DB}" "${OUT}"

echo "=== BioProject audit (studies with 2+ BioProjects) ==="
python scripts/bioproject_audit.py "${DB}" "${OUT}"

echo "=== mapping-rule inventory (slow: one pass over raw_facts) ==="
python scripts/faire_strict/mapping_inventory.py "${DB}" "${OUT}/inventory.csv"
python scripts/faire_strict/mapping_candidates.py "${DB}" "${OUT}/inventory.csv" "${OUT}/candidates.csv"

ls -la "${OUT}"
echo "Job finished: $(date)"
echo "Next: copy ${OUT}/ to the Mac (or paste the summary tables above) for review."
