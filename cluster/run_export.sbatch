#!/bin/bash
# Export the database to FAIRe, then the strict FAIRe v1.0.2 copy and the
# fill summary (2026-11-12). No internet needed.
#
#   sbatch --account=191001-364393 --export=ALL,EXPORT_NAME=faire_2026-11-12 cluster/run_export.sbatch
#
# Writes data/exports/<EXPORT_NAME>/ (pipeline export) and
# data/exports/<EXPORT_NAME>_strict/ (correct_to_faire.py, change log inside).
# If MISSING_PDFS (default data/missing_pdfs.csv) exists, also writes
# <EXPORT_NAME>_strict/study_access.csv and fill_summary.csv; plot the figure
# locally with scripts/faire_strict/plot_fill.py (Arial).
#SBATCH --job-name=faire_export
# Partition: morrill (CPU, no internet). service is the DTN, for downloads only.
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export FAIR_OCEAN_SKIP_SQLITE_WAL=1
# shellcheck source=/dev/null
source cluster/env_activate.sh
DB_FILE="${DB_FILE:-$(pwd)/data/fair_ocean.db}"
export FAIR_OCEAN_DATABASE_URL="sqlite:///${DB_FILE}"
EXPORT_NAME="${EXPORT_NAME:-faire_$(date +%Y-%m-%d)}"
OUT="data/exports/${EXPORT_NAME}"
STRICT="${OUT}_strict"
MISSING_PDFS="${MISSING_PDFS:-data/missing_pdfs.csv}"
echo "Job started: $(date)  db: ${DB_FILE}  out: ${OUT}"

if squeue -u "$USER" -h -o "%j" | grep -qE "^faire_(remap|extraction|curation|fill_capped)$"; then
  echo "A job that writes the database is still running: export after it finishes." >&2
  exit 3
fi

echo "=== export-faire ==="
python -m fair_ocean_agent.cli export-faire --output "${OUT}"
echo "=== strict FAIRe copy ==="
python scripts/faire_strict/correct_to_faire.py "${OUT}" "${STRICT}"
if [ -f "${MISSING_PDFS}" ]; then
  echo "=== fill summary ==="
  python scripts/faire_strict/study_access.py "${DB_FILE}" "${MISSING_PDFS}" "${STRICT}/study_access.csv"
  python scripts/faire_strict/fill_summary.py "${STRICT}" "${STRICT}/fill_summary.csv" "${STRICT}/study_access.csv"
else
  echo "No ${MISSING_PDFS}: skipped study_access / fill_summary."
fi
echo "Job finished: $(date)"
