#!/bin/bash
# Curation (2026-11-11): remove non-marine and foreign material, flag
# non-marine studies. scripts/apply_curation.py; rules and decisions in
# data_integrity_audit.md. Needs the audit folder from run_audits.sbatch
# (same code version) and curation/bioproject_review_2026-11-11.csv.
#
# Two stages:
#   1. dry run (default): writes the plan to data/audits/<audit date>/plan/
#      and prints the summary. Nothing in the database changes.
#        sbatch --account=191001-364393 --export=ALL,AUDIT_DIR=data/audits/2026-11-12 cluster/run_curation.sbatch
#   2. apply, after reviewing the dry-run summary:
#        sbatch --account=191001-364393 --export=ALL,AUDIT_DIR=data/audits/2026-11-12,CURATION_APPLY=1 cluster/run_curation.sbatch
#      Refuses to start while a discovery job runs. Copies the database to
#      data/fair_ocean.db.bak-pre-curation-<date> first (set SKIP_BACKUP=1
#      only if you made your own copy).
#SBATCH --job-name=faire_curation
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/curation_%j.out
#SBATCH --error=logs/curation_%j.err

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
: "${AUDIT_DIR:?set AUDIT_DIR to the run_audits.sbatch output folder, e.g. data/audits/2026-11-12}"
REVIEW="${REVIEW:-curation/bioproject_review_2026-11-11.csv}"
[ -f "${AUDIT_DIR}/bioproject_audit.csv" ] || { echo "missing ${AUDIT_DIR}/bioproject_audit.csv" >&2; exit 2; }
echo "Job started: $(date)  db: ${DB_FILE}  audit: ${AUDIT_DIR}"

if [ "${CURATION_APPLY:-0}" = "1" ]; then
  if squeue -u "$USER" -h -o "%j %T" | grep -E "^faire_(discovery|discovery_parallel|biosample_fill) " | grep -v "^faire_curation"; then
    echo "A discovery or BioSample job is still running (above): stop it first." >&2
    exit 3
  fi
  python -m alembic -c alembic.ini upgrade head
  if [ "${SKIP_BACKUP:-0}" != "1" ]; then
    need=$(stat -c %s "${DB_FILE}")
    free=$(df --output=avail -B1 "$(dirname "${DB_FILE}")" | tail -1)
    if [ "${free}" -lt $((need + need / 10)) ]; then
      echo "Not enough space for a backup copy (${need} bytes needed, ${free} free)." >&2
      exit 4
    fi
    BAK="${DB_FILE}.bak-pre-curation-$(date +%Y%m%d)"
    echo "Backing up to ${BAK}"
    cp "${DB_FILE}" "${BAK}"
  fi
  python scripts/apply_curation.py --db "${DB_FILE}" --audit "${AUDIT_DIR}" --review "${REVIEW}" \
    --out "${AUDIT_DIR}/plan_applied" --apply
else
  python scripts/apply_curation.py --db "${DB_FILE}" --audit "${AUDIT_DIR}" --review "${REVIEW}" \
    --out "${AUDIT_DIR}/plan"
fi
echo "Job finished: $(date)"
