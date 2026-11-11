#!/bin/bash
# Fill BioSamples (past the 300 cap) and ENA runs (past the 500 cap) for the
# projects that survived curation (2026-11-11). scripts/fill_capped_projects.py.
# Order: AFTER run_curation.sbatch with CURATION_APPLY=1, BEFORE run_remap.sbatch.
# Resumable: re-submit after a timeout; what is already in the database is skipped.
#
#   sbatch --account=191001-364393 cluster/run_fill_capped.sbatch
# Needs the internet, so service (the DTN). Light on memory.
#SBATCH --job-name=faire_fill_capped
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fill_capped_%j.out
#SBATCH --error=logs/fill_capped_%j.err

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
echo "Job started: $(date)  host: $(hostname)  db: ${DB_FILE}"

applied=$(python - "${DB_FILE}" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
t = c.execute("select 1 from sqlite_master where type='table' and name='curation_exclusions'").fetchone()
print(c.execute("select count(*) from curation_exclusions").fetchone()[0] if t else 0)
PY
)
if [ "${applied}" = "0" ]; then
  echo "Curation has not been applied (curation_exclusions is empty): run run_curation.sbatch with CURATION_APPLY=1 first." >&2
  exit 3
fi
python scripts/fill_capped_projects.py --dry-run
python scripts/fill_capped_projects.py ${FILL_ARGS:-}
echo "Job finished: $(date)"
echo "Next: run_remap.sbatch (morrill)."
