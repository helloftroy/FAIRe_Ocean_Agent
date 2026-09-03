#!/bin/bash
# Runs the standalone MG-RAST seed-discovery/enrichment staging pipeline.
# This writes only the MG-RAST staging database plus raw MG-RAST metadata;
# it does not ingest into the main FAIRe database.
#
# Resumable/idempotent: re-submitting continues by reusing cached project,
# metadata, and download-manifest JSON unless MGRAST_REFRESH=1 is set.
# Do not run two copies against the same MGRAST_DB at the same time.
#
# Override knobs:
#   sbatch --export=ALL,MGRAST_MAX_PROJECTS=100 cluster/run_mgrast_discovery.sbatch
#   sbatch --export=ALL,MGRAST_PHASE=reports cluster/run_mgrast_discovery.sbatch
#   sbatch --export=ALL,MGRAST_MIN_REQUEST_INTERVAL_SECONDS=5 cluster/run_mgrast_discovery.sbatch
#   sbatch --export=ALL,MGRAST_REFRESH=1 cluster/run_mgrast_discovery.sbatch
#SBATCH --job-name=mgrast_discovery
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mgrast_discovery_%j.out
#SBATCH --error=logs/mgrast_discovery_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
# Python defaults to fully-buffered stdout when it isn't a TTY (always
# true under sbatch, since output goes to a log file) -- a long-running
# job's own periodic progress logging can sit in that buffer for its
# entire runtime with nothing ever reaching the .out file, looking
# exactly like a silently-stuck job even though it's working fine.
export PYTHONUNBUFFERED=1

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON_BIN="$(command -v python3)"
: "${PYTHON_BIN:?python3 not found on PATH after sourcing cluster/env_activate.sh.}"
echo "PYTHON_BIN: ${PYTHON_BIN}"

MGRAST_DB="${MGRAST_DB:-data/seed_discovery/mgrast_paper_seeds.sqlite}"
MGRAST_DATA_DIR="${MGRAST_DATA_DIR:-data/mgrast}"
MGRAST_OUT_CSV="${MGRAST_OUT_CSV:-cluster/seeds_mgrast.csv}"
MGRAST_PHASE="${MGRAST_PHASE:-all}"
MGRAST_MIN_REQUEST_INTERVAL_SECONDS="${MGRAST_MIN_REQUEST_INTERVAL_SECONDS:-2}"
MGNIFY_DB="${MGNIFY_DB:-data/seed_discovery/mgnify_paper_seeds.sqlite}"
QIITA_DB="${QIITA_DB:-data/seed_discovery/qiita_paper_seeds.sqlite}"
GOLD_DB="${GOLD_DB:-data/jgi_gold/gold_sharded.sqlite}"
CNCB_DB="${CNCB_DB:-data/seed_discovery/cncb_gsa_paper_seeds.sqlite}"

echo "MG-RAST database: ${MGRAST_DB}"
echo "MG-RAST data dir: ${MGRAST_DATA_DIR}"
echo "MG-RAST phase: ${MGRAST_PHASE}"
echo "MG-RAST request interval seconds: ${MGRAST_MIN_REQUEST_INTERVAL_SECONDS}"

MAX_PROJECTS_ARGS=()
if [ -n "${MGRAST_MAX_PROJECTS:-}" ]; then
  MAX_PROJECTS_ARGS=(--max-projects "${MGRAST_MAX_PROJECTS}")
  echo "Max projects this run: ${MGRAST_MAX_PROJECTS}"
else
  echo "Max projects this run: unlimited"
fi

REFRESH_ARGS=()
if [ "${MGRAST_REFRESH:-0}" = "1" ]; then
  REFRESH_ARGS=(--refresh)
  echo "Refresh: enabled"
else
  echo "Refresh: disabled"
fi

"${PYTHON_BIN}" run_mgrast_seed_discovery.py \
  --db "${MGRAST_DB}" \
  --data-dir "${MGRAST_DATA_DIR}" \
  --mgnify-db "${MGNIFY_DB}" \
  --qiita-db "${QIITA_DB}" \
  --gold-db "${GOLD_DB}" \
  --cncb-db "${CNCB_DB}" \
  --phase "${MGRAST_PHASE}" \
  --mgrast-min-request-interval-seconds "${MGRAST_MIN_REQUEST_INTERVAL_SECONDS}" \
  "${MAX_PROJECTS_ARGS[@]}" \
  "${REFRESH_ARGS[@]}"

"${PYTHON_BIN}" scripts/mgrast_seeds_to_csv.py \
  --db "${MGRAST_DB}" \
  --out "${MGRAST_OUT_CSV}"

echo "=== MG-RAST output locations ==="
if [ -f "${MGRAST_DATA_DIR}/OUTPUT_LOCATIONS.json" ]; then
  cat "${MGRAST_DATA_DIR}/OUTPUT_LOCATIONS.json"
else
  echo "${MGRAST_DATA_DIR}/OUTPUT_LOCATIONS.json not found"
fi

echo
echo "=== MG-RAST SQLite summary ==="
"${PYTHON_BIN}" - <<PY
import sqlite3
from pathlib import Path

db_path = Path("${MGRAST_DB}")
print(f"mgrast_db={db_path} exists={db_path.exists()} size_mb={(db_path.stat().st_size / 1024 / 1024):.1f}" if db_path.exists() else f"mgrast_db={db_path} missing")
if db_path.exists():
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    for table in ("mgrast_projects", "mgrast_samples", "mgrast_datasets"):
        print(f"{table}: {con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}")
    print("marine_confidence:")
    for row in con.execute("SELECT marine_confidence, count(*) AS n FROM mgrast_projects GROUP BY marine_confidence ORDER BY n DESC"):
        print(f"  {row['marine_confidence']}: {row['n']}")
    print("sequence_accessibility_status:")
    for row in con.execute("SELECT sequence_accessibility_status, count(*) AS n FROM mgrast_projects GROUP BY sequence_accessibility_status ORDER BY n DESC"):
        print(f"  {row['sequence_accessibility_status']}: {row['n']}")
    print(f"paper_seeds: {con.execute('SELECT count(*) FROM paper_seeds').fetchone()[0]}")
    con.close()
PY

echo "Seed CSV: ${MGRAST_OUT_CSV}"
echo "Job finished: $(date)"
