#!/bin/bash
# Runs the standalone CNCB/GSA seed-discovery/enrichment staging pipeline
# on the service partition. This writes only the CNCB/GSA staging database
# by default (data/seed_discovery/cncb_gsa_paper_seeds.sqlite), raw source
# metadata under data/cncb_gsa/, and a standard seed CSV at
# cluster/seeds_cncb_gsa.csv. It does NOT ingest into the main FAIRe DB.
#
# Resumable/idempotent: re-submitting after walltime reuses already cached
# search/project/sample pages unless CNCB_REFRESH=1 is set. SQLite upserts
# make reruns safe. Do not run two copies against the same CNCB_DB.
#
# Override knobs, for example:
#   sbatch --export=ALL,CNCB_MIN_REQUEST_INTERVAL_SECONDS=5 cluster/run_cncb_gsa_discovery.sbatch
#   sbatch --export=ALL,CNCB_MAX_PROJECTS=100 cluster/run_cncb_gsa_discovery.sbatch
#   sbatch --export=ALL,CNCB_MAX_PAGES_PER_QUERY=50 cluster/run_cncb_gsa_discovery.sbatch  # quick/shallow test run; each page is a fixed 10 records regardless of size requested
#   sbatch --export=ALL,CNCB_REFRESH=1 cluster/run_cncb_gsa_discovery.sbatch
#SBATCH --job-name=cncb_gsa_discovery
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cncb_gsa_discovery_%j.out
#SBATCH --error=logs/cncb_gsa_discovery_%j.err

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
: "${PYTHON_BIN:?python3 not found on PATH after sourcing cluster/env_activate.sh -- check your conda env activated correctly.}"
echo "PYTHON_BIN: ${PYTHON_BIN}"

CNCB_DB="${CNCB_DB:-data/seed_discovery/cncb_gsa_paper_seeds.sqlite}"
CNCB_DATA_DIR="${CNCB_DATA_DIR:-data/cncb_gsa}"
CNCB_OUT_CSV="${CNCB_OUT_CSV:-cluster/seeds_cncb_gsa.csv}"
CNCB_MIN_REQUEST_INTERVAL_SECONDS="${CNCB_MIN_REQUEST_INTERVAL_SECONDS:-2}"
CNCB_MAX_PAGES_PER_QUERY="${CNCB_MAX_PAGES_PER_QUERY:-5000}"

echo "CNCB database: ${CNCB_DB}"
echo "CNCB data dir: ${CNCB_DATA_DIR}"
echo "CNCB output CSV: ${CNCB_OUT_CSV}"
echo "CNCB request interval seconds: ${CNCB_MIN_REQUEST_INTERVAL_SECONDS}"
echo "CNCB max pages per query: ${CNCB_MAX_PAGES_PER_QUERY}"

MAX_PROJECTS_ARGS=()
if [ -n "${CNCB_MAX_PROJECTS:-}" ]; then
  MAX_PROJECTS_ARGS=(--max-projects "${CNCB_MAX_PROJECTS}")
  echo "Max projects this run: ${CNCB_MAX_PROJECTS}"
else
  echo "Max projects this run: unlimited within CNCB_MAX_PAGES_PER_QUERY per search term"
fi

REFRESH_ARGS=()
if [ "${CNCB_REFRESH:-0}" = "1" ]; then
  REFRESH_ARGS=(--refresh)
  echo "Refresh: enabled; cached CNCB files will be re-downloaded."
else
  echo "Refresh: disabled; cached CNCB files will be reused."
fi

"${PYTHON_BIN}" run_cncb_gsa_seed_discovery.py \
  --phase all \
  --db "${CNCB_DB}" \
  --data-dir "${CNCB_DATA_DIR}" \
  --cncb-min-request-interval-seconds "${CNCB_MIN_REQUEST_INTERVAL_SECONDS}" \
  --max-pages-per-query "${CNCB_MAX_PAGES_PER_QUERY}" \
  "${MAX_PROJECTS_ARGS[@]}" \
  "${REFRESH_ARGS[@]}"

"${PYTHON_BIN}" scripts/cncb_gsa_seeds_to_csv.py \
  --db "${CNCB_DB}" \
  --out "${CNCB_OUT_CSV}"

echo "=== CNCB/GSA output locations ==="
if [ -f "${CNCB_DATA_DIR}/OUTPUT_LOCATIONS.json" ]; then
  cat "${CNCB_DATA_DIR}/OUTPUT_LOCATIONS.json"
else
  echo "${CNCB_DATA_DIR}/OUTPUT_LOCATIONS.json not found"
fi

echo
echo "=== CNCB/GSA SQLite summary ==="
"${PYTHON_BIN}" - <<PY
import sqlite3
from pathlib import Path

db_path = Path("${CNCB_DB}")
print(f"cncb_db={db_path} exists={db_path.exists()} size_mb={(db_path.stat().st_size / 1024 / 1024):.1f}" if db_path.exists() else f"cncb_db={db_path} missing")
if db_path.exists():
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    for table in ("cncb_projects", "cncb_samples", "cncb_experiments"):
        print(f"{table}: {con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}")
    print("marine_confidence:")
    for row in con.execute("SELECT marine_confidence, count(*) AS n FROM cncb_projects GROUP BY marine_confidence ORDER BY n DESC"):
        print(f"  {row['marine_confidence']}: {row['n']}")
    print("sequence_accessibility_status:")
    for row in con.execute("SELECT sequence_accessibility_status, count(*) AS n FROM cncb_projects GROUP BY sequence_accessibility_status ORDER BY n DESC"):
        print(f"  {row['sequence_accessibility_status']}: {row['n']}")
    print(f"paper_seeds: {con.execute('SELECT count(*) FROM paper_seeds').fetchone()[0]}")
    con.close()
PY

echo "Job finished: $(date)"
