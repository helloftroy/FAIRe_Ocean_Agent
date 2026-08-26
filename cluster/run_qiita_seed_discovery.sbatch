#!/bin/bash
# Runs the standalone Qiita seed-discovery/enrichment staging pipeline on
# the service partition. This writes only the Qiita staging database by
# default (data/seed_discovery/qiita_paper_seeds.sqlite) plus raw Qiita
# metadata under data/qiita/. It does NOT ingest into the main FAIRe DB.
#
# Resumable/idempotent: re-submitting after walltime continues by reusing
# already-downloaded study/sample/prep files unless QIITA_REFRESH=1 is set.
# SQLite upserts make reruns safe. Do not run two copies of this script
# against the same QIITA_DB at the same time.
#
# The MGnify database is read only for the overlap report. For the least
# SQLite drama, avoid running this overlap phase at the exact same time as
# run_seed_discovery_marathon.sbatch if both point at the same MGnify DB.
#
# Override knobs, for example:
#   sbatch --export=ALL,QIITA_MIN_REQUEST_INTERVAL_SECONDS=5 cluster/run_qiita_seed_discovery.sbatch
#   sbatch --export=ALL,QIITA_MAX_STUDIES=100 cluster/run_qiita_seed_discovery.sbatch
#   sbatch --export=ALL,QIITA_PHASE=reports cluster/run_qiita_seed_discovery.sbatch
#   sbatch --export=ALL,QIITA_REFRESH=1 cluster/run_qiita_seed_discovery.sbatch
#SBATCH --job-name=qiita_seed_discovery
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qiita_seed_discovery_%j.out
#SBATCH --error=logs/qiita_seed_discovery_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

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

QIITA_DB="${QIITA_DB:-data/seed_discovery/qiita_paper_seeds.sqlite}"
QIITA_DATA_DIR="${QIITA_DATA_DIR:-data/qiita}"
MGNIFY_DB="${MGNIFY_DB:-data/seed_discovery/mgnify_paper_seeds.sqlite}"
QIITA_PHASE="${QIITA_PHASE:-all}"
QIITA_MIN_REQUEST_INTERVAL_SECONDS="${QIITA_MIN_REQUEST_INTERVAL_SECONDS:-2}"

echo "Qiita database: ${QIITA_DB}"
echo "Qiita data dir: ${QIITA_DATA_DIR}"
echo "MGnify overlap database: ${MGNIFY_DB}"
echo "Qiita phase: ${QIITA_PHASE}"
echo "Qiita request interval seconds: ${QIITA_MIN_REQUEST_INTERVAL_SECONDS}"

MAX_STUDIES_ARGS=()
if [ -n "${QIITA_MAX_STUDIES:-}" ]; then
  MAX_STUDIES_ARGS=(--max-studies "${QIITA_MAX_STUDIES}")
  echo "Max studies this run: ${QIITA_MAX_STUDIES}"
else
  echo "Max studies this run: unlimited (all public Qiita studies visible to the crawler)"
fi

REFRESH_ARGS=()
if [ "${QIITA_REFRESH:-0}" = "1" ]; then
  REFRESH_ARGS=(--refresh)
  echo "Refresh: enabled; cached Qiita files will be re-downloaded."
else
  echo "Refresh: disabled; cached Qiita files will be reused."
fi

"${PYTHON_BIN}" run_qiita_seed_discovery.py \
  --db "${QIITA_DB}" \
  --data-dir "${QIITA_DATA_DIR}" \
  --mgnify-db "${MGNIFY_DB}" \
  --phase "${QIITA_PHASE}" \
  --qiita-min-request-interval-seconds "${QIITA_MIN_REQUEST_INTERVAL_SECONDS}" \
  "${MAX_STUDIES_ARGS[@]}" \
  "${REFRESH_ARGS[@]}"

echo "=== Qiita output locations ==="
if [ -f "${QIITA_DATA_DIR}/OUTPUT_LOCATIONS.json" ]; then
  cat "${QIITA_DATA_DIR}/OUTPUT_LOCATIONS.json"
else
  echo "${QIITA_DATA_DIR}/OUTPUT_LOCATIONS.json not found"
fi

echo
echo "=== Qiita SQLite summary ==="
"${PYTHON_BIN}" - <<PY
import sqlite3
from pathlib import Path

db_path = Path("${QIITA_DB}")
print(f"qiita_db={db_path} exists={db_path.exists()} size_mb={(db_path.stat().st_size / 1024 / 1024):.1f}" if db_path.exists() else f"qiita_db={db_path} missing")
if db_path.exists():
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    for table in ("qiita_studies", "qiita_samples", "qiita_preparations"):
        print(f"{table}: {con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}")
    print("marine_confidence:")
    for row in con.execute("SELECT marine_confidence, count(*) AS n FROM qiita_studies GROUP BY marine_confidence ORDER BY n DESC"):
        print(f"  {row['marine_confidence']}: {row['n']}")
    print("sequence_accessibility_status:")
    for row in con.execute("SELECT sequence_accessibility_status, count(*) AS n FROM qiita_studies GROUP BY sequence_accessibility_status ORDER BY n DESC"):
        print(f"  {row['sequence_accessibility_status']}: {row['n']}")
    print(f"paper_seeds: {con.execute('SELECT count(*) FROM paper_seeds').fetchone()[0]}")
    con.close()
PY

echo "Job finished: $(date)"
