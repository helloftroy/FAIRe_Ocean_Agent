#!/usr/bin/env bash
set -euo pipefail

# Sequential long-running seed-discovery driver for a local Mac terminal.
# Runs only one writer against data/seed_discovery/mgnify_paper_seeds.sqlite
# at a time, then prints DB state after each stage.
#
# Override knobs, for example:
#   EPMC_MODE=no-download ENA_MAX_PAGES=200 ENA_PUBLICATION_MAX_STUDIES=5000 \
#     scripts/run_seed_discovery_marathon.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
SEED_DB="${SEED_DB:-data/seed_discovery/mgnify_paper_seeds.sqlite}"
OUT_CSV="${OUT_CSV:-cluster/seeds_seed_discovery.csv}"
LOG_DIR="${LOG_DIR:-logs/seed_discovery_marathon}"
mkdir -p "$LOG_DIR"

# Defaults are intentionally large enough for an unattended run, but bounded
# so a single publication-resolution stage does not monopolize the machine.
# Set any of these to "none" or empty to omit that max flag.
MGNIFY_DISCOVERY_MAX_PAGES="${MGNIFY_DISCOVERY_MAX_PAGES:-none}"
MGNIFY_DISCOVERY_MAX_STUDIES="${MGNIFY_DISCOVERY_MAX_STUDIES:-none}"
MGNIFY_RESOLVE_MAX_STUDIES="${MGNIFY_RESOLVE_MAX_STUDIES:-10000}"
ENA_MAX_PAGES="${ENA_MAX_PAGES:-500}"
ENA_PUBLICATION_MAX_STUDIES="${ENA_PUBLICATION_MAX_STUDIES:-10000}"
EPMC_MODE="${EPMC_MODE:-bootstrap}" # bootstrap, no-download, or skip

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

optional_max_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" && "$value" != "none" ]]; then
    printf '%s\n%s\n' "$flag" "$value"
  fi
}

print_state() {
  echo
  echo "===== DATABASE STATE: $(timestamp) ====="
  "$PYTHON_BIN" - <<'PY'
import sqlite3
from pathlib import Path

db_path = Path("data/seed_discovery/mgnify_paper_seeds.sqlite")
print(f"seed_db={db_path} exists={db_path.exists()} size_mb={(db_path.stat().st_size / 1024 / 1024):.1f}" if db_path.exists() else f"seed_db={db_path} missing")
for suffix in ("-journal", "-wal", "-shm"):
    side = Path(str(db_path) + suffix)
    if side.exists():
        print(f"{side.name}: {side.stat().st_size} bytes")
if not db_path.exists():
    raise SystemExit(0)

con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
con.row_factory = sqlite3.Row
tables = {row["name"] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
for table in (
    "mgnify_studies",
    "ena_studies",
    "ena_runs",
    "publication_candidates",
    "epmc_accession_links",
    "epmc_article_ids",
):
    if table in tables:
        print(f"{table}: {con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}")

if "mgnify_studies" in tables:
    print("mgnify_status:")
    for row in con.execute(
        "SELECT publication_resolution_status AS status, count(*) AS n "
        "FROM mgnify_studies GROUP BY 1 ORDER BY n DESC"
    ):
        print(f"  {row['status']}: {row['n']}")
if "ena_studies" in tables:
    print("ena_status:")
    for row in con.execute(
        "SELECT publication_resolution_status AS status, count(*) AS n "
        "FROM ena_studies GROUP BY 1 ORDER BY n DESC"
    ):
        print(f"  {row['status']}: {row['n']}")
if "publication_candidates" in tables:
    row = con.execute(
        "SELECT count(*) AS rows, count(DISTINCT lower(doi)) AS distinct_doi "
        "FROM publication_candidates WHERE doi IS NOT NULL AND doi != ''"
    ).fetchone()
    print(f"publication_candidates_with_doi: {row['rows']}")
    print(f"publication_candidates_distinct_doi: {row['distinct_doi']}")
if "crawl_state" in tables:
    print("recent_crawl_state:")
    for row in con.execute(
        "SELECT source, status, updated_at FROM crawl_state "
        "ORDER BY updated_at DESC LIMIT 12"
    ):
        print(f"  {row['updated_at']} {row['source']}: {row['status']}")
con.close()

csv_path = Path("cluster/seeds_seed_discovery.csv")
if csv_path.exists():
    print(f"export_csv={csv_path} size_mb={(csv_path.stat().st_size / 1024 / 1024):.1f}")
PY
  echo "===== END DATABASE STATE ====="
  echo
}

run_step() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${name}.log"
  echo
  echo "===== START ${name}: $(timestamp) ====="
  printf 'Command:'
  printf ' %q' "$@"
  echo
  "$@" 2>&1 | tee "$log_file"
  local status="${PIPESTATUS[0]}"
  echo "===== END ${name}: $(timestamp) status=${status} log=${log_file} ====="
  print_state
  return "$status"
}

echo "Seed discovery marathon starting at $(timestamp)"
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SEED_DB=$SEED_DB"
echo "OUT_CSV=$OUT_CSV"
echo "LOG_DIR=$LOG_DIR"
echo "EPMC_MODE=$EPMC_MODE"
echo "MGNIFY_DISCOVERY_MAX_PAGES=$MGNIFY_DISCOVERY_MAX_PAGES"
echo "MGNIFY_DISCOVERY_MAX_STUDIES=$MGNIFY_DISCOVERY_MAX_STUDIES"
echo "MGNIFY_RESOLVE_MAX_STUDIES=$MGNIFY_RESOLVE_MAX_STUDIES"
echo "ENA_MAX_PAGES=$ENA_MAX_PAGES"
echo "ENA_PUBLICATION_MAX_STUDIES=$ENA_PUBLICATION_MAX_STUDIES"
print_state

if [[ "$EPMC_MODE" == "bootstrap" ]]; then
  run_step "epmc_bootstrap" "$PYTHON_BIN" update_epmc_accession_index.py --bootstrap --db "$SEED_DB"
elif [[ "$EPMC_MODE" == "no-download" ]]; then
  run_step "epmc_ingest_local" "$PYTHON_BIN" update_epmc_accession_index.py --no-download --db "$SEED_DB"
elif [[ "$EPMC_MODE" == "skip" ]]; then
  echo "Skipping Europe PMC bulk update."
else
  echo "Unknown EPMC_MODE=$EPMC_MODE; expected bootstrap, no-download, or skip" >&2
  exit 2
fi

MGNIFY_DISCOVERY_CMD=("$PYTHON_BIN" run_mgnify_seed_discovery.py --no-openalex --db "$SEED_DB")
while IFS= read -r arg; do MGNIFY_DISCOVERY_CMD+=("$arg"); done < <(optional_max_arg --max-pages "$MGNIFY_DISCOVERY_MAX_PAGES")
while IFS= read -r arg; do MGNIFY_DISCOVERY_CMD+=("$arg"); done < <(optional_max_arg --max-studies "$MGNIFY_DISCOVERY_MAX_STUDIES")
run_step "mgnify_discovery" "${MGNIFY_DISCOVERY_CMD[@]}"

MGNIFY_RESOLVE_CMD=("$PYTHON_BIN" run_mgnify_seed_discovery.py --resolve-only --no-openalex --db "$SEED_DB")
while IFS= read -r arg; do MGNIFY_RESOLVE_CMD+=("$arg"); done < <(optional_max_arg --max-studies "$MGNIFY_RESOLVE_MAX_STUDIES")
run_step "mgnify_publications" "${MGNIFY_RESOLVE_CMD[@]}"

ENA_DISCOVERY_CMD=("$PYTHON_BIN" run_ena_seed_discovery.py --date-shards --no-openalex --phase discovery --db "$SEED_DB")
while IFS= read -r arg; do ENA_DISCOVERY_CMD+=("$arg"); done < <(optional_max_arg --max-pages "$ENA_MAX_PAGES")
run_step "ena_discovery" "${ENA_DISCOVERY_CMD[@]}"

ENA_PUBLICATIONS_CMD=("$PYTHON_BIN" run_ena_seed_discovery.py --phase publications --no-openalex --db "$SEED_DB")
while IFS= read -r arg; do ENA_PUBLICATIONS_CMD+=("$arg"); done < <(optional_max_arg --max-studies "$ENA_PUBLICATION_MAX_STUDIES")
run_step "ena_publications" "${ENA_PUBLICATIONS_CMD[@]}"

run_step "export_seed_csv" "$PYTHON_BIN" scripts/seed_discovery_to_csv.py --db "$SEED_DB" --out "$OUT_CSV"

echo "Seed discovery marathon complete at $(timestamp)"
echo "Combined MGnify+ENA seed CSV: $OUT_CSV"
