#!/bin/bash
# Runs ONLY the ENA discovery phase (finding candidate marine ENA studies
# in the first place) to completion -- separate from
# run_ena_publication_resolution.sbatch, which only ever resolves
# publications for studies discovery has already found.
#
# Confirmed live before building this: ENA discovery's own crawl_state
# was "stopped" (not "completed"), with most first_public month-shards
# (2008 through today, --date-shards) still untouched -- publication
# resolution alone was never going to make ENA discovery itself
# complete, since --phase publications never scans for new studies.
#
# NEVER run this at the same time as run_ena_publication_resolution.sbatch,
# run_mgnify_publication_resolution.sbatch, or
# run_seed_discovery_marathon.sbatch -- all four write to the same
# mgnify_paper_seeds.sqlite file, and SQLite only allows one writer at a
# time.
#
# Fully resumable via --resume (default) and --date-shards' own per-shard
# crawl_state -- resubmitting after a walltime cutoff picks up whichever
# month-shards aren't yet completed.
#
# Override knobs, for example:
#   sbatch --export=ALL,ENA_MAX_PAGES=2000 cluster/run_ena_discovery.sbatch
#SBATCH --job-name=ena_discovery
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ena_discovery_%j.out
#SBATCH --error=logs/ena_discovery_%j.err

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

SEED_DB="${SEED_DB:-data/seed_discovery/mgnify_paper_seeds.sqlite}"
echo "Seed database: ${SEED_DB}"

MAX_PAGES_ARGS=()
if [ -n "${ENA_MAX_PAGES:-}" ]; then
  MAX_PAGES_ARGS=(--max-pages "${ENA_MAX_PAGES}")
  echo "Max pages this run: ${ENA_MAX_PAGES}"
else
  echo "Max pages this run: unlimited"
fi

python run_ena_seed_discovery.py \
  --phase discovery \
  --date-shards \
  --no-openalex \
  --db "${SEED_DB}" \
  "${MAX_PAGES_ARGS[@]}"

echo "=== ENA discovery crawl_state ==="
python - <<PY
import sqlite3
conn = sqlite3.connect("${SEED_DB}")
conn.row_factory = sqlite3.Row
for row in conn.execute("SELECT status, count(*) AS n FROM crawl_state WHERE source LIKE 'ena_read_run:%' GROUP BY status ORDER BY n DESC"):
    print(f"{row['status']}: {row['n']}")
overall = conn.execute("SELECT status FROM crawl_state WHERE source = 'ena_read_run'").fetchone()
print(f"overall ena_read_run status: {overall['status'] if overall else 'unknown'}")
conn.close()
PY

echo "Job finished: $(date)"
echo "Next: once overall status is 'completed', run_ena_publication_resolution.sbatch can resolve everything discovery found."
