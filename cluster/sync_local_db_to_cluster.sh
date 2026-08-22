#!/usr/bin/env bash
# Pushes a Mac-side discovery run up to the cluster, so run_discovery.sbatch
# tops up whatever's still missing instead of re-discovering everything from
# a blank database. See cluster/README.md's "Run discovery on your Mac
# first" section for why this exists: without it, the Mac and the cluster
# each independently hit OpenAlex/Crossref/NCBI/ENA for the same papers,
# because they're two separate filesystems with nothing shared but the seed
# CSV. ingest-seeds/enqueue-*-backfill/worker are all idempotent against the
# same database (see cluster/README.md's Troubleshooting section) -- once
# this copy lands, run_discovery.sbatch on the cluster just skips anything
# already resolved and only processes what's new or still pending.
#
# Run this FROM YOUR MAC, after your local worker/auto-fetch run has
# finished (not while one is still going -- see the journal-file check
# below).
#
# Usage: ./cluster/sync_local_db_to_cluster.sh <cluster-ssh-alias> <remote-repo-path>
# Example: ./cluster/sync_local_db_to_cluster.sh morrill /scratch/morrill/users/hmp278/FAIRe_Ocean_Agent
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <cluster-ssh-alias> <remote-repo-path>" >&2
  echo "Example: $0 morrill /scratch/morrill/users/hmp278/FAIRe_Ocean_Agent" >&2
  exit 2
fi
CLUSTER="$1"
REMOTE_PATH="$2"

# A leftover -journal (or -wal) file means SQLite thinks a write was left
# mid-transaction -- normal right after killing a stuck process (SQLite's
# own crash recovery cleans it up on the next connection), but copying the
# main .db file without that recovery having happened yet would ship a
# torn, inconsistent database. Open and close a connection first so SQLite
# replays/discards the journal itself before we copy anything.
if ls data/fair_ocean.db-journal data/fair_ocean.db-wal >/dev/null 2>&1; then
  echo "Found a leftover journal/WAL file -- running one quick connection to let SQLite recover it first." >&2
  python3 -c "import sqlite3; c = sqlite3.connect('data/fair_ocean.db'); c.execute('PRAGMA quick_check'); c.close()"
fi
if ls data/fair_ocean.db-journal data/fair_ocean.db-wal >/dev/null 2>&1; then
  echo "Journal/WAL file is still there after recovery -- is a worker or script still running against data/fair_ocean.db? Stop it first." >&2
  exit 1
fi

echo "Syncing data/fair_ocean.db -> ${CLUSTER}:${REMOTE_PATH}/data/fair_ocean.db"
rsync -avz --progress data/fair_ocean.db "${CLUSTER}:${REMOTE_PATH}/data/fair_ocean.db"

echo "Syncing data/cache/ (warm response cache -- avoids re-fetching anything already cached)"
rsync -avz data/cache/ "${CLUSTER}:${REMOTE_PATH}/data/cache/"

if [ -d data/auto_fetched_pdfs ]; then
  echo "Syncing data/auto_fetched_pdfs/"
  rsync -avz data/auto_fetched_pdfs/ "${CLUSTER}:${REMOTE_PATH}/data/auto_fetched_pdfs/"
fi

if [ -n "${FAIR_OCEAN_LOCAL_PDF_DIR:-}" ] && [ -d "${FAIR_OCEAN_LOCAL_PDF_DIR}" ]; then
  echo "Syncing FAIR_OCEAN_LOCAL_PDF_DIR (${FAIR_OCEAN_LOCAL_PDF_DIR})"
  rsync -avz "${FAIR_OCEAN_LOCAL_PDF_DIR}/" "${CLUSTER}:${REMOTE_PATH}/data/local_pdfs/"
fi

echo
echo "Done. On the cluster, run_discovery.sbatch against this same repo checkout"
echo "will now only process studies that aren't already resolved."
