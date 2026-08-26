#!/usr/bin/env bash
# Pushes everything a Mac-side session has accumulated -- the main
# pipeline DB, the GOLD reference DB, and the MGnify/ENA seed-discovery
# DB -- up to the cluster, so each corresponding sbatch job tops up
# whatever's still missing instead of starting over from a blank
# database. See cluster/README.md's "Run discovery on your Mac first"
# section for why this exists: without it, the Mac and the cluster each
# independently hit the same external sources for the same papers,
# because they're two separate filesystems with nothing shared but the
# seed CSV. ingest-seeds/enqueue-*-backfill/worker, and every
# seed-discovery runner, are all idempotent/resumable against whatever
# database they're pointed at -- once this copy lands, each cluster-side
# job just skips anything already resolved and only processes what's new
# or still pending.
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
# replays/discards the journal itself before we copy anything. Checked for
# every database this script moves, not just the main one -- rsync (unlike
# a plain cp) is also NOT safe to run against a database a process still
# has open for writing, even with this check passed; stop every Mac-side
# worker/discovery process first.
recover_and_check() {
  local db="$1"
  if ls "${db}-journal" "${db}-wal" >/dev/null 2>&1; then
    echo "Found a leftover journal/WAL file for ${db} -- running one quick connection to let SQLite recover it first." >&2
    python3 -c "import sqlite3; c = sqlite3.connect('${db}'); c.execute('PRAGMA quick_check'); c.close()"
  fi
  if ls "${db}-journal" "${db}-wal" >/dev/null 2>&1; then
    echo "Journal/WAL file for ${db} is still there after recovery -- is a worker or script still running against it? Stop it first." >&2
    exit 1
  fi
}

recover_and_check data/fair_ocean.db
echo "Syncing data/fair_ocean.db -> ${CLUSTER}:${REMOTE_PATH}/data/fair_ocean.db"
rsync -avz --progress data/fair_ocean.db "${CLUSTER}:${REMOTE_PATH}/data/fair_ocean.db"

echo "Syncing data/cache/ (warm response cache -- avoids re-fetching anything already cached)"
rsync -avz data/cache/ "${CLUSTER}:${REMOTE_PATH}/data/cache/"

if [ -f data/jgi_gold/gold_sharded.sqlite ]; then
  recover_and_check data/jgi_gold/gold_sharded.sqlite
  echo "Syncing data/jgi_gold/gold_sharded.sqlite (GOLD reference DB -- needed for run_gold_bioproject_search.sbatch)"
  rsync -avz --progress data/jgi_gold/gold_sharded.sqlite "${CLUSTER}:${REMOTE_PATH}/data/jgi_gold/gold_sharded.sqlite"
fi

if [ -f data/seed_discovery/mgnify_paper_seeds.sqlite ]; then
  recover_and_check data/seed_discovery/mgnify_paper_seeds.sqlite
  echo "Syncing data/seed_discovery/mgnify_paper_seeds.sqlite (needed for run_seed_discovery_marathon.sbatch)"
  rsync -avz --progress data/seed_discovery/mgnify_paper_seeds.sqlite "${CLUSTER}:${REMOTE_PATH}/data/seed_discovery/mgnify_paper_seeds.sqlite"
fi

if [ -d data/seed_discovery/europepmc_bulk ]; then
  echo "Syncing data/seed_discovery/europepmc_bulk/ (large -- this is the bulk EPMC accession index the marathon script reads locally instead of hitting the live API)"
  rsync -avz data/seed_discovery/europepmc_bulk/ "${CLUSTER}:${REMOTE_PATH}/data/seed_discovery/europepmc_bulk/"
fi

if [ -d data/auto_fetched_pdfs ]; then
  echo "Syncing data/auto_fetched_pdfs/"
  rsync -avz data/auto_fetched_pdfs/ "${CLUSTER}:${REMOTE_PATH}/data/auto_fetched_pdfs/"
fi

if [ -n "${FAIR_OCEAN_LOCAL_PDF_DIR:-}" ] && [ -d "${FAIR_OCEAN_LOCAL_PDF_DIR}" ]; then
  echo "Syncing FAIR_OCEAN_LOCAL_PDF_DIR (${FAIR_OCEAN_LOCAL_PDF_DIR})"
  rsync -avz "${FAIR_OCEAN_LOCAL_PDF_DIR}/" "${CLUSTER}:${REMOTE_PATH}/data/local_pdfs/"
fi

echo
echo "Done. On the cluster, against this same repo checkout:"
echo "  sbatch cluster/run_discovery.sbatch                  # main pipeline, picks up where the Mac left off"
echo "  sbatch cluster/run_gold_bioproject_search.sbatch      # resumes GOLD BioProject publication search"
echo "  sbatch cluster/run_seed_discovery_marathon.sbatch     # resumes MGnify/ENA seed discovery"
