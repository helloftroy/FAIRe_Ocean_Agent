#!/bin/bash
# Stage 1: discovery + retrieval. No LLM calls at all -- this is the
# HTTPS-capable, CPU-only half of the pipeline (Crossref/Europe PMC/NCBI/
# ENA/OBIS/GBIF/DataCite/BCO-DMO/PANGAEA API calls). Also warms the
# on-disk response cache (data/cache/<source>/*, keyed by request URL) for
# every paper's full text and supplement files -- run_extraction.sbatch
# reads that same cache on the GPU node and never needs its own network
# access as a result. Run this BEFORE run_extraction.sbatch.
#
# Partition name below is copied from multimodal_granite's own
# run_crcp_tile_download.sbatch (its HTTPS-capable job) -- double check
# this is the right partition name for your account before submitting.
#SBATCH --job-name=faire_discovery
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/discovery_%j.out
#SBATCH --error=logs/discovery_%j.err

set -euo pipefail
# $(dirname "$0") is NOT reliable here -- sbatch commonly stages/copies the
# submitted script before running it, so $0 can point at a spool directory
# you don't have write access to, not this repo. SLURM_SUBMIT_DIR is the
# standard, reliable way to get back to wherever `sbatch` was actually run
# from (confirmed live: this exact mkdir failed with "Permission denied"
# using the dirname approach).
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"   # repo root (fair_ocean_agent/), i.e. wherever you ran `sbatch cluster/run_discovery.sbatch` from
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
echo "Working directory: $(pwd)"
python --version

export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
SEED_FILE="${SEED_FILE:-cluster/seeds_five_papers.csv}"
echo "Database: ${FAIR_OCEAN_DATABASE_URL}"
echo "Seed file: ${SEED_FILE}"

# init-db (Base.metadata.create_all) only creates tables that don't exist
# yet -- it never adds a new column to an already-existing table, so an
# already-populated database silently stays on its old schema forever
# after a code update that adds a migration, until something actually
# breaks on the missing column. alembic upgrade head replaces it here:
# confirmed live it builds the exact same full schema from nothing on a
# brand-new database too, so this is safe unconditionally, fresh or not.
python -m alembic -c alembic.ini upgrade head
python -m fair_ocean_agent.cli ingest-seeds "${SEED_FILE}"
python -m fair_ocean_agent.cli enqueue-seed-backfill
# --no-llm is belt-and-suspenders here: none of the three LLM task types
# (EXTRACT_TEXT_FACTS/EXTRACT_SUPPLEMENT_TEXT_FACTS/VALIDATE_EVIDENCE)
# ever get enqueued by anything this script calls -- each is strictly
# behind its own explicit enqueue-*-backfill command, none of which are
# run here -- but the flag costs nothing and guarantees it even if that
# ever changes.
python -m fair_ocean_agent.cli worker --until-empty --no-llm

echo "=== identifiers/full-text discovery done -- warming supplement cache ==="
python -m fair_ocean_agent.cli enqueue-supplement-discovery-backfill
python -m fair_ocean_agent.cli worker --until-empty --no-llm
python -m fair_ocean_agent.cli enqueue-supplement-retrieval-backfill
python -m fair_ocean_agent.cli worker --until-empty --no-llm

# Auto-fetch is a no-op (skips before any network call) for anything that
# already has a PMCID or PDF, so it's safe and cheap to always run here --
# it only ever fills in the gap for a study whose only route to a real
# open-access PDF is this OpenAlex-driven lookup, not the PMCID path
# already tried during discovery above.
echo "=== auto-fetching open-access PDFs for anything discovery didn't already grab ==="
python scripts/auto_fetch_missing_pdfs.py

# The actual answer to "which papers have confirmed samples vs. still
# need a manually-supplied PDF vs. have no sample data at all" -- this is
# the file worth checking after a run, not just the task-queue counts
# below. Lands in data/paper_classification/*.csv on the cluster.
echo "=== classifying: confirmed samples / needs a manual PDF / no sample data ==="
python scripts/classify_papers.py

echo "=== discovery stage complete ==="
python -m fair_ocean_agent.cli status

echo "Job finished: $(date)"
echo "Next: submit run_extraction.sbatch (same FAIR_OCEAN_DATABASE_URL, same repo checkout)."
