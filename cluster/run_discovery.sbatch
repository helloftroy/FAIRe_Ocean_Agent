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
#SBATCH --time=12:00:00
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
python -m fair_ocean_agent.cli worker --until-empty

echo "=== identifiers/full-text discovery done -- warming supplement cache ==="
python -m fair_ocean_agent.cli enqueue-supplement-discovery-backfill
python -m fair_ocean_agent.cli worker --until-empty
python -m fair_ocean_agent.cli enqueue-supplement-retrieval-backfill
python -m fair_ocean_agent.cli worker --until-empty

echo "=== discovery stage complete ==="
python -m fair_ocean_agent.cli status

echo "Job finished: $(date)"
echo "Next: submit run_extraction.sbatch (same FAIR_OCEAN_DATABASE_URL, same repo checkout)."
