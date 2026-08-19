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
cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)
mkdir -p logs

# If your cluster requires modules/conda, uncomment and edit:
# module load anaconda
source .venv/bin/activate

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
python --version

export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
SEED_FILE="${SEED_FILE:-cluster/seeds_five_papers.csv}"
echo "Database: ${FAIR_OCEAN_DATABASE_URL}"
echo "Seed file: ${SEED_FILE}"

python -m fair_ocean_agent.cli init-db
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
