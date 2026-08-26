#!/bin/bash
# Searches Europe PMC + NCBI for the paper behind each GOLD BioProject
# accession (see gold_bioproject_publications.py's own docstring for why:
# GOLD's own bulk publication field is mostly OSTI data-report DOIs, not
# source papers). Same "service"/HTTPS-capable partition as
# run_discovery.sbatch -- deliberately never touches OpenAlex (see that
# runner's docstring), so it's safe to run alongside run_discovery.sbatch
# without doubling load on OpenAlex specifically.
#
# Fully resumable -- every BioProject accession's outcome is recorded as
# it's checked (gold_bioproject_publication_search table), so if this job
# hits its walltime before finishing all ~54,300 accessions, just
# resubmit it; it picks up exactly where it left off. Safe to run
# `squeue` and resubmit as many times as needed.
#SBATCH --job-name=gold_bioproject_search
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gold_bioproject_search_%j.out
#SBATCH --error=logs/gold_bioproject_search_%j.err

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
GOLD_DB="${GOLD_DB:-data/jgi_gold/gold_sharded.sqlite}"
echo "GOLD database: ${GOLD_DB}"

python -m fair_ocean_agent.seed_discovery.run_gold_bioproject_publication_search \
  --db "${GOLD_DB}" \
  --log-level INFO

echo "=== re-ranking primary DOIs against the current (larger) candidate set ==="
python scripts/resolve_gold_primary_publications.py --db "${GOLD_DB}" --low-fanout-threshold 4 --apply

echo "Job finished: $(date)"
