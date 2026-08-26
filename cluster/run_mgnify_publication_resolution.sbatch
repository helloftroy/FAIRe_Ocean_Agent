#!/bin/bash
# Resolves publications for already-discovered MGnify studies
# (data/seed_discovery/mgnify_paper_seeds.sqlite) -- mirrors
# run_ena_publication_resolution.sbatch for the MGnify side. Deliberately
# --resolve-only (skips re-scanning MGnify for new studies) and
# --no-openalex (OpenAlex is opt-in by default anyway now -- kept
# explicit for clarity). Exports the current seed CSV once resolution
# finishes.
#
# NEVER run this at the same time as run_ena_publication_resolution.sbatch,
# run_seed_discovery_marathon.sbatch, or run_ena_discovery.sbatch -- all
# four write to the same mgnify_paper_seeds.sqlite file, and SQLite only
# allows one writer at a time. Run them one at a time; each is
# independently resumable.
#
# Override knobs, for example:
#   sbatch --export=ALL,MGNIFY_RESOLVE_MAX_STUDIES=50000 cluster/run_mgnify_publication_resolution.sbatch
#SBATCH --job-name=mgnify_publication_resolution
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mgnify_publication_resolution_%j.out
#SBATCH --error=logs/mgnify_publication_resolution_%j.err

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

SEED_DB="${SEED_DB:-data/seed_discovery/mgnify_paper_seeds.sqlite}"
OUT_CSV="${OUT_CSV:-cluster/seeds_seed_discovery.csv}"
echo "Seed database: ${SEED_DB}"
echo "Output CSV: ${OUT_CSV}"

MAX_STUDIES_ARGS=()
if [ -n "${MGNIFY_RESOLVE_MAX_STUDIES:-}" ]; then
  MAX_STUDIES_ARGS=(--max-studies "${MGNIFY_RESOLVE_MAX_STUDIES}")
  echo "Max studies this run: ${MGNIFY_RESOLVE_MAX_STUDIES}"
else
  echo "Max studies this run: unlimited (every unresolved MGnify study)"
fi

python run_mgnify_seed_discovery.py \
  --resolve-only \
  --no-openalex \
  --db "${SEED_DB}" \
  "${MAX_STUDIES_ARGS[@]}"

echo "=== exporting seed CSV: $(date) ==="
python scripts/seed_discovery_to_csv.py --db "${SEED_DB}" --out "${OUT_CSV}"

echo "Job finished: $(date)"
