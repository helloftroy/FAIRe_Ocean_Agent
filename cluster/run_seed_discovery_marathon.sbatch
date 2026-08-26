#!/bin/bash
# Runs scripts/run_seed_discovery_marathon.sh (EPMC bootstrap -> MGnify
# discovery/resolution -> ENA discovery/resolution -> CSV export) on the
# service partition instead of your Mac. Same database
# (data/seed_discovery/mgnify_paper_seeds.sqlite), same idempotent/
# resumable design -- resubmitting after a walltime cutoff just continues.
# The marathon script itself already runs everything with OpenAlex
# disabled, so this is safe to run alongside run_discovery.sbatch and
# run_gold_bioproject_search.sbatch without adding OpenAlex load.
#SBATCH --job-name=seed_discovery_marathon
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/seed_discovery_marathon_%j.out
#SBATCH --error=logs/seed_discovery_marathon_%j.err

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

# EPMC_MODE=no-download by default here -- the bulk snapshot is large and
# should already have been synced from the Mac (see
# sync_local_db_to_cluster.sh); only re-bootstrap it if you know the
# cluster doesn't have data/seed_discovery/europepmc_bulk/ yet.
EPMC_MODE="${EPMC_MODE:-no-download}" \
  bash scripts/run_seed_discovery_marathon.sh

echo "Job finished: $(date)"
