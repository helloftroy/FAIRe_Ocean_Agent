#!/bin/bash
# Runs scripts/run_seed_discovery_marathon.sh (EPMC bootstrap -> MGnify
# discovery/resolution -> ENA discovery/resolution -> CSV export) on the
# service partition instead of your Mac. Same database
# (data/seed_discovery/mgnify_paper_seeds.sqlite), same idempotent/
# resumable design -- resubmitting after a walltime cutoff just continues.
# The marathon script itself already runs everything with OpenAlex
# disabled, so this is safe to run alongside run_discovery.sbatch and
# run_gold_bioproject_search.sbatch without adding OpenAlex load.
#
# NEVER run this at the same time as run_ena_publication_resolution.sbatch,
# run_mgnify_publication_resolution.sbatch, or run_ena_discovery.sbatch --
# this script's own mgnify_publications/ena_publications/ena_discovery
# steps write to the exact same mgnify_paper_seeds.sqlite file those jobs
# do, and SQLite only allows one writer at a time.
#SBATCH --job-name=seed_discovery_marathon
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/seed_discovery_marathon_%j.out
#SBATCH --error=logs/seed_discovery_marathon_%j.err

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

# run_seed_discovery_marathon.sh defaults PYTHON_BIN to .venv/bin/python3
# -- the Mac dev convention (a local virtualenv). The cluster activates a
# conda env instead (cluster/env_activate.sh, above), which has no
# .venv/ at all -- point the marathon script at whatever python3
# env_activate.sh just put on PATH instead of its own Mac-oriented default.
PYTHON_BIN="$(command -v python3)"
: "${PYTHON_BIN:?python3 not found on PATH after sourcing cluster/env_activate.sh -- check your conda env activated correctly.}"
echo "PYTHON_BIN: ${PYTHON_BIN}"

# EPMC_MODE=no-download by default here -- the bulk snapshot is large and
# should already have been synced from the Mac (see
# sync_local_db_to_cluster.sh); only re-bootstrap it if you know the
# cluster doesn't have data/seed_discovery/europepmc_bulk/ yet.
PYTHON_BIN="${PYTHON_BIN}" \
  EPMC_MODE="${EPMC_MODE:-no-download}" \
  bash scripts/run_seed_discovery_marathon.sh

echo "Job finished: $(date)"
