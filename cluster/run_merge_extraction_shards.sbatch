#!/bin/bash
# Runs scripts/merge_extraction_shards.py as a batch job so it doesn't need
# a live interactive terminal on a login node -- real gap found live: a
# real 1702-study remap pass took hours, with nothing printing until each
# study committed, which is exactly the kind of run an SSH drop or a
# terminal timeout would kill outright. No GPU needed (this is DB-only --
# no LLM calls at all) and, unlike run_discovery.sbatch, no network calls
# either -- every merge/remap step is local-file SQLite work start to
# finish. "service" (run_discovery.sbatch's own partition) is reserved for
# jobs that genuinely need outbound internet, and on this cluster it's a
# single, heavily-shared node (confirmed live via `sinfo`: 1 node/128 CPUs
# total, vs. "morrill"'s 100 nodes/~6400 CPUs, almost entirely idle) --
# putting a job that needs no network on it would just queue behind
# genuinely internet-bound work for no reason. "morrill" (the general
# compute partition, no special network requirement) is the right choice
# here.
#
# Safe to just resubmit if this hits its walltime before finishing: the
# merge steps themselves are already idempotent (INSERT OR IGNORE / real
# UNIQUE constraints), and the remap loop (scripts/merge_extraction_shards.py's
# _remap_touched_studies) now commits each study as it finishes -- a
# re-run only redoes whichever studies weren't yet committed, not
# everything from scratch, and map_study_to_faire itself is a safe
# delete-then-recreate per study either way.
#
# Usage:
#   sbatch --account=<acct> cluster/run_merge_extraction_shards.sbatch
#   sbatch --account=<acct> --export=ALL,MANIFEST=data/shard_dbs/manifest.json cluster/run_merge_extraction_shards.sbatch
#
# Double check "morrill" is a partition your account can submit non-GPU
# CPU jobs to (`sinfo -o "%R %l %D %C"`) -- swap it for whatever your
# cluster's own general-purpose CPU partition is called if not.
#SBATCH --job-name=faire_merge_extraction_shards
#SBATCH --partition=morrill
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/merge_extraction_shards_%j.out
#SBATCH --error=logs/merge_extraction_shards_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export FAIR_OCEAN_SKIP_SQLITE_WAL=1  # this cluster's Lustre mount cannot do WAL locking -- skip the attempt outright, see database/session.py

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

echo "Job started: $(date)"
echo "Host: $(hostname)"

MANIFEST="${MANIFEST:-data/shard_dbs/manifest.json}"
if [ ! -f "${MANIFEST}" ]; then
  echo "No manifest at ${MANIFEST} -- run 'python scripts/shard_extraction_prep.py --shards N' first." >&2
  exit 2
fi
echo "Manifest: ${MANIFEST}"

python scripts/merge_extraction_shards.py --manifest "${MANIFEST}"

echo "Job finished: $(date)"
echo "Next: check python -m fair_ocean_agent.cli status, then a fresh"
echo "shard_extraction_prep.py round if anything is still pending."
