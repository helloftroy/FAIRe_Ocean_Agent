#!/usr/bin/env bash
# Submits cluster/run_extraction_parallel.sbatch as a chain of sequential
# (never-overlapping) single jobs, instead of a concurrent job array.
#
# Real gap found live: a real --array=1-5 submission failed on EVERY
# array task with "sqlite3.OperationalError: locking protocol" -- not on
# a write, on a plain SELECT, right after the WAL-mode fallback (see
# database/session.py) had already kicked in cleanly. That confirms a
# deeper problem than WAL specifically: this cluster's /scratch mount
# doesn't reliably coordinate SQLite's file locks across DIFFERENT
# COMPUTE NODES at all (a known Lustre limitation -- some mounts scope
# locks to one node only, e.g. a `localflock`-style option). A single job
# never hits this, since only one process ever touches the database at a
# time. Chaining N single jobs back-to-back via --dependency=afterany
# gets you through the same EXTRACT_TEXT_FACTS backlog, one
# EXTRACT_MAX_TASKS-bounded chunk at a time, with zero risk of the
# cross-node locking failure -- at the cost of losing the array version's
# N-way parallel speedup. Revisit true parallelism if this cluster turns
# out to have a node with enough GPUs to pin every array task to one
# node, or if the database moves to a real client-server engine
# (PostgreSQL) that doesn't depend on filesystem locking at all.
#
# Usage:
#   ./cluster/submit_extraction_sequential.sh <rounds> [sbatch args...]
#   ./cluster/submit_extraction_sequential.sh 10 --account=191001-364393
#   ./cluster/submit_extraction_sequential.sh 10 --account=191001-364393 --export=ALL,LLM_BACKEND=vllm,EXTRACT_MAX_TASKS=300
#
# Each round claims up to EXTRACT_MAX_TASKS (default 300) EXTRACT_TEXT_FACTS
# tasks and exits. It's fine to submit more rounds than you think you'll
# need -- enqueue-text-extraction-backfill is idempotent, and a round
# that finds nothing left to claim exits almost instantly, so trailing
# no-op rounds cost nothing but a little queue time. Check progress any
# time with `python -m fair_ocean_agent.cli status` or `squeue -u $USER`.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <rounds> [sbatch args...]" >&2
  echo "Example: $0 10 --account=191001-364393" >&2
  exit 2
fi
ROUNDS="$1"
shift
if ! [[ "${ROUNDS}" =~ ^[0-9]+$ ]] || [ "${ROUNDS}" -lt 1 ]; then
  echo "<rounds> must be a positive integer, got '${ROUNDS}'" >&2
  exit 2
fi

echo "Submitting ${ROUNDS} sequential round(s) of cluster/run_extraction_parallel.sbatch"
echo "(no --array -- each round is a single job; --dependency=afterany chains them so"
echo "they never overlap in time, sidestepping the cross-node SQLite locking issue)."
echo

PREV_JOB_ID=""
for i in $(seq 1 "${ROUNDS}"); do
  if [ -z "${PREV_JOB_ID}" ]; then
    JOB_ID="$(sbatch --parsable "$@" cluster/run_extraction_parallel.sbatch)"
    echo "Round ${i}/${ROUNDS}: submitted job ${JOB_ID}"
  else
    JOB_ID="$(sbatch --parsable --dependency=afterany:"${PREV_JOB_ID}" "$@" cluster/run_extraction_parallel.sbatch)"
    echo "Round ${i}/${ROUNDS}: submitted job ${JOB_ID} (starts after ${PREV_JOB_ID} finishes)"
  fi
  PREV_JOB_ID="${JOB_ID}"
done

echo
echo "All ${ROUNDS} round(s) submitted as a dependency chain."
echo "Watch progress: squeue -u \$USER"
echo "Per-round logs: logs/extraction_parallel_<job_id>_1.out (and .err)"
