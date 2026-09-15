#!/bin/bash
# Parallelizes ONE stage of run_discovery.sbatch's network-bound worker
# loop across several array tasks, all safely claiming from the SAME
# shared data/fair_ocean.db. Safe for the same reason
# run_extraction_parallel.sbatch is: workflow/task_queue.py's SQLite
# claim_next_task used to be single-worker-only (confirmed live producing
# 416 duplicate claims out of 500 under real 8-process contention) --
# fixed with a single atomic UPDATE ... RETURNING claim plus a real
# busy_timeout. Do not run this against a checkout that predates that
# fix. database/session.py also tries to enable WAL mode for better write
# throughput, but degrades gracefully on filesystems that reject it
# (confirmed live: a real Lustre-backed cluster scratch mount) -- the
# atomic claim + busy_timeout above stay correct either way.
#
# Discovery has THREE sequential stages -- identifiers, then supplement
# discovery, then supplement retrieval -- each depending on the previous
# stage's own data (e.g. supplement discovery needs a study's PMCID,
# which only identifier discovery produces). Unlike
# run_extraction_parallel.sbatch's single backlog, this script must be
# run ONCE PER STAGE, in order, each round fully draining (or at least
# making a full pass) before the next stage starts.
#
# IMPORTANT -- how many array tasks is actually safe:
# ncbi_bioproject/ncbi_biosample (hit during the "identifiers" stage
# only) share ONE per-IP eutils rate limit -- 3 req/sec without an
# NCBI_API_KEY, 10/sec with one (see workflow/handlers.py) -- enforced
# independently BY EACH PROCESS with no cross-process coordination, so N
# array tasks multiply that limit by N in aggregate against NCBI's real
# server-side enforcement, which does not scale with your array size.
#   - identifiers, no NCBI_API_KEY set: --array=1 only. Any N>1 already
#     exceeds NCBI's no-key budget on its own.
#   - identifiers, NCBI_API_KEY set: --array=1-3 is a reasonable start
#     (~3.3 eff. req/sec/task, under the shared 10/sec budget with some
#     headroom for retries). Watch the per-array-task logs for repeated
#     429/"rate limit" errors and scale down if you see them; scale up
#     cautiously if you don't.
# supplement-discovery and supplement-retrieval mostly hit Europe PMC and
# individual publisher hosts instead (each adapter throttles itself
# independently, with more generous configured limits -- see
# config/sources.yaml), so a higher --array (e.g. 1-8) is reasonable for
# those two stages regardless of NCBI_API_KEY.
#
# Usage (run once per stage, in order -- wait for each to finish first):
#   export NCBI_API_KEY=...   # optional but recommended, see cluster/README.md
#   sbatch --account=<acct> --array=1-3 --export=ALL,DISCOVERY_STAGE=identifiers cluster/run_discovery_parallel.sbatch
#   sbatch --account=<acct> --array=1-8 --export=ALL,DISCOVERY_STAGE=supplement-discovery cluster/run_discovery_parallel.sbatch
#   sbatch --account=<acct> --array=1-8 --export=ALL,DISCOVERY_STAGE=supplement-retrieval cluster/run_discovery_parallel.sbatch
#   # Then run the ordinary run_discovery.sbatch once (no array) to pick
#   # up any remainder plus auto-fetch/classify/status.
#
# If a stage isn't fully drained after one round, just resubmit the same
# command -- every enqueue-*-backfill command is idempotent and
# claim_next_task naturally picks up wherever the queue left off.
#
# Partition copied from run_discovery.sbatch -- double check this is the
# right partition name for your account before submitting.
#SBATCH --job-name=faire_discovery_parallel
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/discovery_parallel_%A_%a.out
#SBATCH --error=logs/discovery_parallel_%A_%a.err

set -euo pipefail
# SLURM_SUBMIT_DIR, not dirname "$0" -- see run_discovery.sbatch's own
# comment for why.
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

ARRAY_LABEL="${SLURM_ARRAY_TASK_ID:-single}/${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Job started: $(date)"
echo "Array task: ${ARRAY_LABEL} (job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID})"
echo "Host: $(hostname)"

export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
echo "Database (shared across every array task): ${FAIR_OCEAN_DATABASE_URL}"
if [ -n "${NCBI_API_KEY:-}" ]; then
  echo "NCBI_API_KEY is set -- shared ncbi_bioproject/ncbi_biosample client runs at NCBI's with-key 10 req/sec limit."
else
  echo "NCBI_API_KEY is not set -- shared ncbi_bioproject/ncbi_biosample client runs at NCBI's no-key 3 req/sec limit."
fi

DISCOVERY_STAGE="${DISCOVERY_STAGE:-identifiers}"
echo "Stage: ${DISCOVERY_STAGE}"

case "${DISCOVERY_STAGE}" in
  identifiers)
    ENQUEUE_COMMAND="enqueue-seed-backfill"
    TASK_TYPE="DISCOVER_IDENTIFIERS"
    ;;
  supplement-discovery)
    ENQUEUE_COMMAND="enqueue-supplement-discovery-backfill"
    TASK_TYPE="DISCOVER_SUPPLEMENTS"
    ;;
  supplement-retrieval)
    ENQUEUE_COMMAND="enqueue-supplement-retrieval-backfill"
    TASK_TYPE="RETRIEVE_SUPPLEMENTS"
    ;;
  *)
    echo "Unknown DISCOVERY_STAGE='${DISCOVERY_STAGE}' -- use identifiers, supplement-discovery, or supplement-retrieval" >&2
    exit 2
    ;;
esac

# <enqueue command>'s own idempotency (a real DB UNIQUE constraint on
# Task.idempotency_key, with enqueue_task catching the concurrent-insert
# IntegrityError) makes it safe for every array task to call this at
# roughly the same time -- same precedent as run_extraction_parallel.sbatch's
# enqueue-text-extraction-backfill call.
echo "=== ${ENQUEUE_COMMAND} (safe to call from every array task) ==="
python -m fair_ocean_agent.cli "${ENQUEUE_COMMAND}"

# --task-type (not --no-llm) -- the two are mutually exclusive in `worker`,
# and every task type used here is non-LLM anyway, so --task-type alone
# already guarantees no LLM call happens.
echo "=== claiming ${TASK_TYPE} tasks until the queue is empty (array task ${ARRAY_LABEL}) ==="
python -m fair_ocean_agent.cli worker --task-type "${TASK_TYPE}" --until-empty

echo "Job finished: $(date)"
echo "Array task ${ARRAY_LABEL} done for stage '${DISCOVERY_STAGE}'."
