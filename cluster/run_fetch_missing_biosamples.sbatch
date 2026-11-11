#!/bin/bash
# Gap 1 (2026-11-10): fetch NCBI BioSample attributes for the ~332k sample
# entities that never got them (BioProjects past the 300-sample
# MAX_SAMPLES_PER_PROJECT cap). Network only, no LLM. See
# scripts/fetch_missing_biosamples.py for what it does and does not touch.
#
# Safe next to the running discovery job: commits once per 500-sample
# chunk, same SQLite busy_timeout as the workers. Resumable: re-submit after
# a timeout and it skips samples that already have BioSample facts.
#
# Submit from the repo root:
#   sbatch --account=191001-364393 cluster/run_fetch_missing_biosamples.sbatch
# Test first on one study (a few minutes):
#   sbatch --account=191001-364393 --export=ALL,FETCH_ARGS="--study STUDY-12d5ded31994 --max-entities 1000" cluster/run_fetch_missing_biosamples.sbatch
#SBATCH --job-name=faire_biosample_fill
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/biosample_fill_%j.out
#SBATCH --error=logs/biosample_fill_%j.err

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- this script must be run via sbatch, not directly.}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export FAIR_OCEAN_SKIP_SQLITE_WAL=1  # Lustre cannot do WAL locking, see database/session.py

# shellcheck source=/dev/null
source cluster/env_activate.sh
export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"

echo "Job started: $(date)  host: $(hostname)"
echo "Database: ${FAIR_OCEAN_DATABASE_URL}"
python -m alembic -c alembic.ini upgrade head
python scripts/fetch_missing_biosamples.py --dry-run
# shellcheck disable=SC2086
python scripts/fetch_missing_biosamples.py ${FETCH_ARGS:-}
echo "Job finished: $(date)"
echo "Next: the FAIRe re-mapping (gap 2) once the mapping-rule fixes are pushed."
