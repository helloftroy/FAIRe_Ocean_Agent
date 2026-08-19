#!/usr/bin/env bash
# One-time environment setup on the cluster. Run this once from a login
# node (needs internet to pip install and, if you go the Ollama route,
# to download the model) -- not part of any sbatch job.
#
# Usage:
#   git clone https://github.com/helloftroy/FAIRe_Ocean_Agent.git
#   cd FAIRe_Ocean_Agent/fair_ocean_agent
#   ./cluster/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)

# If your cluster uses environment modules, uncomment and edit:
# module load anaconda    # or: module load python/3.11

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Python: $("${PYTHON_BIN}" --version)"

if [ ! -d .venv ]; then
  echo "Creating .venv ..."
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"

# Real, versioned data files this pipeline needs at runtime (FAIRe/BeBOP/
# MIxS schemas etc.) are already part of the repo -- nothing extra to
# fetch there. init-db creates all tables directly from the ORM models
# (fine for a fresh cluster run; use `alembic upgrade head` instead if you
# want migration history tracked).
export FAIR_OCEAN_DATABASE_URL="${FAIR_OCEAN_DATABASE_URL:-sqlite:///$(pwd)/data/fair_ocean.db}"
python -m fair_ocean_agent.cli init-db

echo
echo "Environment ready. Next steps:"
echo "  1. Copy cluster/local.yaml.cluster-example to config/local.yaml and edit"
echo "     the llm.base_url/model to match whichever GPU job you'll run."
echo "  2. If using Ollama: pull the model once, now, while you have internet"
echo "     (see cluster/README.md's 'One-time model download' section)."
echo "  3. Submit cluster/run_discovery.sbatch first, then cluster/run_extraction.sbatch."
echo
echo "Quick self-test (no network, no LLM): python -m pytest tests/unit -q"
