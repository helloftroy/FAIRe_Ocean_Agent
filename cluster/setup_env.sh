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
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "PYTHON_BIN='${PYTHON_BIN}' not found on PATH." >&2
  exit 2
fi
echo "Python: $("${PYTHON_BIN}" --version)"

# This package requires Python >=3.10 (see pyproject.toml) -- fail here
# with a clear message instead of deep inside a cryptic pip dependency-
# resolution error. Many clusters default `python3` to an older system
# Python; a newer one is usually available via `module load`, a direct
# python3.1x binary, or conda. Find one, then re-run with e.g.:
#   PYTHON_BIN=python3.11 ./cluster/setup_env.sh
if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo >&2
  echo "'${PYTHON_BIN}' is too old (this package needs >=3.10). Try:" >&2
  echo "  module avail python 2>&1 | head -30" >&2
  echo "  which -a python3.10 python3.11 python3.12" >&2
  echo "  module avail anaconda   # conda often ships a newer python too" >&2
  echo "then re-run: PYTHON_BIN=<name-or-path> ./cluster/setup_env.sh" >&2
  exit 2
fi

if [ -d .venv ]; then
  VENV_VERSION="$(.venv/bin/python --version 2>&1 || true)"
  if ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "Existing .venv was created with an incompatible Python (${VENV_VERSION})." >&2
    echo "Remove it and re-run: rm -rf .venv && PYTHON_BIN=${PYTHON_BIN} ./cluster/setup_env.sh" >&2
    exit 2
  fi
else
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
