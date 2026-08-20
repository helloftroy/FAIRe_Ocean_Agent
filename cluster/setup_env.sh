#!/usr/bin/env bash
# One-time environment setup on the cluster. Run this once from a login
# node (needs internet to pip install and, if you go the Ollama route,
# to download the model) -- not part of any sbatch job.
#
# Usage (conda, default -- confirmed working when the cluster's own
# python3 is too old for this package's >=3.10 requirement):
#   git clone https://github.com/helloftroy/FAIRe_Ocean_Agent.git
#   cd FAIRe_Ocean_Agent/fair_ocean_agent
#   ./cluster/setup_env.sh
#
# Override the conda env name, Python version, or miniforge/conda
# location:
#   CONDA_ENV_NAME=faire-vllm PYTHON_VERSION=3.12 MINIFORGE_HOME="$HOME/miniforge3" ./cluster/setup_env.sh
#
# If $HOME has a small quota (common on HPC -- vLLM alone pulls in torch +
# CUDA libraries, easily several GB, confirmed live: "Disk quota exceeded"
# partway through `pip install vllm` with the env under the default
# $HOME/miniforge3/envs/...), put the env on scratch space instead with
# CONDA_ENV_PREFIX (this also moves pip's own cache/build directory there,
# which would otherwise hit the same quota independently of where the env
# itself lives):
#   CONDA_ENV_PREFIX=/scratch/$USER/conda_envs/faire-agent ./cluster/setup_env.sh
#
# Or use a plain venv instead of conda:
#   ENV_MANAGER=venv PYTHON_BIN=python3.11 ./cluster/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)

# Remembers ENV_MANAGER/CONDA_ENV_NAME/CONDA_ENV_PREFIX/MINIFORGE_HOME from
# the last successful run, so re-running `./cluster/setup_env.sh` with no
# arguments (e.g. after `git pull`-ing a fix) reuses the SAME environment
# instead of silently defaulting back to a brand-new, empty one -- caught
# live: re-running without repeating CONDA_ENV_PREFIX created/pointed at a
# totally different, package-less env, and a later job failed with "vllm
# not found on PATH" as a result, even though vllm really was installed --
# just in the OTHER, now-orphaned env. Each line below only takes effect
# if the caller didn't already set that variable this time, so an
# explicit override always wins over the remembered one.
ENV_CONFIG_FILE="cluster/.env_config"
if [ -f "${ENV_CONFIG_FILE}" ]; then
  # shellcheck source=/dev/null
  source "${ENV_CONFIG_FILE}"
fi

ENV_MANAGER="${ENV_MANAGER:-conda}"

if [ "${ENV_MANAGER}" = "conda" ]; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-faire-agent}"
  CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
  PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
  MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
  CONDA_SH="${MINIFORGE_HOME}/etc/profile.d/conda.sh"

  if [ ! -f "${CONDA_SH}" ]; then
    echo "conda.sh not found at ${CONDA_SH}." >&2
    echo "Set MINIFORGE_HOME if your (mini)conda/anaconda install lives elsewhere" >&2
    echo "(e.g. MINIFORGE_HOME=\$HOME/miniconda3), or use ENV_MANAGER=venv instead." >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  source "${CONDA_SH}"

  if [ -n "${CONDA_ENV_PREFIX}" ]; then
    CONDA_ENV_TARGET="${CONDA_ENV_PREFIX}"
    CONDA_CREATE_FLAG=(-p "${CONDA_ENV_PREFIX}")
    CONDA_ACTIVATE_TARGET="${CONDA_ENV_PREFIX}"
    # Two SEPARATE caches, neither the same thing as the env itself, both
    # defaulting under $HOME and both able to independently blow the same
    # quota even once the env is on scratch (confirmed live: the very
    # first `conda create` failed with "Disk quota exceeded" writing to
    # $HOME/miniforge3/pkgs/cache/... -- conda's OWN package cache --
    # despite the env's own -p target already being on scratch):
    #   - pip's download/build cache (PIP_CACHE_DIR, default $HOME/.cache/pip)
    #   - conda's package cache (CONDA_PKGS_DIRS, default $HOME/miniforge3/pkgs
    #     and $HOME/.conda/pkgs -- used by `conda create`/`conda install`,
    #     unrelated to pip)
    # Default both alongside the env unless the caller already set them.
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "${CONDA_ENV_PREFIX}")/pip_cache}"
    export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$(dirname "${CONDA_ENV_PREFIX}")/conda_pkgs}"
    mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
    echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
    echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  else
    CONDA_ENV_TARGET="${CONDA_ENV_NAME}"
    CONDA_CREATE_FLAG=(-n "${CONDA_ENV_NAME}")
    CONDA_ACTIVATE_TARGET="${CONDA_ENV_NAME}"
  fi

  if [ -n "${CONDA_ENV_PREFIX}" ]; then
    ENV_EXISTS=0
    [ -d "${CONDA_ENV_PREFIX}" ] && [ -x "${CONDA_ENV_PREFIX}/bin/python" ] && ENV_EXISTS=1
  else
    ENV_EXISTS=0
    conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}" && ENV_EXISTS=1
  fi
  if [ "${ENV_EXISTS}" = "1" ]; then
    echo "conda env '${CONDA_ENV_TARGET}' already exists, reusing it."
  else
    echo "Creating conda env at '${CONDA_ENV_TARGET}' (python=${PYTHON_VERSION})..."
    conda create "${CONDA_CREATE_FLAG[@]}" "python=${PYTHON_VERSION}" -y
  fi
  conda activate "${CONDA_ACTIVATE_TARGET}"

  # A cluster's module-injected LD_LIBRARY_PATH (e.g. a spack-managed
  # toolchain) can shadow this conda env's own bundled libstdc++ with an
  # older system one (/usr/lib64/libstdc++.so.6) -- confirmed live: vllm's
  # own sqlite3 import chain, via a bundled libicui18n.so, failed with
  # "version CXXABI_1.3.15 not found" the moment anything pulled in a C++
  # library, even though the newer libstdc++ needed was sitting right
  # there in the same conda env. The conda env's own lib/ has to come
  # first in the search path.
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  # A shell profile that auto-activates some OTHER virtualenv/conda env
  # (pyenv, pipenv, direnv, a login-script default) can silently win over
  # `conda activate` above -- confirmed live: on a shell with a pipenv env
  # already active, `conda activate` reported success but `python`/`pip`
  # still resolved into the OTHER, unrelated project's virtualenv, and the
  # subsequent `pip install` polluted it instead. Verify the real
  # interpreter path, not just that the command exited 0.
  if [ -n "${CONDA_ENV_PREFIX}" ]; then
    EXPECTED_PREFIX="${CONDA_ENV_PREFIX}"
  else
    EXPECTED_PREFIX="$(conda env list | awk -v name="${CONDA_ENV_NAME}" '$1==name {print $NF}')"
  fi
  ACTUAL_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
  if [ "${ACTUAL_PREFIX}" != "${EXPECTED_PREFIX}" ]; then
    echo >&2
    echo "conda activate '${CONDA_ACTIVATE_TARGET}' did not actually take effect:" >&2
    echo "  expected python from: ${EXPECTED_PREFIX}" >&2
    echo "  got python from:      ${ACTUAL_PREFIX}" >&2
    echo "Something else in your shell profile (another virtualenv, pipenv, direnv, ...)" >&2
    echo "is overriding it. Try 'deactivate' (or whatever your other tool uses) first," >&2
    echo "or run this from a fresh, plain shell." >&2
    exit 2
  fi
  echo "Python: $(python --version) (${ACTUAL_PREFIX})"

  # Consumed by run_discovery.sbatch/run_extraction.sbatch so they
  # activate the exact same environment this script just set up, without
  # duplicating the conda-vs-venv logic in three places. Machine-specific
  # (real $HOME path baked in) -- gitignored, regenerated by this script.
  # Carries the same verify-it-actually-worked check as above, since the
  # exact same silent-override risk applies inside a SLURM job's own shell.
  cat > cluster/env_activate.sh <<EOF
# Auto-generated by setup_env.sh -- do not edit by hand, re-run
# setup_env.sh instead (with the same ENV_MANAGER/CONDA_ENV_NAME/
# CONDA_ENV_PREFIX/etc if you're not using the defaults).
source "${CONDA_SH}"
conda activate "${CONDA_ACTIVATE_TARGET}"
# A cluster's module-injected LD_LIBRARY_PATH can shadow this conda env's
# own bundled libstdc++ with an older system one -- see setup_env.sh's own
# comment above this same line for the real error this fixes.
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
$([ -n "${CONDA_ENV_PREFIX}" ] && echo "export PIP_CACHE_DIR=\"${PIP_CACHE_DIR}\"")
$([ -n "${CONDA_ENV_PREFIX}" ] && echo "export CONDA_PKGS_DIRS=\"${CONDA_PKGS_DIRS}\"")
_expected_prefix="${EXPECTED_PREFIX}"
_actual_prefix="\$(python -c 'import sys; print(sys.prefix)')"
if [ "\${_actual_prefix}" != "\${_expected_prefix}" ]; then
  echo "conda activate '${CONDA_ACTIVATE_TARGET}' did not take effect in this job's shell" >&2
  echo "(expected \${_expected_prefix}, got \${_actual_prefix}) -- check for a" >&2
  echo "shell-profile-activated virtualenv/conda env overriding it." >&2
  exit 2
fi
unset _expected_prefix _actual_prefix
EOF

  cat > "${ENV_CONFIG_FILE}" <<EOF
# Auto-generated by setup_env.sh -- records this run's settings so a later
# re-run with no arguments reuses the SAME environment instead of
# defaulting back to a brand-new one. Safe to delete if you want a clean
# slate; edit by hand if you'd rather just fix one value.
ENV_MANAGER="\${ENV_MANAGER:-conda}"
CONDA_ENV_NAME="\${CONDA_ENV_NAME:-${CONDA_ENV_NAME}}"
CONDA_ENV_PREFIX="\${CONDA_ENV_PREFIX:-${CONDA_ENV_PREFIX}}"
MINIFORGE_HOME="\${MINIFORGE_HOME:-${MINIFORGE_HOME}}"
EOF

elif [ "${ENV_MANAGER}" = "venv" ]; then
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
  # python3.1x binary, or conda (try ENV_MANAGER=conda, the default,
  # instead if that's easier on your cluster).
  if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo >&2
    echo "'${PYTHON_BIN}' is too old (this package needs >=3.10). Try:" >&2
    echo "  module avail python 2>&1 | head -30" >&2
    echo "  which -a python3.10 python3.11 python3.12" >&2
    echo "  ./cluster/setup_env.sh   # ENV_MANAGER defaults to conda, which sidesteps this" >&2
    echo "then re-run: PYTHON_BIN=<name-or-path> ENV_MANAGER=venv ./cluster/setup_env.sh" >&2
    exit 2
  fi

  if [ -d .venv ]; then
    VENV_VERSION="$(.venv/bin/python --version 2>&1 || true)"
    if ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "Existing .venv was created with an incompatible Python (${VENV_VERSION})." >&2
      echo "Remove it and re-run: rm -rf .venv && PYTHON_BIN=${PYTHON_BIN} ENV_MANAGER=venv ./cluster/setup_env.sh" >&2
      exit 2
    fi
  else
    echo "Creating .venv ..."
    "${PYTHON_BIN}" -m venv .venv
  fi

  EXPECTED_PREFIX="$(pwd)/.venv"
  cat > cluster/env_activate.sh <<EOF
# Auto-generated by setup_env.sh -- do not edit by hand, re-run
# setup_env.sh instead.
source "${EXPECTED_PREFIX}/bin/activate"
_expected_prefix="${EXPECTED_PREFIX}"
_actual_prefix="\$(python -c 'import sys; print(sys.prefix)')"
if [ "\${_actual_prefix}" != "\${_expected_prefix}" ]; then
  echo "venv activation did not take effect in this job's shell" >&2
  echo "(expected \${_expected_prefix}, got \${_actual_prefix}) -- check for a" >&2
  echo "shell-profile-activated virtualenv/conda env overriding it." >&2
  exit 2
fi
unset _expected_prefix _actual_prefix
EOF
  # shellcheck source=/dev/null
  source cluster/env_activate.sh

  cat > "${ENV_CONFIG_FILE}" <<EOF
# Auto-generated by setup_env.sh -- records this run's settings so a later
# re-run with no arguments reuses the SAME environment instead of
# defaulting back to a brand-new one. Safe to delete if you want a clean
# slate; edit by hand if you'd rather just fix one value.
ENV_MANAGER="\${ENV_MANAGER:-venv}"
PYTHON_BIN="\${PYTHON_BIN:-${PYTHON_BIN}}"
EOF

else
  echo "Unknown ENV_MANAGER='${ENV_MANAGER}'. Use conda (default) or venv." >&2
  exit 2
fi

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
echo "Environment ready (cluster/env_activate.sh records how run_discovery.sbatch"
echo "and run_extraction.sbatch will activate it -- no need to edit those scripts)."
echo "Next steps:"
echo "  1. Copy cluster/local.yaml.cluster-example (Ollama) or"
echo "     cluster/local.yaml.cluster-vllm-example (vLLM) to config/local.yaml."
echo "  2. Download the model once, now, while you have internet -- see"
echo "     cluster/README.md's 'One-time model download' section."
echo "  3. Submit cluster/run_discovery.sbatch first, then cluster/run_extraction.sbatch."
echo
echo "Quick self-test (no network, no LLM): python -m pytest tests/unit -q"
