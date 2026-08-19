#!/usr/bin/env bash
# One-time model download for the vLLM path. Run this from a login node
# (needs internet) -- NOT part of run_extraction.sbatch, which runs on a
# GPU node with HF_HUB_OFFLINE=1 and expects the model already cached.
#
# Usage: ./cluster/download_vllm_model.sh [model-repo-id]
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (fair_ocean_agent/)

if [ ! -f cluster/env_activate.sh ]; then
  echo "cluster/env_activate.sh is missing -- run ./cluster/setup_env.sh first." >&2
  exit 2
fi
# shellcheck source=/dev/null
source cluster/env_activate.sh

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
export HF_HOME="${HF_HOME:-$(pwd)/cluster/hf_cache}"
mkdir -p "${HF_HOME}"

if ! python -c "import huggingface_hub" 2>/dev/null; then
  echo "Installing huggingface_hub..."
  pip install huggingface_hub
fi

echo "Downloading ${MODEL} into ${HF_HOME} ..."
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL}')
print('Cached at:', path)
"

echo
echo "Done. run_extraction.sbatch with LLM_BACKEND=vllm will find this via"
echo "HF_HOME=${HF_HOME} (set that same value, or leave HF_HOME unset and run"
echo "run_extraction.sbatch from this same repo checkout -- it defaults to"
echo "the same path)."
