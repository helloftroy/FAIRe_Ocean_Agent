# Running this pipeline on a SLURM cluster

## Why two jobs, not one

This pipeline needs two things that (on most university HPC clusters,
including the one these scripts were written against) don't live on the
same node: live HTTPS calls to Crossref/Europe PMC/NCBI/ENA/OBIS/GBIF/
DataCite/BCO-DMO/PANGAEA, and a GPU for LLM extraction.

The split relies on one existing, already-tested piece of this codebase:
every one of those HTTPS calls goes through `sources/base.py`'s
`RateLimitedClient`, which caches every response to disk
(`data/cache/<source>/<hash>.json`, keyed by request URL + params) before
ever hitting the network again for the same request. `run_discovery.sbatch`
(CPU, HTTPS-capable partition) runs every stage that needs the network --
which, as a side effect, warms that cache for every paper's full text and
supplement files. `run_extraction.sbatch` (GPU partition) then runs the
LLM extraction stages; the same code paths still technically call the
same HTTP-fetching functions, but every one of those calls now hits the
warm on-disk cache and never touches the network. **No pipeline code
changes were needed for this split** -- it's purely a matter of running
the existing CLI commands in the right order, on the right partition,
against the same database and cache directory.

Both jobs must share the same filesystem (home or scratch directory) for
this to work -- normal on HPC, since `data/fair_ocean.db` and
`data/cache/` need to be visible to both.

## One-time setup

```bash
# On a login node (needs internet):
git clone https://github.com/helloftroy/FAIRe_Ocean_Agent.git
cd FAIRe_Ocean_Agent
git pull
conda create -p /scratch/morrill/users/hmp278/conda_envs/faire-agent \
  -c conda-forge \
  python=3.11 pip openssl ca-certificates certifi -y

MINIFORGE_HOME=/scratch/morrill/users/hmp278/miniforge3 \
CONDA_ENV_PREFIX=/scratch/morrill/users/hmp278/conda_envs/faire-agent \
  ./cluster/setup_env.sh
```
conda activate /scratch/morrill/users/hmp278/conda_envs/faire-agent


`run_extraction.sbatch` supports three backends via `LLM_BACKEND`
(default `ollama`): `ollama`, `vllm`, or `external` (an endpoint you're
already running/managing some other way -- the script starts nothing,
just trusts `config/local.yaml`'s `base_url`). `config/local.yaml` itself
is gitignored on purpose (same as your local Mac's own copy) -- it's
where the real endpoint lives.

**Ollama** (default -- what this pipeline was validated against on a Mac):

```bash
cp cluster/local.yaml.cluster-example config/local.yaml
```

GPU nodes on most clusters have no internet access (this is normal, and
exactly why `run_extraction.sbatch` never tries to reach one) -- so the
model has to be downloaded once, from somewhere that *does* have
internet, onto storage the GPU node can also see:

```bash
# On a login node, or the same "service"/HTTPS partition as run_discovery.sbatch:
curl -fsSL https://ollama.com/install.sh | sh    # if ollama isn't already available
ollama pull qwen3:4b-instruct-16k                # or whatever config/local.yaml's llm.model is
```

Ollama caches the model under `~/.ollama/models` -- as long as `$HOME` is
the same shared filesystem the GPU job sees (normal on HPC), this only
needs to happen once, not per job.

**vLLM** (if Ollama gives you trouble -- e.g. it needs a real install and
some clusters restrict what login nodes can install/run persistently):

```bash
pip install vllm
# vllm pulls in flashinfer for CUDA kernel fusion; versions before
# 0.6.16.post4 have a type annotation in flashinfer/comm/fd_exchange.py
# that only evaluates on Python 3.12+, so `vllm serve` crashes at model
# load time on Python 3.10/3.11 with "TypeError: type 'array.array' is
# not subscriptable". Pin past the fix:
pip install "flashinfer-python>=0.6.16.post4"
cp cluster/local.yaml.cluster-vllm-example config/local.yaml

# One-time model download (needs internet -- run from a login node):
./cluster/download_vllm_model.sh   # defaults to Qwen/Qwen3-4B-Instruct-2507
```

vLLM loads real Hugging Face Hub model repos, not Ollama's own tags --
`config/local.yaml`'s `model` must be a real HF repo id (the example uses
`Qwen/Qwen3-4B-Instruct-2507`, the closest real equivalent to the Ollama
model this pipeline was validated against). Submit with
`sbatch --export=ALL,LLM_BACKEND=vllm cluster/run_extraction.sbatch`.
`run_extraction.sbatch` sets `HF_HUB_OFFLINE=1` on the GPU node, so a
model that wasn't actually downloaded first fails fast with a clear error
in `logs/vllm_<job_id>.log` instead of hanging trying to reach
huggingface.co.

## Running the 5/6-paper validation set

`cluster/seeds_five_papers.csv` is the same recurring paper set used
throughout this project's own development for spot-checking pipeline
changes (s42003, wrae013, the coral spawning paper, fmicb, the plankton
filtration-bias paper, and PNAS -- the last one is always discovered via
citation-expansion from s42003 rather than seeded directly, so it's
expected to show up even though it's not its own row).

```bash
mkdir -p logs
sbatch --account=191001-364393 cluster/run_discovery.sbatch
# wait for it to complete -- squeue -u $USER, or:
sbatch --dependency=afterok:<job_id_from_above> cluster/run_extraction.sbatch
# or, for vLLM instead of the default Ollama:
sbatch --dependency=afterok:<job_id_from_above> --export=ALL,LLM_BACKEND=vllm cluster/run_extraction.sbatch
sbatch --account=191001-364393 --dependency=afterok:27427 --export=ALL,LLM_BACKEND=vllm cluster/run_extraction.sbatch


```

Check progress:

```bash
tail -f logs/discovery_<job_id>.out      # or extraction_<job_id>.out
python -m fair_ocean_agent.cli status    # task queue counts, any time
```

Results land in `data/exports/audits/cluster_run_<job_id>/` (projectMetadata.csv,
sampleMetadata.csv, experimentRunMetadata.csv, field_reference.csv,
api_paper_corrections.csv). Pull them back to your Mac:

```bash
scp -r <cluster>:<path-to-repo>/fair_ocean_agent/data/exports/audits/cluster_run_<job_id> ./
```

## Scaling to the full ~3000-paper queue

Same two scripts, different seed file: point `SEED_FILE` at
`data/seeds/mdp_all_imported_dois.txt` (needs converting to the
`seed_id,title,doi,...` CSV shape `ingest-seeds` expects first -- see how
`cluster/seeds_five_papers.csv` is built for the column shape) or a
subset of it, e.g.:

```bash
sbatch --export=ALL,SEED_FILE=data/seeds/full_batch_001.csv cluster/run_discovery.sbatch
```

Worth doing in batches of a few hundred rather than all ~3000 at once,
at least for the first real large-scale run -- easier to notice and
retry a stuck batch than to debug a single multi-day job.

## Closed-access papers (local PDFs)

A paper with no PMCID at all (never deposited in Europe PMC/PubMed, even
when it's genuinely freely readable elsewhere -- common for non-
biomedical-focused journals, e.g. AGU's earth-science titles) has no route
through the normal discovery/extraction path: there's no PMCID to fetch
full text with, and no PMID for NCBI's own citation-linking either. A
locally-supplied PDF is the real alternative source for these, for **both**
discovery (mining the PDF's own text for BioProject/SRA/Zenodo/Dryad/etc.
accessions) and extraction (running the LLM over the PDF's own sections)
-- both stages check for one independently, so both `run_discovery.sbatch`
and `run_extraction.sbatch` need the same env var set to get full benefit
from a supplied PDF.

Drop each PDF into one shared directory, named after that paper's own DOI
with `/` replaced by `_` and lowercased -- e.g. `10.1002/2015JG003300`
becomes:

```bash
mkdir -p data/local_pdfs
cp ~/Downloads/jacobs-2021-palmyra.pdf data/local_pdfs/10.1007_s00338-021-02143-5.pdf
```

Then point `FAIR_OCEAN_LOCAL_PDF_DIR` at that directory on **both**
submissions -- a paper with no matching file in the directory (the normal
case for most of a batch) just falls through to the regular Europe PMC
path unchanged, so PDF-supplied and non-PDF papers run together in the
same batch with no other changes:

```bash
sbatch --account=191001-364393 \
  --export=ALL,SEED_FILE=cluster/seeds_fifty_papers.csv,FAIR_OCEAN_LOCAL_PDF_DIR=$(pwd)/data/local_pdfs \
  cluster/run_discovery.sbatch
sbatch --account=191001-364393 --dependency=afterok:<job_id> \
  --export=ALL,LLM_BACKEND=vllm,FAIR_OCEAN_LOCAL_PDF_DIR=$(pwd)/data/local_pdfs \
  cluster/run_extraction.sbatch
```

`FAIR_OCEAN_LOCAL_PDF_PATH` (a single PDF path, no per-study lookup) still
works exactly as before for a quick one-paper test -- it's checked first
and wins over `FAIR_OCEAN_LOCAL_PDF_DIR` whenever both are set, so don't
leave it set by accident in a shell you're about to run a real batch from.

## Troubleshooting

- **`pip install vllm` (or anything else) fails with "Disk quota
  exceeded"**: `$HOME` is full/quota-limited -- see `CONDA_ENV_PREFIX` in
  the "One-time setup" section above to move the env (and pip's cache)
  to scratch space instead. If you already have a partially-installed env
  under `$HOME/miniforge3/envs/...` from before making this switch, free
  the quota it used with `conda env remove -n <name>`.
- **`sbatch: error: Invalid partition`**: the partition names in these
  scripts (`service`, `gpu-a100`) are copied from `multimodal_granite`'s
  own jobs as a starting point -- confirm the real names for your
  account with `sinfo` and edit the `#SBATCH --partition=` lines.
- **`run_extraction.sbatch` exits with "Model '...' is not pulled"**: run
  the one-time `ollama pull` step above from a node with internet access.
- **Ollama giving you trouble generally** (install permissions, service
  won't start, etc.): switch to vLLM -- see the "vLLM" setup section
  above. No pipeline code changes needed either way, just
  `config/local.yaml` and `LLM_BACKEND=vllm` at submit time.
- **`run_extraction.sbatch` with `LLM_BACKEND=vllm` hangs or times out
  waiting for vllm**: check `logs/vllm_<job_id>.log` directly -- a model
  that wasn't downloaded first (see `download_vllm_model.sh`) fails there
  with a clear `HF_HUB_OFFLINE` error rather than in the main log.
- **`logs/vllm_<job_id>.log` shows `TypeError: type 'array.array' is not
  subscriptable` in `flashinfer/comm/fd_exchange.py`**: an old
  `flashinfer-python` (pulled in by `pip install vllm`) that only works on
  Python 3.12+ -- run `pip install "flashinfer-python>=0.6.16.post4"` in
  the same env (see the vLLM setup section above), then resubmit.
- **`run_discovery.sbatch` fails with connection errors**: that
  partition doesn't actually have outbound internet after all -- check
  with your cluster's support docs, or test with a plain `curl
  https://api.crossref.org` from an interactive session on that
  partition before submitting a real job.
- **Re-running is safe**: every stage here is idempotent (re-running
  `ingest-seeds`/`enqueue-*-backfill`/`worker --until-empty` against the
  same database only processes what's actually new or still pending), so
  a failed or interrupted job can just be resubmitted.
