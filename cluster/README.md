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
cd FAIRe_Ocean_Agent/fair_ocean_agent
./cluster/setup_env.sh
```

This creates `.venv`, installs the package, and initializes an empty
database at `data/fair_ocean.db`. Edit `cluster/setup_env.sh` first if
your cluster needs `module load` instead of a plain `python3` on PATH.

### Config and LLM backend

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
sbatch cluster/run_discovery.sbatch
# wait for it to complete -- squeue -u $USER, or:
sbatch --dependency=afterok:<job_id_from_above> cluster/run_extraction.sbatch
# or, for vLLM instead of the default Ollama:
#   sbatch --dependency=afterok:<job_id_from_above> --export=ALL,LLM_BACKEND=vllm cluster/run_extraction.sbatch
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

## Troubleshooting

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
- **`run_discovery.sbatch` fails with connection errors**: that
  partition doesn't actually have outbound internet after all -- check
  with your cluster's support docs, or test with a plain `curl
  https://api.crossref.org` from an interactive session on that
  partition before submitting a real job.
- **Re-running is safe**: every stage here is idempotent (re-running
  `ingest-seeds`/`enqueue-*-backfill`/`worker --until-empty` against the
  same database only processes what's actually new or still pending), so
  a failed or interrupted job can just be resubmitted.
