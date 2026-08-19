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

### Config

```bash
cp cluster/local.yaml.cluster-example config/local.yaml
```

`config/local.yaml` is gitignored on purpose (same as your local Mac's
own copy) -- it's where the real LLM endpoint lives. The example assumes
`run_extraction.sbatch` starts `ollama serve` itself inside the GPU job,
so `base_url` stays `http://localhost:11434/v1`. If your cluster instead
runs vLLM (`vllm serve <model> --port ...`, also OpenAI-compatible) or a
shared, persistent endpoint on a different host, point `base_url` there
instead and set `USE_OLLAMA=0` when submitting `run_extraction.sbatch`
(see below) -- no other pipeline changes needed either way.

### One-time model download

GPU nodes on most clusters have no internet access (this is normal, and
is exactly why `run_extraction.sbatch` never tries to reach one) -- so
the model has to be downloaded once, from somewhere that *does* have
internet, onto storage the GPU node can also see:

```bash
# On a login node, or the same "service"/HTTPS partition as run_discovery.sbatch:
curl -fsSL https://ollama.com/install.sh | sh    # if ollama isn't already available
ollama pull qwen3:4b-instruct-16k                # or whatever config/local.yaml's llm.model is
```

Ollama caches the model under `~/.ollama/models` -- as long as `$HOME` is
the same shared filesystem the GPU job sees (normal on HPC), this only
needs to happen once, not per job.

If `ollama` isn't installable on your cluster at all, vLLM is a plain
`pip install vllm` and serves the same OpenAI-compatible API -- swap
`USE_OLLAMA=0` and adjust `config/local.yaml`'s `base_url`/`model`
accordingly; the rest of this setup is unchanged.

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
- **`run_discovery.sbatch` fails with connection errors**: that
  partition doesn't actually have outbound internet after all -- check
  with your cluster's support docs, or test with a plain `curl
  https://api.crossref.org` from an interactive session on that
  partition before submitting a real job.
- **Re-running is safe**: every stage here is idempotent (re-running
  `ingest-seeds`/`enqueue-*-backfill`/`worker --until-empty` against the
  same database only processes what's actually new or still pending), so
  a failed or interrupted job can just be resubmitted.
