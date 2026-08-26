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

## Running discovery on your Mac first, then syncing to the cluster

`run_discovery.sbatch` doesn't have to be where discovery happens for the
first time -- `ingest-seeds`, `enqueue-*-backfill`, and `worker` are all
idempotent against whatever database they're pointed at (see
Troubleshooting below), so running the non-LLM stages locally first
(`python -m fair_ocean_agent.cli worker --until-empty --no-llm`, plus
`scripts/auto_fetch_missing_pdfs.py` for extra PDF coverage) and then
copying that database up is equivalent to running `run_discovery.sbatch`
cold, just with the network calls made from your Mac instead of the
cluster's service partition.

This matters because your Mac and the cluster are two separate
filesystems with nothing shared but whatever's in the seed CSV -- without
syncing the database itself, `run_discovery.sbatch` has no way to know a
paper's DOI/PDF/samples were already resolved locally, and will
re-resolve them from scratch, hitting OpenAlex/Crossref/NCBI/ENA a second
time for the same papers. Syncing first avoids that:

```bash
# On your Mac, once your local worker/auto-fetch run has finished:
./cluster/sync_local_db_to_cluster.sh <cluster> /scratch/morrill/users/hmp278/FAIRe_Ocean_Agent
```

Then submit `run_discovery.sbatch` as usual -- it'll skip every study the
Mac already resolved and only do network work for what's new or still
pending (a study added since your last sync, a retry, citation-expansion
fanout the Mac's own `--no-llm` run already triggered but didn't finish).
There's no need to also run `ingest-seeds`/`enqueue-seed-backfill`
manually first; `run_discovery.sbatch` already does both, and both are
safe to run again against an already-populated database.

If you'd rather not manage a second discovery pass at all, skip this and
just run `run_discovery.sbatch` cold as in the rest of this doc -- it's
the simpler default. This is worth it specifically when you want to
inspect or filter results locally before committing cluster time (e.g.
`scripts/export_hpc_ready_seeds.py`'s "only studies with confirmed real
samples and a PMCID/PDF" filter needs the discovery to have actually
happened somewhere first).

### Moving GOLD/seed-discovery work to the cluster too

Everything above only moves the main pipeline database. If you've also
been running the GOLD BioProject publication search
(`run_gold_bioproject_publication_search.py`) or the MGnify/ENA seed
discovery marathon (`scripts/run_seed_discovery_marathon.sh`) on your
Mac -- both are long, network-bound, and don't need a GPU, so they're
good candidates to move to the cluster's service partition instead of
relying on your Mac staying online for days:

```bash
# Stop every Mac-side worker/discovery process first -- sync_local_db_to_cluster.sh
# refuses to copy a database another process still has open for writing.
./cluster/sync_local_db_to_cluster.sh <cluster> /scratch/morrill/users/hmp278/FAIRe_Ocean_Agent
```

This now also syncs `data/jgi_gold/gold_sharded.sqlite` and
`data/seed_discovery/mgnify_paper_seeds.sqlite` (plus the bulk Europe PMC
accession index those seed-discovery runners read from locally) if
they're present. Then on the cluster:

```bash
sbatch --account=191001-364393 cluster/run_gold_bioproject_search.sbatch
sbatch --account=191001-364393 cluster/run_seed_discovery_marathon.sbatch
```

Both are fully resumable (every accession/study's outcome is recorded as
it's checked), so if either hits its walltime before finishing, just
resubmit the same command -- it picks up exactly where it left off. Both
deliberately never touch OpenAlex (see each script's own docstring), so
they're safe to run alongside `run_discovery.sbatch` without doubling
load on it.

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

**Automatic fetch for genuinely open-access papers.** Before falling back
to "no PDF, no route" for a study with no PMCID, `run_discovery.sbatch`
also tries OpenAlex's own `best_oa_location.pdf_url` (already fetched as
part of ordinary discovery) and downloads it automatically if OpenAlex
marks the paper truly open-access -- using this project's own honestly-
identifying User-Agent, never a spoofed browser one. This works well for
papers hosted somewhere permissive (PLOS, Frontiers, Nature, preprint
servers); it does **not** get you out of downloading anything from a
publisher that blocks plain automated requests even to open-access
content (confirmed live: Wiley 403s these) -- those still need a manual
download exactly as before. Nothing to configure: successes are saved
straight into `FAIR_OCEAN_LOCAL_PDF_DIR` (or `data/auto_fetched_pdfs/` if
that's not set) using the same DOI-based filename, so a paper it
auto-fetches is indistinguishable from one you supplied by hand, and one
you already supplied by hand is never re-fetched.

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
