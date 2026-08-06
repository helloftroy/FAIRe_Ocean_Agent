# fair-ocean-agent

Evidence-backed, resumable curation pipeline for marine eDNA / molecular
ocean-observing study metadata. See the full design brief for the complete
26-section specification this implements incrementally; this README covers
what's actually built (**Milestones 1-7**) and how to run it.

## What's built

**Milestone 1 (Foundation):** repository scaffold, config loading, the full
schema (all milestones' tables, so later work never needs a breaking
migration -- 16 tables originally, 14 after Milestone 7's simplification,
see "Schema simplification" below), Alembic migrations, identifier normalization +
exact-match deduplication, a CSV/JSONL seed loader, a task queue with
idempotent enqueue/claim/retry, a CLI, structured logging.

**Milestone 2 (initial source resolution):** Crossref, Europe PMC, and
OpenAlex adapters (rate-limited, retried, on-disk cached), wired into the
`DISCOVER_IDENTIFIERS` task handler -- DOI-keyed studies get a real
title/authors/year/journal/OA-status/PMID/PMCID/OpenAlex-ID resolution,
with Stage 2 (explicit-relationship) dedup when a newly-discovered
identifier already belongs to a different canonical study. Validated
against 100 real DOIs (see "Real seed data" below): 100/100 resolved
successfully in ~44s live.

**Milestone 3 (molecular repositories):** NCBI BioProject + BioSample
adapters (E-utilities: esearch/elink/efetch, XML parsing) and an ENA
adapter (study + read_run, clean JSON) added to the same handler --
BioProject/ENA-accession-keyed studies now get real per-sample structured
metadata (collection date, depth, lat/lon, ENVO terms, ...) materialized as
`sample` Entity rows, physical run/file metadata as `sequencing_run`
entities, and sample/assay-specific library instances as `experiment_run`
entities. Explicit entity relationships preserve sample -> library,
library -> assay, and library -> sequencing-run links, including multiple
libraries multiplexed on one physical run. A study with both a DOI and a
BioProject accession gets both resolved in one task. Validated against a
real public BioProject (837 linked BioSamples, 500+ sequencing runs) --
see "Real seed data" below.

NCBI SRA is deliberately not a separate adapter: run-level data is served
via ENA's read_run query instead (same underlying INSDC-shared records,
much cleaner JSON than NCBI's SRA XML) -- see `sources/ncbi.py`'s
docstring and `docs/architecture.md`.

After upgrading an existing database to the experiment/library entity
model, run `fair-ocean enqueue-mapping-backfill` and process those tasks.
Mapping task keys include a mapping version, so this schedules a fresh
pass even when older MAP_FAIRE tasks are already completed; legacy
run-bound raw facts remain intact and are projected onto experiment_run
entities during that pass.

**Milestone 4 (open-access retrieval + provider-independent LLM
extraction):** open-access full-text retrieval (Europe PMC JATS XML only --
never a paywalled source) + deterministic section selection (Methods,
Sampling, DNA extraction, Data Availability, ...), wired into a new
`EXTRACT_TEXT_FACTS` task handler. Every extracted fact requires a verbatim
evidence quote, checked deterministically against the source section
before persistence (`extraction/evidence.py`) -- a fabricated or
paraphrased quote is dropped, never stored.

The LLM side is a **provider-independent `LLMBackend` abstraction**
(`llm/base.py`), not an OpenAI client:

- `OpenAICompatibleHTTPBackend` (`llm/http_backend.py`) speaks the
  OpenAI-compatible chat-completions wire protocol -- the request/response
  *shape*, not a vendor. It never contacts an OpenAI-operated host, never
  needs OpenAI credentials, and sends data only to whatever `base_url` is
  configured (Ollama, vLLM, TGI, an institutional gateway, ...).
- `MockLLMBackend` and `DisabledLLMBackend` for tests and for the default
  (`llm.enabled: false`) state -- a code path that tries to use a disabled
  LLM fails loudly and immediately, never silently.
- **No model is chosen or hard-coded anywhere.** `llm.model` must be set
  explicitly; `build_llm_backend`/`build_benchmark_backend` both reject
  placeholder values rather than defaulting to a real one.
- A **benchmark harness** (`llm/benchmark.py`, `fair-ocean benchmark-models`)
  for comparing multiple candidate open-weight models side by side against
  a set of gold-standard extraction cases (see "Model benchmarking" below)
  -- JSON-validity rate, evidence-verification rate, precision/recall/F1
  against gold, and latency, exported as CSV + JSON for use in a paper.

Validated against a real local Ollama server (qwen2.5:3b) -- see
"Milestone 4 validation" below for the real run and the three bugs it
surfaced (none of them mock-testing could have caught).

**Milestone 5 (data-asset inventory, missingness, validators):**
`INVENTORY_DATA_ASSETS` materializes `DataAsset` rows for ENA sequencing
runs directly from Milestone 3's already-extracted raw_facts (file
accession/size/checksum -- never the file itself); `VALIDATE_LOGIC` checks
coordinate/depth plausibility, collection-date-vs-publication-year
ordering, primer-sequence character validity, and re-validates identifier
formats; `VALIDATE_EVIDENCE` audits that every fact's evidence bookkeeping
matches what its `support_type` promises; `VALIDATE_CROSS_SOURCE` compares
independently-extracted publication titles across Crossref/Europe
PMC/OpenAlex and flags real disagreement rather than silently trusting
whichever source answered first; missingness gets recorded (not just left
as blank cells) for a core set of sampling fields. `fair-ocean validate
--study-id X` shows recorded results and flags anything needing manual
review. See "Milestone 5 validation" below -- this one caught a genuine
third-party data-quality bug in OpenAlex itself, exactly as designed.

**Milestone 6 (FAIRe mapping + export):** a rules-table-driven mapping
layer (`mapping/rules.py`, `mapping/vocabularies.py`, `mapping/units.py`,
`mapping/faire.py`) turns raw_facts into FAIRe (v1.0.2) `StandardizedValue`
rows via the `MAP_FAIRE` task handler (`fair-ocean
enqueue-mapping-backfill`), and `fair-ocean export-faire --output <dir>`
writes one CSV per FAIRe class (`projectMetadata`, `sampleMetadata`,
`ampData`, `stdData`, `experimentRunMetadata`, `eLowQuantData`, `taxaRaw`,
`taxaFinal`), matching the vendored
`FAIRe_checklist_v1.0.2_FULLtemplate.xlsx`'s sheet names/column order
(`schemas/faire/`). Coverage is real but partial by design -- see
"Milestone 6 validation" below for exactly what's mapped, what's
deliberately not, and why.

**Milestone 6b (standards registry):** `schemas/miop/` and `schemas/bebop/`
vendor the MIOP schema and BeBOP protocol templates alongside
`schemas/faire/`; `standards/` (`faire_registry.py`, `miop_registry.py`,
`bebop_templates.py`, `crosswalk.py`, `registry.py`) compiles all three
into a normalized, provenance-tracked registry via `fair-ocean
build-standards-registry` -- `standards/compiled/faire_registry.json`,
`bebop_miop_registry.json`, `term_crosswalk.csv`,
`template_field_usage.csv`, and a self-checking
`standards_validation_report.json`. This is schema compilation, not
mapping: it resolves which upstream term each BeBOP protocol-template
field refers to (MIOP or FAIRe, by a fixed dedup priority -- see
"Milestone 6b" below), it does not map this pipeline's own raw_facts onto
BeBOP/MIOP fields. **BeBOP/MIOP raw_fact mapping (`mapping/bebop.py`,
`exports/bebop.py`) is still not implemented** -- not blocked on a missing
schema anymore, but on an open design question: MIOP/BeBOP fields
describe a *protocol document* (who wrote it, its license, version), not
per-study sample/sequencing facts, so `mapping/faire.py`'s raw_fact ->
StandardizedValue pattern doesn't obviously transfer. See
`mapping/bebop.py`'s docstring.

**Milestone 7 (weekly update: refresh, retry sweeps, quarterly
rediscovery):** `fair-ocean weekly-update [--dry-run]` re-checks every
study's already-known DOI/BioProject/ENA sources for real-world change --
a BioProject genuinely growing past a prior `MAX_SAMPLES_PER_PROJECT`
cap, a citation count updating, embargoed data going public -- via a new
`REFRESH_STUDY_SOURCES` task (`workflow/refresh_handlers.py`) that always
fetches fresh (bypassing the HTTP cache, `RateLimitedClient.clear_cache()`)
and diffs by `content_hash` against the most recent prior `Source`
snapshot: unchanged means nothing new is written; changed means a new
`Source` row is added (linked via `parent_source_id`) with its own fresh
facts, never overwriting the old snapshot -- raw_facts stays an append-
only evidence log. `SourceWatermark` (modeled since Milestone 1, unused
until now) records the outcome of every check. The same command also
implements `SchedulingConfig`'s other cadences: `retry_failed_after_hours`
and `monthly_unresolved_retry` reset long-stale FAILED/MANUAL_REVIEW_REQUIRED
tasks back to PENDING (distinct from `fail_task`'s existing short-term
exponential backoff, which gives up permanently after `max_attempts`),
and `quarterly_full_rediscovery` gives every study one more
DISCOVER_IDENTIFIERS pass roughly every 90 days, catching e.g. a newly-
enabled adapter that existing studies were never checked against.
`WorkflowRun` (also modeled since Milestone 1, also unused until now) gets
a real row per invocation, so `fair-ocean report-run` finally has
something to show. `weekly-update` is enqueue-only, like every other
`enqueue-*` command -- run `fair-ocean worker --until-empty` separately to
process what it queues. See "Milestone 7 validation" below.

**Milestone 7 continued (schema simplification):** the `missingness` and
`publications` tables were folded into `standardized_values` and `sources`
respectively -- two tables that each had to stay in sync with information
that already had (or now has) a natural home, at the user's request for a
simpler, easier-to-debug schema. See "Schema simplification" below for
the full design and the real migration/data-carryover details.

**Milestone 8 (FAIRe-aware extraction, corrected to v3):** the LLM
text-extraction prompt (`extraction/text.py`) went from fully
open-vocabulary -- "extract whatever you find, name it however you like"
-- to structured: `extraction/faire_fields.py` is a single-source-of-truth
taxonomy of ~70 atomic concepts, grouped the way FAIRe groups them (DNA
extraction, PCR/assay setup, controls & replicates, qPCR/standard curve,
sequencing/library prep, bioinformatics workflow, taxonomic assignment
output), each with a short prompt-facing hint. **v2 initially asked the
model to use FAIRe's own exact field spelling (`annealingTemp`, `neg_cont_type`,
`otu_db`, ...) as `fact_type_candidate` itself -- coupling a raw fact's own
identity to one specific standard, the same coupling this pipeline
deliberately avoids everywhere else (a repository adapter's
`fact_type_candidate` is never phrased in Darwin Core/MIxS's own spelling
either). Caught before any real data was built on it, and corrected in
v3:** `fact_type_candidate` is now always a plain, standard-agnostic
native name (`annealing_temperature`, `negative_control_type`,
`reference_database`, `scientific_name`), and the model may additionally
return an *optional* `candidate_standard_fields` hint per fact (e.g.
`{"faire": "annealingTemp"}`), stored in `RawFact.confidence_metadata` (a
pre-existing, previously-unused JSON column -- no migration needed) and
never folded into the fact's own identity. Dropping every hint still
leaves a fully valid, source-native raw fact; standardizing onto FAIRe
remains entirely `mapping/rules.py`'s job, a separate downstream step. An
open fallback is still preserved for explicitly-stated facts that don't
fit any listed concept -- structure is the default, not the only option.
This is what unlocks PCR volumes, primer concentrations, assay names,
controls, replicate structure, thresholds, standard curves, and taxonomy
outputs that the old coarse-blob prompt (`PCR_amplification_conditions` as
one string) could never separate into atomic facts, without coupling
extraction to any one standard's vocabulary.
`extraction/sections.py`'s title patterns and default text budget were
both expanded to match (see "Milestone 8" below for why). The benchmark
harness (`llm/benchmark.py`) now runs every gold case through the exact
same `build_prompt` production code calls, instead of gold cases each
carrying their own free-form `instructions` field that could silently
drift from the real prompt -- see "Model benchmarking" below.

**Supplementary-material and structured-table retrieval layer:**
`DISCOVER_SUPPLEMENTS`/`RETRIEVE_SUPPLEMENTS` task handlers
(`workflow/supplement_handlers.py`) discover `<supplementary-material>`
references in a paper's already-cached JATS XML, fetch Europe PMC's single
supplementary-files zip bundle when the known total size is under a
config cap, and deterministically parse CSV/TSV/XLSX/XLS/JSON/XML members
(`sources/supplement_parsing.py`) into the same provenance-aware
`raw_facts` shape as any structured-API fact. TXT/MD/PDF become reusable
prepared text; a separate, disabled-by-default task can later query only
FAIRe fields still missing after the paper pass. See "Supplementary-
material and structured-table retrieval layer" below for the full design
and what real live validation against PMC7469538 found (and fixed).

**Shared resolve_or_create_study() with confidence tiering:** DOI-seeded
and accession-seeded discovery now route every ambiguous-identifier merge
decision through one function (`identity/resolution.py`), gated by a
three-tier evidence-confidence hierarchy (structured API relation <
regex-matched-then-API-verified accession < LLM prose claim, reusing
`SupportType`) and a new date/location consistency check
(`identity/consistency.py`) before merging into a pre-existing Study --
previously any identifier match, regardless of evidence quality, triggered
an unconditional merge. A new additive `study_sources` join table (plus a
single `create_source()` choke point every Source-creation call site now
routes through) makes a Source belonging to more than one Study
representable for the first time. See "Shared resolve_or_create_study()"
below for the full design and the real 101-study validation run.

**Text-extraction speed fix (v7 -> v8): real Ollama context, collapsed
checklist passes.** Root-caused a genuine, measured ~700-900s-per-section
slowdown to two compounding causes: Ollama's OpenAI-compatible endpoint
silently ignoring `num_ctx` (capping every model at 4096 tokens regardless
of config) and `extraction/text.py`'s 5-topic-focus x recall-retry split
(sized around that same 4096-token ceiling) multiplying out to up to 10
sequential LLM calls per section. Fixed both: a `qwen3:4b-instruct-16k`
Ollama model variant with `num_ctx` genuinely baked into its own Modelfile,
and `extract_facts_from_section`'s default collapsed to a single pass over
the full checklist per chunk, with recall now only firing when a pass
found zero facts. Measured **~12-15x** faster live (mean 61.4s/case vs.
~700-900s/section), with JSON validity 100% and evidence verification
98.5% on the full 18-case gold benchmark. See "Fixing the real Ollama
context limit and collapsing the per-topic passes" below for the honest
precision/recall numbers and why spot-checking traced most of the gap to
pre-existing gold-data drift, and "Gold-case label-drift/completeness
cleanup" below for the fix.

**What it does not do yet:** BeBOP/MIOP raw_fact mapping (see
above), or discovering brand-new studies via keyword search/citation
expansion (`DiscoveryConfig.keyword_search_enabled`/
`citation_expansion_max_depth` exist but are unbuilt --  Milestone 7 only
re-checks studies already in the database). The `discover`, `run-study`,
and `export-bebop` CLI commands exist (matching the brief's full command
list) but exit with a clear "not yet implemented" message rather than
doing nothing silently.

## Install

Requires Python 3.10+.

```bash
cd fair_ocean_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,postgres]"
cp .env.example .env   # edit FAIR_OCEAN_CONTACT_EMAIL at minimum
```

The `postgres` extra installs `psycopg2-binary`; skip it if you're staying
on SQLite for now.

## Quickstart

```bash
fair-ocean init-db                                       # create tables from the ORM models
fair-ocean ingest-seeds data/seeds/studies_template.csv   # 3 fictional example studies
fair-ocean enqueue-seed-backfill                          # queue DISCOVER_IDENTIFIERS per study
fair-ocean status                                          # see queue counts
fair-ocean worker --until-empty                            # resolve via Crossref/Europe PMC/OpenAlex
fair-ocean export-raw-facts --output data/exports/raw_facts.csv
```

`worker` now makes real, rate-limited network calls to Crossref, Europe
PMC, OpenAlex (DOI-keyed), and NCBI/ENA (BioProject/ENA-accession-keyed)
for any study carrying one of those identifiers (see `.env` — set
`FAIR_OCEAN_CONTACT_EMAIL` first; these APIs expect a real contact in the
User-Agent). A study with none of DOI/BioProject-accession/ENA-accession
raises a clear `NotImplementedError` from the handler and lands in
`manual_review_required` after retries exhaust -- expected for
OBIS/GBIF/BCO-DMO/PANGAEA-only studies until later milestones add those
adapters.

The example seed file (`data/seeds/studies_template.csv`) has three
fictional studies exercising different identifier combinations (DOI +
BioProject, DOI only, BioProject only) -- re-run `ingest-seeds` on it again
and note it reports 3 *merged*, not 3 new, because exact-identifier dedup
matches them back to the same canonical studies.

## Real seed data: the mdp_dois_links.md extraction

`data/seeds/studies.csv` is not fictional -- it's 100 real DOIs
stride-sampled (every ~36th entry, to spread across publishers/years rather
than just the first alphabetical chunk) from the "Imported into Paperpile"
section of a user-supplied `mdp_dois_links.md` reading list. Related files
in `data/seeds/`:

- `mdp_all_imported_dois.txt` -- all 3,618 valid DOIs from that section, for
  the eventual full historical backfill (Milestone 7).
- `mdp_malformed_doi_entries.txt` -- 6 entries with copy-paste artifacts
  (e.g. trailing "External Link" text) that fail DOI validation and need
  manual cleanup before use.
- `mdp_not_imported_links.txt` -- 327 non-DOI links (institutional
  repository handles, direct PDFs, Google Scholar search URLs, ProQuest
  view URLs) from the "Not Imported into Paperpile" section. These aren't
  exact identifiers and need URL/citation resolution, not the exact-match
  path -- out of scope until there's a source adapter that can do that.

Validated end-to-end: `fair-ocean ingest-seeds data/seeds/studies.csv` then
`enqueue-seed-backfill` then `worker --until-empty` resolved 100/100 in
~44s live, producing 99 Publications (one DOI had no record in any of the
three sources), 260 Source rows, 1,939 raw_facts, and 300
external_identifiers (100 DOI + 99 OpenAlex ID + 60 PMID + 41 PMCID) with
zero duplicates on repeated ingestion/enqueue.

## Milestone 3 validation: a real public BioProject

`data/seeds/milestone3_validation.csv` seeds one real, public BioProject
accession (`PRJNA1425045`, "SF Bay 18S Metabarcoding Monitoring" -- an
actual eDNA metabarcoding study, not a synthetic test fixture) with no DOI,
specifically to exercise the repository-resolution path end-to-end:

```bash
fair-ocean ingest-seeds data/seeds/milestone3_validation.csv
fair-ocean enqueue-seed-backfill
fair-ocean worker --until-empty
```

This BioProject has 837 linked BioSamples and 500+ sequencing runs --
large enough to exercise the `MAX_SAMPLES_PER_PROJECT`/`MAX_RUNS_PER_STUDY`
truncation caps (300/500) for real, with the truncation logged, not
silent. Live result: 3 Source rows (ncbi_bioproject, ncbi_biosample, ena),
800 Entity rows (300 sample + 500 sequencing_run), 8,115 raw_facts, and 622
unique BioSample-accession external_identifiers (NCBI's and ENA's sample
lists overlap heavily but aren't identical -- both get discovered and
merged into one deduplicated set, not double-counted).

This run is also what surfaced two real bugs, both fixed and covered by
regression tests (`tests/unit/test_worker.py`,
`tests/unit/test_handlers_repository.py`):

1. **Cross-adapter identifier collision.** ncbi_biosample and ena commonly
   report the *same* BioSample accession (they mirror the same underlying
   records). The related-identifier dedup check used to read a
   `study.external_identifiers` ORM collection that goes stale mid-task --
   an identifier added by one adapter's pass through the loop didn't
   retroactively appear in an already-loaded collection, so a second
   adapter reporting the same identifier hit the table's unique constraint.
   Fixed by checking a fresh DB query instead (`find_existing_study_by_identifier`)
   for every identifier, every time.
2. **Worker crash on constraint-violation failures.** That IntegrityError
   left the SQLAlchemy session in a "pending rollback" state; the worker's
   `except` handler called `fail_task()` without rolling back first, so
   `fail_task`'s own writes raised `PendingRollbackError` and crashed the
   whole CLI command instead of marking one task failed and moving on.
   Fixed by rolling back before `fail_task` runs.

## Milestone 4 validation: section-selector coverage against all 41 real PMCIDs

Before trusting the deterministic section-selector (`extraction/sections.py`)
against real papers -- it had only been checked against one paper before
this -- it was run live against every one of the 41 real PMCIDs discovered
in the 100-DOI seed set (no LLM needed for this check; see
`docs/architecture.md` for how to reproduce it):

- 38/41 had full text available in Europe PMC; the other 3 are genuinely
  not open access (`isOpenAccess: N` confirmed directly against Europe
  PMC's API) -- Europe PMC's own `fullTextXML` endpoint 404s for them, so
  this pipeline never sees or attempts to route around that restriction.
- Before a fix: 1/38 papers (2.6%) had a real Methods section that matched
  *none* of the title patterns -- not a paywalled/format problem, a real
  regex gap. Its entire methodology was under a section titled
  **"Experimental procedures"** (a heading style Cell Press-style journals
  use instead of "Methods"/"Materials and Methods"), which none of the
  existing patterns (`method`, `material`, `sampl`, ...) caught.
- Fixed by adding a `procedure` pattern. Re-running: 0/38 misses, and the
  new pattern only matched genuinely relevant titles across the corpus
  ("Experimental procedures", "HTS procedure", "Laboratory procedures") --
  no noisy false positives introduced.
- Across the 38 papers: 1-16 relevant sections matched per paper (mean
  4.9), 888-20,000 characters selected per paper (mean ~8,100, well under
  the 20,000-char cap -- truncation rarely triggers), and **130 distinct
  real section titles** matched in total, confirming the pattern list
  generalizes well beyond the single paper it was originally checked
  against (e.g. `"2.2. Environmental DNA Extraction, Amplification and
  Sequencing"`, `"DNA extraction, library preparation, and sequencing"`,
  `"Nucleic acid extraction"`, `"Quantitative real-time PCR"`, ...).

See `tests/unit/test_extraction_sections.py::test_matches_experimental_procedures_heading`
for the regression test.

## Milestone 4 validation: a real local model (Ollama + qwen2.5:3b)

Everything above this point was tested against mocks. This is the first
real inference-server run: `LOCAL_LLM_BASE_URL=http://localhost:11434/v1`,
`LOCAL_LLM_MODEL=qwen2.5:3b` (set in `.env`; `llm.enabled`/`provider` set
in `config/local.yaml`, a gitignored machine-specific override --
`config/default.yaml` keeps shipping with `llm.enabled: false`).

**It worked end to end with zero code changes** -- factory → HTTP backend
→ JSON-retry loop → evidence verification → persistence, against a real
3B-parameter local model. 6 real papers processed (5 with facts, 1
correctly no-op'd on a non-open-access PMCID), 29 verified facts
persisted, real plausible content (real coordinates, kit names, primers,
platforms) spot-checked by hand.

**Two quality nuances, not bugs:**
- qwen2.5:3b occasionally elides a clause while claiming to quote
  verbatim (e.g. dropped "and finally 10 min at 72°C" mid-sentence). The
  evidence verifier correctly rejected that fact -- exactly as designed --
  but it means this model's effective recall is somewhat lower than its
  raw output would suggest. Worth quantifying against other candidate
  models via `benchmark-models` once more are configured.
- The benchmark's exact-match scoring was too strict for real use (see
  "Model benchmarking" and the assumptions section below) -- fixed.

**This run also surfaced three real bugs, all fixed and regression-tested,
none of them mock-testing could have caught:**

1. **SQLite path resolution depended on the process's cwd, not the repo.**
   `_ensure_sqlite_dir` created the data directory relative to `REPO_ROOT`,
   but the *connection* used the original (possibly relative) URL, which
   `sqlite3` resolves against whatever directory the process happens to be
   running from. Every prior command in this project had been run with
   the repo as cwd, so this never surfaced -- until a one-off analysis
   script run from one directory up got "unable to open database file."
   A cron job or systemd unit (Milestone 7) invoked from a different
   working directory would have hit the same failure. Fixed: `get_engine()`
   now rewrites any relative `sqlite:///` URL to an absolute path anchored
   at `REPO_ROOT` before connecting. See
   `tests/unit/test_database_session.py`.
2. **`config/local.yaml` was documented as gitignored but never actually
   added to `.gitignore`** -- caught while wiring up this real endpoint.
3. **Two tests silently depended on `config/local.yaml` not existing.**
   The moment a real one existed (this one), they broke -- they were
   reading real config off disk instead of controlling their own inputs.
   Both now explicitly isolate themselves from whatever's configured
   locally. This matters beyond just these two tests: `config/local.yaml`
   is *for* per-machine overrides, so any test that calls `load_config()`
   without isolating from it is implicitly assuming nobody running the
   suite has one -- true only by accident, not by design.

## Milestone 5 validation: real data, real bugs, one genuine third-party finding

`fair-ocean enqueue-validation-backfill` + `worker` run against all 101
real studies in the seed database (before any unit test for Milestone 5
existed) surfaced three real issues -- all fixed, all regression-tested:

1. **`populate_missingness_for_study` crashed on every study with a
   matching fact.** `select(RawFact.fact_type_candidate)` (a single-column
   select) returns plain strings via `session.scalars()`, not `RawFact`
   objects -- the code did `f.fact_type_candidate for f in scalars(...)`,
   which raised `AttributeError` immediately. A synthetic empty-study test
   would never have hit this; it only fires when a study actually has a
   matching fact. Fixed, and the regression test is deliberately shaped
   like the real data that triggered it (see
   `tests/unit/test_validation_handlers.py::test_populate_missingness_does_not_crash_on_real_shaped_data`).
2. **Crossref's `title`/`container-title` raw_facts were JSON-encoded
   lists, not plain strings** (`'["Some Title"]'` instead of `"Some
   Title"`) -- Crossref's API returns these as one-item lists, and
   `extract_structured_facts` stored the raw list instead of unwrapping it
   like `parse_publication_fields` already did. This made every crossref
   title compare as "different" from Europe PMC/OpenAlex's plain-string
   titles in cross-source validation, even when the title was identical.
   Fixed in `sources/crossref.py`; 192 already-persisted rows in the real
   database were repaired in place.
3. **Cross-source title comparison had a 61/98 (62%) false-conflict
   rate** -- Europe PMC's `title` field always ends with a trailing period
   that Crossref's and OpenAlex's don't. After fix #2, this was still the
   single largest source of noise. Fixed by stripping trailing punctuation
   before comparing (`validation/cross_source.py`'s `_normalize`) --
   dropped false conflicts from 61 to 14.

**The remaining 14 real conflicts included a genuine, confirmed
third-party data-quality bug, not a bug in this codebase**: for DOI
`10.1111/mec.17318`, OpenAlex's own API
(`api.openalex.org/works/https://doi.org/10.1111/mec.17318`) returns the
title *"Discovery Association Rules in Time Series Data"* -- a
completely different paper. Confirmed directly against OpenAlex's live
API, not an artifact of our adapter. `Publication.title` for this study is
still correct (Crossref is queried first and wins the first-non-null-wins
merge), but this is exactly why cross-source validation exists: if
OpenAlex had answered first for some other paper, its bad title would
have become the "canonical" one with nothing to catch it until this
validator ran. See
`tests/unit/test_validation_cross_source.py::test_genuinely_different_titles_are_conflicting`.

Final state after fixes: 500 data assets inventoried (matching Milestone
3's 500 real sequencing runs), 1,636 validation results (924 confirmed
accession formats, 608 confirmed logical checks, 84 confirmed + 14
conflicting + 3 not-assessed cross-source title comparisons), 303
missingness rows -- all through the real task queue (`claim_next_task` →
handler → `complete_task`), not just direct function calls.

## Milestone 6 validation: FAIRe mapping/export against the real database

The FAIRe schema itself came from the user's clone of
[FAIR-eDNA/FAIRe_checklist](https://github.com/FAIR-eDNA/FAIRe_checklist)
(v1.0.2, commit `042ced519c9a4e3808086e6078c12883cb884cd0`), vendored under
`schemas/faire/` -- see that directory's README for exactly what's
vendored and why. `mapping/rules.py`'s rule table was built directly from
this pipeline's own real, observed raw_facts vocabulary (the actual
`fact_type_candidate`/`entity_level` combinations present in the 101-study
database), not written speculatively -- see that module's docstring for
the full list of what's mapped and, just as importantly, what's
deliberately not (coarse LLM-extracted blobs are mapped to FAIRe's own
`*_method_additional` free-text fallback fields with `review_required`
set, never forced into the atomic fields they don't actually match).

`fair-ocean enqueue-mapping-backfill` + a `MAP_FAIRE`-only worker pass run
against all 101 real studies produced 3,201 `StandardizedValue` rows with
zero handler failures, and `export-faire` produced a 101-row
`projectMetadata.csv` and a 300-row `sampleMetadata.csv` (`ampData`/
`stdData`/`experimentRunMetadata`/`eLowQuantData`/`taxaRaw`/`taxaFinal`
are header-only -- no adapter or extraction step produces that data yet).
No mapping-code bugs turned up, but the run surfaced one genuine,
pre-existing finding:

**`materialSampleID` coverage for the real 837-BioSample/500-run
BioProject (Milestone 3's pressure-test study) is 178/300 (59%), not
178/500.** Every `sample_accession` raw_fact from ENA's 500 sequencing-run
records is looked up against this pipeline's own `sample` Entities before
becoming a `materialSampleID` value (see `mapping/faire.py`'s docstring on
why this is a targeted redirect, never a broadcast) -- and only 300
`sample` Entities exist for that BioProject in the first place, of which
only 178 are ever referenced by a run. This traces directly to
`sources/ncbi.py`'s `MAX_SAMPLES_PER_PROJECT = 300` cap (a deliberate,
already-documented Milestone 3 decision, logged as "not a silent drop" at
fetch time) -- the ENA run-level fetch has no such cap, so it references
322 BioSample accessions the BioSample fetch never pulled down. Not a
mapping bug: `mapping/faire.py` did exactly what it should (skip rather
than fabricate a match), and this is the first time that cap's downstream
effect on FAIRe export completeness has been made concrete. Revisiting
`MAX_SAMPLES_PER_PROJECT` for very large BioProjects is out of scope for
this milestone but worth knowing about before treating any single large
study's FAIRe export as complete.

## Mapping expansion: more structured-source fields, ordered by what's real today

Prompted by a direct review of what NCBI/ENA/OBIS/GBIF/PANGAEA structured
data actually contains vs. what `mapping/rules.py` mapped: many real,
already-fetched structured facts had no rule at all, not because they're
hard to map but because nobody had added the rule yet. Checked adapter
code directly (not assumed) before writing any rule, in three groups:

1. **Immediately real, added now**: `elev`, `samp_collect_device`,
   `samp_size`, `samp_size_unit`, `temp`, `salinity`, `ph`, `diss_oxygen`
   (NCBI BioSample's generic `Attributes/Attribute` passthrough --
   `sources/ncbi.py` already captures *any* named attribute a real
   BioSample record has; these 8 just had no rule yet, same mechanism as
   the already-mapped `geo_loc_name`/`env_broad_scale`/etc.); ENA's
   `read_count`/`fastq_ftp`/`fastq_md5` (already-fetched, previously
   thought "redundant with `DataAsset`" -- true for `base_count`/
   `fastq_bytes`, which really have no FAIRe field, but wrong for these
   three, which map onto real fields: `input_read_count`, `filename`/
   `filename2`, `checksum_filename`/`checksum_filename2`, plus an inferred
   constant `checksum_method` = "MD5" since ENA never reports another
   algorithm); `library_layout` (added to `sources/ena.py`'s `RUN_FIELDS` --
   the one genuinely new small adapter addition here -- mapped to FAIRe's
   `lib_layout`); `citation` (OBIS/GBIF/PANGAEA all already emit this exact
   field) -> `bibliographicCitation`.
2. **Checked, not real yet**: `associatedSequences` -- no adapter surfaces
   genetic-sequence identifiers as their own fact at all.
3. **Checked, contradicts the assumption that OBIS/GBIF already cover
   this**: `target_gene`/primer sequences/PCR conditions/annealing temp/
   amplicon size are **not** in `sources/obis.py` or `sources/gbif.py`'s
   current API calls -- both only fetch basic dataset/occurrence metadata.
   OBIS does have a DNA-derived-data/MIxS extension that can carry these
   fields, but the current adapter doesn't query it. Flagged rather than
   written as dead rules for facts that don't exist yet.
4. **Checked, contradicts the assumption it exists at all**: ENA's
   `LIBRARY_CONSTRUCTION_PROTOCOL` field isn't in `sources/ena.py`'s
   `RUN_FIELDS`. Same as (2) -- needs real adapter work first.

**A real bug this surfaced, not introduced by carelessness but caught by
checking live rather than assuming success:** `mapping/faire.py`'s
`_resolve_entity_id` only ever resolved a real per-entity `entity_id` for
SAMPLE-level facts -- every SEQUENCING_RUN-level fact fell through to the
study-wide broadcast default (`entity_id=None`). Harmless for the two
pre-existing run-level rules (`instrument_platform`/`instrument_model`,
which really are expected to agree across a study's runs and map onto
`projectMetadata`), but the moment a genuinely *per-run* field
(`filename`/`checksum_filename`/`input_read_count`, mapped onto
`experimentRunMetadata`) got its first rule, it silently collapsed all but
one run's value into a single row. Checked directly against the real
101-study database before fixing: only 1 `filename` row existed despite
500 real per-run `fastq_ftp` facts. Fixed `_resolve_entity_id` to also
resolve `SEQUENCING_RUN`-level facts to their own entity; re-ran the
mapping backfill against the real database and confirmed 500 `filename`,
500 `checksum_filename`, and 500 `input_read_count` rows, one per real
run, while `checksum_method`/`lib_layout` (mapped onto `projectMetadata`)
correctly still collapse to one project-wide row each.

Also worth being honest about: **the 8 new BioSample-attribute fields
(`elev`, `temp`, `salinity`, `ph`, `diss_oxygen`, `samp_collect_device`,
`samp_size`, `samp_size_unit`) and `bibliographicCitation` currently
produce zero real rows** against the actual 101-study database -- no
BioSample record in this corpus has reported those specific MIxS
attributes yet, and no OBIS/GBIF/PANGAEA-resolved study has a `citation`
fact captured yet either. The rules are correct and will engage the
moment a study's real source data has them; they just don't today. See
`tests/unit/test_mapping_faire.py` for coverage of every new rule with
synthetic data, and the real-data counts above for what's live right now.

## Milestone 6b validation: the standards registry against real vendored schemas

Unlike every other validation section in this README, there's no synthetic
fixture standing in for real data here -- the registry's whole job is
compiling the *actual* relationships between three real upstream schemas,
so the only meaningful validation is running it against those schemas
directly (`tests/unit/test_standards_registry.py` does exactly this: no
mocked schema, no synthetic BeBOP template).

`fair-ocean build-standards-registry` against the real vendored
`schemas/faire/`, `schemas/miop/`, and `schemas/bebop/` produced 337 FAIRe
terms, 21 MIOP terms (of the 21 slots in `miop/terms.yaml`, all non-
abstract), 0 BeBOP-specific terms, and 222 crosswalk rows -- every one of
which resolved to either `MIOP` (105 rows: 101 at priority 2 via an exact
slot-name match, 4 at priority 4 via a title-alias match) or `FAIRe` (117
rows, all at priority 3, exact slot-name match). None fell to priority 5
("possible duplicate requiring review") or came back `BeBOP-specific`/
`Unresolved`. All six validation checks
(`standards_validation_report.json`) passed: no duplicate canonical IDs,
every crosswalk row explicitly resolved, every MIOP `range` reference that
names an enum actually has one defined, every FAIRe class slot
(`classes.yaml`) resolves to a real term, every term retains full
upstream provenance (repository/source file/field name/commit), and no
BeBOP-specific term duplicates a field FAIRe or MIOP already define.

The one real wrinkle the real templates surfaced: the five protocol
templates don't spell MIOP field names consistently.
`protocol_template_bioinformatics.md` uses the slot's own name (`meth_cat`,
`maturity_level`), while the other four use its title in
underscore-joined form (`methodology_category`, `maturity level`) --
and one of the four (`broad-scale_environmental_context` vs. everyone
else's `broad_scale_environmental_context`) even mixes a hyphen into that.
`standards/miop_registry.py`'s `normalize_field_name` (collapses
hyphen/underscore/space/case) plus matching against *both* a slot's raw
name and its title (`MiopNameLookup`, priority 2 vs. 4) is what makes all
of these resolve to the same term instead of silently creating duplicate
entries for one field spelled two ways -- this wasn't a hypothetical
concern designed in from a spec, it's exactly what the real files needed.

## Milestone 6 follow-ups: FAIRe completeness, exact_mappings, and the BeBOP decision

Three loose ends from Milestone 6/6b, closed out before starting
Milestone 7:

**FAIRe Mandatory-field completeness** (`validation/faire_completeness.py`,
`VALIDATE_FAIRE_COMPLETENESS` / `fair-ocean
enqueue-faire-completeness-backfill`): of FAIRe's 32 fields with a
`requirement_level_condition`, most reference *other* FAIRe fields
(`samp_category`, `assay_type`, `pcr_0_1`, ...) this pipeline doesn't
populate anywhere -- so a completeness check can't evaluate "is this
exempt because samp_category = negative control" when samp_category is
never known. Rather than guess, this only checks **unconditionally**
Mandatory fields, persisted as `Missingness` rows (`target_schema="faire"`).
Run against the real 6 studies with any FAIRe standardized value: 906
rows, zero handler failures. Result, exactly as expected from already-
documented gaps: `eventDate`/`geo_loc_name`/`samp_name` present for all
300 real samples, `project_id` present for all 6; `assay_name`,
`samp_category`, `project_contact`, `recordedBy`, `assay_type`,
`checkls_ver`, `pcr_0_1` came back missing for **every** sample/study
checked -- these have no data source anywhere in this pipeline (no
adapter models a PCR assay as its own Entity, and nothing extracts
project-contact-style metadata). Not a new bug: this turns an existing
README-prose gap into structured, queryable data.

**`exact_mappings` in the FAIRe export** (`exports/faire.py`): alongside
`projectMetadata.csv`/`sampleMetadata.csv`/etc., `export_faire` now also
writes `field_reference.csv` -- one row per FAIRe field with its
requirement level and `exact_mappings` (real cross-standard URIs, e.g.
`env_broad_scale` -> `mixs:0000012`, that came for free with the vendored
FAIRe schema). Built from the same `standards.faire_registry
.build_faire_registry()` Milestone 6b already uses, not a second copy of
the same data. Deliberately a companion file, not extra columns squeezed
into the data CSVs -- those must keep matching FULLtemplate.xlsx's exact
layout.

**BeBOP/MIOP raw_fact mapping: decided out of scope, not unbuilt.**
MIOP/BeBOP describes metadata *about a protocol document* (author,
license, version, maturity level) -- every study this pipeline currently
ingests comes from a published paper or repository record, never an
actual protocol submission, and a paper's free-text methods section
doesn't carry authorship/license/version metadata a real BeBOP submission
would. Forcing LLM-extracted method blobs into MIOP's fields would
fabricate that metadata. Confirmed with the user: this is the right call
for paper-derived input, and deliberately *not* a permanent decision
against BeBOP/MIOP mapping in general -- `mapping/bebop.py` now raises
`BebopMappingNotApplicable` with this reasoning, and says exactly what
would justify revisiting it (this pipeline ingesting an actual protocol
document as an input type).

## Milestone 7 validation: real refresh against real external APIs

`fair-ocean weekly-update --dry-run` against the real 101-study database
correctly reported 101 refreshable studies, 0 stale failed/manual-review
tasks (matching `fair-ocean status`), and that quarterly rediscovery was
due (never run before) against all 101 candidate studies. The real
(non-dry-run) invocation enqueued exactly that: 101 `REFRESH_STUDY_SOURCES`
tasks and 101 fresh `DISCOVER_IDENTIFIERS` tasks (quarterly rediscovery's
own idempotency-key trick -- see `scheduling/rediscovery.py` -- correctly
created *new* tasks alongside the studies' original, already-completed
ones rather than being deduped away).

Processing a real sample against live Crossref/OpenAlex/Europe PMC/NCBI/ENA
(cache intentionally bypassed, so every fetch was a genuine network call)
surfaced real, useful results:

- **4 of 5 studies refreshed cleanly**, correctly reporting `unchanged`
  for every source once the fresh content_hash matched what was already
  stored from Milestones 2/3 -- no duplicate `Source` rows created, exactly
  as designed.
- **1 hit a genuine transient ENA `500 Internal Server Error`.** No new
  handling was needed: the task's own exception propagated up, the
  existing (Milestone 3) `session.rollback()`-before-`fail_task` pattern
  in `worker.py` correctly discarded the partial NCBI work already done
  for that study in the same task, and the task landed in `retry_pending`
  with exponential backoff -- confirming the pre-existing retry
  infrastructure needed no special-casing to work for the new task type.
- `fair-ocean report-run <run_id>` -- dead code before this milestone,
  since nothing ever wrote a `WorkflowRun` row -- now shows a real run's
  summary.

Two real bugs were found and fixed during this validation, not in mock
tests:

1. **`scheduling/weekly.py` called the *real*, un-mockable
   `_build_enabled_adapters()` for its cache-clearing step**, because it
   imported the function by name (`from ...handlers import
   _build_enabled_adapters`) instead of referencing it through the module
   (`handlers._build_enabled_adapters()`). Every test that monkeypatched
   `handlers._build_enabled_adapters` to a fake still hit this one
   unmocked call site -- caught when a test asserting a `RuntimeError`
   should propagate from a broken adapter build instead ran the real
   adapter-building code with no error at all, and, more importantly,
   real test runs had been silently calling `.clear_cache()` against this
   pipeline's actual `data/cache/` directory (a performance cache, not
   real data -- no data loss, just a lost cache hit). Fixed by importing
   the module, not the name.
2. **`is_rediscovery_due` raised `TypeError: can't subtract offset-naive
   and offset-aware datetimes`** the first time a second `weekly-update`
   run needed to check an existing `quarterly_full_rediscovery`
   `WorkflowRun`'s age -- SQLite has no real timezone-aware storage, so a
   `DateTime(timezone=True)` column round-trips as naive even though
   every writer in this codebase is `clock.utcnow()` (always UTC). Fixed
   with a new small helper, `clock.as_aware_utc()`, rather than a one-off
   fix local to this call site, since any future Python-level datetime
   subtraction against an ORM-fetched timestamp would hit the same issue.

## Schema simplification: missingness folded into standardized_values, publications folded into sources

Two tables removed, at the user's request, for the same reason: each had
become a second place that had to stay in sync with information that
already had (or now has) a natural home, which is exactly the kind of
thing that's easy to get subtly wrong while debugging or extending.

**`missingness` -> `standardized_values`.** `standardized_values` gained
`missingness_status`/`sources_inspected`/`reason` columns. Every (study,
entity, target_field) this pipeline checks for now gets exactly one row,
whether or not a value was found -- `standardized_value` is null and
`missingness_status` carries why when nothing was found (same controlled
vocabulary as before); when a value *was* found (by `mapping/faire.py`),
`missingness_status` just gets tagged `"present"` on that same row instead
of a second table needing a second row to say so. One real ordering
consequence worth knowing: `map_study_to_faire`'s idempotent rebuild
(delete-then-recreate every `target_schema="faire"` row for a study) also
clears completeness placeholder rows now that they live in the same
table -- re-running `enqueue-mapping-backfill` after real data changes
still means re-running `enqueue-faire-completeness-backfill` afterward,
same dependency that existed before, just more visible.

**`publications` -> `sources`.** A publication is just one flavor of
source (`source_type="publication_api"`) -- `sources` gained
`authors`/`journal`/`publication_year`/`fulltext_available` columns.
`doi`/`pmid`/`pmcid`/`openalex_id`/`title`/`relationship_to_study` were
**not** recreated: `external_identifier` already holds the DOI for every
publication-type source row, `ExternalIdentifier` already authoritatively
tracks cross-references, `Study.title` already holds the canonical merged
title, and `relationship_to_study` turned out to be unused entirely.
`open_access_status` folds onto `sources.access_status`, which already
used the same enum. The bigger shift: the old `publications` table held
one row per study, *merged* "first non-null wins" across whichever of
crossref/europe_pmc/openalex answered -- exactly the kind of synced
aggregate this fold removes. Each adapter's own `Source` row now carries
its own bibliographic fields as that adapter reported them, no merge at
write time (`workflow/handlers.py`'s `_apply_publication_fields`); a
consumer wanting one answer (e.g. `handle_validate_logic`'s collection-
date-vs-publication-year check) queries across a study's publication-type
sources directly (`_publication_year_for_study`) instead of trusting a
separately-synced value.

**Migration** (`0fc0bf2bf46a_fold_missingness_and_publications.py`)
carries real data forward, not just schema: run against the real 101-study
database, 1,845 `missingness` rows became 1,845 `standardized_values`
rows (930 `not_found_in_inspected_sources`, 912 `present`, 3
`relevant_source_not_inspected` -- either merged onto an existing mapped
row or inserted fresh, whichever applied) and 99 `publications` rows were
attached to one real `Source` row per study (preferring crossref, then
europe_pmc, then openalex, matching the adapters' own priority order).
That per-study attachment is a **one-time best-effort backfill, not a
lossless reconstruction** -- the old design discarded which adapter
contributed which field the moment it merged them, so there was nothing
more precise to recover. Running `weekly-update` + `worker` after the
migration naturally repopulates these fields per-adapter going forward.

## Milestone 8 validation: FAIRe-aware extraction against 6 real local models

**Note:** the run below (and its `fact_type_candidate` values like
`annealingTemp`/`otu_db`) predates the v3 correction described just above
-- it was measured against v2's design, where FAIRe's own field spellings
were used directly as `fact_type_candidate`. Kept here as an accurate
historical record of that run rather than silently rewritten. v3 keeps
the same checklist size/structure (so the group-header-confusion and
small-model-goes-silent findings below are expected to still generalize),
but uses native names instead -- see "Milestone 8 follow-up" below for the
re-validation against v3's actual prompt.

`fair-ocean benchmark-models` run against all 6 real, currently-running
local Ollama models from `config/benchmark_models.yaml`, over all 6 gold
cases (the original example plus the five new FAIRe-aware cases):

| candidate | json_valid | evidence_verif | precision | recall | f1 | mean latency (s) |
|---|---|---|---|---|---|---|
| qwen2.5-3b | 1.00 | 1.00 | 0.40 | 0.04 | 0.07 | 13.3 |
| qwen3-4b-instruct | 1.00 | 0.98 | 0.71 | 0.71 | 0.71 | 25.1 |
| llama3.2-3b | 1.00 | 0.36 | 0.15 | 0.06 | 0.09 | 23.4 |
| phi4-mini-3.8b | 1.00 | 0.47 | 0.76 | 0.27 | 0.39 | 17.9 |
| gemma3-4b | 1.00 | 0.62 | 0.38 | 0.29 | 0.33 | 40.4 |
| granite3.3-8b | 1.00 | 0.76 | 0.59 | 0.45 | 0.51 | 55.3 |

All 6 produced syntactically valid JSON every time -- the real
differentiator across this richer, ~70-field checklist is precision/recall,
which spans a wide range (qwen3-4b-instruct's 0.71 F1 down to qwen2.5-3b's
0.07). Two real, opposite failure modes turned up on inspection of
`detail.json`, not just the summary numbers:

1. **qwen2.5:3b went silent on 4 of 6 cases** -- `json_valid: true`,
   `returned_facts: []`, for source text with 6-11 extractable facts each.
   Not an error: the model chose to report nothing rather than engage with
   the longer, denser checklist. This is the real, direct cost of a
   FAIRe-aware taxonomy replacing a short, open-ended prompt -- the
   smallest model tested (3B) couldn't productively use the extra
   structure at all.
2. **llama3.2:3b confused the checklist's group headers
   ("PCR / assay setup:") with the field names themselves**, e.g.
   returning `{"fact_type_candidate": "PCR / assay setup", "raw_value":
   "assay_name: MiFish-U-12S", ...}` instead of
   `{"fact_type_candidate": "assay_name", "raw_value": "MiFish-U-12S",
   ...}` -- real evidence quotes, correctly-shaped JSON, but the
   fact/value nesting the model invented didn't match the schema at all
   (0 true positives on that case, all real content lost to the wrong
   shape).

**Fix:** added one explicit clarifying paragraph to
`EXTRACTION_INSTRUCTIONS` distinguishing group headings (reference only,
never use as `fact_type_candidate`) from field names (the single
identifier before each bullet's own colon). Re-verified directly against
the real API, not assumed:

- **llama3.2:3b improved but wasn't fully fixed** -- true positives on the
  affected case went from 0/10 to 2/10; roughly half its answers still
  used a group heading. A genuinely weak model's confusion isn't
  eliminated by one clarifying paragraph.
- **qwen2.5:3b got measurably worse on that same case** -- the longer
  prompt pushed it from a valid-but-empty response (33.8s) to output that
  no longer parsed as JSON at all after retries (88.7s). The extra prompt
  length cost this specific small model more than the clarification bought
  it.
- **qwen3-4b-instruct (the strongest performer) was unaffected in the
  direction that matters**: re-checked directly against the same case,
  8 true positives / 0 false positives / 2 false negatives -- confirming
  the fix doesn't cost a model that already understood the schema
  anything, while still being a real, justified prompt-engineering
  improvement for models in between.

Net conclusion, grounded in these real runs rather than assumed: this
FAIRe-aware taxonomy is usable today with **qwen3-4b-instruct or
granite3.3-8b** (the two candidates that engaged productively with the
full checklist); **qwen2.5:3b and llama3.2:3b are not well-suited to this
richer schema** regardless of prompt polish, and reverting to the old,
shorter open-vocabulary prompt would be the only way to get useful output
from models that small -- which would give up exactly the FAIRe-field
coverage this milestone exists to add. This is precisely the kind of
finding `llm/benchmark.py` exists to surface before anyone commits to a
model for real extraction work.

## Milestone 8 follow-up: v2 -> v3, decoupling extraction from FAIRe's own vocabulary

A user review caught a real architecture problem in v2 before any
production data was built on it: v2 asked the model to use FAIRe's own
exact field spellings (`annealingTemp`, `otu_db`, `neg_cont_type`, ...)
directly as `fact_type_candidate` -- coupling a raw fact's own identity to
one specific standard's vocabulary, the same coupling every other raw-fact
source in this pipeline (repository adapters, structured API/XML facts)
deliberately avoids. `standardized_values` mapping is supposed to be a
strictly separate, downstream step over source-native `raw_facts` --
extraction was quietly blurring that boundary.

**Fix:** `extraction/faire_fields.py`'s `FaireExtractionField` now carries
`native_name` (a plain, standard-agnostic description -- what
`fact_type_candidate` is set to, e.g. `annealing_temperature`,
`reference_database`, `negative_control_type`) separately from
`faire_hint` (the FAIRe slot spelling that concept corresponds to, e.g.
`annealingTemp`, `otu_db`, `neg_cont_type`). The prompt (`extraction/text.py`,
now `PROMPT_VERSION = "text-extraction-v3-native-with-hints"`) asks for an
**optional** `candidate_standard_fields` field per fact (e.g.
`{"faire": "annealingTemp"}`), which `extract_facts_from_section` stores
in `RawFactCandidate.confidence_metadata` and, from there,
`RawFact.confidence_metadata` -- a JSON column that already existed on the
ORM model, unused, so this needed **no migration**. Dropping the hint
entirely still leaves a fully valid, standard-agnostic raw fact;
standardizing onto FAIRe (or any other vocabulary) remains entirely
`mapping/rules.py`'s job.

Re-validated live against qwen3-4b-instruct (the strongest performer in
the original run) on all 6 gold cases, now scored against v3's native
names:

| | v2 (FAIRe names as identity) | v3 (native names + optional hints) |
|---|---|---|
| precision | 0.71 | 0.72 |
| recall | 0.71 | 0.73 |
| f1 | 0.71 | 0.73 |
| mean latency (s) | 25.1 | 39.8 |

Precision/recall/F1 are essentially unchanged (slightly better, within
normal run-to-run noise for a live model) -- confirming the rename didn't
cost this model anything, as expected, since the underlying checklist size
and structure are the same. Mean latency increased (25.1s -> 39.8s): the
prompt is longer now (native name + hint + example per concept, plus the
`candidate_standard_fields` output instructions), a real, expected cost of
the correction. Inspecting the raw output (not just the aggregate score)
confirmed the mechanism itself works as designed: the model reliably
returned `candidate_standard_fields` hints that matched
`faire_fields.py`'s own `faire_hint` values (e.g. `annealing_temperature`
paired with `{"faire": "annealingTemp"}`), and every returned
`fact_type_candidate` used a native name, never a FAIRe spelling directly
-- across all 41 facts the model returned, not one instance of the v2
coupling reappeared.

**A real mismatch to flag, not yet resolved:** `mapping/rules.py`'s
current rules are keyed on neither v2's FAIRe-literal names nor v3's
native names -- they use a third, older naming style
(`PCR_forward_primer_sequence`, `PCR_amplification_conditions_temp_cycles`,
...) built from Milestone 4's original coarse, fully open-vocabulary
extraction output, before either FAIRe-aware taxonomy existed. As things
stand, `mapping/rules.py` has zero rules that would match a
`fact_type_candidate` this v3 taxonomy produces (`annealing_temperature`,
`pcr_reaction_volume`, `forward_primer_sequence`, ...), whether or not a
`candidate_standard_fields` hint is attached. This is mapping-owned work
(`mapping/rules.py`), not something this update touches -- flagging it
here since it directly affects whether v3's atomic extraction output ever
reaches a `StandardizedValue` row.

## 100-study stress test: end-to-end audit against the real database

Ran the full pipeline's real output (seed import through validation) against
the real 101-study database, plus real LLM text extraction against 21 of
`claude_studies_tested.txt`'s 50 studies (the complementary half of
`OpenAI_studies_tested.txt`'s 50 -- together the two audits cover all 100
real seeds instead of overlapping). Real bugs found and fixed, each verified
against real data or a real live re-run, not assumed:

1. **A stale, pre-fix DOI.** One study's DOI (`10.1128/msystems.00184-16open_in_new`)
   still carried the copy-paste artifact a recent fix already knows how to
   strip -- the fix was correct, the row just predated it. Re-normalized and
   re-ran discovery live, recovering 3 real sources (crossref/europe_pmc/openalex)
   for a study that had zero.
2. **FAIRe completeness misreported "not found" as "not inspected."**
   `populate_faire_missingness_for_study` always used
   `NOT_FOUND_IN_INSPECTED_SOURCES` for a missing field, even for a study
   with zero sources inspected at all -- misleadingly implying sources were
   checked and the field wasn't there. Fixed to check whether any source has
   actually been inspected first, matching the same distinction
   `populate_missingness_for_study` already made for `core_sampling_metadata`.
3. **Cross-source title comparison false-flagged formatting as disagreement.**
   6 of 7 "conflicting" title checks in the 50-study sample were Crossref/OpenAlex
   inline markup (`<i>`, `<scp>`) or Unicode dash variants that Europe PMC's
   plain-text title never carries -- not a real disagreement. Fixed the
   normalizer; the 7th was a genuine third-party OpenAlex data-quality bug
   (confirmed live against OpenAlex's own API: it really does serve the wrong
   title for that DOI), correctly still flagged rather than smoothed over.
4. **Deleting old raw_facts orphaned the "already processed" signal.**
   Clearing out-of-date `text-extraction-v1` data (a separate, explicitly
   user-approved cleanup) left 38 `article_fulltext` Source rows across the
   database claiming full text was already processed with nothing backing
   that claim -- silently blocking every one of those studies from ever
   re-running extraction. Removed the 38 stale rows (verified zero
   dependents) after explicit confirmation, since it directly affects both
   this audit and the other AI's.
5. **Ollama's default context window silently drops real sections.** Real
   paper sections plus the v3 taxonomy's ~70-concept checklist routinely
   exceed 4096 tokens -- well past anything short synthetic gold-case
   snippets ever approached -- causing real 400 "exceeds context size"
   errors mid-run. Added an opt-in `llm.num_ctx` config option
   (`OpenAICompatibleHTTPBackend` sends it as `{"options": {"num_ctx": ...}}`).
   **Verified live that this does NOT fix it for Ollama specifically** --
   Ollama's OpenAI-compatible endpoint silently ignores the field (confirmed
   by sending the identical request to Ollama's *native* `/api/chat`
   endpoint with the same option, which succeeds where the OpenAI-compat
   route still 400s). The option is still worth setting for other
   OpenAI-compatible servers that do respect it; for Ollama, the real fix is
   server-level (`OLLAMA_CONTEXT_LENGTH` at startup, or a custom Modelfile
   with `PARAMETER num_ctx`) -- deliberately not done automatically here,
   since restarting someone's local Ollama server isn't this project's call
   to make unprompted.

**Verified clean, not just assumed, across the real 101-study database:**
zero duplicate raw_facts, zero duplicate standardized_values, zero duplicate
task idempotency keys anywhere -- confirming re-running ingestion/enqueue
never duplicates work. `depth: "1"` (no unit) for a real BioSample record
was checked directly against NCBI's live API and confirmed to be exactly
what the submitter reported, not a parsing bug.

**Operational resilience, tested live (not just read from code):** killed a
worker mid-task (`kill -9`), confirmed zero partial writes from the
in-flight task, confirmed a second worker correctly skipped the stuck task
and finished the rest, confirmed `release-stale-claims` correctly requeues
it with backoff, confirmed it completes on retry. Re-running the full
ingest+enqueue batch is fully idempotent (0 new studies, 0 new tasks, 0
duplicate facts). **Concurrent SQLite workers do race** -- reproduced live
(two workers, one task, two `ValidationResult` rows for the same check) --
but this is an already-documented, deliberate SQLite limitation (see
`workflow/task_queue.py`'s `claim_next_task` docstring), not a hidden bug.
Verified against a real local PostgreSQL 18 server that `FOR UPDATE SKIP
LOCKED` genuinely eliminates the race: 4 concurrent workers, 20 tasks, zero
duplicates, work evenly distributed.

**A real evidence-quote limitation, surfaced by real papers (not yet
fixed):** one `forward_primer_sequence` fact's `evidence_quote` was a real,
verbatim substring of the paper (passing `verify_evidence_quote` correctly)
but described primer *names* ("784F and 1061R"), not the actual sequence
letters claimed as `raw_value` -- which, on direct inspection, genuinely
does appear elsewhere in the same paper (not fabricated), just in a
different sentence than the one quoted. The deterministic evidence check
confirms a quote is real; it does not confirm a quote *supports* the
specific value paired with it. Flagged here rather than silently
patched -- tightening this (e.g. requiring the quote to literally contain
the raw_value) would break many legitimately-paraphrased facts
(`storage_conditions`-style summaries), so it needs a deliberate design
decision, not a quick fix.

**Still open, mapping-owned (already noted in "Milestone 8 follow-up"
above, re-confirmed here after this run):** `mapping/rules.py` has no rules
for any v3 native name yet, so none of the 222 real facts this run
extracted (`annealing_temperature`, `pcr_reaction_volume`,
`forward_primer_sequence`, `reference_database`, ...) currently reach a
`StandardizedValue` row.

## Model comparison: all 6 installed local models, gold cases + 39 real papers

Ran `fair-ocean benchmark-models`-equivalent comparison of every locally
installed Ollama candidate (qwen2.5:3b, qwen3:4b-instruct, gemma3:4b,
phi4-mini, llama3.2, granite3.3:8b) against two real, complementary test
sets: the 6 curated gold cases (real precision/recall/F1 against
hand-labeled truth) and, separately, one representative section from every
one of the 39 real papers in the database with fetchable full text (real
paper length and complexity, no ground truth -- scored instead on JSON
validity, evidence-verification rate, schema adherence, and latency).
Launched as a detached background job so it could run unattended for
hours without live monitoring; raw results in
`data/exports/model_benchmark_100/` (gitignored, local only).

**Gold cases (precision/recall/F1):**

| candidate | precision | recall | f1 | mean latency (s) |
|---|---|---|---|---|
| qwen2.5-3b | 0.40 | 0.08 | 0.14 | 35.2 |
| qwen3-4b-instruct | 0.72 | 0.73 | 0.73 | 38.0 |
| llama3.2-3b | 0.25 | 0.14 | 0.18 | 26.5 |
| phi4-mini-3.8b | 0.20 | 0.02 | 0.04 | 8.1 |
| gemma3-4b | 0.42 | 0.31 | 0.35 | 37.5 |
| granite3.3-8b | 0.59 | 0.41 | 0.48 | 107.9 |

**39 real papers (no ground truth -- JSON validity, evidence verification,
schema adherence, and reliability under real paper length instead):**

| candidate | json_valid | evidence_verif | schema_adherence | mean latency (s) | timeouts | other errors |
|---|---|---|---|---|---|---|
| qwen2.5-3b | 1.00 | 0.64 | 0.82 | 23.3 | 0 | 0 |
| qwen3-4b-instruct | 0.62 | 0.57 | 0.76 | 43.8 | 0 | 15 |
| llama3.2-3b | 0.92 | 0.28 | 0.56 | 46.1 | 3 | 0 |
| phi4-mini-3.8b | 0.92 | 0.09 | 0.40 | 14.8 | 1 | 1 |
| gemma3-4b | 0.44 | 0.62 | 0.80 | 116.0 | 22 | 0 |
| granite3.3-8b | 0.72 | 0.75 | 0.57 | 90.8 | 10 | 0 |

Two pictures that don't fully agree, and both are real: **qwen3-4b-instruct
is the clear gold-case winner** (highest F1 by a wide margin) but its
real-paper `json_valid` rate drops to 0.62 -- the "other errors" column
(15/39) explains why (see bug #1 below). **qwen2.5-3b, the weakest gold-case
performer, is the most *reliable* one on real papers** (1.00 json_valid, 0
timeouts, 0 other errors, fastest) -- it just extracts fewer facts per
section (2.5 returned vs qwen3's 4.1), consistent with the smallest model
being conservative rather than broken. **gemma3-4b and granite3.3-8b pay a
real latency tax on real paper length** (116s and 91s mean, vs 35-40s on
short gold-case snippets), and both hit the 180s per-request timeout
double-digit times (22 and 10 of 39 respectively) -- a real paper section
is often 5-10x longer than a gold-case snippet, and these two models don't
scale to that gracefully. Net: no single model wins both axes; which one
to actually run depends on whether an unattended batch job values recall
(qwen3-4b-instruct, tolerate the failure rate and retry) or reliability
(qwen2.5-3b, accept a lower per-section yield) more.

**Two real bugs found and fixed during this run:**

1. **`OpenAICompatibleHTTPBackend` swallowed the real reason for every 4xx
   failure.** `httpx.HTTPStatusError`'s own `str()` is just the status
   line, never the response body -- every one of qwen3-4b-instruct's 15
   real-paper failures logged as an identical, generic "400 Bad Request"
   with no way to tell *why* short of a separate manual reproduction
   script. Root-caused by hand once: Ollama's default context window
   (4096 tokens, regardless of a model's real trained context length --
   qwen3-4b-instruct supports up to 262144) is smaller than many real
   paper sections plus the v3 taxonomy's full checklist. Fixed
   `llm/http_backend.py` to include `response.text` (truncated to 500
   chars) in the raised `LLMBackendError` so this is visible immediately
   next time, for any model or any 4xx cause -- covered by a new
   regression test (`test_generate_error_includes_response_body_for_4xx`).
2. **Ollama's context ceiling has no fix from this pipeline's side.** Added
   an optional `LLMConfig.num_ctx` / per-candidate config knob that sends
   `{"options": {"num_ctx": N}}` -- confirmed live that Ollama's
   OpenAI-compatible endpoint (`/v1/chat/completions`) does **not** honor
   it (verified directly: an oversized prompt still 400s at the same
   4096-token ceiling with `num_ctx` set). The real fix has to happen on
   the Ollama side, outside this pipeline's config: either the
   `OLLAMA_CONTEXT_LENGTH` server environment variable, or a per-model
   `PARAMETER num_ctx <N>` in a custom Modelfile. Left as a documented,
   known operational constraint rather than a false "fixed" claim -- this
   pipeline can request a larger context window but cannot force Ollama to
   grant one over its own wire protocol.

## Structured-first extraction: skip asking the LLM about already-resolved fields

Prompted by the model comparison above showing hallucination and context-
window pressure as real, live problems: every extra concept in the prompt
is both more prompt to fit under a model's context ceiling and one more
opportunity for a weaker model to fabricate an answer instead of reporting
"not found." Since structured sources (NCBI/ENA/PANGAEA/...) resolve many
fields deterministically with no hallucination risk at all, there's no
reason to also ask an LLM about a field a structured source already
answered.

`extraction/text.py`'s new `resolved_faire_fields_for_study(session,
study_id)` queries `standardized_values` for every FAIRe `target_field`
already carrying a real value (`missingness_status == "present"`) for that
study -- i.e. whatever a prior `MAP_FAIRE` pass already resolved from
structured facts. `workflow/handlers.py`'s `handle_extract_text_facts`
computes this once per study, before any section's LLM call, and passes it
as `extract_facts_from_section`'s new `exclude_faire_hints` parameter;
`extraction/faire_fields.py`'s `render_field_reference` then omits every
checklist entry whose FAIRe hint is in that set (dropping a group's
heading entirely if every entry in it gets excluded). If `MAP_FAIRE` hasn't
run yet for a study, the resolved set is simply empty and every section's
prompt is exactly as before this existed -- this can only narrow the
checklist, never skip a genuinely-unresolved concept or fabricate a value.

**For this to actually shrink a prompt, `enqueue-mapping-backfill` (+ a
`MAP_FAIRE`-only worker pass) needs to run against a study's structured
facts *before* `enqueue-text-extraction-backfill`** -- the reverse of
either running independently, which is how both already worked (neither is
auto-chained from the other, see "Assumptions and placeholders"). Worth
stating plainly rather than assuming: checked live against the real
101-study database, and **today this has zero measurable effect** --
`mapping/rules.py`'s current rules only ever resolve `eventDate`,
`geo_loc_name`, `samp_name`, and `project_id` from structured sources
(907 real `present` rows, 4 distinct fields, none of them PCR/assay/qPCR/
sequencing/bioinformatics/taxonomy concepts). That's an accurate reflection
of what NCBI/ENA/PANGAEA structured metadata actually contains, not a bug
in this mechanism -- those repositories generally don't report primer
sequences, PCR conditions, or qPCR standards at all, only sample/project-
level Darwin-Core-style fields. The one place this pipeline's structured
sources *do* overlap the extraction taxonomy is sequencing
platform/instrument (`mapping/rules.py` already has `instrument_platform`/
`instrument_model`/`sequencing_platform` rules targeting FAIRe's `platform`/
`instrument`, matching this taxonomy's `sequencing_platform_general`/
`sequencing_instrument` hints) -- but no study in the current database has
had those specific rules produce a `present` row yet either. This
mechanism will start paying off exactly as fast as `mapping/rules.py`'s
structured-source coverage grows to overlap more of the extraction
taxonomy, not before -- tested and confirmed correct in isolation
(`tests/unit/test_extraction_text.py`, `tests/unit/test_handlers_text_extraction.py`)
regardless.

## Diagnosing qwen3-4b-instruct's real-paper JSON failures: input overflow, confirmed live

15 of 39 real papers failed outright for qwen3-4b-instruct in the model
comparison above. Before accepting any fix, reproduced a sample of these
failures directly against the real Ollama endpoint (not guessed) to
distinguish three real candidate causes: input overflow (prompt exceeds
context, needs chunking), output truncation (response cut off mid-JSON,
needs a higher token cap), or an Ollama-specific context-window default.

Reproduced 4 of the 15 failures by rebuilding the exact original prompt
(cached full text + the pre-chunking `build_prompt`) and POSTing it
directly to `qwen3:4b-instruct`. All 4 came back **HTTP 400** in
0.13-0.19s -- consistent with all 15 real logged latencies (0.13-0.47s,
vs. 20-280s for every real generation) -- with an identical body:
`"exceed_context_size_error"`, `n_prompt_tokens` 4386-5742 vs.
`n_ctx: 4096`. This is unambiguous: **input overflow caused by Ollama's
default context window**, not output truncation (which would show real
generation time before failing, not near-zero latency) and not a
mysterious model hiccup. Also confirmed directly that setting `num_ctx` in
the request body does **not** change this via Ollama's OpenAI-compatible
endpoint -- `llm/http_backend.py` already documents this; the config knob
in `config/benchmark_models.yaml` is inert there by design, a genuine fix
needs `OLLAMA_CONTEXT_LENGTH` set server-side or chunking on this
pipeline's side.

Verified the chunking + focused-topic-pass rewrite (`extraction/text.py`
v4-v6: `split_segments_for_calls`, `DEFAULT_MAX_SECTION_CHARS_PER_CALL`,
per-topic focuses) actually fixes this rather than assuming it does: the
same section that 400'd instantly under the old single-shot prompt now
keeps every individual call under ~2,100 tokens (measured directly:
2 chunks x 4 topic focuses = 8 calls, max single-call prompt 8,508 chars),
and re-running the exact same previously-failing case through the new
pipeline against the same live model produced **32 real extracted facts,
zero errors** (took 277s, vs. an instant failure before -- a real latency
cost from many more, smaller calls, worth knowing about before assuming
the fix is free).

## Fixing the real Ollama context limit and collapsing the per-topic passes (v7 -> v8)

The section above's own fix (chunking + 5 topic-focused passes, each with
its own recall retry) worked, but at a real cost the milestone above
already flagged: many more, smaller calls. Measured live on real hardware,
that cost turned out to be much larger than "worth knowing about" implied
-- a single real paper section (well under any char-based chunking
threshold) took **913.8 seconds** end to end, because it triggered up to
10 sequential LLM calls (5 topic focuses x a recall retry each, the retry
firing almost every time since "any checklist concept unmentioned" is true
for nearly every real section). Per-call instrumentation on that same
section showed individual call latencies ranging **12.2s-238.3s** -- a
30-90s+ tax paid separately by every one of those 10 calls.

**Root cause, confirmed live via direct HTTP requests to both models
side by side**: Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`)
silently ignores the `num_ctx` request option, capping every model at its
built-in default (4096 tokens) regardless of `config.py`'s `llm.num_ctx`
setting -- a 5,024-token prompt sent to the plain `qwen3:4b-instruct` model
400s with `"exceeds the available context size (4096 tokens)"`; the exact
same prompt sent to a new `qwen3:4b-instruct-16k` model (built via
`ollama create` from the same base model plus a Modelfile with
`PARAMETER num_ctx 16384` genuinely baked in) succeeds. The 5-focus split
from the previous milestone was sized around that silent 4096-token
ceiling in the first place -- once a real, larger context is available,
the fix is smaller and structural, not just "more/smaller calls":
`extraction/text.py` v8 now defaults `extract_facts_from_section` to a
**single collapsed pass over the full checklist per chunk**
(`focuses=(None,)`) instead of 5 topic-scoped passes, and narrows the
recall retry to only fire when a pass found **zero** facts (not "any
checklist concept went unmentioned," which was true almost every time and
effectively doubled call volume for little benefit). `EXTRACTION_FOCUSES`
remains fully available -- pass it explicitly as `focuses=EXTRACTION_FOCUSES`
-- for a smaller-context model that still needs the old per-topic split.

Config-side, `config/local.yaml`/`.env` now point `llm.model` at
`qwen3:4b-instruct-16k` and raise `extraction_max_chars_per_call` from
1600 to 16000 chars (the old value was conservative for a ~4096-token
effective budget; the full checklist alone is ~3,300 tokens, so 16k chars
of section text plus the checklist and a 2048-token completion budget
comfortably fits in 16384 tokens with real margin for char/token-ratio
estimation error). `config/benchmark_models.yaml` gained a
`qwen3-4b-instruct-16k-ollama` candidate alongside the original
`qwen3-4b-instruct-ollama` entry (kept as a live reference case for the
context-limit finding itself, not removed).

**Verified end to end, not assumed**: re-ran the exact same instrumented
single-section test after the v8 change -- and separately ran the full
18-case gold benchmark (`data/exports/benchmark/qwen3_4b_16k/`) against
`qwen3-4b-instruct-16k`, live:

| | v7 (5-focus, old context) | v8 (collapsed, real 16k context) |
|---|---|---|
| Mean latency per case/section | ~700-900s (measured on 1 section) | **61.4s/case** (median 50s, p95 123s) |
| JSON validity | -- | 100% |
| Evidence verification rate | -- | 98.5% |
| Precision / recall / F1 | -- | 0.447 / 0.575 / 0.503 |

The speed win (roughly **12-15x**) is real and directly measured, not
estimated. The precision/recall numbers are honestly moderate, but
spot-checking the actual returned facts (not just the score) on the
worst-scoring cases shows the gap is mostly **gold-data quality, not a
new extraction regression**: on `real-sponge-edna-sample-sequencing-001`
(scored 6/25 "correct"), every one of the 25 returned facts is a real,
evidence-verified value from the paper -- the gold file only enumerates
17 expected facts, so genuinely-correct extra findings score as false
positives. On `real-sponge-edna-bioinformatics-001`, several "misses" are
gold using stale/informal label names (`storage_conditions`,
`DNA_extraction_method`) that don't match the current taxonomy's real
native names (`sample_storage_conditions`, `dna_extraction_kit`) the model
correctly used instead, plus a few cases where the model copied the
source text's own curly quotation marks verbatim (`'obicut'`) against
gold's stripped value (`obicut`). None of these three effects are
consequences of the v7->v8 change itself -- they're pre-existing
gold-curation drift (the same kind of issue the earlier 12-real-paper
gold-case cleanup this session addressed on a different axis --
fabrication/splicing there, label/completeness drift here) that would
affect scoring under either version of the extraction pipeline. A true
head-to-head against the old 5-focus design wasn't run (would cost
proportionally more of the same slow per-call time this milestone just
fixed) -- fixing the gold-data drift itself is addressed next.

## Gold-case label-drift/completeness cleanup

Followed up on the honest gap flagged above by re-deriving `expected_facts`
for all 12 real-paper gold cases directly against each case's own cached
`source_text` -- not by absorbing every model "extra" wholesale (most
model differences are genuine mislabels, not gold gaps: primer *names*
tagged as primer *sequences*, a reference database mistaken for a
taxonomic-assignment *method*, a sequencer manufacturer mistaken for a
sequencing *location*, a basecalling tool mistaken for a read-merging
tool, values inferred rather than explicitly stated (`library_layout:
"paired"`, `assay_type: "targeted"`), and outright fabrications where a
reaction's own stated component volumes don't sum to the value the model
reported). Roughly 40 genuinely explicit, non-redundant, taxonomy-valid
facts were added across 10 of the 12 cases (2 relabels, the rest
additions), each with a real byte-exact `evidence_quote` sliced directly
out of `source_text` (not hand-typed -- several source texts use `\xa0`
non-breaking spaces and ` ` thin spaces mid-sentence that a
hand-typed quote silently fails to match) and checked against
`verify_evidence_quote` before being written. Two of the 12 cases needed
no changes at all; their existing gold was already complete and correct.

The two relabels are worth calling out because they're the concrete
payoff of the taxonomy near-duplicate-field finding flagged earlier
(fallback vs. primary names can both validly describe the same concept):
`real-sponge-edna-sample-sequencing-001`'s `storage_conditions` ->
`sample_storage_conditions` (the primary field clearly applies, so use
it over the generic fallback) and its `library_concentration_method` ->
`dna_concentration_method` (the source text says extracts, not
libraries, were quantified with the Qubit at that point in the
protocol -- gold's original label was a genuine mislabel, not just a
naming-convention choice).

**Investigated the two cases that scored zero extracted facts, rather
than assuming a bug.** `real-cold-seep-sponge-sequencing-001` turned out
to be a non-issue on closer inspection: rerunning it live against
`qwen3-4b-instruct-16k` twice in a row now reliably returns all 7 real,
evidence-verified facts (the original empty result was a one-off,
almost certainly from whatever the very first cold-start request in that
multi-hour benchmark run hit). `real-methane-sediments-sequencing-001`
is a different story -- **reproducibly** empty (0/0 facts across 2 fresh
runs, valid JSON both times, and unchanged even after retrying at
temperature 0.3 and 0.7, so it isn't sampling noise). Root-caused with a
controlled test: the section is titled "RNA extraction and sequencing"
and every fact in it is genuinely about RNA (`RNA was extracted... using
the RNeasy PowerSoil Total Kit`), but the extraction checklist's fact
names are all DNA-prefixed (`dna_extraction_kit`, `dna_cleanup_method`,
...) with **zero RNA-specific names anywhere in the taxonomy**
(confirmed via `all_field_names()`). Manually substituting `RNA` -> `DNA`
throughout the same section text, with everything else held identical
(same prompt structure, same temperature 0), immediately produced 10
correctly-extracted facts. The model isn't malfunctioning -- given a
strict "only extract what matches a listed concept name" instruction and
a checklist with no RNA-shaped concept to match against, it's making a
literal-minded, highly confident (not just borderline: the empty
response persists at temperature 0.7 too) decision that none of the
checklist applies to an RNA-only section, even though FAIRe's underlying
`nucl_acid_ext_kit`/`nucl_acid_ext_*` fields are nucleic-acid-generic by
design and gold correctly labels this same kind of fact
`dna_extraction_kit` regardless of RNA/DNA. **This is a real taxonomy
naming gap** (no `sample_volume_for_extraction`-style field name signals
"this applies to RNA too"), not a gold-curation issue and not something
this cleanup fixes -- flagged here as a genuine follow-up candidate
(e.g. renaming/aliasing the nucleic-acid-extraction fact names to be
RNA/DNA-neutral) since real marine 'omics data includes metatranscriptomic
studies, not just DNA-based metabarcoding/metagenomics. Left this gold
case's `expected_facts` unchanged -- they're accurate and explicit; the
model's zero-result is the real, now-understood limitation being scored.

## Assay-entity tagging for multi-assay papers

A paper can describe more than one distinct assay run on the same samples
(a 16S rRNA V3-V4 PCR assay and an 18S rRNA V4 assay, say), each with its
own primers, target gene, annealing temperature, PCR conditions. Before
this change, every LLM-extracted fact from a full-text section was
hardcoded to `entity_level=EntityLevel.STUDY` with no entity at all --
every fact was study-wide/broadcast, with no way to know which assay a
given primer or temperature belonged to. Downstream,
`mapping/faire.py::map_study_to_faire`'s dedup key is `(target_table,
target_field, entity_id)`; since every LLM-extracted PCR/assay fact maps to
`target_table="projectMetadata"` and entity resolution unconditionally
returned `None` there, two different assays' primer sets collapsed onto
the identical key -- the first won, the second was silently flagged
`review_required=True` as a "conflict" instead of being recognized as a
second, valid assay. Real, observed data loss, and the concrete bug this
milestone fixes.

Checked directly against the vendored FAIRe schema before designing
anything (`schemas/faire/README.md`): real FAIRe's own `projectMetadata.csv`
export layout is **one wide row per `samp_name`/`assay_name`/`seq_id` join
key** -- FAIRe's actual design already expects one `projectMetadata` row
per assay, not one global row per study. This pipeline's previous
`entity_id=None`-only treatment was a simplification that didn't match
FAIRe's real intent.

**What already existed and got reused, not rebuilt:** `EntityLevel.ASSAY`
and `EntityRelationshipType.USES_ASSAY` were already in the schema (no
migration needed). `sources/base.py::RawFactCandidate` already had
`entity_level`/`entity_external_id`/`entity_label` -- a fully generic "tag
this fact with which entity it belongs to" mechanism, and
`workflow/handlers.py::_materialize_candidate_entity` already
get-or-created an `Entity` from one. That mechanism was already wired into
the supplement-parsing persist path (`workflow/supplement_handlers.py`)
but, tellingly, **not** into the main full-paper `handle_extract_text_facts`
path -- every LLM-extracted fact's `entity_id` was silently `None`
regardless of what `extraction/text.py` set on the candidate. The two
near-duplicate persist loops are now one shared
`workflow/handlers.py::_persist_candidate_facts` helper used by both.

**The fix, end to end:**
- `extraction/faire_fields.py` gains `assay_scoped_field_names()`: the
  native names describing one specific assay's setup (PCR/assay-setup and
  qPCR/standard-curve groups, plus `pcr_replicate_count`) -- deliberately
  excludes `negative_control_type`/`positive_control_type` (they map to
  `sampleMetadata`, not `projectMetadata` -- tagging them would do nothing)
  and `biological_replicate_count` (a sample property, not an assay one).
- `extraction/text.py`'s prompt gained an optional `assay_tag` output
  field: when a section describes more than one assay, the model tags each
  assay-identity/PCR/qPCR fact with a short, consistent tag -- preferring
  the paper's own name for the assay (most likely to be reused
  consistently if the same assay is described again elsewhere in the
  paper) over an arbitrary placeholder. A fact with no `assay_tag` (the
  overwhelming common single-assay case) keeps today's exact
  `entity_level=STUDY` behavior -- strictly additive, backward-compatible
  by construction. `PROMPT_VERSION` bumped to
  `text-extraction-v10-assay-tagging`.
- `mapping/rules.py`'s generated LLM rules get a **parallel** rule at
  `EntityLevel.ASSAY` for every assay-scoped native name, alongside
  (never replacing) the existing `EntityLevel.STUDY` rule.
  Deliberately not widened to "any entity level": `_EXPLICIT_RULES`
  already has literal-FAIRe-spelling rules for
  `target_gene`/`assay_type`/`assay_name` at `EntityLevel.PROJECT`
  (repository/OBIS/GBIF-sourced) sharing the exact fact_type_candidate
  string -- widening would have made the generated rule double-match those
  structured facts too.
- `mapping/faire.py::_resolve_entity_id`: a `projectMetadata`-targeted fact
  with `entity_level=ASSAY` now resolves to its own `entity_id` instead of
  unconditionally `None`, so two assays' facts land in two separate
  `StandardizedValue` rows instead of colliding.
- `exports/faire.py::export_faire`'s `projectMetadata` loop now emits one
  row per assay entity that has real values -- gated on **having at least
  one direct StandardizedValue**, not merely existing, since
  `extraction/experiment_runs.py::materialize_legacy_experiment_runs`
  already creates `ASSAY` entities today purely as link targets for
  structured ENA/BioProject `assay_name` facts (which land on the
  `EXPERIMENT_RUN` row, not the assay entity). Gating on mere existence
  would have made every such structured-only multi-assay study suddenly
  emit N near-duplicate broadcast rows where it previously emitted
  exactly one -- checked for this real regression risk before shipping,
  not just assumed away.

**Verified end to end, not just unit-by-unit**: fed a synthetic two-assay
Methods section ("For 16S, ... annealing temperature of 55C... For 18S,
... annealing temperature of 60C...") through
`extract_facts_from_section` with a mock backend tagging the two facts
`assay_tag="16S"`/`"18S"`, persisted via `_persist_candidate_facts`, ran
`map_study_to_faire`, then `export_faire` -- confirmed real, distinct
`projectMetadata` rows with `annealingTemp`/`assay_name` = 55C/16S and
60C/18S. 517 tests pass (12 new), including two explicit regression guards
(a structured-only assay entity with no direct values still yields exactly
1 project row; a PROJECT/EXPERIMENT_RUN-level structured `assay_name` fact
still broadcasts unaffected).

**Explicit non-goals, not silently skipped:** linking text-derived assay
entities to specific samples/experiment_runs ("which samples were tested
with which assay") -- prose rarely states this per-sample, a substantially
harder problem left for a separate follow-up. Cross-call/cross-section
assay-tag consistency is a real, acknowledged limitation: the model has no
memory between separate section calls or between chunks within one long
section, mitigated (not solved) by biasing the prompt toward the paper's
own given assay name rather than an arbitrary placeholder -- a real fix
would need a separate reconciliation pass.

## Narrowing the sample/experiment LLM checklist to what's realistically in prose

Not every FAIRe field that *can* be populated from a paper's Methods
section is *worth* asking the LLM about. Some are much more reliably
sourced from structured data (NCBI BioSample/ENA/repository experiment
records) than from free prose, and asking anyway just spends prompt
budget and risks a weaker model guessing or hallucinating an answer that
structured data would get right (or correctly report as absent). Following
an explicit review of the FAIRe sample/experiment/project fields, the
`extraction/faire_fields.py` exclusion policy (already used by Codex's
low-value-optional-field work, `LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS`/
`LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS`) grew two more targeted cuts:

- **sampleMetadata's LLM checklist is now deliberately narrow**: only
  `coordinates`/`collection_date`/`depth` remain (->
  `decimalLatitude`/`decimalLongitude`, `eventDate`,
  `minimumDepthInMeters`/`maximumDepthInMeters`). `sample_collection_method`/
  `sample_storage_conditions` (and their narrative-fallback counterparts
  `collection_method`/`storage_conditions`, same FAIRe targets) are
  excluded. `samp_category` (whether a given physical sample is a real
  sample or a control) was **not added to the taxonomy at all** -- it
  varies sample-to-sample in a way a paper's prose essentially never
  states explicitly per-sample; structured sources (BioSample attributes,
  supplementary sample tables) are the only realistic source for it.
  `environmental_context` (already deliberately left unmapped by
  `mapping/rules.py` -- genuinely ambiguous among
  `env_broad_scale`/`env_local_scale`/`env_medium`) is cut too, since it's
  a sample-level narrative concept outside this narrower checklist and was
  contributing zero real mapping value anyway.
- **experimentRunMetadata**: `assay_name` (which actually resolves to
  `projectMetadata` via `mapping/rules.py`'s `_TARGET_TABLE_OVERRIDES`, not
  `experimentRunMetadata` -- checked directly rather than assumed) and
  `library_concentration`/`library_concentration_unit`/
  `library_concentration_method` (-> `lib_conc`/`lib_conc_unit`/
  `lib_conc_meth`) are excluded. `phix_percentage` (-> `phix_perc`) and
  `sequencing_platform_general` (-> `platform`) are deliberately kept --
  these two are worth asking about. Checked the rest of the
  experimentRunMetadata fields the user flagged (`pcr_plate_id`,
  `pcr_well_position`, `lib_id`, `seq_run_id`, `mid_forward`/`mid_reverse`,
  `filename`/`filename2`, `checksum_filename`/`checksum_filename2`,
  `associatedSequences`, `input_read_count`) directly against the taxonomy
  first: none of them were ever part of the LLM checklist to begin with
  (all repository/ENA-structured-only, via `extraction/experiment_runs.py`'s
  `_EXPERIMENT_FACT_TYPES`) -- "should only come from APIs" was already
  true for those, confirmed rather than assumed.

Every excluded field stays fully live for structured adapters -- this is
the same "cut from the LLM's checklist, keep everywhere else" mechanism
already established for the low-value optional project fields, applied to
two more tables. `mapping/rules.py`'s new parallel `EntityLevel.ASSAY`
rule for `assay_name` (added for the multi-assay work above) is
unaffected: it's driven by `assay_scoped_field_names()`, which reads
`FIELD_GROUPS` directly rather than the checklist-rendering exclusion --
correct and inert if the model ever produces an `assay_name` fact anyway
(e.g. from a forced recall pass), consistent with this codebase's existing
"correct and inert until a real source has it" idiom.

Project metadata's own LLM checklist coverage (which fields the LLM
*should* be asked about that it currently isn't) is a separate, explicitly
deferred follow-up, not addressed here.

## Deterministic (no-LLM) project-metadata extraction

Direct follow-up to the checklist-narrowing above: a NOAA-specific FAIRe
checklist review marked a further set of `projectMetadata` fields
(`license`, `rightsHolder`, `accessRights`, `bibliographicCitation`,
`code_repo`, `recordedBy`, `recordedByID`, `project_contact`) "No LLM"
outright -- structured sources first, falling back to deterministic
parsing of the paper's own text, never a model. Before this milestone none
of these had *any* extraction path at all (confirmed against the real
database: zero coverage for the validation paper below).

**Real evidence driving the design, not assumed**: fetched the live
Crossref record for DOI `10.7717/peerj.333` and found `license: null` --
PeerJ always CC-BY-licenses its articles, but Crossref doesn't always
carry that metadata. The paper's own cached JATS full-text XML has the
real answer structurally:
```xml
<permissions><copyright-holder>Davies et al.</copyright-holder>
<license license-type="open-access" xlink:href="http://creativecommons.org/licenses/by/3.0/">...</license>
</permissions>
```
Crossref alone is insufficient; a real JATS-tree-structure parser is
necessary, not optional. The same paper's XML also has two `<contrib-group>`
elements -- one `contrib-type="author"`, one `contrib-type="editor"` -- a
real, live-confirmed risk that a naive "grab every contrib-group" extractor
would silently add the editor to `recordedBy`.

**New module, `extraction/publication_metadata.py`**, three separate
techniques rather than one grab-bag function:
- **JATS tree-structure parsing** (`xml.etree.ElementTree`, not
  flatten-then-regex): `<permissions>/<license>` -> `license` +
  `accessRights`; `<permissions>/<copyright-holder>` -> `rightsHolder`;
  `<contrib-group>/<contrib contrib-type="author">` -> `recordedBy`
  (`" | "`-joined) and `<contrib-id contrib-id-type="orcid">` ->
  `recordedByID` when present; `<contrib corresp="yes">/<email>` ->
  `project_contact`. Explicitly filters to `contrib-type="author"` --
  never editors.
- **Flat-text regex fallback**, `code_repo` only: reuses
  `discovery/text_identifiers.xml_to_text()` rather than reimplementing
  XML flattening, then a GitHub/GitLab/Bitbucket URL pattern.
- **Pure formatter, not extraction**: `bibliographicCitation` composed
  directly from Crossref's own title/authors/year/journal/DOI -- every
  piece already exists as a structured fact by the time this runs.

Deliberately **not** building a JATS/Crossref-derived `institution`
fallback: `map_study_to_faire`'s dedup is "first fact by `created_at` wins
the value, a later disagreeing one only flags `review_required`, never
corrects it" (that module's own docstring is explicit this is not a
source-precedence policy) -- adding a second `institution` source risked
a real ordering bug where a JATS-derived value could silently outrank
ENA's `center_name`, which already covers the large majority of this
pipeline's BioProject/ENA-linked studies structurally. Not worth the risk
for marginal coverage gain.

**`pcr_0_1` (the checklist's `targeted_qPCR_ddPCR_assay_0_1`) needed no
new work at all** -- Codex had just landed
`extraction/search_flags.py::detect_text_search_flags`, a tighter,
better-designed deterministic detector than what this milestone would
have built: technical terms only (PCR/qPCR/ddPCR/amplification/polymerase
chain reaction), which correctly avoids a real false positive found while
researching this -- the checklist's own suggested keywords
("species-specific", "taxon-specific", "diagnostic assay") appear 6 times
in the validation paper in an unrelated coral-settlement-behavior context,
which would have wrongly flagged `pcr_0_1='1'` on a paper that has nothing
to do with qPCR/ddPCR.

**Schema extension**: `expedition_id`/`ship_crs_expocode` aren't part of
the public FAIRe v1.0.2 checklist vendored from upstream -- confirmed via
`grep`, absent from `classes.yaml`/`schema.yaml`/`enums.yaml` entirely.
Added as an explicitly-documented NOAA/SEUS-MBON extension block in both
files (asked the user first rather than silently forking or silently
dropping the fields) -- necessary for more than CSV completeness:
`standards/faire_registry.py::build_faire_registry()` reads only
`schema.yaml`, and an existing test
(`test_registry_validation_report_all_checks_pass_on_real_schemas`)
actively asserts every `classes.yaml` slot has a matching `schema.yaml`
term, so a `classes.yaml`-only addition would have both broken that test
and silently dropped the field from the export CSV.

**Corrects a decision from the prior round**: `platform`/`instrument`/
`lib_layout` (native names `sequencing_platform_general`/
`sequencing_instrument`/`library_layout`) are real `projectMetadata`
fields (confirmed via `mapping/rules.py::_target_table_for_faire_field`),
not experiment-scoped as previously assumed, and this checklist marks them
"No LLM" too -- already 100% covered by ENA's own structured
`instrument_platform`/`instrument_model`/`library_layout` facts wherever a
study has one. Added to `LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS`.

**Wiring**: new `workflow/handlers.py::_discover_publication_metadata_from_sources`,
called inline from `handle_discover_identifiers` right after the existing
`_discover_supplements_from_fulltext`, mirroring that function's exact
shape -- pure parsing of already-fetched/disk-cached text (JATS full text
plus a disk-cached Crossref re-fetch), no new network cost, no LLM, so
(like that function) no reason to keep it manual-only. Idempotency-guarded
via a dedicated `create_source()` existence check, same pattern as every
other Source-creation call site.

**Verified against the real, partially-processed validation paper end to
end**, not just unit-by-unit: ran the new extraction against the real
database (`STUDY-47b4f87d2234`, DOI `10.7717/peerj.333`, backed up first),
then `map_study_to_faire`, then `export_faire`, and confirmed the real
`projectMetadata.csv` row shows exactly the expected values --
`license`/`rightsHolder`/`accessRights`/`bibliographicCitation`/
`recordedBy`/`project_contact` newly populated correctly;
`code_repo`/`recordedByID` correctly empty (this 2014 paper genuinely has
neither a code repository link nor any author ORCID); `platform`/
`instrument`/`lib_layout`/`checksum_method`/`institution` still populated
from ENA exactly as before, untouched. 536 tests pass (16 new, including a
`tests/unit/test_extraction_publication_metadata.py` suite that parses
this same real cached JATS XML end to end).

## Flag-gated LLM checklist: PCR fields + a reusable, generic gating mechanism

The user gave a CSV of PCR-section FAIRe fields, each row's `Conditional`
column saying which deterministic boolean flag(s) (`pcr_0_1`,
`probe_based_qPCR_ddPCR_assay_0_1`) must be true before the LLM should even
be asked about that field -- with an explicit, repeated instruction to
build one generic, reusable gating mechanism now, not a one-off, since more
term batches are coming.

Codex had concurrently built the deterministic half already
(`extraction/search_flags.py`): `TEXT_SEARCH_FLAGS` detects boolean
evidence (`pcr_0_1`, `probe_based_qPCR_ddPCR_assay_0_1`, ...) from paper
text, and `CONTROLLED_SEARCH_FIELDS` does literal-substring matching
against curated term lists for a handful of controlled-vocabulary-style
fields, each already carrying a `required_any_flags: frozenset[str]`. What
was missing: the LLM checklist itself (`extraction/faire_fields.py`) had
no equivalent gate at all -- the "PCR / assay setup" group was shown on
every single call, PCR paper or not.

**Added `required_any_flags` to the LLM taxonomy too** (mirroring
`ControlledSearchField`'s own field name/shape deliberately, one gating
vocabulary shared by both sides, not two): the entire "PCR / assay setup"
group + `pcr_replicate_count` now requires `pcr_0_1`; two new probe fields
additionally accept `probe_based_qPCR_ddPCR_assay_0_1`. `active_flags`
threads through `extraction/text.py`'s `build_extraction_instructions` ->
`build_prompt` -> `extract_facts_from_section`, reaching not just what the
prompt *shows* but also `fact_type_names_for_focus`'s `allowed_fact_types`
filter on both the main and recall passes -- a gated field can never sneak
through a hallucinated response even when it was never shown. Both
production call sites (`workflow/handlers.py`,
`workflow/supplement_handlers.py`) pass the already-computed
`active_flags` in; `supplement_handlers.py` needed a real reorder (it
computed its flags *after* the LLM call, the reverse of `handlers.py`), not
just a parameter add. `llm/benchmark.py::run_case` computes `active_flags`
internally via the same detector against each gold case's own
`source_text`, mirroring production exactly with no gold-case schema
change.

**5 genuinely missing fields added** (`probe_sequence`, `probe_concentration`,
`assay_target_taxa`, `assay_validation`, `study_target_taxonomic_scope`) --
descriptions sourced from `schemas/faire/schema.yaml` directly rather than
the raw CSV, both to avoid real copy-paste encoding artifacts in the CSV
(`Â°C`, `â`) reaching a real prompt, and because two of the CSV's own
proposed native-name spellings (`targetTaxonomicAssay`/`targetTaxonomicScope`)
were literal FAIRe camelCase slot spellings -- exactly the anti-pattern
this taxonomy's own module docstring calls out by name as what a
native_name must never be.

**A second, related problem found and resolved with the user mid-task**: 5
LLM-taxonomy native names (`target_gene`, `thermocycler`,
`commercial_master_mix`, `assay_type`, `biological_replicate_count`)
exactly duplicated `CONTROLLED_SEARCH_FIELDS` entries -- both mechanisms
firing for the same real facts. Checked real gold data before proposing a
fix, not just in theory: the deterministic term-matcher is literal
substring matching against curated lists, not free-text extraction, and it
demonstrably misses or mangles real values (`"MyFi Mix (Meridian
Bioscience)"` matches nothing in `commercial_mm`'s term list; `"Bio-Rad
T100"` gets captured as two disjoint spans `"Bio-Rad | T100"`). Presented
this evidence to the user, who confirmed: **gate these 5 with
`required_any_flags={"pcr_0_1"}` rather than excluding them** -- the LLM
stays the richer, free-text-capable complement, just asked conditionally.
`sequencing_kit` (no natural PCR-adjacent flag -- sequencing is universal
to every eDNA paper) stays excluded, an already-approved trade-off; 4 real
gold cases carry a known, accepted `sequencing_kit` miss as a result, not
fixed here.

**New standing collision-guard test**, the durable fix for "call in
tandem" rather than a one-time hand check: asserts every
`CONTROLLED_SEARCH_FIELDS` term_name that also appears in the LLM taxonomy
is either excluded or flag-gated. It would have caught
`assay_type`/`biological_rep` colliding the moment those landed mid-
research on this very task -- now it catches the next one automatically.

**A real detector bug found by verifying against real gold data, not
trusted on paper**: cross-checked every gold case's `pcr_0_1` detection
against whether its own `expected_facts` include real PCR content, and
found one mismatch -- `example-001` describes explicit PCR content ("...was
amplified using primers... in a 25 uL reaction volume with an annealing
temperature of 57C for 35 cycles...") but never uses the word "PCR" or the
noun "amplification", only the verb "amplified" --
`search_flags.py`'s own `pcr_0_1` pattern (`\bamplification\b`) missed
verb forms entirely. Fixed to `\bamplif\w*\b` (amplify/amplified/
amplifying/amplification/amplifies, no real false-positive risk). Re-ran
the same cross-check after the fix: zero remaining mismatches across all
18 gold cases.

552 tests pass (11 new + ~20 migrated to pass `active_flags` explicitly,
since a `"PCR"`-titled test section with no active flags now correctly
returns nothing for a gated field -- the entire point of this change).

## Duplicate-call audit follow-up, and sample-name replicate detection

**Audit requested**: double-check the PCR-gating work above against
`biological_rep`, `sterilise_method`, `assay_type`, `neg_cont_0_1`/
`pos_cont_0_1`, and the library-prep-sequencing fields for the same
"two independent mechanisms fire for the same real fact" failure mode the
collision-guard test was built to catch. `neg_cont_0_1`/`pos_cont_0_1`,
`sterilise_method`, and `barcoding_pcr_appr`/`lib_screen` checked out clean
(no LLM-taxonomy field maps to their hints at all); `neg_cont_type`/
`pos_cont_type` were confirmed unrelated to `neg_cont_0_1`/`pos_cont_0_1` --
the real schema conditions those two on `samp_category`, a different
per-sample mechanism this task doesn't touch.

**Two real problems found and fixed:**
- `biological_replicate_count` had been gated on `pcr_0_1` in the prior
  round under a blanket "same treatment as the other four overlap fields"
  call. The real schema shows `biological_rep` has no conditional
  requirement at all -- a general sample-design concept, unconditional even
  for non-PCR papers (shotgun metagenomics, etc). Un-gated it (removed
  `required_any_flags`). Verified against gold case
  `controls-replicates-001` first: since the benchmark only exercises the
  LLM checklist path (never the deterministic `CONTROLLED_SEARCH_FIELDS`
  path), *excluding* the LLM field instead of un-gating it would have
  silently broken that already-passing gold case -- concrete evidence for
  un-gating over exclusion, not just schema-reading.
- `forward_sequencing_adapter`/`reverse_sequencing_adapter` (general LLM
  checklist, `adapter_forward`/`adapter_reverse` hints) exactly duplicated
  `search_flags.LLM_JUDGED_SEARCH_FIELDS`'s own narrower, quote-anchored
  pass for the same two fields (explicit "copy the sequence verbatim, only
  accept a quote_id-backed fact" instructions) -- two independent LLM calls
  risking conflicting facts for one real adapter sequence under two
  different `fact_type_candidate` spellings. Excluded the general-checklist
  versions via `LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS`, same pattern as
  `seq_kit`, deferring to the more precise mechanism.

**Collision-guard test extended** to also check
`search_flags.LLM_JUDGED_SEARCH_FIELDS` (it previously only checked
`CONTROLLED_SEARCH_FIELDS`) -- this is exactly the mechanism the adapter
duplicate involved, so the guard would have caught it automatically had it
covered this earlier. Also added `_ACCEPTED_UNCONDITIONAL_OVERLAPS` (just
`biological_rep`) to both guard tests: an explicit, documented exception for
the one field where both an unconditional deterministic detector and an
unconditional LLM field deliberately coexist (a plain replicate count has
low real risk of the two mechanisms disagreeing harmfully, unlike the
adapter-sequence case), rather than silently allowing it to pass the
exclude-or-gate check.

**New feature: `sources/replicate_grouping.py`** -- detects biological
replicate groupings directly from sample-NAME suffix patterns (e.g.
`Site_A_rep1`/`Site_A_rep2`, or `Sample_A`/`Sample_B`/`Sample_C`) in
structured per-sample data, feeding FAIRe's real `biological_rep_relation`
sampleMetadata slot (a per-sample field listing every sibling replicate's
`samp_name`, pipe-joined, per the schema's own worked example). Two
independent, deliberately conservative signals, shared by both adapters
rather than duplicated:
- `EXPLICIT_REP_MARKER` -- an explicit `rep`/`replicate` token plus a digit
  (`_rep1`, `-REP_2`, `_replicate3`), case-insensitive on the token only,
  requires >=2 members.
- `TRAILING_LETTER_SUFFIX` -- a single trailing letter after a separator
  (`_A`, `-b`), gated by requiring >=3 members with consecutive letters so a
  lone `Station_A`/`Station_B` pair (plausibly two different sites, not
  replicates of one) never groups.

Bare trailing digits (`Sample1` vs `Sample12`) are never a signal in either
tier -- deliberately, since that's indistinguishable from arbitrary
incrementing sample numbering. This means FAIRe's own schema example
(`S01_1`/`S01_2`/`S01_3`) isn't itself detected unless the real names also
carry a `rep` token or letter suffix -- an accepted precision-over-recall
tradeoff, confirmed with the user directly (letter-suffix signal on by
default, one fact type for both tiers, always `review_required=True` rather
than splitting into a second, auto-trusted tier).

Wired into `sources/ncbi.py`'s `NcbiBioSampleAdapter` (detects against each
BioSample's `sample_name` attribute, falling back to its title when no
`sample_name` was submitted -- the accession itself is never name-like, so
it's only ever the returned group member/identifier, never the string
matched against) and `sources/supplement_parsing.py` (detects against the
sample-id column's own cell values). ENA is deliberately out of scope: its
`sample_accession` is an INSDC accession, never a free-text name. One new
`MappingRule` (`biological_rep_relation`, SAMPLE-level, `EXACT_LABEL`,
`review_required=True`) routes both adapters' output through the same
existing mapping path as any other structured-adapter fact.

571 tests pass (18 new: `test_replicate_grouping.py`'s 10 pure-function
cases plus adapter/mapping integration tests in
`test_ncbi_ena_parsing.py`/`test_supplement_parsing.py`/`test_mapping_faire.py`).

## Live-run audit against a real paper (PeerJ 10.7717/peerj.333)

**Requested**: the user ran the full pipeline against the same validation
paper used throughout this project and pointed out six concrete-looking
problems in the exported FAIRe CSV. Investigated each against the real
cached PMC full text (fetched live) and the actual raw_facts, rather than
guessing from the symptom alone -- three were real, fixable deterministic-
detector bugs; one was already-correct behavior that needed explaining, not
fixing; one was schema-correct blank output; and one was root-caused all
the way to a live-reproduced LLM extraction failure, not a code bug.

**Fixed, all three confirmed via the actual paper text before changing
anything:**
- `assay_type` returned `"metabarcoding | targeted"` when the paper never
  describes a targeted assay. `_ASSAY_TYPE_CUES`'s bare `species-specific`
  regex cue matched "coral **species-specific** cue preferences" in the
  Discussion -- an ecological statement about coral behavior, not assay
  design. Removed `species-specific`/`taxon-specific` as standalone cues
  (docs/architecture.md's Milestone 15 had already flagged this exact pair
  as unreliable for a different field, `pcr_0_1`, in this same paper); the
  paper's genuine "metabarcoding" classification is untouched, confirmed
  independently by both the deterministic detector and the LLM.
- `commercial_mm` returned `"Bio-Rad"` for a paper whose PCR mixture is
  custom-assembled from separate reagents (ExTaq buffer/polymerase, Pfu
  polymerase) -- "Bio-Rad" only appears as the thermocycler's manufacturer
  ("DNA Engine Tetrad2 Thermal Cycler (Bio-Rad, Hercules, CA, USA)").
  Removed bare manufacturer names (`Thermo Fisher`, `Applied Biosystems`,
  `Bio-Rad`, `NEB`, `KAPA`) from `commercial_mm`'s term list -- these
  manufacturers all sell many non-master-mix products, so a bare brand
  mention can't reliably signal "commercial master mix used." Kept specific
  product/chemistry names (`TaqMan`, `SYBR`, `Luna`, `PowerUp`,
  `QuantiTect`, `SsoAdvanced`) that are unlikely to mean anything else.
- `biological_rep` returned the nonsensical `"4 | 17"`. Root cause: a bare
  `n = <number>` / `N = <number>` pattern, gated only by a loose "the word
  sample(s) appears somewhere in this sentence" context check, matched two
  unrelated numbers in the same paragraph -- `"(n = 4 well replicates per
  cue...)"` (a technical well count) and `"...x N-72C 10 min, with N =
  17-24 depending on the sample"` (a PCR cycle count). Removed the pattern
  (and the now-dead context-check scaffolding it alone used) rather than
  further tighten it -- no simple proximity regex reliably separates a
  genuine biological-replicate count from the many other legitimate uses of
  "n ="/"N =" in PCR/qPCR methods prose. The LLM's unconditional
  `biological_replicate_count` checklist field remains the complementary
  source for well-phrased cases this narrower deterministic detector no
  longer covers.

All three fixes have a dedicated regression test in `test_search_flags.py`
built directly from the real paper's sentences, plus a full-suite rerun
(581 tests pass).

**Investigated and explained, not fixed (already correct):**
- `checksum_method = "MD5"`: not extracted from the paper at all -- it's
  inferred from ENA's own repository metadata (`mapping/rules.py`'s
  `_constant_md5`, whenever ENA's `fastq_md5` is present for a run, since
  ENA's read_run report always uses MD5 but never states the algorithm
  explicitly). Confirmed the real MD5 hex digests are genuinely present in
  this paper's ENA-deposited runs. The user wouldn't see this "in the
  paper" because it's sequencing-repository provenance, not an authorial
  claim -- working as designed.
- `code_repo` blank: the real FAIRe definition is "Link to public
  repository where analysis code is archived" (e.g. a `github.com` URL).
  This paper's only code mention -- "Perl script for rarefaction analysis
  (cca_rarefaction.pl) and R script ... are available in Supplemental
  Information 1" -- is not a public repository link at all, so leaving
  `code_repo` blank is schema-correct, not a miss. `sop_bioinformatics`
  (LLM native name `bioinformatics_sop_reference`) is the more plausible
  target for this exact sentence, but it's also blank in this run -- root
  cause traces to the same chunk-recall gap described next, not a mapping
  bug.

**Investigated via a live reproduction; correctly ruled out timeout/parse/
truncation bugs, but the "genuine model recall limitation" conclusion below
was wrong -- see the follow-up section right after this one, which found
and fixed the real cause**: `ampliconSize`/`amplificationReactionVolume`/
`annealingTemp` came back empty despite the paper explicitly stating a 30
µl reaction volume and a 55 °C anneal step. Fetched the real cached
full-text section ("Metabarcoding of cue communities", 4760 chars,
splitting into 3 real chunks at this run's 2500-char/call setting) and
replayed the exact same chunks against the live local model
(`qwen3:4b-instruct-16k`) end to end. Confirmed `pcr_0_1` correctly gates
true (ruling out a gating bug) and that chunk 1/chunk 3 return nothing
while chunk 2 extracts fine, matching the original run exactly.

## Fixing code_repo, commercial_mm, and the real cause of the missing PCR fields

Direct follow-up to the audit above, after the user reviewed those findings
against the same live pipeline run and asked for three more fixes.

**`code_repo` now captures a non-repository availability statement.**
FAIRe's real definition (`extraction/publication_metadata.py`'s
`extract_code_repo_from_text`) still prefers a genuine GitHub/GitLab/
Bitbucket URL when one exists, but now falls back to the whole sentence
describing where the code/scripts/software are available (e.g. "Perl
script for rarefaction analysis (cca_rarefaction.pl) ... are available in
Supplemental Information 1") when no such URL is present, rather than
leaving the field blank just because it isn't a public-repository link.
Sentence-boundary detection deliberately isn't a single "no periods until
the sentence ends" regex -- this exact real sentence has two periods
embedded in filenames (`cca_rarefaction.pl`, `rarefaction_figs.R`) that
would break a naive version; splits on punctuation-followed-by-whitespace
instead, matching search_flags.py's own `_snippets` approach. Prefers a
real URL over an availability statement when both exist in one text (a
real repo link is always more useful than a pointer to "look in the
supplement").

**`commercial_mm`/new `custom_mm`: broadened detection, whole-sentence
capture, and a commercial-vs-custom classifier.** The user's own
suggestion: search for "PCR mixture"/"PCR mix"/"polymerase chain reaction
(PCR) mixture" and capture the whole quote. Implemented as a new shared
`_match_pcr_mixture_phrase` in `extraction/search_flags.py`: finds a
sentence matching `_PCR_MIXTURE_MARKERS_RE` (PCR/qPCR/reaction mix(ture),
master mix/mastermix, or the parenthetical "polymerase chain reaction
(PCR) mixture" phrasing), then classifies it as `commercial_mm` only if it
also names a specific master-mix product/brand or says "master mix"
outright (`_COMMERCIAL_MASTER_MIX_BRAND_RE`) -- otherwise it's `custom_mm`
(a brand-new `CONTROLLED_SEARCH_FIELDS` entry; FAIRe's real definition is
"custom master mix composition, if a commercial one was not used", exactly
matching this paper's own ExTaq-buffer/Pfu-polymerase composition). One
new `custom_mm` `MappingRule`, mirroring `commercial_mm`'s own existing
one. Verified against the real paper: the PCR-mixture sentence now
populates `custom_mm` with the full composition (not `commercial_mm`,
since no product/brand is named), matching what the user described.

**Root-caused for real this time: the missing PCR fields were a token-
budget truncation bug, not a model recall limitation.** Replayed the exact
same real chunk against the live model with the production prompt
completely unchanged, but printing the model's raw completion instead of
only whether it parsed. The model was NOT returning `[]` -- it was
correctly extracting facts in order (assay_type, target_gene, primer
name/sequence/concentration, ...) and got cut off mid-object at exactly
`max_output_tokens=1024` tokens, producing invalid JSON that the old
all-or-nothing parser discarded entirely, losing every fact the model got
right along with the one it didn't finish. Confirmed the fix by rerunning
with `max_output_tokens=2048` (no prompt changes at all): a complete, valid
15-fact response, including all three missing fields
(`amplicon_size: "~550 bp"`, `pcr_reaction_volume: "30 ul"`,
`annealing_temperature: "55 C"`). Then ran the real `extract_facts_from_section`
end to end using the codebase's own actual `config/local.yaml` defaults
(`max_output_tokens: 2048`, `extraction_max_chars_per_call: 16000`) with no
manual overrides at all -- the whole 4760-char section now fits in ONE
chunk instead of three, and all 17 facts extract correctly. The audit run
script's own `LOCAL_LLM_EXTRACTION_MAX_CHARS_PER_CALL=2500`/
`LOCAL_LLM_MAX_OUTPUT_TOKENS=1024` env overrides had silently shrunk both
values well below `config/local.yaml`'s own already-tuned, already-documented
defaults -- not a codebase bug, but a real, fixable pipeline-configuration
issue. No prompt engineering was needed or attempted; the existing prompt
handles this content correctly once given enough output budget.

**Added a real defense-in-depth hardening regardless of config**:
`llm/base.py`'s `try_parse_json` now recovers as many complete objects as
possible from a top-level JSON array that got cut off mid-object
(`_try_parse_truncated_json_array`, using `json.JSONDecoder.raw_decode` to
decode one complete object at a time and stop cleanly at the first
incomplete fragment), rather than discarding a real, mostly-correct
response just because its last object didn't finish. Only ever applies to
a bare top-level array (the only shape `generate_json`'s callers produce);
a truncated single object, or an array with zero complete objects, still
correctly returns `None` and triggers the existing retry-on-invalid-JSON
path. `generate_json` accepts a partial recovery as final (no wasted retry
call) since a retry at the same token budget is unlikely to fully
succeed either. This protects against a similarly dense paragraph even
under a reasonable token budget, independent of the config fix above.

595 tests pass (9 new: 4 for `code_repo`'s new fallback path, 2 for the
`commercial_mm`/`custom_mm` reclassification, 4 for the JSON-truncation
recovery, minus 1 renamed rather than added). See README's "Fixing
code_repo, commercial_mm, and the real cause of the missing PCR fields".

## An internal_study_id column for tracing rows across multi-study exports

**The problem**: `export_faire()` (`exports/faire.py`) has always merged
every study currently in the database into one shared set of output CSVs
-- `projectMetadata.csv`/`sampleMetadata.csv`/`experimentRunMetadata.csv`
each get rows from every study appended together, with no per-study filter
or grouping anywhere. This was invisible with one study at a time, but the
user is now running multiple papers concurrently (PeerJ 10.7717/peerj.333
plus two more via a parallel process) and had no way to tell, just from
the exported CSVs, which sample/experiment rows belonged to which project
row.

**Investigated whether "the same real sample shared across two papers"
already has a home before designing the fix**: it doesn't. `Entity.study_id`
(`database/models.py`) is a single-valued foreign key with a unique
constraint on `(study_id, entity_level, external_identifier)`. `Source`
already has an explicit `StudySource` many-to-many join table for exactly
the "one real thing referenced by multiple studies" case -- there is no
equivalent for `Entity`. So today, if two different papers reference the
same real NCBI BioSample or ENA run, the pipeline creates two independent
`Entity` rows, not one shared one. The user's own request anticipated a
pipe-separated multi-study column for this exact scenario; presented the
real correctness wrinkle directly before building anything: grouping rows
across studies by a shared `external_identifier` string is only safe for a
genuine global repository accession (NCBI/ENA), never for a paper-native
label a supplement table or the LLM assigned (e.g. two papers could both
independently use "S1" without meaning the same physical sample). User
confirmed: build the simple, always-correct version now (one
`internal_study_id` value per row, matching its real owner), treat true
cross-study entity merging as a separate future feature.

**Implementation**: a new `internal_study_id` column, prepended as the
first column in all three CSVs, set from the already-in-scope `study` in
each of `export_faire`'s three row-building loops. Deliberately **not** a
real FAIRe field -- not added to `schemas/faire/classes.yaml` (unlike the
genuine NOAA/SEUS-MBON domain extensions `expedition_id`/
`ship_crs_expocode` added in an earlier milestone, this must never be
mistaken for something submittable to NOAA/GBIF/OBIS as part of the real
checklist) and deliberately excluded from `field_reference.csv` (which
documents real FAIRe fields with real requirement levels/`exact_mappings`
-- this column has neither). The three `_write_csv` call sites prepend the
column inline (`[INTERNAL_STUDY_ID_FIELD, *project_columns]`, etc.)
without mutating the underlying `project_columns`/`sample_columns`/
`experiment_columns` variables, since those same lists are reused
afterward to build `field_reference.csv`'s column inventory -- keeping
that file scoped to genuine FAIRe fields only.

**Verified against the user's own real two-study database**
(`fair_ocean_audit_2_current_rerun`, containing exactly the two papers
mentioned above): `projectMetadata.csv`'s two rows each carry their own
distinct `internal_study_id`; all 68 combined sample/experiment rows
correctly split 25/43 across the two studies by `internal_study_id`,
matching each row's real owner; confirmed `internal_study_id` does not
appear in `field_reference.csv`.

597 tests pass (2 new: one exact-column-order test updated for the new
prepended column, one new end-to-end two-study traceability test). See
README's "An internal_study_id column for tracing rows across multi-study
exports".

## Fixing a real taxon-extraction bug and two missed-field gaps, found via two more real papers

The user ran two more real papers through the pipeline (ISME J
10.1093/ismejo/wrae013, PLOS ONE 10.1371/journal.pone.0303937) and found
four more concrete problems.

**Fixed: `assay_target_taxa`/`targetTaxonomicAssay` producing outright
garbage.** `extraction/taxonomic_assay.py` (a new module, built by Codex,
that deliberately reads only title/abstract/keywords -- "easy to overfill
from a whole paper" per its own docstring) pulled `"Diversity studies |
The device was"` for the PLOS paper and nothing for the ISME paper.
Root-caused directly against the real Crossref abstract: its
`_taxa_from_metadata_item` fell back to `_taxon_mentions(sentence)` -- a
whole-sentence scan for anything matching "Capitalized word + lowercase
word(s)" (`_SCIENTIFIC_NAME_RE`, meant to catch genuine binomial species
names) -- whenever a sentence passed the loose assay-context gate (mention
of "PCR"/"amplicon"/etc.) without containing an explicit target/detect/
amplify phrase. That fallback matches ordinary English constantly: the
PLOS abstract's own opening two words ("Diversity studies") and a
different sentence's opening three words ("The device was") both
satisfied it, with zero relationship to any real taxon. Removed the
fallback entirely -- only the narrower target-phrase capture (an explicit
"targeting X"/"designed to amplify X"/etc. phrase) is now trusted for
sentence-level extraction; a sentence with assay-context but no such
phrase now correctly contributes nothing instead of guessing. Separately
confirmed via the real ISME full text that its blank result was already
correct, not a bug: its RT-qPCR methods only ever name the marker gene
("16S rRNA", primers 515F/805R) -- a `target_gene` fact, not a taxon name
-- and never state an explicit target taxon anywhere in the paper, so
"nothing extracted" is the right, non-hallucinated answer for this paper
specifically, independent of the abstract-only scoping choice.

**Fixed: `seq_kit` missing "TruSeq Stranded mRNA kit (Illumina)".**
`_SEQUENCING_KIT_PATTERNS` (`extraction/search_flags.py`) had no pattern
for the TruSeq family (Stranded mRNA, Nano, PCR-Free, DNA, ...) -- one of
the most common Illumina library-prep kit names in real papers. Added a
generic `TruSeq <1-4 words> kit` pattern.

**Investigated, partially improved: forward/reverse primer volume missed
for "...and 1 uL of each primer".** The ISME paper's RT-qPCR mixture
sentence states one aggregate volume covering both primers, never naming
forward/reverse separately -- unlike `template_dna_volume`, which the
model correctly pulled from the adjacent "1 uL of cDNA" in the same
sentence. Added explicit guidance to `forward_primer_volume`/
`reverse_primer_volume`/`forward_primer_concentration`/
`reverse_primer_concentration`'s checklist descriptions: reuse an
aggregate "each primer"/"both primers" value for both fields when the text
doesn't name them separately. Confirmed live against the real sentence in
isolation that the model now reliably fills both fields; in the full,
busier chunk (competing with primer names/sequences/annealing temp/cycle
count/master-mix name all in one call) it did not consistently pick this
up on every replay -- a real, but harder to fully close, attention-budget
effect distinct from the max_output_tokens truncation bug fixed last
round.

**A standing test caught two more real duplicate-mechanism gaps mid-task,
exactly as designed**: Codex's new `adapter_trimming_method`/
`length_filtering_tool`/`minimum_read_length` deterministic detectors
(narrow Trimmomatic/MINLEN-specific regexes) duplicate the LLM taxonomy's
own native names for the same three concepts, and the collision-guard test
built earlier this session failed immediately on the current branch.
Checked real data before resolving: excluding the LLM versions would have
lost `adapter_trimming_method` entirely for the PeerJ paper (custom Perl
script) and the ISME paper (SeqPrep) -- neither is Trimmomatic, so the
narrow deterministic detector alone would never catch them; on the one
real Trimmomatic paper (PLOS) both mechanisms already agree today. Added
all three to `_ACCEPTED_UNCONDITIONAL_OVERLAPS`, the same low-conflict-risk,
deliberate-redundancy precedent as `biological_rep`.

607 tests pass (10 new since the last commit, including one
taxonomic_assay.py false-positive regression test built from the real
PLOS abstract and one primer-volume/concentration hint-wording test from
this round). See README's "Fixing a real taxon-extraction bug and two
missed-field gaps, found via two more real papers".

## Disambiguating trim tools, catching a reagent-listing gap, and building out the second-PCR field family

Direct follow-up after the user reviewed the previous round's fixes
against the same two real papers.

**Fixed: Trimmomatic wrongly stamped onto `adapter_trimming_method` too.**
The user pointed out the ISME paper genuinely uses *both* SeqPrep (adapter
removal) and Trimmomatic (quality/length trimming) -- but only Trimmomatic
was showing up, and it was showing up for BOTH fields. Root cause: the
deterministic Trimmomatic detector was completely context-blind -- it
fired on both `adapter_trimming_method` and `length_filtering_tool`
whenever "Trimmomatic" appeared anywhere in the text, regardless of what
the surrounding sentence said it was used for. Rewrote it as a single
`_match_trim_tool(purpose=...)` function that only attributes a tool name
to a field when context matching that purpose ("adapter"/"ILLUMINACLIP"/
"barcode" vs. "quality"/"length"/"MINLEN"/"LEADING"/"TRAILING") appears
within `_TRIM_TOOL_CONTEXT_WINDOW` (90 chars) of that specific mention --
a whole-sentence check wasn't tight enough, since the real paper describes
both tools in one long compound sentence with numbered clauses
("(i) ...SeqPrep...; (ii) ...; and (iii) ...Trimmomatic..."), ~250
characters apart. Added a `_SEQPREP_RE` pattern so `adapter_trimming_method`
can now recognize SeqPrep at all (it previously had no adapter-tool
pattern beyond Trimmomatic itself). Confirmed live: `adapter_trimming_method`
now correctly gets "SeqPrep 1.2" and `length_filtering_tool` gets
"Trimmomatic 0.39" for this exact paper -- no pipe-joining needed, since
each tool now lands on the field matching what the text actually says it
was used for.

**Fixed: `custom_mm` missing a reagent-listing sentence with no "mixture"
word at all.** The PLOS ONE paper's real PCR-composition sentence --
"...performed with 0.02 U/ul of Phusion High Fidelity DNA polymerase, 1X
Phusion HF Buffer and 200 uM of dNTPs (New England Biolabs, USA)" -- never
uses "mixture"/"mix", so the existing `_PCR_MIXTURE_MARKERS_RE` trigger
(built two rounds ago) missed it entirely. Added a second, ANDed trigger:
a sentence naming a polymerase (a known brand -- Taq/Phusion/Pfu/ExTaq/
KAPA/Q5/GoTaq/Platinum/AmpliTaq/HotStar/iProof -- or a generic "X
polymerase" phrase) together with "buffer" or "dNTP" is just as much a
PCR-mixture description as one using the word "mixture" itself. Confirmed
live: this sentence now correctly populates `custom_mm` (no master-mix
brand named, so not `commercial_mm`).

**Built out the second-PCR (`pcr2_*`) field family**, per the user's
request after noticing the PLOS ONE paper's two-step protocol has "quite a
few open fields" with nowhere to go. Added 9 new native names --
`second_pcr_reaction_volume`/`second_pcr_template_dna_volume`/
`second_pcr_thermocycler`/`second_pcr_annealing_temperature`/
`second_pcr_cycle_count`/`second_pcr_conditions`/
`second_pcr_commercial_master_mix`/`second_pcr_custom_master_mix`/
`second_pcr_plate_id` -- mirroring the first PCR's own fields one for one
against the real FAIRe schema's `pcr2_*` slots (`pcr2_analysis_software`/
`pcr2_method_additional` were already deliberately excluded from an
earlier round, matching the first PCR's own exclusion of those two
concepts). Gated on `pcr_0_1` like the rest of the group, rather than a
dedicated two-step-only flag -- a one-step-PCR paper simply won't have any
second-PCR text for the model to (correctly) find nothing in.

**A live validation of the new fields caught a real hallucination bug
before it shipped**: replaying the real PeerJ second-PCR text showed the
model copying this module's own example values ("22 uL", "2 uL") verbatim
into `raw_value` when the real text didn't state those specific
quantities for the second PCR -- exactly the "never copy an example into
raw_value" failure mode the extraction prompt already warns against, just
newly visible here. Removed the example values from the four new
quantity-shaped second-PCR fields (matching how other fields in this
module already omit an example rather than risk one), and added explicit
"the SECOND PCR's own X only, never the first PCR's" (and the reverse, on
every corresponding first-PCR field) disambiguation to prevent
cross-attribution when a paper describes both PCR steps together.
Re-verified live against the full first+second-PCR paragraph: all 18
facts extract correctly, each quantity attributed to the right PCR step,
zero hallucinated examples.

610 tests pass (3 new: two Trimmomatic/SeqPrep disambiguation regression
tests, one reagent-listing-without-"mixture" regression test -- the
second-PCR field additions are covered by the existing
`test_render_field_reference_includes_faire_hints`-style completeness
tests, which enumerate every native name/hint automatically).

## Excluding MAG BioSamples and parsing filter/pore-size/depth out of free-text attributes

Running two more real papers (10.3389/fmicb.2024.1295149,
10.1038/s42003-024-06136-2) surfaced three real problems in
`sources/ncbi.py`.

**Filter fields blank for fmicb -- and a real category error caught before
it shipped.** NCBI's `samp_mat_process` attribute (schema-documented to
hold a full free-text processing narrative, e.g. `"0.22 µm cartridge
filtration followed by DNA extraction"`) passed through verbatim but was
never parsed for the FAIRe fields it actually describes. Checked the real
schema before writing a parser: `filter_diameter` is explicitly "Diameter
of a filter **if circular**. Unit = mm" -- the physical filter disc size
(e.g. a 47mm membrane) -- a completely different concept and unit from
pore size. `"0.22 µm"` is a pore size, whose real home is a separate,
previously-unmapped field, `size_frac` ("Filtering pore size ... Unit =
µm"). New `_derive_filter_facts` in `sources/ncbi.py` parses
`samp_mat_process`'s text for `size_frac` (µm value near "pore"/"filt"
context), `filter_diameter` (mm value near "filter"/"disc"/"membrane"
context -- a genuinely different pattern/context/unit from the pore-size
one, never conflated), `filter_material` (against the real
`filter_material_enum` term list), `filter_name` (a short curated brand
list -- Sterivex, Millipore, Whatman, ...), and `filter_passive_active_0_1`
(`"1"` for "cartridge"/"pump"/"peristaltic", `"0"` for "passive"/
"submerged"/"gravity", otherwise left unset -- never guessed). Confirmed
live against the real BioProject: correctly produces `size_frac`/
`filter_passive_active_0_1` for the water samples and nothing at all for
the sediment samples' `samp_mat_process` value ("DNA extraction from
sediment samples" -- no filtration mentioned).

**fmicb's depth was identical across every sample despite a real depth
profile (30m to 5200m).** No sample carries a literal `depth` attribute;
the real per-sample depth is embedded as free text inside a differently-
named attribute, `source_material_id` (this submitter's own convention,
e.g. `"3500 m V3-V4"`, `"Overlaying water V3-V4"`). With nothing mapping
that to depth, every sample fell back to one study-wide broadcast
LLM-extracted depth value -- wrong for all but whichever one sample it
happened to match. New `_derive_depth_from_source_material_id`: a leading
number (optionally "m") becomes a `depth` fact, reusing the *existing*
`depth` `MappingRule`/`to_meters` transform (no new rule needed);
`"Overlaying water"` (no leading number) is deliberately left blank rather
than guessed as 0. Confirmed live: 20 distinct real per-sample depths (30m
through 5200m) now populate correctly, matching the paper's own described
depth profile exactly.

**MAG (metagenome-assembled genome) BioSamples were polluting
sampleMetadata for s42003.** Traced live (a real `efetch` against a real
accession) that a BioProject can legitimately link to both the true raw
environmental BioSamples *and* a family of cross-linked MAG BioSamples
from an entirely different downstream assembly project -- the MAG records
carry assembly-specific attributes (`assembly software`/
`completeness score`/`binning software`) a raw sample never has, and none
of the real per-sample data (depth/lat-lon). Confirmed BioSample XML
carries an authoritative, INSDC-standard structural marker for this that
the adapter didn't parse: the `<BioSample package="...">` root attribute
(a MAG's package is `MIMAG.*`, the real "Minimum Information about a MAG"
checklist) and `<Models><Model>MIMAG</Model></Models>`. New
`_is_mag_biosample` (checks `package`/`model`, with a title-text fallback
for a record that doesn't carry either) now excludes a MAG-identified
BioSample from becoming a SAMPLE entity entirely, at the top of
`extract_structured_facts`'s per-sample loop.

**A separate, more fundamental bug found while verifying all of the
above against s42003's real BioProject, reported to the user rather than
silently fixed (out of this task's approved scope)**: `esearch` for
`PRJNA529480` returns *two* UIDs, and `_esearch_first_uid` blindly takes
the first one -- which turned out to be `PRJEB73262` ("TPA metagenomic
assembly of the PRJNA529480 data set"), a *different*, MAG-only downstream
project that just happens to mention the real accession in its own title,
not `PRJNA529480` itself (confirmed via `esummary`: UID `1356142` ->
`PRJEB73262`; the real, correct UID `529480` -> `PRJNA529480`, 99 real
linked samples). This is the true root cause of s42003's missing
sample-level data -- the adapter was fetching an entirely different
project's 8 BioSamples (all MAGs, now correctly excluded by the fix
above) instead of the real project's 99. Affects both
`NcbiBioProjectAdapter` and `NcbiBioSampleAdapter` (`fetch_record` in
both classes calls the same naive `_esearch_first_uid`), and plausibly any
other paper where NCBI's esearch returns more than one UID for an
accession search. Flagged for a follow-up fix (verify each candidate UID's
own accession via `esummary` and pick the exact match) rather than
implemented here, given the scope and risk of changing core BioProject/
BioSample resolution used by every paper this pipeline processes.

620 tests pass (8 new: MAG exclusion via package/title, a real-sample
BioSample confirmed unaffected, filter/size_frac derivation from both real
`samp_mat_process` values plus a constructed filter-diameter case, and
depth derivation from `source_material_id`).

## Multi-study entity sharing + citation-expansion discovery

Direct follow-up to the previous section's `_esearch_first_uid` finding.
The user's own framing, escalated into a full architectural task: with
~3,000 papers about to go through initial discovery, and papers routinely
reanalyzing/reusing each other's deposited samples, the pipeline needed a
real way to represent "the same real BioSample/experiment/sequencing run
is used by more than one paper" -- not just a smarter UID pick. Two
consequential scoping choices were made explicitly with the user before
building anything: a newly-discovered citing paper should **fully
auto-expand** into its own Study (not just be recorded), and traversal
depth should be **configurable multi-hop**, not fixed at one hop -- both
raise real uncontrolled-growth risk at 3,000-paper scale, addressed with
concrete caps below rather than left implicit.

**The esearch fix, for real this time.** `sources/ncbi.py`'s
`_esearch_verified_uid` replaces `_esearch_first_uid`: a single-UID
esearch result returns unchanged (no extra call); a multi-UID result gets
each candidate's own accession fetched via a new shared `_esummary_records`
helper and matched against what was actually searched for -- exactly the
check that distinguishes `PRJNA529480` (UID `529480`) from `PRJEB73262`
(UID `1356142`, the wrong project from the previous section). No match at
all falls back to the first UID but records a review-flagged
`ambiguous_uid_resolution` `RawFact` rather than guessing silently. A
second, independent signal was added alongside it: `NcbiBioSampleAdapter`
now confirms one already-fetched sample's own `biosample->bioproject`
reverse-elink points back to the same BioProject UID used to fetch it --
this is the specific check that would have caught the `PRJEB73262` case
directly, since a downstream MAG-reanalysis BioSample's reverse elink
points to *its own* BioProject, not the one whose accession search
happened to surface it first.

Confirmed live against three independent real ambiguous cases, not just
the original bug report: `PRJNA529480` (UID `529480` correctly picked over
the wrong `1356142`/`PRJEB73262`, 99 real samples, reverse-elink agrees);
`PRJEB1787`/Tara Oceans prokaryote size-fraction data (correct UID
`196960` picked over `1350728`/`PRJEB92205`, a SPIRE-produced TPA
reanalysis); `PRJNA385854`/bioGEOTRACES (correct UID `385854` picked over
three other SPIRE/EMG-produced TPA reanalysis projects, 480 linked
samples, reverse-elink agrees). The SPIRE/EMG automated-reanalysis
pattern polluting esearch's own relevance ranking turned out to be
systemic across real, heavily-reused public datasets, not a one-off --
exactly the kind of thing worth confirming against more than one example
before trusting a fix at 3,000-paper scale.

**Data model: `entity_studies`.** New table, an exact structural mirror of
the existing `study_sources` (`identity/source_linking.py`'s own
precedent for "one real thing linked to multiple studies"). `Entity.study_id`
stays the fast "home" pointer, unchanged everywhere else in the codebase;
`entity_studies` is what makes an Entity belonging to more than one Study
representable at all. Only `SAMPLE`/`EXPERIMENT_RUN`/`SEQUENCING_RUN` are
shareable (`SHAREABLE_ENTITY_LEVELS`) -- `PROJECT` stays single-study
deliberately (a shared BioProject accession is already
`resolve_or_create_study`'s job, and a paper's own bioinformatics-pipeline
facts are PROJECT-level and paper-specific, so making PROJECT shareable
too would create a second, competing mechanism for the same situation).
A new partial unique index on `entities` (`entity_level`,
`external_identifier`, shareable levels only) makes a global
get-or-create lookup safe -- confirmed against the real production
database before adding it that zero cross-study duplicates existed among
7,787 real entities across 101 studies.

A single new choke point, `identity/entity_linking.py`, replaced what
turned out to be **two independent, divergent** `_get_or_create_entity`
implementations -- one in `workflow/handlers.py` (the main fact-
materialization path) and a second, separately-written copy in
`extraction/experiment_runs.py` (the legacy sequencing-run-to-experiment-
run materialization path). Only the first got updated for cross-study
sharing initially; the second would have silently kept creating
un-linked, un-shareable duplicate entities for exactly the papers this
whole feature targets. Caught by running the real end-to-end verification
below, not by static review -- a live `sampleMetadata.csv` export showed
an empty `internal_study_id` for an auto-materialized experiment entity,
traced to the second implementation never having written an `entity_studies`
home row at all.

**Node-adding discovery.** New `DISCOVER_CITING_STUDIES` task
(`workflow/handlers.py::handle_discover_citing_studies`), enqueued right
after a BioProject accession resolves (idempotency key is the accession
alone, so two studies sharing one accession only trigger this once). One
`bioproject->pubmed` elink call returns every citing PMID at once --
genuinely different from the reverse-elink UID-verification signal above,
and cheap by construction (no per-sample fan-out). A citing paper not
already known gets its own full Study row
(`discovery_depth`/`discovery_parent_study_id`/`discovery_root_study_id`/
`discovery_trigger` provenance columns) and re-enters the normal
`DISCOVER_IDENTIFIERS` pipeline via the task queue's own idempotency --
not an in-process recursive traversal. Two safety valves make the
approved aggressive-expansion choices safe at scale:
`DiscoveryConfig.citation_expansion_max_depth` (wired up for real for the
first time -- previously defined but never read anywhere; default `1` for
the initial discovery run) and a new `max_citing_papers_per_bioproject`
(default 25) -- both record a review-flagged `citing_pmid_not_expanded`
fact for whatever they cap, never a silent drop. A recurring pass
(`enqueue-citation-rediscovery-backfill`, auto-wired into `weekly-update`
alongside the existing `quarterly_full_rediscovery` block) re-checks every
known accession on a 90-day default cadence, since the accession-only
idempotency key above means a paper published or indexed *after* a
study's first resolution would otherwise never get picked up.

**Three more real bugs, found only by running the real thing end to end**
(seeding the real s42003 paper and letting `DISCOVER_IDENTIFIERS`/
`DISCOVER_CITING_STUDIES` run for real against live Crossref/Europe PMC/
OpenAlex/NCBI/ENA -- a mocked unit test suite alone did not, and structurally
could not, have caught any of these three):

1. **Full-text-mined accession re-discovery treated as a study-identity
   conflict.** The citing paper found via citation-discovery
   (10.1073/pnas.2005917117) has its own open-access full text, which --
   unsurprisingly, since that's often *why* it was flagged as citing in
   the first place -- also mentions `PRJNA529480` directly. The
   pre-existing full-text identifier scanner tried to reconcile that
   mention against the study that already owns the accession, couldn't
   establish "consistent" (no single Source to compare evidence against),
   and fell back to its own pre-existing safety behavior: split off an
   empty placeholder sibling Study and flag a `PENDING` `CandidateMatch`
   for human review. Not data corruption, but this would recur for
   essentially every citing paper at ~3,000-paper scale, since a citing
   paper's own text plausibly reiterating the accession it cites is the
   *common* case, not an edge case. Fixed with a new
   `_linked_via_discovery_lineage` check in `identity/resolution.py`,
   checked *before* confidence tier: once two studies are already known
   (via `discovery_parent_study_id`/`discovery_root_study_id`) to be a
   deliberately-separate citing/cited pair, no identifier match between
   them is grounds for a merge or a sibling-split, regardless of which
   adapter reports it or at what confidence.
2. **Far more seriously: an unconditional silent Study merge on shared
   BioSamples.** `NcbiBioSampleAdapter`/`EnaAdapter`'s `find_related()`
   both report every fetched sample's accession as a default-
   `STRUCTURED_SOURCE` `RelatedIdentifier` -- and `resolve_or_create_study`
   treats any `STRUCTURED_SOURCE` match against a different Study as
   strong enough to merge unconditionally, no consistency check, by
   design (correct for a DOI/PMID/BioProject accession, where that really
   is strong duplicate-submission evidence). The moment the citing
   paper's own BioSample resolution reached a sample the original study
   already owned, this silently merged the two Study rows together --
   exactly the outcome this entire feature exists to prevent, and the
   most serious of the three findings (real data corruption, not just
   review-queue noise, since it would have merged genuinely different
   papers' bibliographic and project-level facts). Fixed by excluding
   `BIOSAMPLE_ACCESSION` from Study-identity resolution entirely: a
   shared BioSample accession is now always recorded as informational
   (`SHARES_ACCESSION_WITH`), never merge/sibling-split evidence, since
   the real "this sample belongs to more than one Study" fact is already
   fully represented at the Entity level via `entity_studies`.
3. **A global-vs-study-scoped constraint mismatch in `EntityRelationship`.**
   `workflow/handlers.py`'s `_get_or_create_entity_relationship` looked up
   an existing relationship scoped by `study_id`, but the table's own
   `uq_entity_relationship` constraint is global on `(from_entity_id,
   to_entity_id, relationship_type)`, with no `study_id` in it. Once two
   studies can legitimately resolve the same shared run/sample/experiment
   entities (fix #2, above, is what let this actually happen), the
   second study's own resolution pass reached the identical global triple
   its study-scoped lookup couldn't see, producing a real `UNIQUE
   constraint failed` mid-task. Fixed by making the lookup global,
   matching the real constraint -- a physical relationship between two
   entities doesn't change depending on which Study is asking, exactly as
   physically invariant as the entities themselves.

**Final verified state**, after all three fixes: seeding s42003
(`10.1038/s42003-024-06136-2`) end to end through `ingest-seeds` ->
`enqueue-seed-backfill` -> `worker` produces exactly two Study rows --
the original (`discovery_depth=0`, correct title/DOI/`PRJNA529480`/83 real
samples) and the citing PNAS paper found via citation-discovery
(`discovery_depth=1`, correct `discovery_parent_study_id`/
`discovery_root_study_id`/`discovery_trigger="bioproject_pubmed_citation"`)
-- zero merges, zero `CandidateMatch` review-queue noise, and 231 real
entities (samples, experiment runs, sequencing runs) correctly linked to
*both* studies via `entity_studies` while keeping their original home.
`exports/faire.py`'s `sampleMetadata.csv` reflects this correctly: each
shared sample appears exactly once (keyed by home `Entity.study_id`, so no
double-counting), with a pipe-joined `internal_study_id` listing both
studies and neither study's own paper-specific broadcast default merged
onto the row (`broadcast = {} if len(linked_study_ids) != 1 else
_study_wide_values(...)`, matching `identity/consistency.py`'s own
"unknown is preferable to guessing" stance) -- only the sample's own
entity-level facts, which stay correctly unconditional.

645 tests pass (23 new): the esearch/reverse-elink fix (multi-UID
disambiguation, review-flagging, reverse-elink mismatch detection);
cross-study entity sharing (`_get_or_create_entity`'s global lookup,
`exports/faire.py`'s pipe-joined `internal_study_id`/broadcast gating);
citation-discovery (new-study creation with correct provenance, the
already-known-DOI no-op case, both safety-valve caps); the recurring
node-adding backfill (accession-scoped idempotency, cadence gating in
`weekly-update`); and the two Study-identity regression tests
(discovery-lineage suppression, BIOSAMPLE_ACCESSION never merging or
flagging) built directly from the real live-run findings above.

## Root-paper determination and two-phase discovery/mapping gating

Direct follow-up after reviewing the previous section's live results. Two
sharp questions from the user: for a shared entity, which linked study's
facts should actually fill its blanks (today, none did, arbitrarily) --
and shouldn't the whole discovery network settle before any table-filling
happens at all, even if that takes a long time for one heavily-reused
seed.

**Clarifying how seeding actually works, first.** The user's mental model
was "we process one paper's whole network before starting the next" --
not true. All ~3,000 seeded papers get their own `Study` row up front, in
one pass, before any discovery runs; every one of them gets a
`DISCOVER_IDENTIFIERS` task enqueued at the same time, same priority.
Whichever task the worker happens to claim first decides `Entity.study_id`
("home") for any accession the two share. For a citation-discovered pair
specifically (one paper found via citing another's BioProject), the cited
paper is guaranteed to resolve first by construction -- but for two
*independently seeded* papers that happen to share a BioProject with no
citation relationship between them, home is pure queue-order luck, not a
claim about who collected the data. `discovery_depth==0` means "was in the
initial seed batch," nothing more -- a depth-0 seed can itself be a
reanalysis paper if the seed file happens to include one.

**Root-determination algorithm**, exactly as scoped with the user:
earliest publication year among linked studies wins outright (reusing the
existing `_publication_year_for_study` helper, not duplicating it -- a
paper cannot analyze data that doesn't exist yet); the BioProject's own
registration date is read as corroboration/plausibility context only,
never used to override the year-based pick on its own; submitter/
institution matching is explicitly deferred, since no adapter in this
codebase parses structured author affiliations into anything usable today
(Crossref only captures given/family name; OpenAlex's raw `authorships`
blob, which does carry institutions in the real API, is stored unparsed
here); genuinely ambiguous cases (same year, or no publication-date signal
at all for the tied studies) get flagged `NEEDS_REVIEW`, never guessed.
`identity/entity_linking.py::create_entity` sets an entity's root to
itself, eagerly, at creation -- the common (non-shared) case needs no
algorithm at all -- and flips back to `PENDING` the moment a second study
links to it, so a stale answer never lingers past the point it stopped
being true.

**Two-phase gating, without touching the already-live-validated discovery
machinery.** A background design agent explored the exact task-chaining
structure before anything was built, and confirmed `MAP_FAIRE` really is
only ever enqueued from one place (`enqueue_mapping_backfill`) -- discovery
never auto-chains into it, already true and deliberate. That made the
right insertion point clear: gate the *existing* mapping boundary, not
rewrite `DISCOVER_IDENTIFIERS`/`DISCOVER_CITING_STUDIES` themselves. New
`identity/component.py::compute_study_component` computes the connected
component a study belongs to via BFS over two edge types -- shared
`EntityStudy` links AND discovery-lineage columns -- because neither alone
is sufficient: two independently-seeded siblings sharing an entity have no
lineage edge between them at all, while a freshly-created citing study has
*zero* `EntityStudy` rows until its own discovery task actually completes,
so shared-entity edges alone would miss it. A new self-rescheduling poll
task, `CHECK_COMPONENT_SETTLED`, is the barrier mechanism this flat,
dependency-free task queue doesn't otherwise have: it recomputes the
component, checks for anything still discovering anywhere in it, and
either reschedules itself (fresh idempotency key per poll generation, a
delayed `available_after`) or -- once nothing is left in flight and
membership stopped changing -- runs root determination and finally
enqueues `MAP_FAIRE` for every member. A capped generation count marks a
pathologically stuck component `stalled` rather than polling forever; a
settled component that grows again later (a new citing paper found by the
existing quarterly rediscovery pass) flips back to `PENDING` and starts a
fresh cycle. `enqueue_mapping_backfill` now routes any study with a
shareable entity through this gate instead of enqueueing `MAP_FAIRE`
directly; a study with none never touches this machinery and proceeds
immediately, exactly as before -- satisfying "studies with no shareable
entities proceed immediately, no waiting" precisely.

That gate turned out to be the efficiency layer, not the sole correctness
guarantee, though -- the same design agent's exploration caught a real
fact worth escalating rather than quietly designing around:
`handle_extract_text_facts` (Codex's concurrent work) calls
`map_study_to_faire` *inline*, a second, independent `StandardizedValue`
writer that lives entirely outside the `MAP_FAIRE` task type. That means
gating only `MAP_FAIRE` cannot, by itself, guarantee a shared entity's
broadcast is always root-aware. `exports/faire.py`'s own read-time gate
(`_entity_broadcast_is_authoritative`, checked at export time regardless
of when or how many times mapping ran for a study) is what actually
guarantees correctness -- replacing the previous milestone's blunt "suppress
if linked to more than one study" rule with "show only the *root* study's
broadcast, never a non-root's, never while determination is still pending
or ambiguous." A genuinely new, unconditional (not settle-gated, since
it's a structural fact about entity ownership rather than a cross-study
priority judgment) rule was added alongside it per the user's own explicit
ask: a study that links to shared samples/runs but is home to none of them
did no original data collection, just reanalysis, and never gets a
`projectMetadata` row -- its `Study` row and `EntityStudy` links stay
untouched in the network; `sampleMetadata`/`experimentRunMetadata` already
correctly never emitted rows for entities such a study doesn't home.

**A real bug caught immediately, before it ever reached a live run.**
Adding `Entity.root_study_id` (a second foreign key from `entities` to
`studies`, alongside the existing `study_id`) broke the ORM the moment the
full test suite ran: `AmbiguousForeignKeysError` on the `Study.entities`/
`Entity.study` relationship, since SQLAlchemy could no longer infer which
column to join on. Fixed with an explicit `foreign_keys=` on both sides,
pinning the relationship to `study_id` specifically -- `root_study_id`
stays a plain reference column, never used for ORM traversal.

**Live end-to-end re-run**, same real s42003/PNAS pair already used to
validate the previous milestone, now pushed all the way through the new
pipeline: discovery -> settle-check drain -> root determination ->
`MAP_FAIRE` -> export. The result was genuinely informative, not just a
pass/fail: root determination picked the **PNAS paper (published 2020) as
root**, not s42003 (published 2024, the actual seed paper) -- direct proof
that root is decided by real publication-date evidence, completely
independent of which paper happened to be seeded or discovered first,
exactly the decoupling the user asked for between `Entity.study_id`
("home," an implementation detail) and `root_study_id` (a deliberate
answer). The same run *naturally* exercised the analysis-only exclusion
without it being specifically engineered for the test: PNAS links to all
83 shared samples but is home to none of them, so it correctly produced
zero `projectMetadata` rows, while `sampleMetadata.csv` still correctly
listed all 83 samples exactly once each with pipe-joined
`internal_study_id`; both studies reached `entity_component_status=settled`
with zero ambiguity flags raised.

668 tests pass (18 new): component BFS (both edge types, including the
case lineage-only would miss); root determination (earliest-year wins,
same-year tie flags ambiguous, non-shared entities skip the algorithm
entirely); the settle-check handler (reschedule-while-in-flight, settle-
and-enqueue-MAP_FAIRE, the stalled-generation cap, component reopening);
`exports/faire.py`'s root-aware broadcast gate and the analysis-only
`projectMetadata` exclusion; and `enqueue_mapping_backfill`'s deferral
behavior for studies with shareable entities.

## Mapping expansion: the rest of FAIRe's Environment section

Beyond the 8 BioSample attributes already mapped (elev/samp_collect_device/
samp_size/samp_size_unit/temp/salinity/ph/diss_oxygen), checked the
vendored schema systematically for every other field tagged
`in_subset: Environment` (70 total) and added the remaining 63 --
`chlorophyll`, `turbidity`, `nitrate`/`nitrite`, the `host_*` fields
(for host-associated samples), `tidal_stage`, `wind_speed`, and each of
their `_unit`/method companions -- all via the same generic NCBI BioSample
`Attributes/Attribute` passthrough as everything else in this table, all
`EXACT_LABEL` (BioSample MIxS attribute name == FAIRe field name). A new
regression test (`test_every_faire_environment_field_has_a_sample_level_rule`)
checks this stays true against the vendored schema directly, so a future
schema update that adds a new Environment field gets caught rather than
silently missed.

Checked honestly, not assumed: **re-ran the mapping backfill against the
real 101-study database and confirmed zero new rows** from this batch --
no BioSample record in this corpus has reported any of these 63 attributes
yet (same as the original 8 before them). The rules are correct and
inert until a real source has the attribute.

**PANGAEA `variableMeasured` is now decomposed when present.** The current
101-study database still has no real `adapter:pangaea` facts, but live
PANGAEA/DataCite checks found real marine/eDNA payloads with populated
parameter metadata. In particular, `10.1594/PANGAEA.935870` ("Operational
taxonomic units of deep-sea fishes from environmental DNA...") exposes
`variableMeasured` entries for `DATE/TIME`, latitude/longitude, `DEPTH,
water` (`unitText: "m"`), `Water volume, filtered` (`unitText: "m**3"`),
`Sample ID`, genetics accessions, and OTU columns with WoRMS LSIDs. The
PANGAEA adapter now keeps the original `variableMeasured` blob and also
emits decomposed project-level facts such as `pangaea_variable_name`,
`pangaea_variable_unit`, `pangaea_variable_measurement_technique`, and
defined-term identifiers/URLs with source locators like
`pangaea.jsonld.variableMeasured[1].unitText`.

Conservative boundary: these are parameter/column definitions, not the
per-sample measured values themselves. They should inform source discovery,
review, and later tabular-data retrieval, but they are not automatically
mapped as present FAIRe sample values until the adapter also fetches and
parses the PANGAEA textfile rows.

## Supplementary-material and structured-table retrieval layer

Until now, the pipeline only ever read a paper's main JATS full-text body --
`<supplementary-material>` references (linked tables, spreadsheets, or
externally-hosted repository files) were visible in the XML but never
retrieved or parsed, so real per-sample/environmental data sitting in a
paper's own supplementary tables was invisible. Three task types close
this gap, run in the recommended order **structured APIs -> main-paper LLM
pass -> retrieve/parse/prepare supplements -> supplement LLM pass over
still-missing fields -> validation**:

- `DISCOVER_SUPPLEMENTS` (`fair-ocean enqueue-supplement-discovery-backfill`)
  parses `<supplementary-material>` tags out of the already-cached article
  XML (no new network call), deduplicating by filename (real articles
  repeat the same list verbatim under two different sections). Creates one
  `Source` row (`source_type=supplement`, created once with a terminal
  status, same convention as every other Source-creation call site) plus
  one companion `DataAsset` row per distinct file. Externally-hosted
  repository supplements (an `<ext-link>` DOI rather than a Europe-PMC-
  hosted `<media>` element) are surfaced as `RelatedIdentifier`s, so the
  *existing* `DISCOVER_IDENTIFIERS` -> structured-adapter pipeline picks
  them up automatically -- no separate fetch/parse path needed for that
  case.
- `RETRIEVE_SUPPLEMENTS` (`fair-ocean enqueue-supplement-retrieval-backfill`)
  fetches Europe PMC's single `{pmcid}/supplementaryFiles` zip bundle (there
  is no per-file fetch API) once the summed *already-known* size (from the
  XML's own `<?size?>` values, never a guess) is under `supplements.
  max_bundle_bytes` -- otherwise every pending file is marked
  `not_accessible` and the bundle is never downloaded. Each member is then
  routed by extension: CSV/TSV/XLSX/XLS go through deterministic table
  parsers (`sources/supplement_parsing.py`), JSON/XML through a structured
  walk, and TXT/MD/PDF into persisted `prepared_source_texts` rows. This
  stage never invokes an LLM. Anything else (DOCX, nested archives) is
  inventoried only. A member over
  `supplements.max_member_bytes` stays available-but-unparsed rather than
  read into memory. One file's parse failure is caught and recorded on
  that file's own `DataAsset` row -- it never aborts the rest of the task.
- `EXTRACT_SUPPLEMENT_TEXT_FACTS`
  (`fair-ocean enqueue-supplement-text-extraction-backfill`) is the separate,
  explicit model stage. It is off by default and queues nothing until
  `FAIR_OCEAN_SUPPLEMENT_LLM_ENABLED=true` (or the matching local YAML
  setting) is set. By default, only studies with a completed main-paper
  pass are eligible. The handler rebuilds FAIRe mapping from structured +
  paper facts, removes every already-present field from the supplement
  prompt, runs the focused/recall extractor over prepared text, then
  remaps after each document so later supplements ask an even smaller
  question. The exact prepared text remains in `prepared_source_texts`;
  each retained `RawFact.evidence_quote` is copied from its model-selected
  segment rather than generated by the model.

**Discovery (not retrieval) also runs inline during a DOI-driven
`DISCOVER_IDENTIFIERS` pass, for free.** `_discover_supplements_from_fulltext`
(`workflow/handlers.py`) reuses the exact same open full text already
fetched for `_discover_identifiers_from_fulltext`'s regex-based repository-
accession mining (a cache hit, not a second real request) and calls the
same `discover_supplements_for_study` logic the standalone
`DISCOVER_SUPPLEMENTS` backfill uses -- so a normal DOI-first discovery run
against the ~3,600-paper seed list also surfaces supplementary-material
references without any extra step. It's gated `if doi:`, matching the
existing identifier-mining gate exactly: a study discovered via
BioProject/ENA/dataset-DOI *first* (no DOI yet at that task's run time)
doesn't get this inline pass until a later `DISCOVER_IDENTIFIERS` re-run
happens with a DOI present (e.g. after a Stage 2 merge attaches one, or via
`quarterly_full_rediscovery`) -- same as identifier mining, not a new gap.
`RETRIEVE_SUPPLEMENTS` (the actual zip download + deterministic parsing/
text preparation) deliberately stays a separate, explicitly-enqueued
backfill. Supplement inference has its own second opt-in, so preparing
files never silently spends model time.

**The user's 6 named states (referenced/available/retrieved/parsed/
inaccessible/parse_failed) live entirely on the companion `DataAsset`
row's existing `access_status`/`inspection_level` columns** (+
`description` disambiguating `retrieved` vs `parse_failed`, which share
the same status/level pair). Prepared textual content lives in the new
`prepared_source_texts` table; `DataAsset.inspection_level=lightweight`
means text is ready but not yet model-inspected, and `full` means the
opt-in pass completed. `Source`
itself is set once at creation and never mutated again, matching every
other existing call site (`validation_handlers.py` already treats "a
Source row exists" as a proxy for "inspected"; mutating it per retrieval
attempt would have silently broken that).

Table/JSON/XML parsing only ever alias-matches a small, curated set of
known headers (`collection_date`, `depth`, `temp`, `salinity`, `ph`,
`diss_oxygen`, `samp_collect_device`, `samp_size`, `latitude`/`longitude`,
...) onto this pipeline's own existing native names -- never one `RawFact`
per raw, uncontrolled column header. A recognized identifier-like column
(`sample_id`, `run_accession`, ...) binds that row's facts to a
SAMPLE/SEQUENCING_RUN `Entity`; otherwise facts bind at the study level.
Unrecognized columns are reported (name + count) in the `DataAsset`
description for a human to see, never turned into facts. `source_locator`
carries full cell-level provenance, e.g.
`supplement.Supplementary_Table_2.xlsx#sample_metadata!H42` -- the exact
same provenance-aware shape as any API-derived `raw_facts` row. Because
supplement-derived facts are ordinary `RawFactCandidate`s flowing through
the normal `mapping/rules.py` path. `present_faire_fields_for_study()`
recomputes the supplement exclusion set after the paper and after each
prepared document, including explicit paper facts that remain flagged for
human review.

**Missingness now distinguishes "referenced but not yet inspected" from
"actually absent."** `populate_missingness_for_study` and
`populate_faire_missingness_for_study` previously decided
`not_found_in_inspected_sources` vs. `relevant_source_not_inspected` using
only "does any `Source` row exist" -- never checking whether a referenced
supplement had actually been parsed. A study with a supplement `DataAsset`
still below `inspection_level=full` now gets `source_not_accessible`
(if `access_status=not_accessible`) or `relevant_source_not_inspected`
instead of wrongly reporting `not_found_in_inspected_sources` for a field
the supplement might still resolve.

**Live validation against a real article already in the database**
(`STUDY-c47dd2ce9a62` / PMC7469538) surfaced two real bugs, both fixed
before this was called done:

1. The real bundle summed to ~32.5MB (4 small XLSX tables + one unrelated
   32MB dataset) -- over the initial `max_bundle_bytes` default of 25MB,
   which would have blocked the whole bundle (including the small, useful
   tables) from ever being downloaded. Raised the default to 50MB so a
   bundle shaped like this real one downloads fine, while
   `max_member_bytes` (10MB, unchanged) still correctly keeps the 32MB
   member itself inventory-only rather than parsed.
2. Real supplementary XLSX exports commonly carry a caption/title row
   (`"Supplementary Table S5. Accession Number."`) above the actual header
   row. The parser originally read row 1 unconditionally as the header,
   silently misaligning every column against the real header+data one row
   down. Fixed by skipping leading rows with fewer than 2 non-blank cells
   before selecting the header row (a real header names >=2 columns; a
   caption has exactly one).

After both fixes, the run against PMC7469538 correctly discovered 9
distinct files (deduped from 18 raw tags across two repeated sections),
marked the 4 DOCX files `retrieved`/not-parsed (unsupported type), parsed
all 4 XLSX files without error, and marked the 32MB dataset
`available`/not-retrieved (over the per-member cap) -- exactly the shape
the design predicted. Zero facts were extracted from this particular
paper's XLSX tables, and that's the honest, correct outcome: inspection
showed `Table_5.xlsx` is a taxa-by-sample OTU abundance matrix (samples as
*columns*, not the per-row-per-sample shape this parser targets) and
`Table_6`-`8.xlsx` are statistical comparison tables -- neither shape
contains alias-matchable per-sample metadata, so reporting "0 recognized
columns" rather than fabricating or misaligning values is the correct
behavior, not a bug.

## Shared resolve_or_create_study(): confidence-tiered merging across both discovery paths

Studies get discovered two ways -- DOI-first (Crossref/Europe PMC/OpenAlex,
then mining related repository accessions) and accession-first (a study
seeded with just a BioProject/ENA accession, publication discovered later
or never). A paper can be tied to many projects/samples and vice versa, so
whenever either path discovers an identifier that already belongs to a
*different* Study, a merge decision has to be made. Previously both paths
funneled this into the same call (`merge_study_into`) unconditionally --
any identifier match, regardless of evidence quality, triggered an
immediate, irreversible merge. `identity/resolution.py`'s
`resolve_or_create_study()` is now the single function both paths call
(via `workflow/handlers.py`'s `_apply_related_identifiers`, now a thin
wrapper), so DOI-led and accession-led discovery can never silently
diverge in how they reconcile ambiguous matches.

**Evidence-tier gating** reuses `SupportType` as a shared confidence
vocabulary end to end (not a second parallel one) -- added as a new
`confidence` field on `RelatedIdentifier` (`sources/base.py`), defaulting
to tier 1 for every existing adapter (`ncbi.py`, `ena.py`, `pangaea.py`,
`bcodmo.py`, `datacite.py`, `obis.py`, `gbif.py`, `europe_pmc.py`,
`openalex.py`), all of which already surface genuinely structured-API
evidence with zero code changes needed:

- **Tier 1** (`STRUCTURED_SOURCE`) -- a structured API relation field
  (DataCite's own `relatedIdentifiers`, Europe PMC's PMID/PMCID). Safe to
  auto-link, no consistency check, matching every adapter's prior
  unconditional-merge behavior exactly.
- **Tier 2** (`DETERMINISTICALLY_DERIVED`) -- a regex-matched accession
  pulled from prose (`discovery/text_identifiers.py`), now tagged as tier 2
  and gated behind a brand new `verify_deterministic_identifier()` step
  that confirms the hit actually resolves against that identifier type's
  own source API (BioProject -> ncbi_bioproject/ena, dataset DOI ->
  pangaea/bcodmo/datacite, filtered by the DOI's own prefix) *before* it's
  trusted at all -- an unverifiable hit is silently dropped and logged, not
  treated as an error. Once verified, it's still consistency-checked
  (below) before merging into an existing Study.
- **Tier 3** (`INFERRED`) -- an LLM-extracted prose claim ("we reused data
  from..."). Plumbing only -- no extractor producing these exists yet.
  Always consistency-checked and never sufficient to merge alone even when
  consistent; it always lands a `CandidateMatch` for human review either way.

**Deliberately not `SupportType.EXPLICIT`** for tier 1 -- that value
already means something different and established in this codebase (an
LLM-extracted `RawFact` with a verbatim evidence quote, see
`extraction/text.py` and the two validators that branch on it) and reusing
it here would silently collide with that meaning.

**The consistency check** (`identity/consistency.py`) compares a
pre-existing Study's recorded sampling date range and geographic extent
against the newly-discovered evidence's own, mirroring
`validation/logical.py`'s "unparseable is NOT_ASSESSED, never a false
conflict" principle. Concrete, documented thresholds (same "generously
conservative, round-number" style as `DEEPEST_OCEAN_POINT_METERS`):
±365 days of adjacency around an existing date range (a single field
season/cruise commonly straddles a calendar-year boundary), a 2-degree
geographic margin (~220km, covers normal within-region sampling spread), and
a default-conservative rule for the no-signal-either-way case -- if new
evidence has no parseable date at all and the existing study's registered
span already exceeds 2 years, that's treated as inconsistent by default
(a registered span that wide looks more like an umbrella accession
accumulating multiple deposits than one paper's actual field work).
**Deliberately never reads assay/marker-gene/primer/target-gene fields as a
signal** -- FAIRe's own data model treats a project running multiple
assays (16S + 18S on the same samples) as normal single-project structure,
not evidence of two investigations.

**The hard new case**: when a match fails the consistency check (or is
tier 3), the newly-discovered evidence must NOT be merged in --
`resolve_or_create_study` splits off a brand-new sibling `Study`, moves
just *that one Source's* `RawFact` rows onto it (unambiguous, since
`RawFact.source_id` already identifies exactly which facts are new),
links the ambiguous identifier via a new `RelationshipType.SHARES_ACCESSION_WITH`
value, and drops a `PENDING` row in `candidate_matches` -- a table that's
existed since Milestone 1 but had never been written to by any code path
until now. **Entity ownership rule** for the genuinely hard sub-problem
(`Entity` has no `source_id` of its own, only study-scoped): an `Entity`
only moves to the sibling if *100% of its `RawFact`s* came from this one
source; if any fact on it came from a different source under the same
original study, it stays put and relies on the flag for a human to sort
out -- a false negative (stays behind) is recoverable later, a false
positive (wrongly split) would silently scatter one physical sample's
facts across two studies, which is strictly worse.

**Schema**: additive only, per an explicit design decision -- `sources.study_id`
stays exactly as it was (every existing adapter/handler/test keeps working
unchanged), and a new `study_sources(study_id, source_id, relationship_type,
confidence)` join table is added *alongside* it, backfilled once via a new
migration (`relationship_type='is_home_of'`, `confidence='structured_source'`
for every pre-existing `Source` row -- genuinely tier-1 provenance, not a
guess, since every existing Source came from a directly-resolved adapter
fetch or a structurally-parsed reference). Going forward, a single new
choke-point function, `identity/source_linking.py`'s `create_source()`,
writes both `sources.study_id` and the `study_sources` row atomically for
every new Source -- all four existing Source-creation call sites
(`workflow/handlers.py`'s `_persist_source_and_facts` and its
`europe_pmc_fulltext` Source row, `workflow/supplement_handlers.py`'s
`discover_supplements_for_study`, `workflow/refresh_handlers.py`'s
`_persist_refreshed_source_and_facts`) route through it now. Full removal
of `sources.study_id` remains a future organic cleanup, not part of this
work.

**Verified against the real 101-study database**: applied the migration
(385 pre-existing `sources` rows, each backfilled to exactly one
`study_sources` row, confirmed via direct query), then re-ran
`DISCOVER_IDENTIFIERS` for every one of the 101 real studies through the
new `resolve_or_create_study`-wired pipeline -- 101/101 completed with zero
errors, 162 new `Source` rows were created along the way (bringing the
total to 547) and `study_sources` stayed exactly 1:1 with `sources`
throughout (547/547 after the run, confirming `create_source`'s choke
point held for every new row), and zero `CandidateMatch` rows were created
(expected: a well-behaved, already-correctly-resolved real corpus
shouldn't surface any genuine ambiguity).

**Explicit non-goals**: building brand-new keyword-search discovery
adapters (NCBI BioProject text search, OBIS/GBIF/BCO-DMO/PANGAEA search --
confirmed unbuilt, `DiscoveryConfig.keyword_search_enabled` remains an
unused stub) is a separate follow-up; this task wired the shared function
into the two entry points that already exist (DOI-first and the existing
accession-first path). Two of the confidence-tiering design's own tier-1
examples turned out to be real, pre-existing gaps once checked against the
live code -- NCBI's BioProject adapter doesn't implement PMID->BioProject
elink, and BCO-DMO's adapter doesn't mine its own `dcterms:bibliographicCitation`
field for a linkable identifier -- both flagged as natural follow-ups, not
fixed here.

## Loading your own seed list

Put your file at `data/seeds/studies.csv` (or `.jsonl`) -- copy
`data/seeds/studies_template.csv` for the exact column layout:

```text
seed_id,title,doi,pmid,pmcid,bioproject_accession,ena_study_accession,dataset_id,repository,url,notes
```

Not every column needs a value; leave cells blank. Then:

```bash
fair-ocean ingest-seeds data/seeds/studies.csv
fair-ocean enqueue-seed-backfill
fair-ocean worker --until-empty
```

## Database

SQLite by default (`data/fair_ocean.db`, gitignored). To move to PostgreSQL:

1. Install with the Postgres extra:

   ```bash
   pip install -e ".[dev,postgres]"
   ```

2. Create a local database. With Homebrew PostgreSQL on macOS:

   ```bash
   brew install postgresql@16
   brew services start postgresql@16
   createdb fair_ocean
   ```

   Any existing local/server PostgreSQL instance is fine; the app only
   needs a normal database URL.

3. Set the database URL in `.env`:

   ```bash
   FAIR_OCEAN_DATABASE_URL=postgresql+psycopg2://localhost/fair_ocean
   ```

   For a username/password database, use:

   ```bash
   FAIR_OCEAN_DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/fair_ocean
   ```

4. Run migrations:

   ```bash
   alembic upgrade head
   ```

Use `alembic upgrade head` rather than `fair-ocean init-db` for
PostgreSQL. Migrations are the source of truth for shared/server
databases; `init-db` is only a local SQLite/dev shortcut that creates
tables directly from the current ORM models.

The ORM uses SQLAlchemy's generic JSON type with a PostgreSQL JSONB
variant, so SQLite remains easy locally while PostgreSQL stores document
columns such as `tasks.payload`, `raw_facts.confidence_metadata`,
`workflow_runs.sources_queried`, and
`standardized_values.sources_inspected` as JSONB. The migration
`5b9e1d3c7a20_postgres_jsonb_and_worker_indexes.py` also adds composite
indexes for task claiming and high-volume study/fact/schema lookups.

```bash
alembic upgrade head        # apply migrations
alembic revision --autogenerate -m "description"   # after changing models.py
```

For multi-worker mode, use PostgreSQL and start more than one worker
process with distinct IDs:

```bash
fair-ocean worker --until-empty --worker-id worker-a
fair-ocean worker --until-empty --worker-id worker-b
```

The task queue's `claim_next_task` branches on dialect: PostgreSQL gets
`SELECT ... FOR UPDATE SKIP LOCKED`, so workers do not block each other on
the same queue row. SQLite gets a plain claim query and remains
single-worker-safe only because SQLite has no row-level locking.

## Resuming interrupted work

Every CLI command that touches the DB commits per logical unit (the worker
commits after each task, not at the end of a batch), and the task queue is
the resumability mechanism:

- Re-run `ingest-seeds` any time -- exact-identifier matches merge rather
  than duplicate.
- Re-run `enqueue-seed-backfill` any time -- already-queued studies are not
  re-queued (idempotency key = task type + study id).
- If a worker is killed mid-task, its claim is orphaned. Run
  `fair-ocean release-stale-claims --stale-after-minutes 30` to return those
  tasks to `retry_pending` (or `manual_review_required` if attempts are
  exhausted).

## Inspecting and retrying failed tasks

```bash
fair-ocean status                 # counts by status
fair-ocean status --json          # machine-readable
```

There's no dedicated "list failed tasks" command yet (Milestone 1 didn't
need one for the seed-ingestion path) -- query the `tasks` table directly
for now, e.g. `sqlite3 data/fair_ocean.db "select task_id, task_type, last_error from tasks where status='manual_review_required'"`.

## Running a local open-weight model endpoint

1. Start whatever inference server you're using (Ollama, vLLM, TGI's
   OpenAI-compatible router, an institutional gateway, ...) and note its
   base URL and the exact model name it expects.
2. In `.env`, set `LOCAL_LLM_BASE_URL` (e.g. `http://localhost:11434/v1`)
   and `LOCAL_LLM_MODEL` to that model name -- there's no default to fall
   back on; `build_llm_backend` raises clearly if `llm.model` is still the
   placeholder value.
3. In `config/default.yaml`, set `llm.enabled: true` and
   `llm.provider: openai_compatible`.
4. `fair-ocean enqueue-text-extraction-backfill` then `fair-ocean worker
   --until-empty` will now actually call your model for any study with a
   known PMCID (discovered during a prior `DISCOVER_IDENTIFIERS` run).

"OpenAI-compatible" here names the HTTP request/response shape only (POST
`{base_url}/chat/completions`) -- nothing in this pipeline talks to an
OpenAI-operated host, needs an OpenAI account, or sends data anywhere
except your configured `base_url`.

## Optional independent LLM evidence verifier

Text extraction and evidence verification are deliberately decoupled. The
primary extractor can be a faster model such as qwen3, while
`VALIDATE_EVIDENCE` can optionally ask a separate verifier model whether a
stored `evidence_quote` actually supports the extracted
`fact_type_candidate` and `raw_value`.

For example, to use Granite through Ollama as the verifier while keeping a
different extractor:

```bash
LOCAL_LLM_VERIFIER_ENABLED=true
LOCAL_LLM_VERIFIER_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_VERIFIER_MODEL=granite3.3:8b
LOCAL_LLM_VERIFIER_MAX_OUTPUT_TOKENS=512
```

The verifier writes one `ValidationResult` per checked LLM text fact, with
`validator_name` like `llm_evidence_support:granite3.3:8b` and status
`supported`, `unsupported`, or `not_assessed`. The deterministic evidence
checks still run first: facts with missing evidence quotes are flagged
without asking the verifier.

## Model benchmarking

This project doesn't pick a model -- it gives you a harness to compare
candidates and report metrics, since the actual choice is yours to make
later (validated first against Claude-curated cases, eventually against
your own manual review).

1. Start one or more candidate model servers.
2. Fill in `config/benchmark_models.yaml` with a `label`/`base_url`/`model`
   entry per candidate (as many as you want compared side by side) --
   every field defaults to a `REPLACE_WITH_...` placeholder that the
   command refuses to run against.
3. Add gold-standard cases under `data/benchmark/gold/*.json`. Each file:

   ```json
   {
     "case_id": "unique-id",
     "source_text": "the exact paper section text",
     "section_title": "PCR amplification",
     "expected_facts": [
       {"fact_type_candidate": "annealing_temperature", "raw_value": "57C", "evidence_quote": "verbatim quote from source_text"}
     ],
     "curated_by": "claude"
   }
   ```

   `section_title` defaults to `"Methods"` if omitted. There's no
   `instructions` field -- every case is run through the exact same
   `extraction.text.build_prompt` production code uses, so a gold case
   can't silently test a different prompt than the real pipeline sends
   (see "Milestone 8" below for why that used to be possible and isn't
   anymore). `data/benchmark/gold/example-001.json` is a fictional worked
   example (clearly marked, not a real paper); five more
   (`pcr-assay-detail-001`, `controls-replicates-001`,
   `qpcr-standard-curve-001`, `sequencing-library-001`,
   `bioinformatics-taxonomy-001`) exercise the FAIRe-aware field groups
   (`extraction/faire_fields.py`) specifically. Add real, Claude-curated
   (or later, manually-reviewed) cases before drawing conclusions meant
   for a paper.

4. Run it:

   ```bash
   fair-ocean benchmark-models
   ```

This writes `data/exports/benchmark/summary.csv` (one row per candidate --
`json_validity_rate`, `evidence_verification_rate`, `precision`, `recall`,
`f1`, `mean/median/p95_latency_seconds`, `errors` -- paste straight into a
results table) and `detail.json` (every case's raw/verified facts, for
spot-checking disagreements). Matching against `expected_facts` is exact
(fact type + normalized value), and evidence verification is the same
deterministic quote-in-source check used in the real extraction pipeline
-- a model that hallucinates a quote gets caught the same way a
hallucinated extraction would in production.

## Testing

```bash
pytest
```

289 tests, all offline (in-memory SQLite, mocked HTTP transport, mock LLM
backend, no live network, no live inference server). Covers identifier
normalization/validation, seed loading (CSV + JSONL, exact-match merge,
idempotent re-ingestion), task queue idempotency/claiming/retry/stale-claim
recovery, config loading/precedence, Stage 1+2 deduplication (including the
identifier-collision-merge path), adapter response parsing (fixture-based,
no network, Milestone 2's JSON adapters and Milestone 3's NCBI-XML/ENA-JSON
adapters), the HTTP layer (rate limiting/retry/404/caching, via
`httpx.MockTransport`), NCBI's real esearch/elink/efetch orchestration
(mocked transport, real XML parsing), the `DISCOVER_IDENTIFIERS` handler's
orchestration for both the publication and repository paths (per-sample
Entity creation, cross-adapter identifier-collision dedup, retry
idempotency), and Milestone 4's `LLMBackend` abstraction (JSON-retry logic,
the OpenAI-compatible wire protocol via `httpx.MockTransport` -- including
that it never sends credentials unless configured and never assumes a
default model), evidence-quote verification (exact quotes pass, paraphrases
and fabrications don't), deterministic section selection (including that
nested JATS `<sec>` elements don't get their text double-counted), the
benchmark harness's metrics (a hallucinated fact is caught by evidence
verification and scored accordingly), and the `EXTRACT_TEXT_FACTS`
handler's orchestration (fabricated-quote rejection, retry idempotency, and
that a disabled LLM fails loudly rather than silently no-oping). Milestone
5 adds: data-asset inventory from real-shaped ENA raw_facts (idempotent
retry, skipping runs with no listed file), logical validators against
every real coordinate/depth/date/primer format observed live (a primer
*name* is NOT_ASSESSED, never flagged as an invalid sequence), evidence
and cross-source validators (including the real trailing-period and
wrong-OpenAlex-title regression tests above), missingness population, and
the validation task handlers' idempotency through the real task queue.

None of this hits Crossref/Europe PMC/OpenAlex/NCBI/ENA/an inference server
live; those were exercised manually against real data (the 100-DOI seed
set, a real BioProject, and a real Europe PMC full-text fetch with a mock
model response -- see above), not in CI.

## Server deployment (Linux, always-on)

On a server: install as above, point `FAIR_OCEAN_DATABASE_URL` at
PostgreSQL, run `alembic upgrade head`, and run `fair-ocean weekly-update`
from cron/systemd timer to enqueue refresh/retry work. Run one or more
`fair-ocean worker --until-empty --worker-id <unique-id>` processes under
whatever process supervisor you already use (systemd, supervisord, tmux).
Multiple workers require PostgreSQL; SQLite remains single-worker local
development mode.

## Security, copyright, and data-handling constraints (apply from Milestone 1 onward)

- No large sequence files (FASTQ/BAM/CRAM) are ever downloaded -- only
  accessions/checksums/metadata get recorded (`data_assets` table).
- No paywalled or access-controlled content is ever fetched.
- Secrets only ever come from environment variables / `.env` (see
  `.env.example`); nothing in `config/*.yaml` should ever hold a real key.
- LLM facts require an evidence quote or they don't get created (Milestone
  4 enforces this at the extraction-schema level; the `raw_facts.evidence_quote`
  column already exists to make it structurally impossible to skip).

## How evidence and missingness are represented

Every `raw_facts` row carries `support_type` (`explicit`,
`structured_source`, `deterministically_derived`, `normalized`, `inferred`,
`conflicting`, `not_found`) and, for text-derived facts, an
`evidence_quote`. Missing information is never just a blank cell --
`standardized_values.missingness_status` uses the controlled vocabulary
from the brief (`not_found_in_inspected_sources`, `source_not_accessible`,
`relevant_source_not_inspected`, `not_applicable`, `conflicting`,
`mapping_unresolved`, `reported_ambiguously`, `manual_review_required`).
There's no separate `missingness` table (folded away, see "Schema
simplification" below) -- every (study, entity, target_field) this
pipeline checks for gets exactly one `standardized_values` row, whether or
not a value was actually found.

## Assumptions and placeholders (read before relying on this in Milestone 2+)

- **ID format**: business-key IDs (`study_id`, `fact_id`, `task_id`, ...) are
  `PREFIX-<12 hex chars>` (uuid4-based), not the brief's illustrative
  `STUDY-000001` sequential style. This was a deliberate choice for
  collision-free concurrent-worker inserts (a shared counter would
  serialize writes across every worker). See
  `src/fair_ocean_agent/database/ids.py` if strictly sequential IDs turn out
  to matter later -- nothing else depends on the format.
- **FAIRe is vendored and mapped (Milestone 6); BeBOP/MIxS/Darwin Core are
  not.** `schemas/faire/` has the real v1.0.2 `schema.yaml`/`classes.yaml`/
  `enums.yaml`. BeBOP/MIOP mapping is blocked on the `miop` schema repo
  (see "Milestone 6 validation" above and `mapping/bebop.py`); MIxS/Darwin
  Core mapping was never in scope for this milestone and `config/
  models.yaml` still has placeholder version strings for both.
- **FAIRe mapping coverage is real but partial by design**, not yet-total-
  but-eventually-complete. This note predates Milestone 8: at the time it
  was written, extraction was still fully open-vocabulary, so every
  LLM-extracted fact could only ever map to FAIRe's free-text fallback
  fields (flagged `review_required`), never an atomic one. Milestone 8
  (corrected in v3, see above) gives the extractor a structured checklist
  of atomic concepts with standard-agnostic `fact_type_candidate` names
  (`pcr_conditions`, `pcr_cycle_count`, `annealing_temperature`, ...) plus
  an optional `candidate_standard_fields` hint suggesting the matching
  FAIRe field (e.g. `annealingTemp`) -- closing the extraction-side half of
  this gap. Whether `mapping/rules.py` actually has rules consuming these
  new native names (with or without their hints) yet is a separate,
  mapping-owned question this note doesn't track -- check `mapping/rules.py`
  directly rather than assuming from this paragraph. `assay_name` (a
  FAIRe-mandatory join key on `ampData`/`stdData`/`experimentRunMetadata`/
  `eLowQuantData`) has no data source at all in this pipeline yet, since
  no adapter models a PCR assay as its own Entity -- those four tables
  export header-only.
- **Only Crossref/Europe PMC/OpenAlex/NCBI-BioProject/NCBI-BioSample/ENA
  exist**; OBIS/GBIF/BCO-DMO/PANGAEA/DataCite (`config/sources.yaml` has
  placeholder entries, `enabled: false`) are Milestone 4+. A study with
  none of DOI/BioProject-accession/ENA-accession raises `NotImplementedError`
  and eventually lands in `manual_review_required`, which is correct, not
  a bug.
- **NCBI SRA has no dedicated adapter.** Run-level data (library
  strategy/platform/file accessions) is served via the `ena` adapter's
  `read_run` query instead -- ENA mirrors the same INSDC-shared records
  via much cleaner JSON than NCBI's SRA XML (`EXPERIMENT_PACKAGE_SET`).
  `ncbi_sra` stays `enabled: false` in config with a comment explaining
  why, rather than being silently absent.
- **`MAX_SAMPLES_PER_PROJECT` (300) and `MAX_RUNS_PER_STUDY` (500) cap
  per-task work** against unusually large BioProjects/ENA studies (some
  eDNA time series run into the thousands of samples). Truncation is
  logged as a warning and recorded as a `raw_facts` coverage note, never
  silent -- but it does mean a huge project's *complete* sample/run list
  needs a dedicated backfill pass (Milestone 7), not just one
  `DISCOVER_IDENTIFIERS` task.
- **Adapter rate limiting is per-worker-process, not global.** Each adapter
  instance tracks its own last-request time and is cached for the life of
  one `fair-ocean worker` invocation (`workflow/handlers.py`'s
  `_build_enabled_adapters` cache, cleared by `reset_adapter_cache()` at
  shutdown). `ncbi_bioproject` and `ncbi_biosample` share one `RateLimitedClient`
  instance (not one each) since both hit `eutils.ncbi.nlm.nih.gov`, which
  enforces its per-IP limit across all eutils calls combined -- two
  independent throttles would let their combined rate exceed NCBI's real
  limit. Two worker *processes* running concurrently would still each
  enforce their own limits independently, not jointly -- fine for the
  single-worker SQLite mode this is designed for; would need a shared
  limiter (e.g. Redis-backed) for true multi-worker Postgres deployments.
- **`DISCOVER_IDENTIFIERS` is idempotent per (study, adapter, queried
  identifier), not per field.** Retrying a task skips re-creating a
  `Source`/`RawFact`/`Entity` set already recorded for that combination,
  but always re-applies that adapter's own bibliographic fields
  (authors/journal/publication_year/fulltext_available, straight onto its
  own `Source` row -- see "Schema simplification" below) and repository-
  adapter related-identifiers from a fresh (cache-hit, so free)
  `fetch_record()` call, so a retry that only got partway through still ends
  up complete. Related-identifier existence is checked via a live DB query
  every time (not a cached ORM collection) specifically so two adapters
  discovering the same identifier in one task -- which happens routinely
  between ncbi_biosample and ena -- don't collide. Re-running
  `DISCOVER_IDENTIFIERS` deliberately (not as a retry) to pick up upstream
  corrections is not yet distinguished from a retry -- there's no "force
  refresh" flag; that's a Milestone 7 (weekly update / staleness) concern.
- **Worker is single-process-in-a-loop**, not a daemon/supervisor. Running
  it repeatedly (cron, a shell loop, or a supervisor) is left to the
  deployment, per Milestone 7.
- **`release-stale-claims` is manual**, not automatic -- no scheduler exists
  yet to run it periodically.
- **The contact email** is real (`FAIR_OCEAN_CONTACT_EMAIL` in `.env`, not
  committed) since Milestone 2 makes live calls to Crossref/Europe
  PMC/OpenAlex; `.env.example`'s placeholder is only for a fresh clone.
- **Text extraction only sources full text from Europe PMC**, and only
  when Europe PMC itself reports it as open-access (`inEPMC: Y`) --
  section 1's "no paywalled or access-controlled source" constraint is
  enforced by only ever calling Europe PMC's `fullTextXML` endpoint, never
  a publisher site.
- **`enqueue-text-extraction-backfill` is separate and opt-in**, not
  auto-chained from `DISCOVER_IDENTIFIERS` finding a PMCID. With
  `llm.enabled: false` (the default), nothing about running the seed
  pipeline suddenly creates `EXTRACT_TEXT_FACTS` tasks that are guaranteed
  to fail against a disabled backend -- you enable text extraction
  deliberately once a real model endpoint is configured.
- **Section selection is a fixed-pattern, not LLM-assisted, pass.** Section
  10 allows an LLM fallback "only when necessary"; that fallback isn't
  built. Validated live against all 41 real PMCIDs in the seed set (see
  "Milestone 4 validation" below): 0 misses out of 38 full-text papers
  after fixing one real gap ("Experimental Procedures" headings). If a
  future paper's headings still don't match the pattern list in
  `extraction/sections.py`, no sections get selected and the task no-ops
  rather than sending the whole paper -- worth re-running that same
  pressure test periodically as the seed corpus grows, rather than
  assuming today's 100% holds forever.
- **The benchmark harness's fact-matching forgives formatting, not
  content.** `fact_type_candidate` is compared underscore/hyphen/space/
  case-insensitively (`"collection_date"` == `"collection date"`), and
  `raw_value` is compared as an exact normalized string *or* as the same
  calendar date if both sides parse as a date (`"14 March 2021"` ==
  `"2021-03-14"`) -- unifying date formats for real is a later pipeline
  stage (mapping/standardization); at the raw-extraction benchmark stage,
  a model shouldn't be marked wrong just for using a different date
  format. Non-date values (kit names, primer sequences, ...) still require
  exact normalized-string equality -- no substring/fuzzy matching, since
  that would inflate agreement and defeat the point of comparing models.
  Date parsing uses a fixed sentinel anchor (`datetime(1, 1, 1)`), not
  today's date, so a partial date like `"March 2021"` resolves
  deterministically and only matches an equally-precise gold value, not
  today-dependent noise. See `llm/benchmark.py`'s `_values_equivalent` and
  `_normalize_label` docstrings, and
  `tests/unit/test_llm_benchmark.py::test_qwen_real_world_case_now_scores_correctly`
  for the real qwen2.5:3b result (precision/recall 0.00 → 0.60/0.75) that
  motivated this.
- **`data/benchmark/gold/example-001.json` is fictional**, not a real
  paper -- replace it (or add alongside it) with real Claude-curated cases
  before drawing conclusions from a benchmark run. Nothing prevents
  running the harness against the example case; it just isn't evidence of
  anything about real models on real papers.
- **Data-asset inventory only covers ENA sequencing runs.** Other asset
  types from section 13's list (ASV/OTU tables, sample metadata tables,
  code repositories, protocol files, ...) need either a dedicated adapter
  (OBIS/GBIF/BCO-DMO/PANGAEA, Milestone 6+) or LLM-based classification of
  Data Availability Statement text, neither built yet.
- **Logical validation only covers a handful of fact types**
  (coordinates, depth, collection_date-vs-publication-year, primer
  sequences, identifier formats) -- matched by exact/substring
  `fact_type_candidate` name, not a schema. Extending this to every
  MIxS/FAIRe field is a Milestone 6 concern (once there's a standards
  mapping to validate against).
- **Cross-source comparison only checks `title`.** It's the one concept
  named identically by all three publication adapters
  (`fact_type_candidate="title"` in all of crossref/europe_pmc/openalex);
  journal/venue name differs per adapter ("container-title" vs
  "journalTitle" vs none for OpenAlex) and harmonizing those is a mapping
  concern (Milestone 6), not validation.
- **Missingness is tracked at two levels**: a small, pre-FAIRe
  `core_sampling_metadata` check (`collection_date`, `lat_lon`, `depth`,
  this pipeline's own vocabulary) from Milestone 5, and a full
  unconditionally-Mandatory-FAIRe-field check (`target_schema="faire"`)
  from the Milestone 6 follow-up -- see `validation/faire_completeness.py`
  for why conditionally-Mandatory fields are excluded rather than guessed
  at. Both live as `standardized_values` rows (see "Schema simplification"
  below), not a separate table.
- **`VALIDATE_EVIDENCE` only persists failures, not passing checks** (a
  study can have thousands of facts; a passing fact's evidence bookkeeping
  is already visible on the `raw_facts` row itself). `VALIDATE_CROSS_SOURCE`
  and `VALIDATE_LOGIC`/accession-format checks persist every outcome
  including confirmations, since those are one-or-few rows per study, not
  one per fact -- no bloat concern, and a confirmation is itself useful
  audit evidence.
- **No mirror-collapsing logic exists yet for cross-source comparison**
  (section 16: "do not count mirrored NCBI and ENA records as fully
  independent evidence") -- not needed yet, since this pipeline doesn't
  currently have two sources describing the *same* record (ncbi_biosample
  gives sample attributes, ena gives run metadata: overlapping scope, not
  duplicate records). If a future source mirrors an existing one, collapse
  via `sources.mirror_group` before counting independent sources, rather
  than double-counting.
