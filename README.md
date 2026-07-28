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
`sample` Entity rows, and per-run sequencing metadata (library
strategy/source, platform, file accession/size/checksum -- never the file
itself) as `sequencing_run` Entity rows. A study with both a DOI and a
BioProject accession gets both resolved in one task. Validated against a
real public BioProject (837 linked BioSamples, 500+ sequencing runs) --
see "Real seed data" below.

NCBI SRA is deliberately not a separate adapter: run-level data is served
via ENA's read_run query instead (same underlying INSDC-shared records,
much cleaner JSON than NCBI's SRA XML) -- see `sources/ncbi.py`'s
docstring and `docs/architecture.md`.

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

**What it does not do yet:** OBIS/GBIF/BCO-DMO/PANGAEA/DataCite (no
adapter yet -- a study with none of DOI/BioProject/ENA accession raises
`NotImplementedError` from the handler), BeBOP/MIOP raw_fact mapping (see
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
