# Architecture

## Layering

```
sources/        -- base.py (SourceAdapter ABC + HTTP/rate-limit/cache) + crossref.py/europe_pmc.py/openalex.py (M2) + ncbi.py/ena.py (M3); OBIS/GBIF/BCO-DMO/PANGAEA/DataCite are M5+
discovery/       -- seed loading done; citation expansion, keyword search, relevance are M5+/M7
identity/        -- identifiers.py (normalize/validate), deduplication.py (Stage 1 exact-match + Stage 2 merge_study_into) done; Stage 3 candidate_matches scoring is M5+
extraction/      -- evidence.py, sections.py, text.py (M4, LLM-based text extraction); structured (API/XML) extraction lives in each sources/*.py adapter's extract_structured_facts(), not here
llm/             -- base.py (LLMBackend ABC), http_backend.py (OpenAI-compatible wire protocol, any base_url), mock.py, disabled.py, factory.py, benchmark.py (M4) -- done
mapping/         -- raw_facts -> standardized_values (FAIRe/BeBOP/vocab) (M6, not yet written)
validation/      -- evidence/schema/logical/cross-source checks (M5, not yet written)
assets/          -- data-asset inventory (M5, not yet written)
scheduling/      -- weekly watermarked discovery, run reports (M7, not yet written)
exports/         -- raw_facts.py done; faire.py/bebop.py/study_summary.py are M6
workflow/        -- task_queue.py, retries.py, worker.py, states.py (M1) + handlers.py (M2: publication resolution; M3: + repository resolution; M4: + EXTRACT_TEXT_FACTS)
database/        -- models.py (full schema + M3's entities.external_identifier column), enums.py, session.py, ids.py, migrations/ -- done
```

The dependency direction is one-way: `database` has no dependents inside
the package; `identity`/`workflow` depend only on `database`; `sources/`
depends only on `config`/`database.enums`/`logging_setup`/`identity.identifiers`
(ENA's `find_related` uses `guess_identifier_type` to disambiguate
SRA-vs-ENA study accessions); `workflow.handlers` is the composition root
that wires `sources/*` adapters to `identity` dedup and persists into
`database.models` -- `cli.py` is the only module allowed to depend on
everything. This is what makes it safe to add OBIS/GBIF/etc. adapters or
the extraction/mapping/validation layers later without touching
`database/models.py` except to add columns/tables via a migration.

## Why the DISCOVER_IDENTIFIERS handler lives in workflow/, not sources/ or discovery/

`workflow/handlers.py` is orchestration (call each enabled adapter, persist
Source/RawFact/Publication/ExternalIdentifier rows, run Stage 2 dedup on
newly-discovered identifiers) -- it contains zero source-specific parsing
logic (that's in each adapter's `extract_structured_facts`/
`parse_publication_fields`/`find_related`), which is what section 6 asks
for ("Do not put source-specific parsing logic in the central
orchestrator"). It's registered into `workflow.worker.TASK_HANDLERS` via an
import side effect in `cli.py`, so `worker.py` itself never needs to know
which handlers exist.

## Adapter idempotency and rate-limiter lifetime (Milestone 2/3)

`RateLimitedClient` (in `sources/base.py`) tracks `last_request_at` on
itself and caches every response permanently under `data/cache/<adapter>/`.
Because of this, `workflow/handlers.py` caches adapter *instances* for the
life of one worker process (`_build_enabled_adapters`'s module-level
cache) rather than building fresh ones per task -- rebuilding per task
would reset the rate limiter's throttle every task and never actually
enforce `config/sources.yaml`'s `rate_limit_per_second` across a run.
`ncbi_bioproject` and `ncbi_biosample` additionally share ONE
`RateLimitedClient` (via `SourceAdapter.__init__`'s `http=` override),
since both hit `eutils.ncbi.nlm.nih.gov`, whose per-IP rate limit applies
across all eutils calls combined, not per logical adapter.

`handle_discover_identifiers` is safe to retry: `_persist_source_and_facts`
checks for an existing `Source` row (study, adapter, queried identifier)
before creating one, so a task that fails partway through (e.g. Crossref
succeeds, then Europe PMC times out) and gets retried by the queue won't
duplicate that adapter's Source/RawFact/Entity rows. Publication-field
merging and related-identifier discovery always re-run from a fresh
`fetch_record()` call regardless (cheap, since it's a cache hit), so a
Publication row or the set of discovered identifiers ends up complete even
if it was first assembled across two attempts.

Related-identifier existence (`_apply_related_identifiers`) is checked via
a live `find_existing_study_by_identifier` DB query, not the
`study.external_identifiers` ORM relationship collection -- that
collection can go stale mid-task once multiple identifier-adding passes
happen in one `handle_discover_identifiers` call (one per adapter). This
is not a hypothetical: `ncbi_biosample` and `ena` routinely report the
*same* BioSample accession (they mirror the same underlying INSDC
records), and a live validation run against a real 837-sample BioProject
hit exactly this -- a `UNIQUE constraint failed` on
`external_identifiers` because the second adapter's "is this already
recorded" check consulted a collection that didn't yet reflect the first
adapter's inserts. Fixed by always re-querying the DB. See
`tests/unit/test_handlers_repository.py::test_two_adapters_discovering_the_same_related_identifier_does_not_raise`.

That same live run also surfaced a worker-level bug: the IntegrityError
above left the SQLAlchemy session in a "pending rollback" state, and
`workflow/worker.py`'s exception handler called `fail_task()` (which
writes to the session) *before* rolling back, so `fail_task` itself raised
`PendingRollbackError` and crashed the whole CLI command instead of
cleanly marking one task failed. Fixed by rolling back first. See
`tests/unit/test_worker.py`.

## Why raw_facts and standardized_values are separate tables

Extraction (Pass 1) writes `raw_facts` in whatever vocabulary the source
used, tied to `evidence_quote` + `source_locator`. Mapping (Pass 2) reads
accepted `raw_facts` and writes `standardized_values` against a specific
`target_schema`/`target_schema_version`. Bumping the FAIRe schema version
therefore means re-running mapping, not re-running extraction against every
paper again -- extraction is the expensive, LLM-involving step; mapping is
deterministic rule application over already-extracted facts.

## Why `entities` is one generic table, not one table per level

`EntityLevel` (project, sampling_event, sample, assay, experiment_run,
sequencing_run, protocol, bioinformatics_workflow, data_asset) is a column, not a set of
separate tables with separate foreign keys. A new entity level is a new
enum value, not a migration. Level-specific attributes live in
`raw_facts`/`standardized_values` keyed by `entity_id`, not as columns on
`entities` itself.

`entities.external_identifier` (added Milestone 3, via a real migration,
using `batch_alter_table` for SQLite compatibility since SQLite can't
`ALTER TABLE ADD CONSTRAINT` in place) is the entity's own accession when a
source assigns one below the study level -- a BioSample accession for a
`sample` entity, an ENA/SRA run accession for a `sequencing_run` entity.
It's what makes `workflow/handlers.py`'s `_get_or_create_entity`
idempotent: get-or-create by `(study_id, entity_level, external_identifier)`,
enforced by a real unique constraint (`uq_entity_study_level_external_id`).
NULL is exempt from that constraint in both SQLite and PostgreSQL (`NULL
<> NULL` in standard SQL), so entity levels with no independent accession
(sampling_event, assay, protocol, ...) can have any number of NULL rows.

`experiment_run` is the row-owning entity for FAIRe
`experimentRunMetadata`: one sample- and assay-specific library, identified
by `lib_id` when the source reports one. It is deliberately distinct from
`sequencing_run`, the physical run/file event. `entity_relationships`
records `derived_from_sample`, `uses_assay`, and `sequenced_in_run`, so two
library entities can share one sequencing run without merging their PCR
well, MID/barcode, concentration, or file metadata. A source that omits
`lib_id` gets an `internal:` entity key for traceability, but no fabricated
FAIRe `lib_id` value.

`RawFactCandidate.entity_external_id`/`entity_label` (also Milestone 3) is
how an adapter tells the handler "this fact is about sub-entity X, not the
study as a whole" without the adapter needing to know anything about
`Entity` rows itself -- e.g. `NcbiBioSampleAdapter.fetch_record()` returns
one `SourceRecord` for an entire BioProject's worth of samples (bounding
`Source`-row growth to one row per project regardless of sample count),
and `extract_structured_facts()` tags each attribute fact with the
specific BioSample accession it came from.

## Why the LLM layer is a provider-independent abstraction, not an OpenAI client (Milestone 4)

`llm/base.py`'s `LLMBackend` ABC and its `OpenAICompatibleHTTPBackend`
implementation exist specifically so this pipeline never depends on a
particular vendor or model. "OpenAI-compatible" names the HTTP
request/response *wire protocol* (`POST {base_url}/chat/completions` with
an OpenAI-shaped payload) -- the same shape Ollama, vLLM, TGI's
OpenAI-compatible router, and most institutional inference gateways all
speak. `OpenAICompatibleHTTPBackend` never contacts an OpenAI-operated
host, never requires OpenAI credentials, and sends data only to whatever
`base_url` is configured. `build_llm_backend`/`build_benchmark_backend`
(`llm/factory.py`) both reject placeholder model names rather than
defaulting to a real one -- there is no model this project ships with
opinions about.

This is also why the benchmark harness (`llm/benchmark.py`) exists as
first-class scaffolding rather than an afterthought: the choice of model is
explicitly deferred to the user, validated first against Claude-curated
gold cases (`GoldCase.curated_by`) and eventually manual review, with
metrics (JSON-validity rate, evidence-verification rate, precision/recall/F1,
latency) computed the same way regardless of which candidate is being
measured -- `run_benchmark` and `summarize` have no per-model branches.

## Why extraction/text.py's facts get evidence_quote but sources/*.py's don't

`RawFactCandidate.evidence_quote` (added Milestone 4, on the shared model
in `sources/base.py`) is populated only by LLM-derived facts
(`extraction/text.py`). Structured API/XML facts (Milestone 2/3 adapters)
already have unambiguous evidence in the form of `source_locator` (a
JSON/XML path back to the exact field) -- there's no free-text "quote" to
extract from a JSON response. `extraction/evidence.py`'s
`verify_evidence_quote` is what makes evidence_quote meaningful rather than
just an unverified string the model could fabricate: it's checked against
the exact section text a fact claims to come from, and a fact that fails
verification is dropped in `extract_facts_from_section` before it ever
reaches the handler -- persistence-layer code (`handle_extract_text_facts`)
never sees an unverified candidate.

## Why EXTRACT_TEXT_FACTS is separate from, not chained off, DISCOVER_IDENTIFIERS

`enqueue_text_extraction_backfill` is a distinct, explicitly-triggered
step, not something `handle_discover_identifiers` queues automatically
when it discovers a PMCID. With `llm.enabled: false` (the default), every
`EXTRACT_TEXT_FACTS` task would otherwise fail against `DisabledLLMBackend`
-- correct behavior, but noisy, and it would silently start happening the
moment this milestone's code merged rather than when a user deliberately
configures a real model endpoint. Text extraction is opt-in per the
existing `fair-ocean enqueue-seed-backfill` / `fair-ocean
enqueue-text-extraction-backfill` split.

## Why sqlite:/// URLs get rewritten to an absolute path (found via real-model testing)

`database/session.py`'s `_resolve_sqlite_url` rewrites a relative
`sqlite:///data/fair_ocean.db` URL to an absolute one anchored at
`REPO_ROOT` before `create_engine` ever sees it. Without this, `sqlite3`
resolves a relative path against the *process's current working
directory* at connect time -- which only matches `REPO_ROOT` if the
process happens to be launched from inside the repo. Every `fair-ocean`
CLI invocation always was, so this was invisible until a one-off Python
analysis script (checking real extraction output during Milestone 4's
live model testing) was run with cwd one directory up and got "unable to
open database file." The exact same failure mode would hit a cron job or
systemd unit (Milestone 7) invoked from `/` or a service working
directory that isn't the repo -- this fix is what makes the database
location deployment-invariant rather than an accident of how you happen
to launch the process.

## Why the benchmark harness's matching forgives formatting but not content

`llm/benchmark.py`'s `_score_case` was originally exact-string-match on
`(fact_type_candidate, normalized raw_value)`. Live testing against a real
qwen2.5:3b run produced a fully-correct extraction (5/5 facts, all
evidence-verified) that scored precision=recall=0.00, purely because the
model wrote `"collection date"` / `"14 March 2021"` where the gold case
(written by Claude) happened to write `"collection_date"` /
`"2021-03-14"`. Two fixes, deliberately scoped narrowly:

- `_normalize_label` treats underscores/hyphens/spaces/case as the same
  label -- a category name is a naming choice, not part of the fact being
  verified.
- `_values_equivalent` additionally treats two values as equal if both
  parse as the same calendar date via `dateutil`, using a fixed sentinel
  anchor (`datetime(1, 1, 1)`) rather than dateutil's default of *today* --
  otherwise a partial date like `"March 2021"` would resolve differently
  depending on what day the benchmark happened to run, and could
  spuriously match or fail to match today-dependent noise.

Non-date values still require exact normalized-string equality --
`dateutil`'s strict mode (`fuzzy=False`) reliably rejects non-date strings
(kit names, primer sequences, instrument names) as unparseable rather than
guessing, so this never loosens matching for anything that isn't
genuinely a reformatted date. Unifying date formats *for real* (so a
report always shows ISO dates regardless of what a source said) is a
later pipeline concern (mapping/standardization, Milestone 6) -- this
fix only prevents the raw-extraction benchmark from penalizing a model
for a formatting choice that later pipeline stages will normalize anyway.

## Milestone 5: validation/assets are pure functions, persistence is the handler's job

`validation/logical.py`, `validation/evidence.py`, and
`validation/cross_source.py` never touch the database directly -- each
takes plain values (a raw string, a support_type, a study_id + session for
the query-heavy cross-source case) and returns a small result object.
`workflow/validation_handlers.py` is the only place that turns those
results into `ValidationResult` rows, same separation as `sources/*.py`
(parsing) vs `workflow/handlers.py` (persistence) from Milestones 2-4. This
is what made the modules trivially testable against exact real-data
formats (see README's "Milestone 5 validation" section) before ever
touching a database session.

`assets/inventory.py` deliberately derives `DataAsset` rows from
already-persisted `raw_facts` rather than adding a new adapter call -- 
Milestone 3's ENA adapter already extracts `fastq_ftp`/`fastq_bytes`/
`fastq_md5` per sequencing_run entity; inventorying them is a read-and-
reshape of data already paid for, not a new network round-trip. This
pattern (mine what's already extracted before reaching for a new source)
is why Milestone 5 shipped with real, non-synthetic validation on day one:
101 real studies' worth of `raw_facts` already existed to inventory and
validate against.

`VALIDATE_CROSS_SOURCE` only compares `title` for now -- it's the one
concept named identically (`fact_type_candidate="title"`) across all
three Milestone 2 publication adapters. Journal name differs per adapter
("container-title" vs "journalTitle" vs none for OpenAlex); reconciling
field-name differences across sources is what the mapping layer
(Milestone 6, raw_facts → standardized_values) is *for* -- validation
compares what's already named the same, mapping is what makes differently-
named things comparable in the first place.

`VALIDATE_EVIDENCE` only persists a `ValidationResult` for facts that
*fail* the check, unlike `VALIDATE_LOGIC`/`VALIDATE_CROSS_SOURCE`, which
persist every outcome including confirmations. The difference is
cardinality: a study can have thousands of raw_facts (evidence-consistency
is checked per-fact), so recording every pass would bloat the table for
no benefit -- a passing fact's evidence is already visible on the
`raw_facts` row itself. Cross-source/accession-format checks produce at
most a handful of rows per study, so recording confirmations there is
cheap and is itself useful audit evidence (e.g. "924 accession formats
confirmed, 0 unsupported" is a real, reassuring number from Milestone 5's
live validation run, not just an absence of complaints).

## Why FAIRe entity_id resolution is table-shaped, not fact-shaped (Milestone 6)

`mapping/faire.py` doesn't simply copy a RawFact's own `entity_id` onto the
StandardizedValue it produces. FAIRe's `projectMetadata` table is a
study-wide singleton -- one row per project, never per-sample or per-run --
but this pipeline's own raw_facts for that same information (e.g.
`instrument_platform`) live on `sequencing_run` entities, repeated
identically once per run (up to 500 times for one real BioProject). If
entity_id were copied straight from the source fact, mapping would either
produce 500 duplicate rows for a single-valued project field, or (if
naively deduplicated by value alone) silently merge two runs that actually
used different platforms. Instead, resolution is keyed off what the
*target table* means: `projectMetadata`-targeted values always get
`entity_id=None` and are deduplicated by `(target_table, target_field,
entity_id)` with the first value winning -- but a later disagreeing value
flags the existing row `review_required=True` rather than being silently
dropped or silently overwriting. The one deliberate exception is
`sample_accession` -> `materialSampleID`: that value belongs to one
specific sample, so it's redirected via an `Entity` lookup by
`external_identifier`, never broadcast or defaulted to `None`.

The same table-shaped logic handles the opposite mismatch: a study-level
LLM-extracted fact (no entity at all) mapped onto a `sampleMetadata`
field gets `entity_id=None` too, which `exports/faire.py` reads as "apply
this value to every sample row in the study that doesn't have a more
specific value of its own" -- a reasonable modeling choice for a single-
protocol study where such a fact (extraction method, storage conditions)
really does apply uniformly, and one that keeps `mapping/faire.py` from
having to guess which specific sample an un-scoped fact belongs to.

## Why FAIRe's rules table only maps what real raw_facts data actually has

`mapping/rules.py` was built by querying this pipeline's own real,
persisted `raw_facts` for the actual `(fact_type_candidate, entity_level)`
combinations present in the 101-study database, not by reading FAIRe's
~300-slot schema top-down and guessing at correspondences. The two most
consequential decisions that fell out of that: (1) FAIRe's atomic
PCR/extraction fields (`pcr_primer_forward`, `pcr_cond`, `annealingTemp`,
`nucl_acid_ext_kit`, ...) had no rule at all, because the Milestone 4
extraction prompt produced one coarse free-text blob per concept
(`PCR_amplification_conditions`) -- mapping that blob into six atomic
fields would mean inventing structure the source text doesn't give
deterministic evidence for, so it was mapped only to FAIRe's own
`*_method_additional` fallback field, always flagged `review_required`.
**This predates Milestone 8** (see "Why the extraction taxonomy is one
module the prompt is rendered from" below): extraction now produces
atomic, standard-agnostic `fact_type_candidate` values
(`pcr_conditions`, `annealing_temperature`, `dna_extraction_kit`, ...) with
an optional FAIRe-field hint attached, closing the extraction-side half of
this gap. Whether `mapping/rules.py` has rules consuming these new native
names yet is separate, ongoing mapping-side work -- check `mapping/rules.py`
directly for current coverage rather than assuming from this paragraph.
(2) `eventDate` needed a precision-aware ISO formatter
(`mapping/units.py`'s `to_iso_event_date`, using the dual-anchor
dateutil trick) rather than reusing `dates.try_parse_date` directly --
that function's single fixed anchor is correct for *comparing* two dates
(Milestone 4 benchmark scoring, Milestone 5 date-ordering validation) but
would silently fabricate day-of-month precision a source string like
"March 2021" never actually reported.

## Why the standards registry (Milestone 6b) is a separate concern from mapping/faire.py

It would be tempting to fold `standards/`'s FAIRe/MIOP/BeBOP crosswalk into
`mapping/`, since both deal with "which external schema does this field
correspond to." They're deliberately kept apart because they resolve two
different kinds of question. `mapping/faire.py` answers "which FAIRe field
does *this pipeline's own raw_fact* correspond to" -- its inputs are this
project's real, extracted data. `standards/registry.py` answers "which
upstream term does *this BeBOP protocol template's field* correspond to"
-- its inputs are three vendored upstream schemas, and it never touches
this pipeline's own database at all (`build_registry()` takes no
session). Conflating them would make `mapping/` depend on `standards/`'s
compiled output for no benefit, and would make it unclear which layer is
responsible for a given piece of provenance.

The crosswalk's 5-level dedup priority (exact identifier > exact MIOP name
> exact FAIRe name > known alias/title > normalized-name-only review
candidate, never an automatic merge) exists because the real BeBOP
protocol templates don't spell MIOP field names consistently --
`protocol_template_bioinformatics.md` uses a slot's structural name
(`meth_cat`) while the other four templates use its title in
underscore-joined form (`methodology_category`), and one of those four
even mixes in a hyphen (`broad-scale_environmental_context`). A single
"normalize and compare" pass would conflate real fields with genuinely
different meanings just as easily as it reconciles spelling variants of
the same field -- there's no way to tell those apart from the string alone.
Splitting the match into "exact structural name" (priority 2, high
confidence) versus "exact title" (priority 4, still confident but a
weaker signal) versus "normalized-only" (priority 5, flagged for human
review, never silently merged) means a real spelling variant resolves
automatically while a coincidental name collision gets surfaced instead of
silently merged -- see `standards/crosswalk.py`'s module docstring and
`tests/unit/test_standards_crosswalk_synthetic.py` for the priority-4-vs-5
distinction made concrete.

## Why REFRESH_STUDY_SOURCES reuses DISCOVER_IDENTIFIERS's traversal instead of duplicating it (Milestone 7)

`workflow/handlers.py`'s `_resolve_publication_sources` and
`_resolve_repository_sources` do a lot: iterate enabled adapters, call
`fetch_record`, merge publication fields, discover related identifiers,
perform Stage 2 merges. Milestone 7's refresh path needs all of that
*except* the one-line persistence decision at the end ("skip if a Source
row already exists" vs. "diff by content_hash and persist if changed").
Rather than fork the whole traversal into a second, slowly-diverging
copy, both functions took a `persist_fn` parameter (defaulting to the
original `_persist_source_and_facts`, so `handle_discover_identifiers`'s
behavior is unchanged) that `workflow/refresh_handlers.py` overrides with
`_persist_refreshed_source_and_facts`. This is the same shape as
Milestone 6's `mapping/faire.py` reusing `standards.faire_registry` rather
than re-parsing `schema.yaml` a second way -- shared traversal, swappable
policy at the one point where discovery and refresh genuinely differ.

## Why a refreshed Source becomes a new row (parent_source_id), never an overwrite

Every other place raw_facts gets consumed (Milestone 5's validators,
Milestone 6's FAIRe mapping) already treats it as an append-only evidence
log, not a mutable "current value" cache -- a fact is "what source X said
at time Y," and disagreement between sources is something to surface
(`VALIDATE_CROSS_SOURCE`), not silently resolve. Refresh follows the same
principle: when a re-fetched record's `content_hash` differs from the
most recent prior `Source` row for that (study, adapter, identifier), a
*new* `Source` row is created (linked via `parent_source_id`) with its own
fresh `RawFact`s, rather than mutating the old row or deleting it. This
means a BioProject growing from 300 to 400 samples produces two Source
snapshots history can distinguish, not one row silently changed in place.
The known gap this leaves (documented in `workflow/refresh_handlers.py`'s
docstring): `mapping/faire.py`'s dedup is "first raw_fact encountered
wins," so it doesn't yet prefer the newest snapshot when a field has
facts from more than one Source generation -- re-running
`enqueue-mapping-backfill` after a refresh with real changes is still a
manual step, not automatic.

## Why weekly-update's cache-bypass is a per-adapter clear_cache(), not a fetch_record(force_refresh=...) parameter

Getting genuinely fresh data on refresh requires bypassing
`RateLimitedClient`'s on-disk cache, which never expires on its own
(correct for the normal discovery/retry path, where a cached response
being stale is never the point). The alternative to a blanket
`clear_cache()` -- threading a `force_refresh` flag through
`fetch_record()`'s signature -- would touch six adapters' method
signatures and the shared `_esearch_first_uid`/`_elink_ids` helpers in
`sources/ncbi.py` for the same effect, and the caller still can't target
individual cache files from outside without duplicating each adapter's
URL-building logic (exactly what section 6 says not to do). Clearing an
adapter's entire cache directory once per weekly-update run is coarser
(it also invalidates cache entries for studies not being refreshed this
pass) but adapter-agnostic and touches nothing inside `sources/*.py`.

## Why quarterly_full_rediscovery's cadence lives in WorkflowRun history, not a new table

`SourceWatermark` is scoped to one `(source_name, query_identifier)` pair
-- it has no natural place to record "when did the *quarterly full sweep*
last run," which is a global, not per-source, fact. Rather than add a new
table for one boolean question, `scheduling/rediscovery.py`'s
`is_rediscovery_due` queries `WorkflowRun` for the most recent completed
`run_type="quarterly_full_rediscovery"` row directly -- exactly the kind
of thing `WorkflowRun` already exists to record. The same reasoning is why
`retry_stale_failed_tasks`/`retry_stale_manual_review_tasks`
(`scheduling/retry_policies.py`) don't need any new bookkeeping either:
"has this FAILED/MANUAL_REVIEW_REQUIRED task been stale long enough"
is answered directly from the `Task` row's own `updated_at`, so those
sweeps self-pace regardless of how often `weekly-update` itself is
invoked.

## Why publications folded per-adapter onto Source rows, not into one merged row

The old `publications` table held one row per study, built by merging
whichever of crossref/europe_pmc/openalex answered first-non-null-wins.
That merge step (`_merge_publication_field`) is exactly the shape of bug
this schema otherwise avoids: two independent representations of the same
fact (a Source row per adapter already existed; a second, synced
Publication row duplicated part of what those rows implied) that could
drift if one write path updated one and not the other. The fold instead
lets each adapter's own Source row carry its own bibliographic fields as
that adapter reported them (`workflow/handlers.py`'s
`_apply_publication_fields`) -- no merge at write time. A consumer that
needs one answer across sources (validation/logical.py's collection-date-
vs-publication-year check) computes it at query time
(`_publication_year_for_study`: first non-null across a study's
publication-type Source rows) instead of reading a value some earlier
write path already decided. This is the same principle Milestone 7's
`parent_source_id` snapshot-on-change design already established for
refreshed sources: prefer recomputing a merged view over storing one that
can go stale.

`doi`/`pmid`/`pmcid`/`openalex_id` were not recreated as Source columns
during this fold specifically because they were already fully redundant
before it: `Source.external_identifier` already holds the DOI for every
publication-type row (every publication adapter is queried by DOI, see
`_resolve_publication_sources`), and `ExternalIdentifier` was already the
authoritative store for every other cross-reference. The old
`publications` table had been duplicating both without either being a
genuine second source of truth -- removing the duplication was strictly a
subtraction, not a redesign.

## Why the extraction taxonomy is one module the prompt is rendered from, not prose written directly into the prompt (Milestone 8)

`extraction/faire_fields.py` exists as a separate, structured module
(`FaireExtractionField` dataclasses grouped in `FIELD_GROUPS`) rather than
just writing the ~70-concept checklist directly into
`extraction/text.py`'s prompt string, for the same reason
`mapping/rules.py`'s `RULES` table is data instead of inline logic: the
taxonomy is something three different things need to agree on --
the prompt (`render_field_reference()`), gold-case validation
(`all_field_names()`, used by `test_gold_cases_use_native_taxonomy_names_or_a_documented_fallback`),
and any future mapping-layer work that wants to know exactly which
concepts extraction can even produce. Encoding it once and rendering the
prompt from it means those three can't quietly drift out of sync the way
gold cases and the production prompt used to (see below).

**v2 -> v3 correction.** v2's field names were FAIRe's own exact slot
spellings (`annealingTemp`, not "PCR annealing temperature"), on the
reasoning that an exact-label mapping rule is more reliable than a fuzzy
one. That coupled `fact_type_candidate` -- a raw fact's own identity -- to
one specific standard's vocabulary, which is exactly the coupling this
pipeline avoids everywhere else in the raw-facts layer (a repository
adapter's `fact_type_candidate` is never phrased in Darwin Core or MIxS's
own spelling; standardizing onto any vocabulary is `mapping/rules.py`'s
job alone, a separate downstream step over source-native raw_facts). A
user caught this before any real database held rows built on it. v3 keeps
every `FaireExtractionField` but splits it into two attributes: `native_name`
(a plain, standard-agnostic description -- what `fact_type_candidate`
actually gets set to, e.g. `annealing_temperature`) and `faire_hint` (the
FAIRe slot spelling that concept corresponds to, e.g. `annealingTemp` --
rendered into the prompt only as a bracketed suggestion, never as the
concept's own name). The model may optionally return a
`candidate_standard_fields` hint per fact (e.g. `{"faire": "annealingTemp"}`),
which `extract_facts_from_section` stores in `RawFactCandidate.confidence_metadata`
and, from there, `RawFact.confidence_metadata` -- a JSON column that
already existed, unused, so this needed no migration. A hint is
deliberately non-authoritative: a raw fact with no hint at all, or a hint
a caller ignores entirely, is still a fully valid, standard-agnostic raw
fact. `mapping/rules.py` can choose to trust a hint as a fast path or
ignore it and match purely on `native_name`/`entity_level` the way every
other adapter's facts are matched -- that choice is entirely
`mapping/rules.py`'s to make, not decided at extraction time.

Before this taxonomy existed at all (pre-Milestone-8), the extraction
prompt had no reason to produce anything but ad hoc names, so every atomic
PCR/sequencing/bioinformatics fact collapsed into one coarse per-concept
blob (`PCR_amplification_conditions`) that could only ever map onto
FAIRe's free-text `*_method_additional` fallback fields, flagged for
manual review. The v3 taxonomy fixes that same gap without re-introducing
the standard-coupling problem v2 introduced while fixing it.

## Why the benchmark harness builds its prompt from extraction/text.py instead of a gold-case-local `instructions` field

Before this milestone, `GoldCase` carried its own `instructions` string,
rendered by a second, benchmark-local prompt template
(`EXTRACTION_PROMPT_TEMPLATE`). That meant the benchmark harness could
score a candidate model against a prompt that wasn't actually the one
`handle_extract_text_facts` sends in production -- a change to
`extraction/text.py`'s real instructions wouldn't be reflected in a
benchmark run until someone remembered to copy the change into every gold
case file too. `GoldCase.section_title` (defaulting to `"Methods"`) plus
`run_case` calling `extraction.text.build_prompt` directly removes that
second copy entirely: whatever the real pipeline's prompt says right now
is exactly what the next `fair-ocean benchmark-models` run tests, by
construction, not by discipline.

## Why the checklist-vs-group-header clarification was kept despite a real cost to the weakest model

Live benchmarking (README's "Milestone 8 validation") found two opposite
real failure modes across 6 local models: llama3.2:3b confused the
checklist's group headings ("PCR / assay setup:") with the field names
themselves, while qwen2.5:3b simply went silent on more than half the
cases rather than engage with the longer, denser prompt at all. A single
clarifying paragraph added to `EXTRACTION_INSTRUCTIONS` (distinguishing
headings from field names) measurably helped llama3.2:3b (0/10 -> 2/10
true positives on the affected case) but measurably hurt qwen2.5:3b
further on that same case (a valid-but-empty response became one that
didn't parse as JSON at all, and took nearly 3x longer). Re-checked
directly against qwen3-4b-instruct (the strongest performer) to confirm
the addition cost it nothing: 8 true positives / 0 false positives / 2
false negatives on the same case, unaffected in the direction that
matters.

The fix was kept anyway. Neither weak model produces usable output for
this task with or without it -- qwen2.5:3b's failure mode shifting from
"silent" to "malformed" doesn't change whether its output is usable (it
isn't, either way), while the clarification is a real, independently
justified correctness improvement for every model capable enough to need
it. Tuning the prompt to rescue a 3B model's failure mode would optimize
for a candidate this taxonomy was never going to work well with in the
first place, at a real cost (a longer, more hedged prompt) paid by every
other candidate.

## Why supplement retrieval state lives on DataAsset, not Source (supplementary-material layer)

`DISCOVER_SUPPLEMENTS` creates one `Source` row per referenced
supplementary file, but that row is set once at creation
(`inspection_status=inspected`, `inspection_level=metadata_only`) and never
mutated again -- matching every other `Source`-creation call site in this
codebase. This was a deliberate constraint, not an oversight: other code
(`validation_handlers.py`) already treats "a `Source` row exists for this
study" as a proxy for "this source was inspected." If a not-yet-retrieved
supplement's `Source` row were created the same way and only *later*
mutated to reflect real progress, every existing call site relying on that
proxy would start silently misreporting supplements that are merely
*referenced* as fully inspected.

Instead, the six retrieval states the design calls for (referenced /
available / retrieved / parsed / inaccessible / parse_failed) live on a
companion `DataAsset` row (`source_id` FK to the `Source` row) using
columns that already existed for exactly this purpose --
`DataAsset.access_status` and `DataAsset.inspection_level` -- decoupled
from `Source`'s own, intentionally-static fields. `retrieved` and
`parse_failed` happen to share the same `(access_status, inspection_level)`
pair; `DataAsset.description` (a pre-existing free-text column)
disambiguates them. Net result: the full 6-state model needed zero new
database columns and zero migrations.

`populate_missingness_for_study`/`populate_faire_missingness_for_study`
read this same `DataAsset.inspection_level`/`access_status` pair to decide
`relevant_source_not_inspected` vs. `source_not_accessible` vs. falling
through to `not_found_in_inspected_sources` -- the first real consumer of
`InspectionLevel` for missingness purposes; before this, the enum existed
on both `Source` and `DataAsset` but nothing actually read it when
computing missingness.

## Why resolve_or_create_study() reuses merge_study_into instead of replacing it (shared study resolution)

`identity/resolution.py`'s `resolve_or_create_study()` is the new single
merge-decision function both DOI-seeded and accession-seeded discovery
call, but it deliberately does **not** reimplement `merge_study_into`'s
own FK-reassignment machinery for the "attach" outcome (tier-1 auto-link,
or tier-2/3 after a consistent check) -- it calls `merge_study_into`
unchanged. That function already does exactly the right thing (full-study
absorption, `ExternalIdentifier` reconciliation, bulk FK reassignment
across `_STUDY_FK_MODELS`, `canonical_status=MERGED`) for a real "these
two Study rows describe the same thing" case. `resolve_or_create_study` is
a thin orchestration layer on top of it, adding only the NEW decision this
task needed: whether to reach `merge_study_into` at all, or instead split
off a sibling Study and flag for review.

The genuinely new logic is the *split* case (`_create_sibling_and_flag`),
needed because `merge_study_into`'s machinery operates on a whole absorbed
study's worth of rows -- it has no notion of "move just this one Source's
evidence." Two design decisions there are worth being explicit about:

- **`StudySource` is deliberately excluded from `merge_study_into`'s
  `_STUDY_FK_MODELS` tuple.** A blanket bulk `UPDATE study_id` is wrong for
  a join table where the target `study_id` might already have its own row
  for the same `source_id` (a unique-constraint collision `Entity`/
  `RawFact`/etc. don't have, since those aren't unique per study). It gets
  its own explicit drop-on-collision/re-point block instead, mirroring the
  `ExternalIdentifier` block a few lines above it in the same function.
- **Entity ownership on split**: `Entity` has no `source_id` column of its
  own (only study-scoped, via `external_identifier`), so "does this Entity
  belong with the evidence being split off" is genuinely ambiguous when an
  Entity has facts from more than one Source. The chosen rule -- move an
  Entity only if 100% of its RawFacts came from the one Source being
  split off, otherwise leave it in place and rely on the flagged
  `CandidateMatch` for a human to sort out -- trades a recoverable false
  negative (Entity stays behind when it arguably should move) against an
  unrecoverable false positive (silently scattering one physical sample's
  facts across two Study rows). The asymmetry in cost is why the
  conservative rule was chosen without much debate.

`sources.study_id` itself was kept as a plain, unchanged FK rather than
replaced by the new `study_sources` join table, specifically to avoid
touching the ~15 files across the adapter/handler layer that read
`Source.study_id` directly today. `study_sources` is purely additive: a
single new choke point (`identity/source_linking.py`'s `create_source()`)
writes both columns atomically for every new Source, and the join table's
existence is what makes "a Source belongs to more than one Study"
representable at all going forward -- but nothing existing had to change
to make room for it.

## Why BIOSAMPLE_ACCESSION is excluded from Study-identity resolution (multi-study entity sharing)

`identity/resolution.py`'s `resolve_or_create_study()` (see the section
above) treats a `STRUCTURED_SOURCE`-confidence `RelatedIdentifier` match
against a different Study as strong enough evidence to merge the two Study
rows unconditionally, no consistency check needed -- correct for a DOI/
PMID/BioProject accession, where two Study rows both claiming the exact
same one is genuinely a strong signal they're duplicate submissions of the
same underlying study.

That assumption breaks for `BIOSAMPLE_ACCESSION` specifically, once
Milestone 24 made real sample/experiment/sequencing-run reuse across
genuinely different papers a normal, expected occurrence (a second paper
reanalyzing an earlier paper's deposited data) rather than a rare
coincidence. `NcbiBioSampleAdapter`/`EnaAdapter`'s `find_related()` both
report *every* fetched sample's accession as a `RelatedIdentifier`
(default confidence `STRUCTURED_SOURCE`) -- so the moment a second,
citing paper's own BioSample/ENA resolution reaches a sample an earlier
paper already owns, the unconditional-merge branch fires and silently
collapses two distinct papers into one Study row, discarding whichever
one lost. Confirmed live: this is exactly what happened running a real
end-to-end citation-discovery pass (10.1038/s42003-024-06136-2 /
10.1073/pnas.2005917117, both resolving PRJNA529480) before this fix.

`resolve_or_create_study()` now special-cases `BIOSAMPLE_ACCESSION`
entirely, ahead of the merge/sibling-split branching: a match against a
different Study is recorded as an informational `ExternalIdentifier`
(`relationship_type=SHARES_ACCESSION_WITH`, mirroring the sibling-split
case's own convention) and the loop moves on -- never a merge, never a
sibling+`CandidateMatch` flag either, since there's nothing ambiguous
about it. The real "this sample belongs to more than one Study" fact is
represented at the *Entity* level instead, via `entity_studies` (see
below) -- Study-level `ExternalIdentifier` rows for `BIOSAMPLE_ACCESSION`
are purely a discoverability convenience now, never identity evidence.

## `entity_studies`: representing a shared physical sample/run across Studies

`Entity.study_id` (a plain FK, like `Source.study_id`) stays the fast
"home" pointer everywhere in the codebase -- unchanged. The new
`entity_studies` table (exact structural mirror of the pre-existing
`study_sources`, see the section above) is what makes an Entity belonging
to more than one Study representable at all: every Entity gets a home row
here at creation (`relationship_type=IS_HOME_OF`), and a second paper
resolving the same real BioSample/experiment/sequencing-run accession gets
an additional, non-home row (`relationship_type=SHARES_ACCESSION_WITH`)
pointing the SAME Entity row at its own Study, rather than creating a
duplicate Entity.

Only `SAMPLE`/`EXPERIMENT_RUN`/`SEQUENCING_RUN` are shareable
(`database/enums.py`'s `SHAREABLE_ENTITY_LEVELS`) -- deliberately not
`PROJECT`: a shared BioProject accession is already `resolve_or_create_study`'s
job (a real Study-identity signal, see above), and a paper's own
bioinformatics-pipeline/assay-interpretation facts are PROJECT-level and
paper-specific by the user's own stated model, so making PROJECT
shareable too would create a second, competing mechanism for the same
situation. `ASSAY` stays single-study for the same "interpretive, not
physical" reason.

A single new choke point, `identity/entity_linking.py`'s
`create_entity()`/`link_entity_to_study()`/`get_or_create_entity()`,
replaced what used to be **two independent, divergent**
`_get_or_create_entity` implementations (`workflow/handlers.py`'s
fact-materialization path and `extraction/experiment_runs.py`'s legacy
sequencing-run-to-experiment-run path) -- only one of which had been
updated to know about `entity_studies` before this consolidation, exactly
the kind of silent gap a single choke point (mirroring
`identity/source_linking.py`'s own `create_source`) exists to prevent.
`get_or_create_entity()`'s lookup is global (no `study_id` filter) for
shareable levels, made safe by a new partial unique index on `entities`
(`entity_level`, `external_identifier`, shareable levels only) --
confirmed against the real production database before adding it that zero
existing cross-study duplicates existed among 7,787 real entities.

`workflow/handlers.py`'s `_get_or_create_entity_relationship` needed a
matching fix: its existence check used to be scoped by `study_id`, but
`entity_relationships`' own `uq_entity_relationship` constraint is global
on `(from_entity_id, to_entity_id, relationship_type)` -- once two Studies
can legitimately resolve the same shared entities, a study-scoped lookup
tries to re-insert the identical global triple and hits that constraint.
Fixed by making the lookup global too, matching the real constraint (a
physical relationship between two entities -- this run WAS sequenced from
this sample -- doesn't change depending on which Study is asking, exactly
as physically invariant as the entities themselves).

`identity/deduplication.py::merge_study_into` and
`identity/resolution.py::_create_sibling_and_flag` both needed companion
`entity_studies`-aware updates for the same reason `merge_study_into`
already special-cases `study_sources` (see the section above): `Entity` IS
in `merge_study_into`'s bulk-reassigned `_STUDY_FK_MODELS`, so an
absorbed entity's home study_id pointer gets updated correctly, but its
`entity_studies` home row needs its own drop-on-collision/re-point block
to stay in sync, never a blind bulk `UPDATE` (which could violate
`uq_entity_study` if the surviving Study already independently shares that
same entity). `_create_sibling_and_flag`'s Entity-move block gets the
identical treatment when a moved entity's home changes.

## Citation-expansion ("node-adding") discovery

New `DISCOVER_CITING_STUDIES` task
(`workflow/handlers.py::handle_discover_citing_studies`), enqueued right
after `handle_discover_identifiers` resolves any BioProject accession
(idempotency key is the accession alone, so two Studies sharing one
accession only trigger this once for free). One `bioproject->pubmed`
elink call returns every citing PMID at once -- cheap by construction, no
per-sample fan-out, and a genuinely different NCBI capability from
`sources/ncbi.py`'s own `biosample->bioproject` reverse-elink (UID-
*correctness* verification, not citation discovery).

Auto-expansion is deliberately aggressive, per an explicit user choice
after reviewing the tradeoff: a citing paper not already known gets its
own full `Study` row (`discovery_depth`/`discovery_parent_study_id`/
`discovery_root_study_id`/`discovery_trigger` provenance columns on
`Study`) and re-enters the normal `DISCOVER_IDENTIFIERS` pipeline
recursively -- via the task queue's own idempotency/resumability, not an
in-process recursive traversal. Real safety valves make that safe at
~3000-paper scale: `DiscoveryConfig.citation_expansion_max_depth` (wired
up for the first time -- previously defined but never read anywhere) caps
how many hops deep auto-expansion goes (default 1 for the initial
discovery run); `max_citing_papers_per_bioproject` caps how many of one
accession's citing PMIDs expand per run. Both caps record a review-flagged
`citing_pmid_not_expanded` `RawFact` for whatever they skip, never a
silent drop.

The recurring pass (`scheduling/rediscovery.py::enqueue_citation_rediscovery_backfill`,
CLI: `enqueue-citation-rediscovery-backfill`, auto-wired into
`weekly-update` alongside the existing `quarterly_full_rediscovery` block
via `SchedulingConfig.citation_rediscovery_enabled`/
`citation_rediscovery_interval_days`) exists because the accession-only
idempotency key above means a paper published or indexed by PubMed *after*
a Study's first resolution would otherwise never get picked up again --
this re-enqueues `DISCOVER_CITING_STUDIES` for every known accession with
a fresh, run-scoped key.

## Root determination and two-phase discovery/mapping gating

`Entity.study_id` ("home") is decided by whichever study's task-queue
processing happened to reach an accession first -- for two independently-
seeded studies (both directly in the seed batch, no citation relationship
between them) that happen to share a BioProject, that's processing-order
luck, not a claim about which paper actually collected the physical
sample. `identity/root_determination.py` computes a genuinely separate,
deliberate answer -- `Entity.root_study_id`/`root_status`
(`EntityRootStatus`) -- for exactly the one purpose `exports/faire.py`
needs it for: which linked study's broadcast-style (study-wide LLM/text)
facts should fill a shared entity's blanks. Priority, in order: (1)
earliest publication year among linked studies (reusing
`workflow/validation_handlers.py::_publication_year_for_study` rather than
duplicating it -- a paper cannot analyze data that doesn't exist yet); (2)
the BioProject's own registration date (the `submitted` `RawFact`
`NcbiBioProjectAdapter` already writes) as corroboration/plausibility
context only, never used to override (1) on its own; (3) submitter/
institution matching is explicitly deferred -- no adapter in this codebase
parses structured author affiliations today (Crossref only captures
given/family name; OpenAlex's `authorships` blob, which does carry
institutions in the real API, is stored unparsed here); (4) genuinely
ambiguous (same year, or no publication-date signal at all for the tied
studies) -> flagged `NEEDS_REVIEW`, never guessed, matching this
codebase's established flag-don't-guess convention
(`_record_citation_expansion_capped_fact`, `CandidateMatch`-on-ambiguity).
`identity/entity_linking.py::create_entity` sets `root_study_id=self`/
`DETERMINED` eagerly for the common (non-shared) case -- no algorithm
needed -- and flips back to `PENDING` the moment a second study links to
an already-existing entity.

Root determination only runs once every study sharing an entity has
finished discovering; an earlier answer could be invalidated by a
not-yet-processed sibling still in the queue. Detecting "finished" needs a
**connected component**, not just the two studies involved directly --
`identity/component.py::compute_study_component` BFS walks two distinct
edge types, and neither alone is sufficient: shared-`EntityStudy` edges
miss a freshly-created citing study (from
`handle_discover_citing_studies`), which has zero `EntityStudy` rows until
its own `DISCOVER_IDENTIFIERS` task actually completes; discovery-lineage
edges (`discovery_parent_study_id`/`discovery_root_study_id`) miss two
independently-seeded siblings with no citation relationship between them
at all -- confirmed directly against `identity/resolution.py::_linked_via_discovery_lineage`,
which only ever checks those two columns.

This flat, async task queue (`workflow/task_queue.py`) has no built-in
task-dependency/barrier mechanism, so a new self-rescheduling poll task,
`CHECK_COMPONENT_SETTLED` (`workflow/settle_handlers.py`), *is* that
mechanism: it recomputes the component and checks for any non-terminal
`DISCOVER_IDENTIFIERS`/`DISCOVER_CITING_STUDIES` task anywhere in it; if
one exists, or the component's membership grew since the last check, it
reschedules itself with a fresh idempotency key (a monotonic `generation`
counter carried in its own payload, not a timestamp, so retries within one
generation never multiply tasks) and a delayed `available_after`
(`DiscoveryConfig.component_settle_poll_interval_seconds`, default 300s --
cheap, no network calls). A `max_settle_check_generations` cap (default
100) marks a pathologically stuck component `stalled` (flagged, not polled
forever) rather than looping indefinitely. A settled component that grows
again later (a new citing paper found via
`scheduling/rediscovery.py`'s quarterly pass, or the eager reopen hook in
`handle_discover_citing_studies` right after a new citing study is
created) flips every member back to `PENDING` and starts a fresh cycle,
so root answers don't go permanently stale.

`workflow/mapping_handlers.py::enqueue_mapping_backfill` routes any study
with >=1 shareable-level entity through this settle-check gate instead of
enqueueing `MAP_FAIRE` directly -- satisfying the user's explicit "build
the network before pulling any data into tables" request. A study with no
shareable entities never touches this machinery at all and proceeds
immediately, unblocked, exactly as before. This gate is the efficiency/
consistency layer, not the sole correctness guarantee, though: a design
agent's exploration confirmed `handle_extract_text_facts` calls
`map_study_to_faire` *inline* (a second, independent `StandardizedValue`
writer outside the `MAP_FAIRE` task type entirely), so `exports/faire.py`'s
own root-aware read-time gate (`_entity_broadcast_is_authoritative`) is
what actually guarantees correctness regardless of when or how many times
mapping ran for a given study.

`exports/faire.py` also gained an unconditional (not settle-gated, since
it's a structural fact about entity ownership rather than a cross-study
priority judgment) analysis-only-paper exclusion: a study that links to
shared samples/runs via `EntityStudy` but is home to none of them did no
original data collection of its own, just reanalysis, and never gets a
`projectMetadata` row -- its `Study` row and `EntityStudy` links stay
untouched, and `sampleMetadata`/`experimentRunMetadata` already correctly
never emit rows for entities it doesn't home.

### Why dataset accessions are excluded from Study-identity resolution entirely

A 10-seed pressure test (Milestone 26) found that `identity/resolution.py`'s
merge/sibling-split machinery still had two more gaps beyond what
Milestones 24-25 fixed, both only reachable once more than one
independently-seeded pair was in play at once:

1. `_linked_via_discovery_lineage` (used inside `_resolve_against_one_candidate`)
   only recognizes a citing/cited relationship the *pipeline itself*
   discovered via citation-expansion -- it has nothing to say about two
   studies that were both seeded directly, with no citation relationship
   between them at all, which is exactly the "two independently seeded
   papers sharing a BioProject" case this whole feature was built around.
2. Even after excluding non-`STRUCTURED_SOURCE` `BIOPROJECT_ACCESSION`
   matches specifically, `EnaAdapter`'s own *structured* `find_related()`
   still re-confirms the identical `study_accession`/
   `secondary_study_accession` at `STRUCTURED_SOURCE` confidence the
   moment both studies independently, fully resolve the shared accession
   -- which happens for *any* two papers that legitimately reuse the same
   real dataset, not only a genuine duplicate submission. "STRUCTURED_SOURCE
   still means duplicate" turned out to be a false premise specifically
   for repository dataset accessions, even though it remains true and
   necessary for genuinely paper-identifying types.

The fix generalizes rather than patches: `_DATASET_ACCESSION_IDENTIFIER_TYPES`
(`BIOPROJECT_ACCESSION`/`BIOSAMPLE_ACCESSION`/`SRA_STUDY_ACCESSION`/
`ENA_STUDY_ACCESSION`) is excluded from Study-identity resolution
entirely, at every confidence tier -- these identify a *deposited
submission*, never the paper itself, so a match against a different Study
is never Study-identity evidence, full stop. Only genuinely
paper-identifying types (`DOI`/`PMID`/`PMCID`/`OPENALEX_ID`) still drive
merge/sibling-split; a paper accidentally seeded twice is still caught via
those, independent of this exclusion, so no real "same paper, duplicate
seed row" detection capability is lost. `DATASET_DOI`/`OBIS_DATASET_UUID`/
`GBIF_DATASET_KEY`/`BCODMO_DATASET_ID`/`PANGAEA_ID`/`NCEI_ACCESSION` (a
different adapter family entirely) were deliberately left untouched --
the same underlying risk plausibly applies to them too, but extending the
exclusion on suspicion rather than confirmed evidence would be exactly
the kind of unproven generalization this whole finding warns against;
left as a flagged, explicit follow-up for whenever those adapters get
their own pressure test.

## Task queue

See `workflow/task_queue.py` docstrings for the idempotency-key scheme and
the SQLite-vs-PostgreSQL claim-query branch. `workflow/worker.py`'s
`TASK_HANDLERS` registry is empty in Milestone 1 by design -- see
README.md's "Assumptions and placeholders" section.

## Milestone status

| Milestone | Status |
|---|---|
| 1: Foundation | Done |
| 2: Crossref / Europe PMC / OpenAlex / dedup | Done -- validated against 100 real DOIs, 100/100 resolved |
| 3: NCBI BioProject/BioSample, ENA (SRA via ENA mirror) | Done -- validated against a real 837-sample/500+-run BioProject |
| 4: Open-access text retrieval, provider-independent LLM abstraction, benchmark harness | Done -- validated against a real local Ollama server (qwen2.5:3b), found/fixed 3 real bugs |
| 5: Data-asset inventory, missingness, validators | Done -- validated against all 101 real studies through the real task queue, found/fixed 3 real bugs including a confirmed third-party OpenAlex data-quality issue |
| 6: FAIRe mapping, exports | Done -- validated against all 101 real studies (3,201 standardized values, 0 handler failures), surfaced one pre-existing Milestone 3 coverage gap (see README's "Milestone 6 validation"). |
| 6b: standards registry (FAIRe + MIOP + BeBOP crosswalk) | Done -- validated against the real vendored schemas (337 FAIRe terms, 21 MIOP terms, 222 crosswalk rows, all 6 validation checks passed; see README's "Milestone 6b validation"). |
| 6 follow-ups: FAIRe completeness, exact_mappings, BeBOP decision | Done -- see README's "Milestone 6 follow-ups". BeBOP/MIOP raw_fact mapping is a decided-out-of-scope call (paper-derived input doesn't carry protocol-document metadata), not an open question or a blocker. |
| 7: Refresh known studies, watermarks, retry/rediscovery cadences, weekly updates | Done -- validated against the real 101-study database and live Crossref/OpenAlex/Europe PMC/NCBI/ENA (found/fixed 2 real bugs: a monkeypatch-escaping import in scheduling/weekly.py, and a naive/aware datetime subtraction in is_rediscovery_due). Brand-new-study discovery via keyword search/citation expansion remains unbuilt (`DiscoveryConfig.keyword_search_enabled`/`citation_expansion_max_depth`). |
| 7 continued: schema simplification (missingness -> standardized_values, publications -> sources) | Done -- migration `0fc0bf2bf46a` applied to the real database (1,845 missingness rows + 99 publication rows carried forward, not discarded); see README's "Schema simplification". |
| 8: FAIRe-aware extraction (raw fact taxonomy, targeted prompt, benchmark harness alignment) | Done, corrected to v3 -- see README's "Milestone 8 validation" for the original real multi-model benchmark run, and "Milestone 8 follow-up" for the v2 -> v3 architecture correction (native, standard-agnostic `fact_type_candidate` names + optional `candidate_standard_fields` hints, replacing v2's direct use of FAIRe's own field spellings). |
| 9: Supplementary-material and structured-table retrieval layer | Done -- validated against the real PMC7469538 article (9 deduped supplementary files discovered from 18 raw tags; 4 XLSX deterministically parsed with no errors, 4 DOCX inventoried, 1 32MB dataset correctly kept inventory-only); found/fixed 2 real bugs (an under-sized `max_bundle_bytes` default that would have blocked this exact real bundle, and a caption-row-above-header misalignment in the table parser). See README's "Supplementary-material and structured-table retrieval layer". |
| 10: Shared resolve_or_create_study() with confidence tiering and consistency checking | Done -- validated against the real 101-study database (migration backfilled 385 pre-existing `sources` rows 1:1 into the new `study_sources` table; a full `DISCOVER_IDENTIFIERS` re-run across all 101 studies completed with zero errors, 162 new Source rows created, `study_sources` stayed 1:1 with `sources` throughout, zero `CandidateMatch` rows -- expected for an already-correctly-resolved corpus). See README's "Shared resolve_or_create_study()". |
| 11: Text-extraction speed fix (v7 -> v8): real Ollama context + collapsed checklist passes | Done -- root-caused a measured ~700-900s/section slowdown to Ollama silently ignoring `num_ctx` (capping every model at 4096 tokens) compounding with a 5-topic-focus x recall-retry split sized around that same ceiling (up to 10 sequential calls/section). Fixed via a `qwen3:4b-instruct-16k` Modelfile variant with `num_ctx` genuinely baked in, plus collapsing `extract_facts_from_section`'s default to one full-checklist pass per chunk with recall firing only on zero facts. Measured ~12-15x faster live (61.4s/case mean vs. ~700-900s/section) on the full 18-case gold benchmark, 100% JSON validity, 98.5% evidence verification. See README's "Fixing the real Ollama context limit and collapsing the per-topic passes". |
| 12: Old-model raw_fact quarantine + gold-case label-drift/completeness cleanup | Done -- `map_study_to_faire` was mapping every `RawFact` regardless of `review_status` with no `ORDER BY`, so a stale pre-v8 fact could silently outrank (or non-deterministically tie with) a fresh v8 one; fixed to exclude `REJECTED` facts and order by `created_at` explicitly. Quarantined 381 real `RawFact` rows extracted under a superseded prompt/model version (plus 149 rows + 9 Sources of untraceable provenance -- zero hits across git log/branches/dangling objects) by setting `review_status='rejected'`, then re-ran `map_study_to_faire` + missingness population for all 21 affected studies; confirmed zero `StandardizedValueEvidence` rows still cite a rejected fact. Separately re-derived `expected_facts` for all 12 real-paper gold cases against their own cached source text (~40 genuinely explicit facts added/relabeled across 10 cases, verified via byte-exact `evidence_quote` slicing + `verify_evidence_quote`), and root-caused the 2 gold cases that had scored zero extracted facts: one was a non-reproducible one-off (now reliably returns all facts), the other is a real, reproducible taxonomy gap -- the extraction checklist has zero RNA-specific fact names, so the model confidently (unchanged even at temperature 0.7) refuses to map an RNA-extraction section's facts onto DNA-prefixed field names, confirmed by a controlled RNA->DNA text substitution that immediately unlocked 10 correctly-extracted facts. See README's "Gold-case label-drift/completeness cleanup". |
| 13: Assay-entity tagging for multi-assay papers | Done -- a paper describing more than one distinct assay (e.g. separate 16S and 18S PCR protocols on the same samples) previously had every PCR/primer/qPCR fact collapse onto one broadcast `projectMetadata` row (`entity_id=None` unconditionally), silently losing the second assay's data as a "conflict" flag. Confirmed via the vendored FAIRe schema that real `projectMetadata.csv` is one row per `assay_name`, not one row per study. Fixed end to end: `extraction/text.py`'s prompt gained an optional per-fact `assay_tag` (native names scoped via new `extraction/faire_fields.assay_scoped_field_names()`), `mapping/rules.py` gained a parallel `EntityLevel.ASSAY` rule alongside every existing `EntityLevel.STUDY` one, `mapping/faire.py::_resolve_entity_id` now resolves an assay-tagged `projectMetadata` fact to its own entity_id, `exports/faire.py` emits one row per assay entity that has real values (gated on having a direct `StandardizedValue`, not mere existence, to avoid regressing the pre-existing structured-only ENA/BioProject assay-linkage path), and the previously-unwired `_materialize_candidate_entity` call was added to `handle_extract_text_facts` (factored into a new shared `_persist_candidate_facts` helper also used by the supplement-parsing path, which already had it). Verified end to end with a synthetic two-assay section through extraction -> mapping -> export, confirming two distinct, correctly-valued `projectMetadata` rows. 517 tests pass (12 new). See README's "Assay-entity tagging for multi-assay papers". |
| 14: Narrow the sample/experiment LLM checklist to what's realistically in prose | Done -- per an explicit user review of FAIRe sample/experiment fields, excluded `sample_collection_method`/`sample_storage_conditions` (+ narrative-fallback counterparts `collection_method`/`storage_conditions`) and `environmental_context` from the LLM's sampleMetadata checklist (only `coordinates`/`collection_date`/`depth` remain), and `assay_name`/`library_concentration`/`library_concentration_unit`/`library_concentration_method` from experimentRunMetadata-adjacent fields (kept `phix_percentage`/`sequencing_platform_general`). `samp_category` was deliberately never added to the taxonomy -- it varies sample-to-sample in a way prose essentially never states, structured sources only. Confirmed the rest of the experimentRunMetadata fields flagged (`pcr_plate_id`, `lib_id`, `seq_run_id`, `filename`/`filename2`, `checksum_filename`/`checksum_filename2`, `associatedSequences`, `input_read_count`, ...) were never in the LLM checklist to begin with (structured/ENA-only already). Uses the same `LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS`/`LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS` mechanism Codex's low-value-optional-field work established, applied to two more tables. 518 tests pass (1 new + 2 updated for the broadened exclusion scope). Project metadata's own checklist coverage is a separate, explicitly deferred follow-up. See README's "Narrowing the sample/experiment LLM checklist to what's realistically in prose". |
| 15: Deterministic (no-LLM) project-metadata extraction | Done -- a NOAA-specific FAIRe checklist review marked `license`/`rightsHolder`/`accessRights`/`bibliographicCitation`/`code_repo`/`recordedBy`/`recordedByID`/`project_contact` "No LLM"; none had any extraction path at all before this (confirmed zero coverage for the real validation paper, DOI `10.7717/peerj.333`). New `extraction/publication_metadata.py`: JATS `<permissions>`/`<contrib-group>` tree-structure parsing (`xml.etree.ElementTree`, filtered to `contrib-type="author"` -- confirmed live this exact paper's XML has a separate editor contrib-group that must never be conflated with `recordedBy`), a flat-text regex fallback for `code_repo`, and a pure Crossref-composed `bibliographicCitation` formatter. Confirmed live that Crossref alone is insufficient (this paper's Crossref record has `license: null` despite PeerJ's real CC-BY policy; the JATS XML has the real answer structurally). Added `expedition_id`/`ship_crs_expocode` to the vendored schema as an explicit, documented NOAA/SEUS-MBON extension (not part of upstream FAIRe v1.0.2) after confirming `classes.yaml`-only would both break `test_registry_validation_report_all_checks_pass_on_real_schemas` and silently drop the field from exports. Corrected a prior-round decision: `platform`/`instrument`/`lib_layout` are real `projectMetadata` fields (not experiment-scoped) and are now excluded from the LLM checklist too, matching this same checklist's "No LLM" marking -- already 100% ENA-covered. `pcr_0_1` needed no new work: Codex's concurrently-landed `extraction/search_flags.py::detect_text_search_flags` already covers it, and more precisely than this milestone would have (avoids a real false-positive found while researching this -- the checklist's own suggested "species-specific"/"taxon-specific" keywords appear 6 times in the validation paper in an unrelated context). Wired via new `_discover_publication_metadata_from_sources`, called inline from `handle_discover_identifiers` mirroring `_discover_supplements_from_fulltext`'s exact shape (no LLM, no new network cost, idempotency-guarded). Verified end to end against the real database: the real `projectMetadata.csv` row for the validation study now shows every expected value, with `code_repo`/`recordedByID` correctly empty (genuinely absent from that paper) and ENA-sourced fields untouched. 536 tests pass (16 new). See README's "Deterministic (no-LLM) project-metadata extraction". |
| 16: Flag-gated LLM checklist: PCR fields + a reusable, generic gating mechanism | Done -- a NOAA FAIRe checklist marked every PCR-section field conditional on deterministic boolean flags (`pcr_0_1`/`probe_based_qPCR_ddPCR_assay_0_1`, Codex's `extraction/search_flags.py`), with an explicit user instruction to build one reusable gating mechanism, not a one-off. Added `required_any_flags: frozenset[str]` to `extraction/faire_fields.py`'s `FaireExtractionField` (mirroring `search_flags.ControlledSearchField`'s own field deliberately), gating the entire "PCR / assay setup" group on `pcr_0_1`; threaded `active_flags` through `extraction/text.py` (prompt content AND `allowed_fact_types` accept-filter on both main and recall passes), both production handlers (`supplement_handlers.py` needed a real reorder, not just a parameter add -- it computed flags after the LLM call), and `llm/benchmark.py::run_case` (computed from each gold case's own text, no schema change). Added 5 genuinely missing fields (`probe_sequence`, `probe_concentration`, `assay_target_taxa`, `assay_validation`, `study_target_taxonomic_scope`) with descriptions from `schema.yaml` rather than the raw CSV (avoided both encoding artifacts and two FAIRe-camelCase-spelled proposed native names). Found a real overlap (5 LLM native names duplicating `CONTROLLED_SEARCH_FIELDS` entries) and, after checking real gold data proved the deterministic matcher isn't extraction-equivalent, gated rather than excluded them per the user's decision; added a standing collision-guard test that already caught two more collisions (`assay_type`/`biological_rep`) landing mid-task. Found and fixed a real detector bug by cross-checking every gold case's `pcr_0_1` detection against its own expected facts: `\bamplification\b` missed verb forms ("was amplified") that a real gold case uses, widened to `\bamplif\w*\b`; zero mismatches remain across all 18 gold cases after the fix. 552 tests pass (11 new + ~20 migrated to pass `active_flags` explicitly). See README's "Flag-gated LLM checklist: PCR fields + a reusable, generic gating mechanism". |
| 17: Duplicate-call audit follow-up + sample-name replicate detection | Done -- audited milestone 16's gating decisions plus `sterilise_method`/`neg_cont_0_1`/`pos_cont_0_1`/library-prep-sequencing fields for the same duplicate-mechanism failure mode; most checked out clean, but found `biological_replicate_count` had been over-gated on `pcr_0_1` (real schema shows it's unconditional -- confirmed via gold case `controls-replicates-001`, which would have silently broken had the field been excluded instead of un-gated) and `forward_sequencing_adapter`/`reverse_sequencing_adapter` exactly duplicated `search_flags.LLM_JUDGED_SEARCH_FIELDS`'s own narrower, quote-anchored `adapter_forward`/`adapter_reverse` pass -- fixed by un-gating the former and excluding the latter (same `LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS` pattern as `seq_kit`). Extended the collision-guard test to also cover `LLM_JUDGED_SEARCH_FIELDS` (previously only checked `CONTROLLED_SEARCH_FIELDS`) plus a documented `_ACCEPTED_UNCONDITIONAL_OVERLAPS` exemption for `biological_rep`'s deliberate always-on duplication. New feature: `sources/replicate_grouping.py`, a pure, shared sample-name-suffix detector (an explicit `_rep1`/`_rep2`-style marker signal, and a `_A`/`_B`/`_C` trailing-letter signal gated to >=3 consecutive-lettered members to avoid merging genuinely different sites) feeding FAIRe's real per-sample `biological_rep_relation` sampleMetadata slot; wired into `sources/ncbi.py`'s `NcbiBioSampleAdapter` (via each BioSample's `sample_name` attribute, falling back to title) and `sources/supplement_parsing.py` (via the sample-id column), with one new SAMPLE-level `MappingRule` (`review_required=True`). ENA deliberately excluded -- its `sample_accession` is an INSDC accession, never a free-text name. 571 tests pass (18 new). See README's "Duplicate-call audit follow-up, and sample-name replicate detection". |
| 18: Live-run audit against a real paper (PeerJ 10.7717/peerj.333) | Done -- investigated 6 concrete problems the user found in a full pipeline export against the real cached PMC full text. Fixed 3 real deterministic-detector false positives, each confirmed against the actual paper sentence before changing anything: `assay_type`'s bare `species-specific`/`taxon-specific` cues matched an ecological "coral species-specific cue preferences" statement with zero PCR content nearby (removed as standalone cues, docs' own Milestone 15 had already flagged this pair as unreliable for a different field in this same paper); `commercial_mm`'s bare manufacturer names (`Bio-Rad`, `Thermo Fisher`, `Applied Biosystems`, `NEB`, `KAPA`) matched a thermocycler brand mention for a paper whose PCR mix was actually custom-assembled (removed the bare brand names, kept specific product names); `biological_rep`'s bare `n = <number>` pattern matched two unrelated numbers in one paragraph (a well count and a PCR cycle count), producing a nonsensical `"4 | 17"` (removed the pattern and its now-dead context-check scaffolding). Explained two as already-correct, not bugs: `checksum_method="MD5"` is legitimately inferred from ENA's own repository metadata, not the paper's text; `code_repo` blank is schema-correct since the paper's only code mention is a supplement-file reference, not a public repository link. Investigated but did not yet find the real cause of the missing `ampliconSize`/`amplificationReactionVolume`/`annealingTemp` (corrected in Milestone 19, next). 581 tests pass (3 new regression tests built from the real paper's sentences). See README's "Live-run audit against a real paper (PeerJ 10.7717/peerj.333)". |
| 19: Fixing code_repo, commercial_mm, and the real cause of the missing PCR fields | Done -- direct follow-up after the user reviewed Milestone 18's findings. `code_repo` now falls back to capturing the whole sentence describing where code/scripts are available (e.g. "available in Supplemental Information 1") when no public-repository URL exists, rather than leaving the field blank -- sentence-splitting deliberately avoids a naive "no periods until the end" regex, since this exact real sentence has two periods embedded in filenames (`cca_rarefaction.pl`, `rarefaction_figs.R`). Added a new shared `_match_pcr_mixture_phrase` detector in `extraction/search_flags.py`: finds a PCR-mixture-description sentence (broader markers than before, matching the user's own suggestion), then classifies it as `commercial_mm` only if it also names a specific master-mix product/brand, otherwise a brand-new `custom_mm` `CONTROLLED_SEARCH_FIELDS` entry (with its own new `MappingRule`) -- correctly routes this exact paper's ExTaq/Pfu custom mixture to `custom_mm`, not `commercial_mm`. Most importantly, corrected Milestone 18's "genuine model recall limitation" conclusion: replayed the same real chunk against the live model printing the raw completion instead of just whether it parsed, and found the model was NOT returning `[]` -- it was correctly extracting facts and getting cut off mid-object at exactly `max_output_tokens=1024`, and the old all-or-nothing JSON parser discarded every correct fact along with the one that didn't finish. Confirmed the fix live: raising to `max_output_tokens=2048` with the identical, unmodified prompt produced a complete 15-fact response including all three missing fields; then ran the real `extract_facts_from_section` end to end using the codebase's own actual `config/local.yaml` defaults (`max_output_tokens: 2048`, `extraction_max_chars_per_call: 16000`, no manual overrides) and got all 17 facts correctly from a single chunk -- the audit run script's own env var overrides (`2500`/`1024`) had silently shrunk both values below the codebase's own already-tuned defaults. No prompt engineering was needed. Also added real defense-in-depth regardless of config: `llm/base.py::try_parse_json` now recovers as many complete objects as possible from a truncated top-level JSON array (via `json.JSONDecoder.raw_decode`) instead of discarding the whole response, while a truncated non-array or zero-complete-objects still correctly fails closed to the existing retry path. 595 tests pass (9 new). See README's "Fixing code_repo, commercial_mm, and the real cause of the missing PCR fields". |
| 20: An internal_study_id column for tracing rows across multi-study exports | Done -- `export_faire()` has always merged every study in the database into one shared set of output CSVs with no per-study filter; invisible with one study at a time, but the user is now running multiple papers concurrently and had no way to tell which sample/experiment rows belonged to which project row. Investigated whether "the same real sample shared across two papers" already has a home (it doesn't -- `Entity.study_id` is single-valued with a unique `(study_id, entity_level, external_identifier)` constraint, unlike `Source`'s existing `StudySource` many-to-many join table for exactly this case) and surfaced a real correctness risk directly to the user before building anything: grouping rows across studies by a shared `external_identifier` is only safe for a genuine global repository accession (NCBI/ENA), never for a paper-native label two papers could coincidentally reuse (e.g. both labeling a sample "S1"). User confirmed building the simple, always-correct version now (one `internal_study_id` value per row) and treating true cross-study entity merging as a separate future feature. Added a new `internal_study_id` column, prepended as the first column in `projectMetadata.csv`/`sampleMetadata.csv`/`experimentRunMetadata.csv`, deliberately not a real FAIRe field (not added to `classes.yaml`, excluded from `field_reference.csv`). Verified against the user's own real two-study database (`fair_ocean_audit_2_current_rerun`): both project rows carry distinct `internal_study_id`s, and all 68 combined sample/experiment rows correctly split 25/43 across the two studies. 597 tests pass (2 new). See README's "An internal_study_id column for tracing rows across multi-study exports". |
| 21: Fixing a real taxon-extraction bug and two missed-field gaps, found via two more real papers | Done -- audited two more real papers (ISME J 10.1093/ismejo/wrae013, PLOS ONE 10.1371/journal.pone.0303937). Fixed a real garbage-output bug in Codex's new `extraction/taxonomic_assay.py` (abstract/title/keyword-only `assay_target_taxa` extractor): its whole-sentence fallback (`_taxon_mentions(sentence)`, a "Capitalized word + lowercase word(s)" regex meant to catch binomial species names) fired whenever a sentence merely mentioned "PCR"/"amplicon" without an explicit target/detect/amplify phrase, producing `"Diversity studies | The device was"` for the PLOS paper from its own opening sentence fragments -- removed the fallback, keeping only the narrower target-phrase-gated capture. Separately confirmed the ISME paper's blank result was already correct (its RT-qPCR methods only ever name the marker gene "16S rRNA", never an explicit target taxon), independent of the abstract-only scoping. Added a missing `TruSeq <kit>` pattern to `_SEQUENCING_KIT_PATTERNS` (confirmed missing on the ISME paper's "TruSeq Stranded mRNA kit (Illumina)"). Added explicit "each primer"/"both primers" aggregate-value guidance to the four primer volume/concentration checklist descriptions, confirmed live to fix extraction from the ISME paper's "...1 uL of each primer" phrasing in isolation (though not consistently in the full, busier chunk -- a harder attention-budget effect, not the max_output_tokens bug from Milestone 19). The standing collision-guard test (Milestone 17) caught two more real duplicate-mechanism gaps mid-task exactly as designed -- Codex's new narrow Trimmomatic/MINLEN-specific deterministic detectors for `adapter_trimming_method`/`length_filtering_tool`/`minimum_read_length` duplicate the LLM taxonomy's own native names; checked real data (excluding the LLM versions would have lost these entirely for the PeerJ and ISME papers, which use a custom Perl script and SeqPrep respectively, not Trimmomatic) before adding all three to `_ACCEPTED_UNCONDITIONAL_OVERLAPS` alongside `biological_rep`. 607 tests pass (10 new since the last commit). See README's "Fixing a real taxon-extraction bug and two missed-field gaps, found via two more real papers". |
| 22: Disambiguating trim tools, catching a reagent-listing gap, and building out the second-PCR field family | Done -- direct follow-up after the user reviewed Milestone 21's fixes. Fixed a real context-blind bug: the deterministic Trimmomatic detector fired for both `adapter_trimming_method` and `length_filtering_tool` whenever "Trimmomatic" appeared anywhere in the text, even though the ISME paper actually uses SeqPrep for adapter removal and Trimmomatic only for quality/length trimming, described ~250 characters apart in one long compound sentence -- a whole-sentence context check wasn't tight enough to tell the two clauses apart. Rewrote as a single `_match_trim_tool(purpose=...)` function gated on context within a 90-char proximity window of each specific tool mention, and added a missing `_SEQPREP_RE` pattern so `adapter_trimming_method` can recognize SeqPrep at all. Fixed a real miss: `custom_mm` requires a "mixture"/"mix" word to trigger, but the PLOS ONE paper's real PCR-composition sentence ("...0.02 U/ul of Phusion High Fidelity DNA polymerase, 1X Phusion HF Buffer and 200 uM of dNTPs...") never uses either word -- added a second trigger (a named polymerase brand co-occurring with "buffer"/"dNTP" in the same sentence). Built out the second-PCR (`pcr2_*`) field family per the user's request: 9 new native names mirroring the first PCR's own fields one for one against the real FAIRe schema, gated on `pcr_0_1`. A live validation of the new fields caught a real hallucination bug before it shipped: the model copied this module's own example values ("22 uL", "2 uL") verbatim into `raw_value` when the real second-PCR text didn't state those quantities -- removed the examples from the four new quantity-shaped fields and added explicit "the SECOND PCR's own X only, never the first PCR's" (and the reverse, on every first-PCR counterpart) disambiguation guidance; re-verified live against a full first+second-PCR paragraph that all 18 facts now extract correctly with zero hallucinated examples. 610 tests pass (3 new). See README's "Disambiguating trim tools, catching a reagent-listing gap, and building out the second-PCR field family". |
| 23: Excluding MAG BioSamples and parsing filter/pore-size/depth out of free-text attributes | Done -- running two more real papers (10.3389/fmicb.2024.1295149, 10.1038/s42003-024-06136-2) surfaced three real problems in `sources/ncbi.py`. Fixed a real category error caught before it shipped: `samp_mat_process`'s free text (e.g. "0.22 um cartridge filtration...") was never parsed at all, and a naive parser would have written the pore size into `filter_diameter` -- checked the real schema first and found `filter_diameter` is the physical filter *disc* diameter in mm, a different concept/unit entirely; the real home for pore size is a separate, previously-unmapped field, `size_frac`. New `_derive_filter_facts` parses `size_frac`/`filter_diameter`/`filter_material`/`filter_name`/`filter_passive_active_0_1`, each gated on its own specific pattern plus nearby context, never guessed. Fixed fmicb's depth being identical across every sample despite a real depth profile (30m-5200m): no sample has a literal `depth` attribute, but the real per-sample depth is embedded inside a differently-named attribute (`source_material_id`, e.g. "3500 m V3-V4") this submitter repurposed -- new `_derive_depth_from_source_material_id` reuses the *existing* `depth` MappingRule/`to_meters` transform, no new rule needed; confirmed live that 20 distinct real depths now populate correctly. Fixed MAG (metagenome-assembled genome) BioSamples polluting sampleMetadata for s42003: confirmed live via a real efetch that a BioProject can link to both real environmental BioSamples and a family of cross-linked MAG BioSamples from an entirely different downstream project, carrying assembly-specific attributes and none of the real per-sample data -- new `_is_mag_biosample` (checks the real INSDC-standard `package="MIMAG.*"`/`Models/Model` structural marker, with a title-text fallback) now excludes these from becoming SAMPLE entities at all. Found, but did not fix (out of this task's approved scope, reported to the user instead), a more fundamental bug while verifying against s42003's real BioProject: `esearch` for `PRJNA529480` returns two UIDs, and `_esearch_first_uid` blindly takes the first one, which turned out to be a *different* project (`PRJEB73262`, a MAG-only downstream project that just mentions the real accession in its own title) -- confirmed via `esummary` that the correct UID (`529480`) has 99 real linked samples, while the wrong one (`1356142`) has only the 8 MAG records the adapter was actually fetching. Affects both `NcbiBioProjectAdapter` and `NcbiBioSampleAdapter`, and plausibly any paper where esearch returns more than one UID for an accession. 620 tests pass (8 new). See README's "Excluding MAG BioSamples and parsing filter/pore-size/depth out of free-text attributes". |
| 24: Multi-study entity sharing + citation-expansion ("node-adding") discovery | Done -- direct follow-up to Milestone 23's `_esearch_first_uid` finding, escalated by the user into a full architectural feature: real papers reuse other papers' deposited samples, and the pipeline had no way to represent that. Fixed the UID bug for real (`_esearch_verified_uid`: multi-UID esearch results now cross-checked against esummary's own `project_acc`/`accession` field, falling back to a review-flagged first-UID guess only when no candidate matches; plus an independent `biosample->bioproject` reverse-elink signal), confirmed live against PRJNA529480 *and* two more independently-ambiguous real BioProjects (PRJEB1787/Tara Oceans, PRJNA385854/bioGEOTRACES -- the SPIRE/EMG-produced TPA-reanalysis pattern polluting esearch rankings turned out to be systemic, not a one-off). New `entity_studies` join table (exact structural mirror of the existing `study_sources`) lets a SAMPLE/EXPERIMENT_RUN/SEQUENCING_RUN `Entity` belong to more than one `Study`, via a single new choke point (`identity/entity_linking.py`'s `create_entity`/`link_entity_to_study`/`get_or_create_entity`, replacing two independent, divergent `_get_or_create_entity` implementations that used to live in `workflow/handlers.py` and `extraction/experiment_runs.py`). New `DISCOVER_CITING_STUDIES` task (`bioproject->pubmed` elink, one call per accession) auto-expands a citing paper into its own full `Study` row via the existing task queue's own idempotency machinery, with real safety valves for ~3000-paper scale (`DiscoveryConfig.citation_expansion_max_depth`, wired up for the first time; new `max_citing_papers_per_bioproject`), plus a recurring node-adding backfill pass (`enqueue-citation-rediscovery-backfill`, wired into `weekly-update`) for papers published after a study's first resolution. A live end-to-end run against the real s42003 paper -- seeding it and letting `DISCOVER_IDENTIFIERS`/`DISCOVER_CITING_STUDIES` run for real against live Crossref/Europe PMC/OpenAlex/NCBI/ENA -- surfaced 3 more real bugs beyond the planned scope, none caught by mocked unit tests: (1) a citing paper's own full-text scan re-discovering the accession that citation-discovery already used to link it was treated as a study-identity conflict, spinning up an empty placeholder sibling Study + `CandidateMatch` review item for essentially every citing paper -- fixed with `_linked_via_discovery_lineage`, checked *before* confidence tier (not after), so two studies already known to be a deliberate citing/cited pair never re-litigate that relationship regardless of which adapter or confidence level reports the shared accession; (2) far more seriously, `NcbiBioSampleAdapter`/`EnaAdapter`'s `find_related()` reports every BioSample accession as a default-`STRUCTURED_SOURCE` `RelatedIdentifier`, which used to *unconditionally merge* two Study rows the moment they shared one real sample -- exactly the outcome this whole feature exists to prevent -- fixed by excluding `BIOSAMPLE_ACCESSION` from Study-identity resolution entirely (a shared BioSample is now informational, recorded via `SHARES_ACCESSION_WITH`, never identity evidence); (3) `EntityRelationship`'s existing `_get_or_create_entity_relationship` looked up prior rows scoped by `study_id`, but its own DB constraint (`uq_entity_relationship`) is global on `(from_entity_id, to_entity_id, relationship_type)` -- once two studies could legitimately resolve the same shared run/sample/experiment entities, this mismatch produced a real `UNIQUE constraint failed` the moment the second study's own ENA/NCBI resolution reached them; fixed by making the lookup global too, matching the real constraint. Final verified state: two genuinely separate `Study` rows (correct titles/DOIs/`discovery_depth`/`discovery_parent_study_id`/`discovery_root_study_id`/`discovery_trigger`), zero merges, zero `CandidateMatch` review-queue noise, 231 entities correctly linked to both studies via `entity_studies`, and `sampleMetadata.csv`'s shared-sample row showing the correct pipe-joined `internal_study_id` (`STUDY-.../STUDY-...`) with the broadcast-merge correctly suppressed (Milestone 20's `internal_study_id` column, gated per Section 5 of this feature). 645 tests pass (23 new). See README's "Multi-study entity sharing + citation-expansion discovery". |
| 25: Root-paper determination + two-phase discovery/mapping gating | Done -- direct follow-up after the user reviewed Milestone 24's live-run results and asked two sharp questions: for a shared entity, which linked study's facts should actually fill its blanks (not an arbitrary/undefined one), and shouldn't the full discovery network settle before any table-filling happens at all. Clarified a real misunderstanding first: seeded papers are NOT processed "one full network at a time" -- all ~3000 seeds get their own Study row up front, and whichever `DISCOVER_IDENTIFIERS` task the queue happens to claim first decides `Entity.study_id` ("home"), which is processing-order luck, not data authorship, for two independently-seeded papers sharing a BioProject with no citation link between them. New `entity_studies`-shaped concepts: `Entity.root_study_id`/`root_status` (`EntityRootStatus`: not_shared/pending/determined/ambiguous) is a deliberate, evidence-based answer -- earliest linked-study publication year wins outright (`workflow/validation_handlers.py`'s existing `_publication_year_for_study`, reused not duplicated), BioProject registration date used only as corroboration context on an ambiguity flag, submitter/institution matching explicitly deferred (no adapter parses structured author affiliations today) -- same/no-signal ties get flagged `NEEDS_REVIEW`, never guessed. New `identity/component.py::compute_study_component` BFS walks BOTH `EntityStudy` edges (shared entities) AND `discovery_parent_study_id`/`discovery_root_study_id` lineage edges, since neither alone is sufficient (confirmed: two independently-seeded siblings have no lineage edge at all; a freshly-created citing study has zero `EntityStudy` rows until its own discovery completes). New self-rescheduling `CHECK_COMPONENT_SETTLED` task (`workflow/settle_handlers.py`) is the barrier mechanism this flat, dependency-free task queue doesn't otherwise have -- polls (fresh idempotency key per generation, not the task's own retry/backoff) until nothing in the component is still discovering, then runs root determination and finally enqueues `MAP_FAIRE`; `enqueue_mapping_backfill` now routes any study with a shareable entity through this gate instead of enqueueing `MAP_FAIRE` directly, while a study with none proceeds immediately, unblocked. `exports/faire.py`'s broadcast gate (Milestone 24's blunt "suppress if linked to >1 study") is now root-aware: a shared entity's broadcast shows only the root study's values, never an arbitrary or every linked study's; a new, unconditional (not settle-gated, since it's a structural ownership fact) analysis-only-paper exclusion drops `projectMetadata` rows for any study that links to shared entities but is home to none of them. A design agent's exploration caught a real fact worth escalating rather than silently designing around: `handle_extract_text_facts` calls `map_study_to_faire` *inline* (Codex's concurrent work), a second `StandardizedValue` writer outside the `MAP_FAIRE` task type entirely -- confirmed this means only `exports/faire.py`'s own root-aware read-time gate can guarantee correctness regardless of when/how mapping ran; the settle-gated `MAP_FAIRE` enqueue is the efficiency/consistency layer on top, not the sole correctness mechanism. A real `AmbiguousForeignKeysError` surfaced immediately from adding `Entity.root_study_id` (a second FK from `entities` to `studies` alongside `study_id`) -- fixed with explicit `foreign_keys=` on both sides of the `Study.entities`/`Entity.study` relationship. Live end-to-end re-run of the same real s42003/PNAS pair through the full new pipeline (discovery -> settle-check drain -> root determination -> `MAP_FAIRE` -> export) produced a genuinely informative result: root determination picked the **PNAS paper (published 2020) as root**, not s42003 (published 2024, the seed) -- proving root is decided by real evidence, independent of which paper was seeded or discovered first, exactly the decoupling the user asked for. The same run naturally (not artificially) exercised the analysis-only exclusion: PNAS links to all 83 shared samples but homes none of them, so it correctly produced zero `projectMetadata` rows while `sampleMetadata.csv` still correctly listed all 83 samples once each with pipe-joined `internal_study_id`; both studies reached `entity_component_status=settled` with zero ambiguity flags. 668 tests pass (18 new). See README's "Root-paper determination and two-phase discovery/mapping gating". |
| 26: Generalizing dataset-accession exclusion from Study-identity resolution, found via a 10-seed pressure test | Done -- direct follow-up after the user asked to pressure-test the whole architecture with ~10 real, never-before-seeded papers (6 known to carry real NCBI data, including the already-validated s42003/PNAS pair, plus 4 fresh marine eDNA papers found live via Crossref search). Immediately surfaced two more real bugs, neither reachable by the single-pair tests used to validate Milestones 24-25. (1) The empty-orphan-sibling-Study bug from Milestone 24 was back, for a related but distinct case `_linked_via_discovery_lineage` doesn't cover: this batch seeded PNAS *directly* (not discovered via citation-expansion), so with no lineage relationship recorded between it and s42003 at all, its own full-text mention of `PRJNA529480` fell through to the sibling-split path again. (2) Fixing that (extending the informational-only carve-out to non-`STRUCTURED_SOURCE` `BIOPROJECT_ACCESSION` matches) immediately surfaced a deeper problem on the very next re-run: s42003 silently MERGED into the PNAS study. Root cause: `EnaAdapter`'s own *structured* `find_related()` re-confirms the identical `study_accession`/`secondary_study_accession` (`STRUCTURED_SOURCE` confidence) the moment both papers independently, fully resolve a shared accession -- which happens for *any* two papers that legitimately reuse the same real dataset, not just a genuine duplicate submission, making "STRUCTURED_SOURCE still means duplicate" a false premise for accession-level identifiers specifically. Generalized the fix into a principled rule rather than another one-off patch: `identity/resolution.py`'s new `_DATASET_ACCESSION_IDENTIFIER_TYPES` (`BIOPROJECT_ACCESSION`/`BIOSAMPLE_ACCESSION`/`SRA_STUDY_ACCESSION`/`ENA_STUDY_ACCESSION`) is now excluded from Study-identity resolution entirely, at every confidence tier -- these identify a *deposited submission*, never the paper itself, so a match against a different Study is never Study-identity evidence; only genuinely paper-identifying types (DOI/PMID/PMCID/OpenAlex ID) still drive merge/sibling-split, and still correctly catch a paper accidentally seeded twice independent of this exclusion. `DATASET_DOI`/`OBIS_DATASET_UUID`/`GBIF_DATASET_KEY`/`BCODMO_DATASET_ID`/`PANGAEA_ID`/`NCEI_ACCESSION` (a different adapter family, not pressure-tested this round) were deliberately left untouched rather than extended on unconfirmed suspicion -- flagged as a plausible but unproven risk for a future pressure-test round. Final verified state, same 10-seed batch re-run clean: 10 distinct Study rows (zero merges, zero orphan siblings, zero `CandidateMatch` noise, zero task failures), 231 entities correctly shared between s42003 and PNAS and *consistently* resolved to the same evidence-based root across all of them, both components `settled`, `sampleMetadata.csv` totaling exactly 181 rows (83+24+6+43+25, the shared 83 counted once, under their home study) with no duplication, and `projectMetadata.csv` correctly excluding PNAS (analysis-only, 231 links, 0 homed) while correctly including s42003 (83 homed samples of its own). 688 tests pass (2 new; 6 existing tests updated to exercise the general merge/consistency-check machinery via `PMID` instead of a now-exempted dataset-accession type). See README's "Generalizing dataset-accession exclusion, found via a 10-seed pressure test". |
