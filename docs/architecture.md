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
