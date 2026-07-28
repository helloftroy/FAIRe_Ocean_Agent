"""Task handlers registered into workflow.worker.TASK_HANDLERS. Handlers
orchestrate adapters + persistence only -- all source-specific parsing
lives in the adapters themselves (sources/*.py), per section 6 ("Do not put
source-specific parsing logic in the central orchestrator").

Importing this module has the side effect of registering handlers; it's
imported once from cli.py before the worker runs.
"""
from __future__ import annotations

from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.config import load_config, load_sources_config
from fair_ocean_agent.database.enums import (
    AccessStatus,
    EntityLevel,
    IdentifierType,
    InspectionLevel,
    InspectionStatus,
    ReviewStatus,
    SourceType,
    TaskType,
)
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Source, Study, Task
from fair_ocean_agent.extraction.sections import select_relevant_sections
from fair_ocean_agent.extraction.text import PROMPT_VERSION, extract_facts_from_section
from fair_ocean_agent.identity.deduplication import find_existing_study_by_identifier, merge_study_into
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier
from fair_ocean_agent.llm.base import LLMBackend
from fair_ocean_agent.llm.factory import build_llm_backend
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import (
    RateLimitedClient,
    RelatedIdentifier,
    SourceAdapter,
    SourceConfig,
    SourceRecord,
    SourceRecordNotFoundError,
    hash_payload,
)
from fair_ocean_agent.sources.crossref import CrossrefAdapter
from fair_ocean_agent.sources.ena import EnaAdapter
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.sources.ncbi import NcbiBioProjectAdapter, NcbiBioSampleAdapter
from fair_ocean_agent.sources.openalex import OpenAlexAdapter
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)

# Publication-oriented (DOI-keyed, populate `sources`' bibliographic
# columns) vs repository-oriented (BioProject/ENA-accession-keyed,
# populate `entities` + study-wide raw_facts) adapters are resolved by two
# different helper functions below, since they persist differently.
_PUBLICATION_ADAPTER_CLASSES: dict[str, type[SourceAdapter]] = {
    "crossref": CrossrefAdapter,
    "europe_pmc": EuropePmcAdapter,
    "openalex": OpenAlexAdapter,
}
_REPOSITORY_ADAPTER_CLASSES: dict[str, type[SourceAdapter]] = {
    "ncbi_bioproject": NcbiBioProjectAdapter,
    "ncbi_biosample": NcbiBioSampleAdapter,
    "ena": EnaAdapter,
}

# Process-lifetime cache: rate limiting is per-adapter-instance state
# (RateLimitedClient tracks last_request_at on itself), so a worker
# processing many tasks in a loop must reuse the same adapter instances --
# rebuilding fresh ones per task would silently reset the throttle every
# task and never actually enforce config/sources.yaml's rate_limit_per_second
# across a run. Cleared via reset_adapter_cache() at worker shutdown.
_adapter_cache: dict[str, SourceAdapter] | None = None


def _build_enabled_adapters() -> dict[str, SourceAdapter]:
    global _adapter_cache
    if _adapter_cache is not None:
        return _adapter_cache

    retrieval_config = load_config().retrieval
    sources_config = load_sources_config()
    adapters: dict[str, SourceAdapter] = {}

    def is_enabled(name: str) -> bool:
        entry = sources_config.get(name)
        return entry is not None and entry.enabled

    def make_config(name: str) -> SourceConfig:
        entry = sources_config[name]
        return SourceConfig(
            name=name,
            enabled=entry.enabled,
            base_url=entry.base_url,
            rate_limit_per_second=entry.rate_limit_per_second,
            priority=entry.priority,
        )

    for name, cls in _PUBLICATION_ADAPTER_CLASSES.items():
        if is_enabled(name):
            adapters[name] = cls(make_config(name), retrieval_config)

    # ncbi_bioproject and ncbi_biosample both hit eutils.ncbi.nlm.nih.gov,
    # which enforces its per-IP rate limit across ALL eutils calls combined,
    # not per logical "source" -- two independent RateLimitedClient
    # instances would each allow their own configured rate, and together
    # could exceed NCBI's real limit. They share one client instead.
    ncbi_names = [n for n in ("ncbi_bioproject", "ncbi_biosample") if is_enabled(n)]
    if ncbi_names:
        shared_entry = sources_config[ncbi_names[0]]
        shared_http = RateLimitedClient("ncbi_eutils", retrieval_config, shared_entry.rate_limit_per_second)
        for name in ncbi_names:
            adapters[name] = _REPOSITORY_ADAPTER_CLASSES[name](make_config(name), retrieval_config, http=shared_http)

    if is_enabled("ena"):
        adapters["ena"] = EnaAdapter(make_config("ena"), retrieval_config)

    _adapter_cache = adapters
    return adapters


def reset_adapter_cache() -> None:
    """Close all cached adapters' HTTP clients and clear the cache. Call
    once at worker shutdown (cli.py does this after run_worker returns) --
    not per-task, since the whole point of the cache is surviving across
    tasks within one worker process."""
    global _adapter_cache
    if _adapter_cache:
        for adapter in _adapter_cache.values():
            adapter.close()
    _adapter_cache = None


# Same rationale as _adapter_cache: reused across tasks within one worker
# process so the LLM backend's HTTP client isn't rebuilt per task.
_llm_backend_cache: LLMBackend | None = None


def _build_llm_backend_cached() -> LLMBackend:
    global _llm_backend_cache
    if _llm_backend_cache is None:
        _llm_backend_cache = build_llm_backend(load_config().llm)
    return _llm_backend_cache


def reset_llm_backend_cache() -> None:
    """Close the cached LLM backend and clear the cache. Call once at
    worker shutdown, same as reset_adapter_cache()."""
    global _llm_backend_cache
    if _llm_backend_cache is not None:
        _llm_backend_cache.close()
    _llm_backend_cache = None


def _apply_publication_fields(source: Source, fields: dict) -> None:
    """Sets this one adapter's own bibliographic fields directly onto its
    own Source row -- deliberately no cross-adapter merge (see Source's
    class docstring for why). A consumer wanting one merged answer across
    a study's publication-type sources queries for it directly instead of
    reading a separately-synced aggregate."""
    if fields.get("authors"):
        source.authors = fields["authors"]
    if fields.get("journal"):
        source.journal = fields["journal"]
    if fields.get("publication_year"):
        source.publication_year = fields["publication_year"]
    if fields.get("fulltext_available"):
        source.fulltext_available = fields["fulltext_available"]
    if fields.get("open_access_status") == AccessStatus.OPEN.value:
        source.access_status = AccessStatus.OPEN.value


def _get_or_create_entity(
    session: Session,
    study_id: str,
    entity_level: EntityLevel,
    external_identifier: str,
    label: str | None,
) -> Entity:
    existing = (
        session.query(Entity)
        .filter_by(study_id=study_id, entity_level=entity_level.value, external_identifier=external_identifier)
        .one_or_none()
    )
    if existing is not None:
        if label and not existing.label:
            existing.label = label
        return existing

    entity = Entity(
        study_id=study_id,
        entity_level=entity_level.value,
        external_identifier=external_identifier,
        label=label,
    )
    session.add(entity)
    session.flush()
    return entity


PersistFn = Callable[[Session, Study, SourceAdapter, SourceType, str, SourceRecord], bool]


def _persist_source_and_facts(
    session: Session,
    study: Study,
    adapter: SourceAdapter,
    source_type: SourceType,
    query_identifier: str,
    record: SourceRecord,
) -> bool:
    """Idempotency guard: skip RawFact/Entity creation if this exact
    (study, adapter, identifier) combination was already recorded by a prior
    attempt at this task. fetch_record() itself is safe (and cheap) to
    repeat regardless -- the on-disk response cache makes it a non-network
    hit -- so callers still use its result for related-identifier discovery
    even when fact persistence here is a no-op.

    For publication-type sources, this always applies this adapter's own
    bibliographic fields (authors/journal/publication_year/
    fulltext_available/access_status) onto its own Source row, whether that
    row is newly created or already existed -- these are plain scalar
    columns on one row per adapter, not a growing fact list, so re-setting
    them on retry carries none of the duplication risk the fact-persistence
    guard exists to prevent.

    Returns True if a new Source/facts row was actually created (Milestone
    7's refresh path checks this to decide whether anything needs a fresh
    look; the normal discovery path ignores the return value)."""
    source = (
        session.query(Source)
        .filter_by(study_id=study.study_id, source_name=adapter.name, external_identifier=query_identifier)
        .first()
    )
    already_recorded = source is not None

    if source is None:
        source = Source(
            study_id=study.study_id,
            source_type=source_type.value,
            source_name=adapter.name,
            external_identifier=query_identifier,
            url=record.url,
            access_status=AccessStatus.UNKNOWN.value,
            retrieved_at=record.retrieved_at,
            content_hash=record.content_hash,
            inspection_status=InspectionStatus.INSPECTED.value,
            inspection_level=InspectionLevel.METADATA_ONLY.value,
        )
        session.add(source)
        session.flush()

    if source_type == SourceType.PUBLICATION_API:
        _apply_publication_fields(source, adapter.parse_publication_fields(record))

    if already_recorded:
        return False

    for fact in adapter.extract_structured_facts(record):
        entity_id = None
        if fact.entity_external_id:
            entity_id = _get_or_create_entity(
                session, study.study_id, fact.entity_level, fact.entity_external_id, fact.entity_label
            ).entity_id
        session.add(
            RawFact(
                study_id=study.study_id,
                entity_id=entity_id,
                source_id=source.source_id,
                source_locator=fact.source_locator,
                raw_field_name=fact.raw_field_name,
                raw_value=fact.raw_value,
                fact_type_candidate=fact.fact_type_candidate,
                entity_level=fact.entity_level.value,
                support_type=fact.support_type.value,
                extraction_method=f"adapter:{adapter.name}",
                review_status=ReviewStatus.ACCEPTED.value,
            )
        )

    return True


def _apply_related_identifiers(
    session: Session, study: Study, related: list[RelatedIdentifier], source_name: str
) -> Study:
    """Adds newly-discovered identifiers to the study, performing a Stage 2
    (explicit-relationship) merge if one of them already belongs to a
    *different* canonical study. Returns the study to keep using (may be a
    different object than the one passed in, if a merge happened).

    Existence is checked via a fresh DB query (find_existing_study_by_identifier)
    rather than the `study.external_identifiers` ORM collection, which can go
    stale mid-task: multiple adapters commonly discover the *same* related
    identifier in one task (e.g. ncbi_biosample and ena both surface the same
    BioSample accessions for a study), and rows added via session.add() in an
    earlier iteration of this same loop -- or by a different adapter earlier
    in _resolve_repository_sources -- don't retroactively appear in an
    already-loaded relationship collection, so a naive "is it in
    study.external_identifiers" check would re-attempt the same insert and
    hit the table's unique constraint.
    """
    for rel in related:
        try:
            normalized_value = normalize_identifier(rel.identifier_type, rel.value)
        except IdentifierError:
            continue

        existing_study = find_existing_study_by_identifier(session, rel.identifier_type, normalized_value)
        if existing_study is None:
            session.add(
                ExternalIdentifier(
                    study_id=study.study_id,
                    identifier_type=rel.identifier_type.value,
                    identifier_value=normalized_value,
                    source=source_name,
                    verified=True,
                )
            )
            session.flush()
        elif existing_study.study_id != study.study_id:
            study = merge_study_into(session, absorb=existing_study, into=study)
        # else: existing_study.study_id == study.study_id -- already recorded, nothing to do.
    return study


def _resolve_publication_sources(
    session: Session, study: Study, doi: str, persist_fn: PersistFn = _persist_source_and_facts
) -> Study:
    """`persist_fn` defaults to the normal skip-if-already-recorded
    behavior; Milestone 7's refresh path (workflow/refresh_handlers.py)
    passes in a diff-aware variant instead, reusing all of this function's
    adapter-traversal/related-identifier logic rather than duplicating it.

    `study.title` is the one piece of bibliographic data that's still
    merged across adapters here (first adapter to report a title wins,
    same as always) -- everything else each adapter reports
    (authors/journal/publication_year/fulltext_available/access_status)
    goes straight onto that adapter's own Source row inside `persist_fn`,
    no merge (see Source's class docstring)."""
    adapters = _build_enabled_adapters()

    for name in _PUBLICATION_ADAPTER_CLASSES:
        adapter = adapters.get(name)
        if adapter is None:
            continue
        try:
            record = adapter.fetch_record(doi)
        except SourceRecordNotFoundError:
            logger.info("no %s record for DOI %s", name, doi)
            continue

        persist_fn(session, study, adapter, SourceType.PUBLICATION_API, doi, record)

        title = adapter.parse_publication_fields(record).get("title")
        if title and not study.title:
            study.title = title

        study = _apply_related_identifiers(session, study, adapter.find_related(record), name)

    return study


def _resolve_repository_sources(
    session: Session,
    study: Study,
    bioproject_accession: str | None,
    ena_accession: str | None,
    persist_fn: PersistFn = _persist_source_and_facts,
) -> Study:
    adapters = _build_enabled_adapters()
    any_repository_adapter_enabled = any(adapters.get(n) for n in _REPOSITORY_ADAPTER_CLASSES)
    if not any_repository_adapter_enabled:
        logger.info(
            "study %s has a BioProject/ENA accession but no repository adapter "
            "(ncbi_bioproject/ncbi_biosample/ena) is enabled in config/sources.yaml",
            study.study_id,
        )
        return study

    if bioproject_accession:
        for name in ("ncbi_bioproject", "ncbi_biosample"):
            adapter = adapters.get(name)
            if adapter is None:
                continue
            try:
                record = adapter.fetch_record(bioproject_accession)
            except SourceRecordNotFoundError:
                logger.info("no %s record for %s", name, bioproject_accession)
                continue
            persist_fn(session, study, adapter, SourceType.REPOSITORY_API, bioproject_accession, record)
            study = _apply_related_identifiers(session, study, adapter.find_related(record), name)

            # Repository-only studies (no DOI, so _resolve_publication_sources
            # never runs) would otherwise never get a study.title at all --
            # read it directly off the BioProject record rather than adding a
            # parse_publication_fields-style method for a single field.
            if name == "ncbi_bioproject" and not study.title:
                title = record.raw.get("title")
                if title:
                    study.title = title

    ena_adapter = adapters.get("ena")
    ena_query_identifier = bioproject_accession or ena_accession
    if ena_adapter is not None and ena_query_identifier is not None:
        try:
            record = ena_adapter.fetch_record(ena_query_identifier)
        except SourceRecordNotFoundError:
            logger.info("no ena record for %s", ena_query_identifier)
        else:
            persist_fn(session, study, ena_adapter, SourceType.REPOSITORY_API, ena_query_identifier, record)
            study = _apply_related_identifiers(session, study, ena_adapter.find_related(record), "ena")
            if not study.title:
                title = record.raw.get("study", {}).get("study_title")
                if title:
                    study.title = title

    return study


def handle_discover_identifiers(session: Session, task: Task) -> None:
    """Resolves whichever of DOI / BioProject accession / ENA study
    accession the study already has:

    - DOI -> Crossref/Europe PMC/OpenAlex (bibliographic Source rows + study-level facts).
    - BioProject/ENA accession -> NCBI BioProject/BioSample + ENA (Entity
      rows per sample/run + entity-level facts), with a Stage 2 merge if a
      newly-discovered identifier already belongs to a different study.

    A study can have both and gets both resolved in one task. Raises
    NotImplementedError only if none of these identifiers are present at
    all (e.g. OBIS/GBIF/BCO-DMO/PANGAEA-only studies -- later milestones).
    """
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    def identifier_value(identifier_type: IdentifierType) -> str | None:
        ei = next((e for e in study.external_identifiers if e.identifier_type == identifier_type.value), None)
        return ei.identifier_value if ei else None

    doi = identifier_value(IdentifierType.DOI)
    bioproject_accession = identifier_value(IdentifierType.BIOPROJECT_ACCESSION)
    ena_accession = identifier_value(IdentifierType.ENA_STUDY_ACCESSION)

    if not any((doi, bioproject_accession, ena_accession)):
        raise NotImplementedError(
            "DISCOVER_IDENTIFIERS currently resolves studies with a DOI "
            "(crossref/europe_pmc/openalex) or a BioProject/ENA study accession "
            "(ncbi_bioproject/ncbi_biosample/ena). This study has none of those -- "
            "OBIS/GBIF/BCO-DMO/PANGAEA-only resolution is a later milestone."
        )

    if not _build_enabled_adapters():
        raise RuntimeError("No source adapters are enabled in config/sources.yaml")

    if doi:
        study = _resolve_publication_sources(session, study, doi)
    if bioproject_accession or ena_accession:
        study = _resolve_repository_sources(session, study, bioproject_accession, ena_accession)

    session.flush()


def handle_extract_text_facts(session: Session, task: Task) -> None:
    """Open-access full-text retrieval + deterministic section selection +
    LLM-based fact extraction (Milestone 4, section 10/12). Requires a
    PMCID (discovered via Europe PMC during a prior DISCOVER_IDENTIFIERS
    run) and a full text available in Europe PMC -- neither a paywalled
    source nor any full-text source besides Europe PMC is ever consulted.

    With the default config (llm.enabled: false), building the LLM backend
    returns a DisabledLLMBackend, and generation raises immediately -- this
    task then fails/retries/lands in manual_review_required exactly like
    any other not-yet-configured capability, rather than silently no-oping.
    """
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    pmcid_identifier = next(
        (ei for ei in study.external_identifiers if ei.identifier_type == IdentifierType.PMCID.value), None
    )
    if pmcid_identifier is None:
        raise NotImplementedError(
            "EXTRACT_TEXT_FACTS requires a PMCID (discovered via Europe PMC during "
            "DISCOVER_IDENTIFIERS) to fetch open-access full text -- this study has none."
        )
    pmcid = pmcid_identifier.identifier_value

    europe_pmc = _build_enabled_adapters().get("europe_pmc")
    if europe_pmc is None:
        raise RuntimeError("europe_pmc adapter is not enabled in config/sources.yaml")

    already_recorded = (
        session.query(Source)
        .filter_by(study_id=study.study_id, source_name="europe_pmc_fulltext", external_identifier=pmcid)
        .first()
        is not None
    )
    if already_recorded:
        logger.info("full text for %s already processed for study %s", pmcid, study.study_id)
        return

    try:
        fulltext_xml = europe_pmc.fetch_fulltext_xml(pmcid)
    except SourceRecordNotFoundError:
        logger.info("no open-access full text available for %s", pmcid)
        return

    sections = select_relevant_sections(fulltext_xml)
    if not sections:
        logger.info("no relevant sections found in full text for %s", pmcid)
        return

    backend = _build_llm_backend_cached()

    source = Source(
        study_id=study.study_id,
        source_type=SourceType.ARTICLE_FULLTEXT.value,
        source_name="europe_pmc_fulltext",
        external_identifier=pmcid,
        access_status=AccessStatus.OPEN.value,
        retrieved_at=utcnow(),
        content_hash=hash_payload({"pmcid": pmcid, "section_titles": [s["title"] for s in sections]}),
        inspection_status=InspectionStatus.INSPECTED.value,
        inspection_level=InspectionLevel.FULL.value,
    )
    session.add(source)
    session.flush()

    for section in sections:
        facts, _response = extract_facts_from_section(backend, section["title"], section["text"])
        for fact in facts:
            session.add(
                RawFact(
                    study_id=study.study_id,
                    source_id=source.source_id,
                    source_locator=fact.source_locator,
                    raw_field_name=fact.raw_field_name,
                    raw_value=fact.raw_value,
                    evidence_quote=fact.evidence_quote,
                    fact_type_candidate=fact.fact_type_candidate,
                    entity_level=fact.entity_level.value,
                    support_type=fact.support_type.value,
                    extraction_method="llm_text_extraction",
                    model_name=backend.label,
                    prompt_version=PROMPT_VERSION,
                    review_status=ReviewStatus.ACCEPTED.value,
                )
            )

    session.flush()


def enqueue_text_extraction_backfill(session: Session) -> int:
    """Queue an EXTRACT_TEXT_FACTS task for every study that has a PMCID
    (discovered via a prior DISCOVER_IDENTIFIERS run) and doesn't already
    have one (idempotent via enqueue_task's default idempotency key). This
    is deliberately a separate, explicitly-triggered step -- not
    auto-chained from DISCOVER_IDENTIFIERS -- so enabling text extraction
    stays opt-in: with llm.enabled: false (the default), nothing about
    running the seed pipeline suddenly starts creating tasks that are
    guaranteed to fail against a disabled LLM."""
    from fair_ocean_agent.workflow.task_queue import enqueue_task

    study_ids = (
        session.execute(
            select(ExternalIdentifier.study_id).where(
                ExternalIdentifier.identifier_type == IdentifierType.PMCID.value
            )
        )
        .scalars()
        .all()
    )
    for study_id in study_ids:
        enqueue_task(session, TaskType.EXTRACT_TEXT_FACTS, study_id=study_id)
    return len(study_ids)


TASK_HANDLERS[TaskType.DISCOVER_IDENTIFIERS] = handle_discover_identifiers
TASK_HANDLERS[TaskType.EXTRACT_TEXT_FACTS] = handle_extract_text_facts
