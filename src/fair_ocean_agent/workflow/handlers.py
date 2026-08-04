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
    EntityRelationshipType,
    IdentifierType,
    InspectionLevel,
    InspectionStatus,
    ReviewStatus,
    SourceType,
    TaskType,
)
from fair_ocean_agent.database.models import Entity, EntityRelationship, ExternalIdentifier, RawFact, Source, Study, Task
from fair_ocean_agent.discovery.text_identifiers import (
    extract_repository_identifiers_from_text,
    verify_deterministic_identifier,
    xml_to_text,
)
from fair_ocean_agent.extraction.sections import select_relevant_sections
from fair_ocean_agent.extraction.search_flags import detect_controlled_search_facts, detect_text_search_flags
from fair_ocean_agent.extraction.text import (
    PROMPT_VERSION,
    extract_facts_from_section,
    resolved_faire_fields_for_study,
)
from fair_ocean_agent.identity.resolution import resolve_or_create_study
from fair_ocean_agent.identity.source_linking import create_source
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError, try_parse_json
from fair_ocean_agent.llm.factory import build_llm_backend
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.mapping.faire import map_study_to_faire
from fair_ocean_agent.sources.base import (
    RateLimitedClient,
    RawFactCandidate,
    RelatedIdentifier,
    SourceAdapter,
    SourceConfig,
    SourceRecord,
    SourceRecordNotFoundError,
    hash_payload,
)
from fair_ocean_agent.sources.bcodmo import BcoDmoAdapter
from fair_ocean_agent.sources.crossref import CrossrefAdapter
from fair_ocean_agent.sources.datacite import DataCiteAdapter
from fair_ocean_agent.sources.ena import EnaAdapter
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.sources.gbif import GbifAdapter
from fair_ocean_agent.sources.ncbi import NcbiBioProjectAdapter, NcbiBioSampleAdapter
from fair_ocean_agent.sources.obis import ObisAdapter
from fair_ocean_agent.sources.openalex import OpenAlexAdapter
from fair_ocean_agent.sources.pangaea import PangaeaAdapter
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
    "bcodmo": BcoDmoAdapter,
    "pangaea": PangaeaAdapter,
    "datacite": DataCiteAdapter,
    "obis": ObisAdapter,
    "gbif": GbifAdapter,
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

    for name in ("bcodmo", "pangaea", "datacite", "obis", "gbif"):
        if is_enabled(name):
            adapters[name] = _REPOSITORY_ADAPTER_CLASSES[name](make_config(name), retrieval_config)

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


def _get_or_create_entity_relationship(
    session: Session,
    study_id: str,
    from_entity_id: str,
    to_entity_id: str,
    relationship_type: EntityRelationshipType,
) -> EntityRelationship:
    existing = session.scalar(
        select(EntityRelationship).where(
            EntityRelationship.study_id == study_id,
            EntityRelationship.from_entity_id == from_entity_id,
            EntityRelationship.to_entity_id == to_entity_id,
            EntityRelationship.relationship_type == relationship_type.value,
        )
    )
    if existing is not None:
        return existing
    relationship = EntityRelationship(
        study_id=study_id,
        from_entity_id=from_entity_id,
        to_entity_id=to_entity_id,
        relationship_type=relationship_type.value,
    )
    session.add(relationship)
    session.flush()
    return relationship


def _materialize_candidate_entity(
    session: Session,
    study_id: str,
    fact: RawFactCandidate,
    internal_namespace: str | None = None,
) -> Entity | None:
    if not fact.entity_external_id:
        return None
    external_identifier = fact.entity_external_id
    if internal_namespace and external_identifier.startswith("internal:"):
        external_identifier = f"{external_identifier}:source:{internal_namespace}"
    entity = _get_or_create_entity(
        session,
        study_id,
        fact.entity_level,
        external_identifier,
        fact.entity_label,
    )
    for link in fact.entity_links:
        target = _get_or_create_entity(
            session,
            study_id,
            link.entity_level,
            link.external_identifier,
            link.label,
        )
        _get_or_create_entity_relationship(
            session,
            study_id,
            entity.entity_id,
            target.entity_id,
            link.relationship_type,
        )
        if link.relationship_type == EntityRelationshipType.DERIVED_FROM_SAMPLE:
            if entity.parent_entity_id is None:
                entity.parent_entity_id = target.entity_id
            elif entity.parent_entity_id != target.entity_id:
                logger.warning(
                    "entity %s is linked to conflicting sample parents %s and %s; preserving the first",
                    entity.entity_id,
                    entity.parent_entity_id,
                    target.entity_id,
                )
    return entity


def _persist_candidate_facts(
    session: Session,
    study_id: str,
    source_id: str,
    facts: list[RawFactCandidate],
    *,
    extraction_method: str,
    model_name: str | None = None,
    prompt_version: str | None = None,
    review_status: str | None = None,
    internal_namespace: str | None = None,
) -> None:
    """Shared RawFactCandidate -> RawFact persistence for both LLM-driven
    persist paths (full-paper text extraction here, supplement table/text
    parsing in supplement_handlers.py). Always materializes the fact's
    entity via _materialize_candidate_entity first, so a fact tagged with
    a real entity_external_id (e.g. extraction/text.py's per-assay
    assay_tag) gets a real Entity + entity_id rather than landing as an
    untethered study-wide fact."""
    for fact in facts:
        entity = _materialize_candidate_entity(session, study_id, fact, internal_namespace=internal_namespace)
        entity_id = entity.entity_id if entity is not None else None
        kwargs = dict(
            study_id=study_id,
            entity_id=entity_id,
            source_id=source_id,
            source_locator=fact.source_locator,
            raw_field_name=fact.raw_field_name,
            raw_value=fact.raw_value,
            evidence_quote=fact.evidence_quote,
            confidence_metadata=fact.confidence_metadata,
            fact_type_candidate=fact.fact_type_candidate,
            entity_level=fact.entity_level.value,
            support_type=fact.support_type.value,
            extraction_method=extraction_method,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        if review_status is not None:
            kwargs["review_status"] = review_status
        session.add(RawFact(**kwargs))


PersistFn = Callable[[Session, Study, SourceAdapter, SourceType, str, SourceRecord], tuple[bool, Source]]


def _persist_source_and_facts(
    session: Session,
    study: Study,
    adapter: SourceAdapter,
    source_type: SourceType,
    query_identifier: str,
    record: SourceRecord,
) -> tuple[bool, Source]:
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

    Returns (created, source): `created` is True if a new Source/facts row
    was actually created (Milestone 7's refresh path checks this to decide
    whether anything needs a fresh look; the normal discovery path ignores
    it); `source` is always the Source row for this (adapter, identifier)
    pair, new or pre-existing -- callers thread it into
    _apply_related_identifiers so identity/resolution.py's
    resolve_or_create_study() knows exactly which evidence is "new"."""
    source = (
        session.query(Source)
        .filter_by(study_id=study.study_id, source_name=adapter.name, external_identifier=query_identifier)
        .first()
    )
    already_recorded = source is not None
    same_content_source = None
    if source is None and record.content_hash:
        same_content_source = (
            session.query(Source)
            .filter_by(study_id=study.study_id, source_name=adapter.name, content_hash=record.content_hash)
            .first()
        )
        if same_content_source is not None:
            source = same_content_source
            already_recorded = True

    if source is None:
        source = create_source(
            session,
            Source(
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
            ),
        )
    elif same_content_source is not None:
        logger.info(
            "source %s for study %s already recorded with same content hash via %s; "
            "skipping duplicate facts for alias %s",
            adapter.name,
            study.study_id,
            same_content_source.external_identifier,
            query_identifier,
        )

    if source_type == SourceType.PUBLICATION_API:
        _apply_publication_fields(source, adapter.parse_publication_fields(record))

    if already_recorded:
        return False, source

    for fact in adapter.extract_structured_facts(record):
        entity = _materialize_candidate_entity(session, study.study_id, fact)
        entity_id = entity.entity_id if entity is not None else None
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
                confidence_metadata=fact.confidence_metadata,
            )
        )

    return True, source


def _apply_related_identifiers(
    session: Session, study: Study, related: list[RelatedIdentifier], source_name: str, source: Source | None = None
) -> Study:
    """Delegates entirely to resolve_or_create_study
    (identity/resolution.py) so DOI-seeded and accession-seeded discovery
    never diverge in how they reconcile a newly-discovered identifier
    against pre-existing studies. `source_name` is kept for call-site
    signature stability but is no longer read here -- every RelatedIdentifier
    already carries its own `.source` label, which resolve_or_create_study
    uses directly. `source` is the Source row this evidence came from --
    None for call sites with no single triggering Source (fulltext-mined
    identifiers, supplement discovery), which resolve_or_create_study
    degrades gracefully (see its own docstring)."""
    del source_name
    return resolve_or_create_study(session, study, related, source)


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

        _created, source = persist_fn(session, study, adapter, SourceType.PUBLICATION_API, doi, record)

        title = adapter.parse_publication_fields(record).get("title")
        if title and not study.title:
            study.title = title

        study = _apply_related_identifiers(session, study, adapter.find_related(record), name, source)

    return study


def _discover_identifiers_from_fulltext(session: Session, study: Study, adapters: dict[str, SourceAdapter]) -> Study:
    """Mine open full text for repository accessions after DOI metadata has
    discovered a PMCID.

    This deliberately does not create an `europe_pmc_fulltext` Source row:
    that row is the idempotency marker for EXTRACT_TEXT_FACTS, and identifier
    discovery must not cause later paper extraction to no-op.
    """
    europe_pmc = adapters.get("europe_pmc")
    if not isinstance(europe_pmc, EuropePmcAdapter):
        return study

    pmcid = _identifier_value(session, study.study_id, IdentifierType.PMCID)
    if pmcid is None:
        return study

    try:
        fulltext_xml = europe_pmc.fetch_fulltext_xml(pmcid)
    except SourceRecordNotFoundError:
        logger.info("no open-access full text available for identifier scan: %s", pmcid)
        return study

    related = extract_repository_identifiers_from_text(
        xml_to_text(fulltext_xml),
        source_name="europe_pmc_fulltext_identifier_scan",
    )
    verified = [r for r in related if verify_deterministic_identifier(adapters, r)]
    for dropped in related:
        if dropped not in verified:
            logger.info(
                "regex-matched %s %s could not be confirmed against its source API; discarding",
                dropped.identifier_type.value, dropped.value,
            )
    if verified:
        study = _apply_related_identifiers(
            session, study, verified, "europe_pmc_fulltext_identifier_scan", source=None
        )
    return study


def _discover_supplements_from_fulltext(session: Session, study: Study, adapters: dict[str, SourceAdapter]) -> Study:
    """Discovers <supplementary-material> references (and any externally-
    hosted repository supplement DOIs) from the same open full text used
    for identifier mining above, so a DOI-driven DISCOVER_IDENTIFIERS pass
    also surfaces supplement metadata for free.

    Deliberately mirrors _discover_identifiers_from_fulltext's own
    fetch-fulltext-again-if-needed shape rather than threading the XML
    through from that function -- fetch_fulltext_xml is disk-cached
    (RetrievalConfig.cache_enabled, on by default), so this is a cache hit,
    not a second real request.

    This only *discovers* supplements (pure XML parsing, no extra network
    call or LLM invocation) -- RETRIEVE_SUPPLEMENTS (the actual zip
    download + parsing) stays a separate, explicitly-enqueued backfill,
    matching EXTRACT_TEXT_FACTS's own opt-in convention (no task type
    auto-chains into a costlier one).

    Imports discover_supplements_for_study locally to avoid a circular
    import: supplement_handlers.py already imports several helpers
    (_apply_related_identifiers, _build_enabled_adapters, ...) from this
    module at module level.
    """
    europe_pmc = adapters.get("europe_pmc")
    if not isinstance(europe_pmc, EuropePmcAdapter):
        return study

    pmcid = _identifier_value(session, study.study_id, IdentifierType.PMCID)
    if pmcid is None:
        return study

    try:
        fulltext_xml = europe_pmc.fetch_fulltext_xml(pmcid)
    except SourceRecordNotFoundError:
        return study

    from fair_ocean_agent.workflow.supplement_handlers import discover_supplements_for_study

    discover_supplements_for_study(session, study, europe_pmc, fulltext_xml)
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
            _created, source = persist_fn(session, study, adapter, SourceType.REPOSITORY_API, bioproject_accession, record)
            study = _apply_related_identifiers(session, study, adapter.find_related(record), name, source)

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
            _created, source = persist_fn(session, study, ena_adapter, SourceType.REPOSITORY_API, ena_query_identifier, record)
            study = _apply_related_identifiers(session, study, ena_adapter.find_related(record), "ena", source)
            if not study.title:
                title = record.raw.get("study", {}).get("study_title")
                if title:
                    study.title = title

    return study


def _identifier_values(session: Session, study_id: str, identifier_type: IdentifierType) -> list[str]:
    rows = session.scalars(
        select(ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.study_id == study_id)
        .where(ExternalIdentifier.identifier_type == identifier_type.value)
        .order_by(ExternalIdentifier.created_at)
    ).all()
    return list(dict.fromkeys(rows))


def _identifier_value(session: Session, study_id: str, identifier_type: IdentifierType) -> str | None:
    values = _identifier_values(session, study_id, identifier_type)
    return values[0] if values else None


def _fetch_and_persist_repository_record(
    session: Session,
    study: Study,
    adapters: dict[str, SourceAdapter],
    adapter_name: str,
    identifier: str,
    persist_fn: PersistFn,
) -> Study:
    adapter = adapters.get(adapter_name)
    if adapter is None:
        return study
    try:
        record = adapter.fetch_record(identifier)
    except SourceRecordNotFoundError:
        logger.info("no %s record for %s", adapter_name, identifier)
        return study

    _created, source = persist_fn(session, study, adapter, SourceType.REPOSITORY_API, identifier, record)
    study = _apply_related_identifiers(session, study, adapter.find_related(record), adapter_name, source)
    if not study.title:
        title = (
            record.raw.get("title")
            or record.raw.get("name")
            or record.raw.get("headline")
            or (((record.raw.get("data") or {}).get("attributes") or {}).get("titles") or [{}])[0].get("title")
        )
        if title:
            study.title = title
    return study


def _resolve_dataset_sources(
    session: Session,
    study: Study,
    *,
    dataset_doi: str | None,
    bcodmo_id: str | None,
    pangaea_id: str | None,
    obis_uuid: str | None,
    gbif_key: str | None,
    persist_fn: PersistFn = _persist_source_and_facts,
) -> Study:
    adapters = _build_enabled_adapters()

    if dataset_doi:
        # DataCite is the generic DOI metadata authority for datasets.
        study = _fetch_and_persist_repository_record(session, study, adapters, "datacite", dataset_doi, persist_fn)
        # Repository-specific DOI prefixes are also resolved against their
        # native metadata API when available, preserving both sources.
        if "pangaea" in dataset_doi.lower():
            study = _fetch_and_persist_repository_record(session, study, adapters, "pangaea", dataset_doi, persist_fn)
        if "bco-dmo" in dataset_doi.lower() or "bcodmo" in dataset_doi.lower():
            study = _fetch_and_persist_repository_record(session, study, adapters, "bcodmo", dataset_doi, persist_fn)

    if bcodmo_id:
        study = _fetch_and_persist_repository_record(session, study, adapters, "bcodmo", bcodmo_id, persist_fn)
    if pangaea_id:
        study = _fetch_and_persist_repository_record(session, study, adapters, "pangaea", pangaea_id, persist_fn)
    if obis_uuid:
        study = _fetch_and_persist_repository_record(session, study, adapters, "obis", obis_uuid, persist_fn)
    if gbif_key:
        study = _fetch_and_persist_repository_record(session, study, adapters, "gbif", gbif_key, persist_fn)

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

    doi = _identifier_value(session, study.study_id, IdentifierType.DOI)
    dataset_dois = _identifier_values(session, study.study_id, IdentifierType.DATASET_DOI)
    bioproject_accessions = _identifier_values(session, study.study_id, IdentifierType.BIOPROJECT_ACCESSION)
    ena_accessions = _identifier_values(session, study.study_id, IdentifierType.ENA_STUDY_ACCESSION)
    sra_accessions = _identifier_values(session, study.study_id, IdentifierType.SRA_STUDY_ACCESSION)
    bcodmo_ids = _identifier_values(session, study.study_id, IdentifierType.BCODMO_DATASET_ID)
    pangaea_ids = _identifier_values(session, study.study_id, IdentifierType.PANGAEA_ID)
    obis_uuids = _identifier_values(session, study.study_id, IdentifierType.OBIS_DATASET_UUID)
    gbif_keys = _identifier_values(session, study.study_id, IdentifierType.GBIF_DATASET_KEY)

    if not any((doi, dataset_dois, bioproject_accessions, ena_accessions, sra_accessions, bcodmo_ids, pangaea_ids, obis_uuids, gbif_keys)):
        raise NotImplementedError(
            "DISCOVER_IDENTIFIERS currently resolves studies with a DOI "
            "(crossref/europe_pmc/openalex), a dataset DOI "
            "(datacite plus native repository adapters where recognized), "
            "a BioProject/ENA study accession (ncbi_bioproject/ncbi_biosample/ena), "
            "or a BCO-DMO/PANGAEA/OBIS/GBIF dataset identifier. This study has none of those."
        )

    if not _build_enabled_adapters():
        raise RuntimeError("No source adapters are enabled in config/sources.yaml")

    if doi:
        study = _resolve_publication_sources(session, study, doi)
        study = _discover_identifiers_from_fulltext(session, study, _build_enabled_adapters())
        study = _discover_supplements_from_fulltext(session, study, _build_enabled_adapters())

    dataset_dois = _identifier_values(session, study.study_id, IdentifierType.DATASET_DOI)
    bioproject_accessions = _identifier_values(session, study.study_id, IdentifierType.BIOPROJECT_ACCESSION)
    ena_accessions = _identifier_values(session, study.study_id, IdentifierType.ENA_STUDY_ACCESSION)
    sra_accessions = _identifier_values(session, study.study_id, IdentifierType.SRA_STUDY_ACCESSION)
    bcodmo_ids = _identifier_values(session, study.study_id, IdentifierType.BCODMO_DATASET_ID)
    pangaea_ids = _identifier_values(session, study.study_id, IdentifierType.PANGAEA_ID)
    obis_uuids = _identifier_values(session, study.study_id, IdentifierType.OBIS_DATASET_UUID)
    gbif_keys = _identifier_values(session, study.study_id, IdentifierType.GBIF_DATASET_KEY)

    for bioproject_accession in bioproject_accessions:
        study = _resolve_repository_sources(session, study, bioproject_accession, None)
    for sequencing_accession in [*ena_accessions, *sra_accessions]:
        study = _resolve_repository_sources(session, study, None, sequencing_accession)
    for dataset_doi in dataset_dois:
        study = _resolve_dataset_sources(
            session,
            study,
            dataset_doi=dataset_doi,
            bcodmo_id=None,
            pangaea_id=None,
            obis_uuid=None,
            gbif_key=None,
        )
    for bcodmo_id in bcodmo_ids:
        study = _resolve_dataset_sources(
            session,
            study,
            dataset_doi=None,
            bcodmo_id=bcodmo_id,
            pangaea_id=None,
            obis_uuid=None,
            gbif_key=None,
        )
    for pangaea_id in pangaea_ids:
        study = _resolve_dataset_sources(
            session,
            study,
            dataset_doi=None,
            bcodmo_id=None,
            pangaea_id=pangaea_id,
            obis_uuid=None,
            gbif_key=None,
        )
    for obis_uuid in obis_uuids:
        study = _resolve_dataset_sources(
            session,
            study,
            dataset_doi=None,
            bcodmo_id=None,
            pangaea_id=None,
            obis_uuid=obis_uuid,
            gbif_key=None,
        )
    for gbif_key in gbif_keys:
        study = _resolve_dataset_sources(
            session,
            study,
            dataset_doi=None,
            bcodmo_id=None,
            pangaea_id=None,
            obis_uuid=None,
            gbif_key=gbif_key,
        )

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

    backend = _build_llm_backend_cached()
    extraction_version = f"{PROMPT_VERSION}:{backend.label}"
    already_recorded = (
        session.query(Source)
        .filter_by(
            study_id=study.study_id,
            source_name="europe_pmc_fulltext",
            external_identifier=pmcid,
            source_version=extraction_version,
        )
        .first()
        is not None
    )
    if already_recorded:
        logger.info(
            "full text for %s already processed for study %s with %s",
            pmcid,
            study.study_id,
            extraction_version,
        )
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

    source = create_source(
        session,
        Source(
            study_id=study.study_id,
            source_type=SourceType.ARTICLE_FULLTEXT.value,
            source_name="europe_pmc_fulltext",
            external_identifier=pmcid,
            access_status=AccessStatus.OPEN.value,
            retrieved_at=utcnow(),
            content_hash=hash_payload({"pmcid": pmcid, "section_titles": [s["title"] for s in sections]}),
            source_version=extraction_version,
            inspection_status=InspectionStatus.INSPECTED.value,
            inspection_level=InspectionLevel.FULL.value,
        ),
    )

    # Rebuild mapping immediately before extraction so this ordering does
    # not depend on a caller remembering to run a separate MAP_FAIRE task.
    # FAIRe fields already resolved from structured sources
    # (NCBI/ENA/PANGAEA/...) are dropped from every section's checklist --
    # less prompt, fewer questions, no chance of the LLM contradicting an
    # already-resolved structured value.
    map_study_to_faire(session, study.study_id)
    session.flush()
    already_resolved = resolved_faire_fields_for_study(session, study.study_id)
    llm_config = load_config().llm

    section_texts = tuple((section["title"], section["text"]) for section in sections)
    locator_prefix = f"europe_pmc_fulltext:{pmcid}"
    flag_facts = detect_text_search_flags(section_texts, locator_prefix=locator_prefix)
    controlled_search_facts = detect_controlled_search_facts(
        section_texts,
        locator_prefix=locator_prefix,
        active_flags=frozenset(fact.fact_type_candidate for fact in flag_facts),
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        [*flag_facts, *controlled_search_facts],
        extraction_method="deterministic_text_search_flagging",
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )

    for section in sections:
        try:
            facts, response = extract_facts_from_section(
                backend,
                section["title"],
                section["text"],
                exclude_faire_hints=already_resolved,
                max_section_chars_per_call=llm_config.extraction_max_chars_per_call,
                max_output_tokens=llm_config.max_output_tokens,
            )
        except LLMBackendError as exc:
            logger.warning(
                "text extraction failed for study %s section %r: %s",
                study.study_id,
                section["title"],
                exc,
            )
            raise
        if response is not None and try_parse_json(response.text) is None:
            raise LLMBackendError(
                f"{backend.label}: text extraction returned invalid JSON after retries "
                f"for study {study.study_id} section {section['title']!r}"
            )
        _persist_candidate_facts(
            session,
            study.study_id,
            source.source_id,
            facts,
            extraction_method="llm_text_extraction",
            model_name=backend.label,
            prompt_version=PROMPT_VERSION,
            review_status=ReviewStatus.ACCEPTED.value,
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
