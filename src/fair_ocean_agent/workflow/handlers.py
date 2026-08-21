"""Task handlers registered into workflow.worker.TASK_HANDLERS. Handlers
orchestrate adapters + persistence only -- all source-specific parsing
lives in the adapters themselves (sources/*.py), per section 6 ("Do not put
source-specific parsing logic in the central orchestrator").

Importing this module has the side effect of registering handlers; it's
imported once from cli.py before the worker runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.config import MIN_LLM_MAX_OUTPUT_TOKENS, load_config, load_sources_config
from fair_ocean_agent.database.enums import (
    AccessStatus,
    CanonicalStatus,
    ComponentStatus,
    DataAvailabilityStatus,
    EntityLevel,
    EntityRelationshipType,
    IdentifierType,
    InspectionLevel,
    InspectionStatus,
    ReviewStatus,
    SourceType,
    SupportType,
    TaskType,
)
from fair_ocean_agent.database.models import (
    Entity,
    ExternalIdentifier,
    RawFact,
    Source,
    Study,
    Task,
)
from fair_ocean_agent.discovery.text_identifiers import (
    extract_dataset_repository_identifiers_from_text,
    extract_repository_identifiers_from_text,
    resolve_sra_run_accessions_to_studies,
    resolve_sra_sample_accessions_to_studies,
    verify_deterministic_identifier,
    xml_to_text,
)
from fair_ocean_agent.extraction.sections import select_relevant_sections
from fair_ocean_agent.extraction.api_verification import detect_and_correct_elev_depth_mislabeling
from fair_ocean_agent.extraction.downstream_analysis import detect_downstream_analysis_techniques
from fair_ocean_agent.extraction.faire_fields import suppress_resolved_faire_hints_for_text
from fair_ocean_agent.extraction.pdf import extract_pdf_sections, extract_pdf_text
from fair_ocean_agent.extraction.publication_metadata import (
    extract_primer_reference_citations,
    extract_publication_metadata_facts,
    generate_funding_source,
    generate_rights_holder,
)
from fair_ocean_agent.extraction.search_flags import (
    detect_controlled_search_facts,
    detect_llm_judged_search_facts,
    detect_phix_percentage_facts,
    detect_text_search_flags,
)
from fair_ocean_agent.extraction.section_categories import (
    derive_pcr_0_1_from_category_detection,
    detect_section_categories_present,
)
from fair_ocean_agent.extraction.section_category_extraction import (
    extract_pulled_env_var_facts,
    extract_section_category_facts,
    normalize_controlled_sample_prep_facts,
)
from fair_ocean_agent.extraction.study_factor import (
    generate_study_factor,
    generate_study_target_taxonomic_scope,
)
from fair_ocean_agent.extraction.text import (
    PROMPT_VERSION,
    extract_facts_from_section,
    resolved_faire_fields_for_study,
)
from fair_ocean_agent.identity.deduplication import find_existing_study_by_identifier
from fair_ocean_agent.identity.entity_linking import get_or_create_entity as _get_or_create_entity
from fair_ocean_agent.identity.entity_linking import (
    get_or_create_entity_relationship as _get_or_create_entity_relationship,
)
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier
from fair_ocean_agent.identity.resolution import resolve_or_create_study
from fair_ocean_agent.identity.source_linking import create_source
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError, try_parse_json
from fair_ocean_agent.llm.factory import build_llm_backend
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.mapping.faire import map_study_to_faire
from fair_ocean_agent.mapping.primer_library import PRIMER_NAME_TO_SEQUENCE_FIELD, corpus_primer_sequence, study_primer_name
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
from fair_ocean_agent.sources.dryad import DryadAdapter
from fair_ocean_agent.sources.ena import EnaAdapter
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.sources.figshare import FigshareAdapter
from fair_ocean_agent.sources.gbif import GbifAdapter
from fair_ocean_agent.sources.ncbi import (
    NcbiBioProjectAdapter,
    NcbiBioSampleAdapter,
    _elink_ids,
    _esearch_verified_uid,
    _esummary_records,
)
from fair_ocean_agent.sources.obis import ObisAdapter
from fair_ocean_agent.sources.openalex import OpenAlexAdapter
from fair_ocean_agent.sources.osf import OsfAdapter
from fair_ocean_agent.sources.pangaea import PangaeaAdapter
from fair_ocean_agent.sources.sequence_file_heuristics import SequenceDataStatus
from fair_ocean_agent.sources.zenodo import ZenodoAdapter
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)

LOCAL_PDF_PATH_ENV = "FAIR_OCEAN_LOCAL_PDF_PATH"

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
    "zenodo": ZenodoAdapter,
    "dryad": DryadAdapter,
    "figshare": FigshareAdapter,
    "osf": OsfAdapter,
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

    for name in ("bcodmo", "pangaea", "datacite", "obis", "gbif", "zenodo", "dryad", "figshare", "osf"):
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


# pcr_primer_reference_forward/reverse -> the sequence field a corpus
# lookup would need to already have before there's no point chasing the
# citation further -- see corpus_primer_sequence.
_PRIMER_REFERENCE_FIELD_TO_SEQUENCE_FIELD = {
    "pcr_primer_reference_forward": "pcr_primer_forward",
    "pcr_primer_reference_reverse": "pcr_primer_reverse",
}


def _resolve_and_seed_primer_references(
    session: Session,
    study: Study,
    source: Source,
    fulltext_xml: str | None,
    *,
    locator_prefix: str,
) -> None:
    """Resolves a primer's own bibliographic citation to a real DOI (see
    extraction/publication_metadata.py::extract_primer_reference_citations)
    when this paper names a primer without giving its own sequence, and
    queues the referenced paper for the normal discovery pipeline to seed
    (handle_discover_primer_reference_study below decides whether it's
    actually worth chasing -- corpus already knowing the sequence, an
    already-known Study for that DOI, and the depth cap are all checked
    there, at task-processing time, not here, since the corpus can change
    between now and whenever that task actually runs). Per an explicit
    user request: "put it next in the queue of papers to seed... go all
    levels deep." No-op for the local-PDF path (fulltext_xml is None there
    -- no structured bibliography to resolve against)."""
    if fulltext_xml is None:
        return
    primer_names = {
        name_field: study_primer_name(session, study.study_id, name_field)
        for name_field in PRIMER_NAME_TO_SEQUENCE_FIELD
    }
    if not any(primer_names.values()):
        return
    reference_facts = extract_primer_reference_citations(fulltext_xml, primer_names, locator_prefix=locator_prefix)
    if not reference_facts:
        return
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        reference_facts,
        extraction_method="primer_reference_citation",
        review_status=ReviewStatus.ACCEPTED.value,
    )
    session.flush()

    from fair_ocean_agent.workflow.task_queue import enqueue_task

    for fact in reference_facts:
        if not fact.raw_value.startswith("doi: "):
            continue  # only a fallback reference title, nothing resolvable to chase
        sequence_field = _PRIMER_REFERENCE_FIELD_TO_SEQUENCE_FIELD.get(fact.fact_type_candidate)
        primer_name = (fact.confidence_metadata or {}).get("primer_name")
        if sequence_field is None or not primer_name:
            continue
        doi = fact.raw_value[len("doi: ") :]
        try:
            normalized_doi = normalize_identifier(IdentifierType.DOI, doi)
        except IdentifierError:
            continue
        enqueue_task(
            session,
            TaskType.DISCOVER_PRIMER_REFERENCE_STUDIES,
            study_id=study.study_id,
            payload={"doi": normalized_doi, "primer_name": primer_name, "sequence_field": sequence_field},
            idempotency_key=f"DISCOVER_PRIMER_REFERENCE:{normalized_doi}",
        )


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
                review_status=fact.review_status or ReviewStatus.ACCEPTED.value,
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
    discovered a PMCID -- or, when FAIR_OCEAN_LOCAL_PDF_PATH is set, from a
    locally-supplied PDF instead. A paper with no PMCID at all (never
    deposited in PMC, even when freely readable elsewhere -- see
    handle_extract_text_facts' own docstring) previously had NO route to
    ever surface its BioProject/BioSample accessions: this function
    required a PMCID unconditionally, same latent gap as EXTRACT_TEXT_FACTS
    had before it also learned to prefer a local PDF.

    This deliberately does not create an `europe_pmc_fulltext` Source row:
    that row is the idempotency marker for EXTRACT_TEXT_FACTS, and identifier
    discovery must not cause later paper extraction to no-op.
    """
    local_pdf_path = os.environ.get(LOCAL_PDF_PATH_ENV)
    if local_pdf_path:
        pdf_path = Path(local_pdf_path)
        if not pdf_path.exists():
            raise SourceRecordNotFoundError(f"local PDF does not exist: {pdf_path}")
        text = extract_pdf_text(pdf_path)
        source_name = "local_pdf_identifier_scan"
    else:
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
        text = xml_to_text(fulltext_xml)
        source_name = "europe_pmc_fulltext_identifier_scan"

    # Pass 1: deterministic accessions (BioProject, SRA/ENA study-level,
    # and sample-/run-level resolved up to their parent study). Kept
    # unconditional -- cheap, deterministic, already the existing behavior.
    related = extract_repository_identifiers_from_text(text, source_name=source_name)
    # SRA/ENA/DDBJ *sample*- and *run*-level accessions (SRS/ERS/DRS,
    # SRR/ERR/DRR, often cited as a range) are common in Data Availability
    # statements but aren't a repository identifier this pipeline stores
    # directly -- resolved here to whichever real parent study/BioProject
    # accession they belong to instead (confirmed live: 10.1073/pnas.2103275118
    # cites only "SRS7105074 - SRS7105095" and never states its own study
    # accession).
    related += resolve_sra_sample_accessions_to_studies(adapters, text, source_name=source_name)
    related += resolve_sra_run_accessions_to_studies(adapters, text, source_name=source_name)
    verified = [r for r in related if verify_deterministic_identifier(adapters, r)]

    # Pass 2: general-purpose dataset repositories (Zenodo/Dryad/Figshare/
    # OSF) -- only attempted once Pass 1 has come up empty, per an explicit
    # user request for a staged search rather than always querying every
    # repository. Real motivating cases where Pass 1 finds nothing at all:
    # 10.1038/s41598-024-60762-8 (Zenodo DOI only) and 10.1002/edn3.184
    # (Dryad dataset URL only).
    if not verified:
        pass2_related = extract_dataset_repository_identifiers_from_text(text, source_name=source_name)
        pass2_verified = [r for r in pass2_related if verify_deterministic_identifier(adapters, r)]
        related += pass2_related
        verified += pass2_verified

        # Pass 3: DataCite's own relatedIdentifiers for the ARTICLE's own
        # DOI (not a dataset DOI already found in text) -- a structurally
        # different, structured-API discovery channel that can surface a
        # dataset the paper's own text never names at all. Only attempted
        # once Pass 2 has also come up empty.
        if not pass2_verified:
            article_doi = _identifier_value(session, study.study_id, IdentifierType.DOI)
            datacite = adapters.get("datacite")
            if article_doi and isinstance(datacite, DataCiteAdapter):
                pass3_related = [
                    RelatedIdentifier(
                        identifier_type=IdentifierType.DATASET_DOI,
                        value=dataset_doi,
                        relationship_type=RelationshipType.IS_DATASET_FOR,
                        source="datacite_related_identifiers",
                        confidence=SupportType.STRUCTURED_SOURCE,
                    )
                    for dataset_doi in datacite.find_datasets_citing(article_doi)
                ]
                pass3_verified = [r for r in pass3_related if verify_deterministic_identifier(adapters, r)]
                related += pass3_related
                verified += pass3_verified

    for dropped in related:
        if dropped not in verified:
            logger.info(
                "regex-matched %s %s could not be confirmed against its source API; discarding",
                dropped.identifier_type.value, dropped.value,
            )
    if verified:
        study = _apply_related_identifiers(session, study, verified, source_name, source=None)
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


def _discover_publication_metadata_from_sources(
    session: Session, study: Study, adapters: dict[str, SourceAdapter], doi: str
) -> Study:
    """Deterministic (no-LLM) project-metadata extraction
    (extraction/publication_metadata.py) from the same open full text used
    for identifier/supplement mining above, plus a disk-cached re-fetch of
    the Crossref record, so a DOI-driven DISCOVER_IDENTIFIERS pass also
    resolves license/accessRights/recordedBy/project_contact/
    bibliographicCitation/code_repo for free.

    `rightsHolder` now lives in the full-text LLM stage because real
    rights sections vary between authors, "The Author(s)", journals,
    societies, and publishers, often with the year as part of the holder
    phrase.

    Deliberately mirrors _discover_supplements_from_fulltext's shape: pure
    parsing of already-fetched/disk-cached text, no new network cost, no
    LLM -- so, like that function, there's no cost reason to keep this
    manual-only. Idempotency-guarded the same way _persist_source_and_facts
    is (an early existence check on this function's own dedicated
    `source_name`), checked before the (cheap but non-zero) re-fetch/parse
    work rather than after.
    """
    already_recorded = (
        session.query(Source.source_id)
        .filter_by(study_id=study.study_id, source_name="publication_metadata_extraction", external_identifier=doi)
        .first()
        is not None
    )
    if already_recorded:
        return study

    europe_pmc = adapters.get("europe_pmc")
    fulltext_xml: str | None = None
    if isinstance(europe_pmc, EuropePmcAdapter):
        pmcid = _identifier_value(session, study.study_id, IdentifierType.PMCID)
        if pmcid is not None:
            try:
                fulltext_xml = europe_pmc.fetch_fulltext_xml(pmcid)
            except SourceRecordNotFoundError:
                fulltext_xml = None

    crossref_raw: dict | None = None
    crossref = adapters.get("crossref")
    if crossref is not None:
        try:
            crossref_raw = crossref.fetch_record(doi).raw
        except SourceRecordNotFoundError:
            crossref_raw = None

    locator_prefix = f"publication_metadata:{doi}"
    facts = extract_publication_metadata_facts(fulltext_xml, crossref_raw, locator_prefix=locator_prefix)
    if not facts:
        return study

    source = create_source(
        session,
        Source(
            study_id=study.study_id,
            source_type=SourceType.PUBLICATION_API.value,
            source_name="publication_metadata_extraction",
            external_identifier=doi,
            retrieved_at=utcnow(),
            content_hash=hash_payload({"doi": doi, "fact_count": len(facts)}),
            inspection_status=InspectionStatus.INSPECTED.value,
            inspection_level=InspectionLevel.FULL.value,
        ),
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        facts,
        extraction_method="deterministic_publication_metadata",
    )
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
        if "zenodo" in dataset_doi.lower():
            study = _fetch_and_persist_repository_record(session, study, adapters, "zenodo", dataset_doi, persist_fn)
        if "dryad" in dataset_doi.lower():
            study = _fetch_and_persist_repository_record(session, study, adapters, "dryad", dataset_doi, persist_fn)
        if "figshare" in dataset_doi.lower():
            study = _fetch_and_persist_repository_record(session, study, adapters, "figshare", dataset_doi, persist_fn)
        if "osf.io" in dataset_doi.lower():
            study = _fetch_and_persist_repository_record(session, study, adapters, "osf", dataset_doi, persist_fn)

    if bcodmo_id:
        study = _fetch_and_persist_repository_record(session, study, adapters, "bcodmo", bcodmo_id, persist_fn)
    if pangaea_id:
        study = _fetch_and_persist_repository_record(session, study, adapters, "pangaea", pangaea_id, persist_fn)
    if obis_uuid:
        study = _fetch_and_persist_repository_record(session, study, adapters, "obis", obis_uuid, persist_fn)
    if gbif_key:
        study = _fetch_and_persist_repository_record(session, study, adapters, "gbif", gbif_key, persist_fn)

    return study


def _has_accessible_sequence_data_signal(session: Session, study: Study) -> bool:
    """Checked once, at the end of handle_discover_identifiers, to decide
    Study.data_availability_status -- see that column's own docstring.

    Pass 1 success: real per-sample/run entities exist (BioProject/ENA/SRA
    structured resolution creates SAMPLE/EXPERIMENT_RUN/SEQUENCING_RUN
    entities directly, see _resolve_repository_sources). Pass 2/3 success:
    a dataset-repository adapter's own file-listing check found something
    (sources/sequence_file_heuristics.py's CONFIRMED/LIKELY tiers -- ABSENT
    doesn't count, matching that module's own "don't waste time on a
    record with nothing" framing)."""
    has_sample_level_entity = (
        session.query(Entity.entity_id)
        .filter(
            Entity.study_id == study.study_id,
            Entity.entity_level.in_(
                (EntityLevel.SAMPLE.value, EntityLevel.EXPERIMENT_RUN.value, EntityLevel.SEQUENCING_RUN.value)
            ),
        )
        .first()
        is not None
    )
    if has_sample_level_entity:
        return True
    positive_statuses = (SequenceDataStatus.CONFIRMED.value, SequenceDataStatus.LIKELY.value)
    return (
        session.query(RawFact.fact_id)
        .filter(
            RawFact.study_id == study.study_id,
            RawFact.fact_type_candidate == "sequence_data_status",
            RawFact.raw_value.in_(positive_statuses),
            RawFact.review_status != ReviewStatus.REJECTED.value,
        )
        .first()
        is not None
    )


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
        study = _discover_publication_metadata_from_sources(session, study, _build_enabled_adapters(), doi)

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
    # ENA/SRA records often surface the BioProject accession only after the
    # sequencing-accession pass above. Resolve any newly-added BioProject
    # immediately so DOI-led runs also collect NCBI BioProject/BioSample
    # sample metadata in the same worker task.
    refreshed_bioproject_accessions = _identifier_values(session, study.study_id, IdentifierType.BIOPROJECT_ACCESSION)
    for bioproject_accession in refreshed_bioproject_accessions:
        if bioproject_accession in bioproject_accessions:
            continue
        for name in ("ncbi_bioproject", "ncbi_biosample"):
            adapter = _build_enabled_adapters().get(name)
            if adapter is None:
                continue
            try:
                record = adapter.fetch_record(bioproject_accession)
            except SourceRecordNotFoundError:
                logger.info("no %s record for %s", name, bioproject_accession)
                continue
            _created, source = _persist_source_and_facts(
                session, study, adapter, SourceType.REPOSITORY_API, bioproject_accession, record
            )
            study = _apply_related_identifiers(session, study, adapter.find_related(record), name, source)
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

    # Node-adding discovery hook: for every BioProject and BioSample
    # accession this study now has (however it got resolved above), enqueue
    # a PubMed citation/reuse-discovery pass. idempotency_key is the
    # accession plus NCBI database -- two different studies resolving the
    # same BioProject or BioSample only trigger this once for free, no new
    # dedup table needed (see handle_discover_citing_studies).
    from fair_ocean_agent.workflow.task_queue import enqueue_task

    for bioproject_accession in _identifier_values(session, study.study_id, IdentifierType.BIOPROJECT_ACCESSION):
        enqueue_task(
            session,
            TaskType.DISCOVER_CITING_STUDIES,
            study_id=study.study_id,
            payload={"bioproject_accession": bioproject_accession},
            idempotency_key=f"DISCOVER_CITING_STUDIES:bioproject:{bioproject_accession}",
        )
    for biosample_accession in _identifier_values(session, study.study_id, IdentifierType.BIOSAMPLE_ACCESSION):
        enqueue_task(
            session,
            TaskType.DISCOVER_CITING_STUDIES,
            study_id=study.study_id,
            payload={"biosample_accession": biosample_accession},
            idempotency_key=f"DISCOVER_CITING_STUDIES:biosample:{biosample_accession}",
        )

    # Two-phase discovery/mapping: for a study with any shareable-level
    # entity, start (or no-op if already started) the settle-check poll
    # that gates root determination and MAP_FAIRE until this study's whole
    # connected component (identity/component.py) stops growing -- see
    # workflow/settle_handlers.py.
    from fair_ocean_agent.workflow.settle_handlers import maybe_enqueue_settle_check

    maybe_enqueue_settle_check(session, study.study_id)

    # Give-up tracking, per an explicit user request: after the staged
    # repository search above (Pass 1/2/3) has had its full chance, record
    # whether anything accessible was ever found. A primer-reference-
    # chased study's only goal was a primer sequence or the next reference
    # in the citation chain, not full sample data, so this question simply
    # doesn't apply to it -- discovery_trigger is checked, not walked up
    # the full lineage (see DataAvailabilityStatus's own docstring for why
    # the simpler per-study check was chosen over a lineage walk).
    if study.discovery_trigger != "primer_reference_citation":
        study.data_availability_status = (
            DataAvailabilityStatus.ACCESSIBLE.value
            if _has_accessible_sequence_data_signal(session, study)
            else DataAvailabilityStatus.NOT_ACCESSIBLE.value
        )

    session.flush()


def _pubmed_doi_from_esummary(record: dict) -> str | None:
    for article_id in record.get("articleids", []):
        if article_id.get("idtype") == "doi":
            return article_id.get("value")
    return None


def _record_citation_expansion_capped_fact(
    session: Session,
    study: Study,
    accession: str,
    *,
    accession_db: str,
    reason: str,
    extra: dict | None = None,
) -> None:
    """Records a review-flagged PROJECT-level RawFact instead of silently
    dropping citing papers that exceeded a safety-valve cap
    (citation_expansion_max_depth or max_citing_papers_per_bioproject, see
    config.py's DiscoveryConfig) -- matches this codebase's established
    "flag, never silently drop" convention (identity/resolution.py's own
    CandidateMatch-on-ambiguity, sources/ncbi.py's ambiguous_uid_resolution).
    source_id is None: this isn't attributable to one Source row, it's a
    property of the citation-discovery pass itself."""
    session.add(
        RawFact(
            study_id=study.study_id,
            source_id=None,
            source_locator=f"ncbi_{accession_db}_pubmed_citation.{accession}",
            raw_field_name="citation_expansion_capped",
            raw_value=reason,
            fact_type_candidate="citing_pmid_not_expanded",
            entity_level=EntityLevel.PROJECT.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method=f"ncbi_{accession_db}_pubmed_citation",
            review_status=ReviewStatus.NEEDS_REVIEW.value,
            confidence_metadata=extra,
        )
    )


def handle_discover_citing_studies(session: Session, task: Task) -> None:
    """Forward citation/reuse discovery: given a BioProject or BioSample
    accession this study already resolved, finds NEW papers that cite/reuse
    that exact NCBI record via an accession-db->pubmed elink call. This is
    the "node-adding" half of discovery the user asked for: new research
    linking to old projects or samples gets its own Study and re-enters the
    normal discovery pipeline.

    Auto-expansion is deliberately aggressive per an explicit user scoping
    decision: a newly-discovered citing paper not already in the database
    gets its own full Study row and re-enters the normal
    DISCOVER_IDENTIFIERS pipeline recursively -- via the task queue's own
    idempotency/resumability, not an in-process recursive BFS, so a citing
    paper that itself cites further BioProjects keeps expanding through
    ordinary task processing. citation_expansion_max_depth and
    max_citing_papers_per_bioproject (config.py's DiscoveryConfig) are the
    real safety valves that make this safe at ~3000-paper batch scale --
    capped work is flagged for review (_record_citation_expansion_capped_fact),
    never silently dropped.
    """
    parent = session.get(Study, task.study_id)
    if parent is None:
        raise ValueError(f"Study {task.study_id} not found")

    payload = task.payload or {}
    accession_db = "bioproject" if payload.get("bioproject_accession") else "biosample"
    accession = payload.get(f"{accession_db}_accession")
    if not accession:
        raise ValueError(
            "DISCOVER_CITING_STUDIES task requires a bioproject_accession or biosample_accession in its payload"
        )

    adapters = _build_enabled_adapters()
    adapter_name = f"ncbi_{accession_db}"
    ncbi_adapter = adapters.get(adapter_name)
    if ncbi_adapter is None:
        logger.info(
            "%s adapter disabled; skipping citation discovery for %s", adapter_name, accession
        )
        return

    discovery_config = load_config().discovery
    if parent.discovery_depth >= discovery_config.citation_expansion_max_depth:
        _record_citation_expansion_capped_fact(
            session, parent, accession, accession_db=accession_db,
            reason=(
                f"discovery_depth {parent.discovery_depth} >= "
                f"citation_expansion_max_depth {discovery_config.citation_expansion_max_depth}: "
                "citing papers not expanded"
            ),
        )
        session.flush()
        return

    # Shares the same rate-limited http client _build_enabled_adapters()
    # already constructed for ncbi_bioproject/ncbi_biosample -- NCBI's rate
    # limit is per-IP across all eutils calls combined (see that function's
    # own comment); a separate client here would silently double the real
    # request rate.
    http = ncbi_adapter.http
    base_url = ncbi_adapter.config.base_url

    resolution = _esearch_verified_uid(http, base_url, accession_db, accession, expected_accession=accession)
    if resolution is None:
        logger.info("no %s UID found for %s during citation discovery", accession_db, accession)
        return

    citing_pmids = _elink_ids(http, base_url, accession_db, "pubmed", resolution.uid)
    if not citing_pmids:
        return

    max_citing = discovery_config.max_citing_papers_per_bioproject
    within_cap, over_cap = citing_pmids[:max_citing], citing_pmids[max_citing:]
    pubmed_records = _esummary_records(http, base_url, "pubmed", within_cap)
    root_study_id = parent.discovery_root_study_id or parent.study_id

    new_study_count = 0
    for pmid in within_cap:
        record = pubmed_records.get(pmid)
        doi = _pubmed_doi_from_esummary(record) if record else None
        if doi is None:
            continue
        try:
            normalized_doi = normalize_identifier(IdentifierType.DOI, doi)
        except IdentifierError:
            continue
        if find_existing_study_by_identifier(session, IdentifierType.DOI, normalized_doi) is not None:
            continue  # already known, whether via this same accession or independently

        citing_study = Study(
            canonical_status=CanonicalStatus.CANDIDATE.value,
            discovery_depth=parent.discovery_depth + 1,
            discovery_parent_study_id=parent.study_id,
            discovery_root_study_id=root_study_id,
            discovery_trigger=f"{accession_db}_pubmed_citation",
        )
        session.add(citing_study)
        session.flush()
        session.add(
            ExternalIdentifier(
                study_id=citing_study.study_id,
                identifier_type=IdentifierType.DOI.value,
                identifier_value=normalized_doi,
                source=f"ncbi_{accession_db}_pubmed_citation",
                verified=True,
            )
        )
        session.flush()

        from fair_ocean_agent.workflow.task_queue import enqueue_task

        enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=citing_study.study_id)
        new_study_count += 1

    if over_cap:
        _record_citation_expansion_capped_fact(
            session, parent, accession, accession_db=accession_db,
            reason=(
                f"{len(over_cap)} of {len(citing_pmids)} citing PMIDs exceeded "
                f"max_citing_papers_per_bioproject ({max_citing})"
            ),
            extra={"excess_pmid_count": len(over_cap)},
        )

    # A component that already SETTLED (root determination ran, MAP_FAIRE
    # enqueued) can grow again right here -- a brand-new citing study just
    # got added to it. Reopen so root determination re-runs against the
    # now-larger, correct membership instead of going stale. No-op if the
    # parent's component was never settled in the first place (still
    # PENDING, or NOT_APPLICABLE because nothing shareable exists yet).
    if new_study_count > 0 and parent.entity_component_status == ComponentStatus.SETTLED.value:
        from fair_ocean_agent.workflow.settle_handlers import reopen_component_settle_check

        reopen_component_settle_check(session, parent.study_id)

    session.flush()
    logger.info(
        "citation discovery for %s %s: %d citing PMIDs, %d new studies created, %d capped",
        accession_db, accession, len(citing_pmids), new_study_count, len(over_cap),
    )


def _record_primer_reference_expansion_capped_fact(
    session: Session, study: Study, doi: str, *, reason: str
) -> None:
    """Same "flag, never silently drop" shape as
    _record_citation_expansion_capped_fact above, for the backward
    (reference-chasing) direction instead of the forward (citing-papers)
    one."""
    session.add(
        RawFact(
            study_id=study.study_id,
            source_id=None,
            source_locator=f"primer_reference_citation.{doi}",
            raw_field_name="primer_reference_expansion_capped",
            raw_value=reason,
            fact_type_candidate="primer_reference_not_expanded",
            entity_level=EntityLevel.PROJECT.value,
            support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
            extraction_method="primer_reference_citation",
            review_status=ReviewStatus.NEEDS_REVIEW.value,
        )
    )


def handle_discover_primer_reference_study(session: Session, task: Task) -> None:
    """Backward reference-chasing discovery: given a primer's own
    bibliographic citation (a real DOI resolved from the paper's own JATS
    <ref-list>, see extraction/publication_metadata.py::
    extract_primer_reference_citations), seeds the referenced paper as its
    own Study and re-enters the normal DISCOVER_IDENTIFIERS pipeline --
    same task-queue-driven, non-in-process-recursive shape as
    handle_discover_citing_studies above, mirrored for the opposite
    direction (papers a primer's own citation points BACK to, not papers
    that cite this one). If that paper's own methods text also just cites
    the primer's origin instead of stating the sequence,
    _resolve_and_seed_primer_references fires again on it once its own
    EXTRACT_TEXT_FACTS runs, and this handler runs again -- ordinary task
    processing is what makes "chase it all the way down" safe, capped by
    DiscoveryConfig.primer_reference_expansion_max_depth (a separate,
    smaller cap than citation_expansion_max_depth -- a different
    discovery vector, expected to bottom out in a handful of hops per an
    explicit user observation: "sometimes it takes 5 papers")."""
    parent = session.get(Study, task.study_id)
    if parent is None:
        raise ValueError(f"Study {task.study_id} not found")

    payload = task.payload or {}
    doi = payload.get("doi")
    primer_name = payload.get("primer_name")
    sequence_field = payload.get("sequence_field")
    if not doi or not primer_name or not sequence_field:
        raise ValueError(
            "DISCOVER_PRIMER_REFERENCE_STUDIES task requires doi, primer_name, and sequence_field in its payload"
        )

    # The corpus may already have this primer's sequence by the time this
    # task actually runs (queues can lag well behind extraction), even if
    # it didn't at enqueue time -- re-check now rather than trusting a
    # possibly-stale decision.
    if corpus_primer_sequence(session, primer_name, sequence_field) is not None:
        return

    discovery_config = load_config().discovery
    if parent.discovery_depth >= discovery_config.primer_reference_expansion_max_depth:
        _record_primer_reference_expansion_capped_fact(
            session, parent, doi,
            reason=(
                f"discovery_depth {parent.discovery_depth} >= "
                f"primer_reference_expansion_max_depth {discovery_config.primer_reference_expansion_max_depth}: "
                "referenced paper not seeded"
            ),
        )
        session.flush()
        return

    if find_existing_study_by_identifier(session, IdentifierType.DOI, doi) is not None:
        return  # already known, whether via this same reference or independently

    root_study_id = parent.discovery_root_study_id or parent.study_id
    referenced_study = Study(
        canonical_status=CanonicalStatus.CANDIDATE.value,
        discovery_depth=parent.discovery_depth + 1,
        discovery_parent_study_id=parent.study_id,
        discovery_root_study_id=root_study_id,
        discovery_trigger="primer_reference_citation",
    )
    session.add(referenced_study)
    session.flush()
    session.add(
        ExternalIdentifier(
            study_id=referenced_study.study_id,
            identifier_type=IdentifierType.DOI.value,
            identifier_value=doi,
            source="primer_reference_citation",
            verified=True,
        )
    )
    session.flush()

    from fair_ocean_agent.workflow.task_queue import enqueue_task

    enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=referenced_study.study_id)
    session.flush()
    logger.info(
        "primer reference discovery for %s (%s): seeded new study %s for DOI %s",
        primer_name, sequence_field, referenced_study.study_id, doi,
    )


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

    local_pdf_path = os.environ.get(LOCAL_PDF_PATH_ENV)
    # A closed-access paper with no PMCID at all (never deposited in PMC,
    # even though it may be freely readable elsewhere -- confirmed live:
    # several real eDNA papers OpenAlex marks as open-access still show
    # Europe PMC's own isOpenAccess=N/inEPMC=N) previously could never
    # reach the local_pdf_path branch below at all: this PMCID check ran
    # unconditionally before it, regardless of whether a local PDF was
    # supplied as the real alternative source. Only require a PMCID when
    # there's no local PDF to fall back on instead.
    pmcid: str | None = None
    if not local_pdf_path:
        pmcid_identifier = next(
            (ei for ei in study.external_identifiers if ei.identifier_type == IdentifierType.PMCID.value), None
        )
        if pmcid_identifier is None:
            raise NotImplementedError(
                "EXTRACT_TEXT_FACTS requires a PMCID (discovered via Europe PMC during "
                "DISCOVER_IDENTIFIERS) to fetch open-access full text -- this study has none. "
                f"Set {LOCAL_PDF_PATH_ENV} to process a locally-supplied PDF instead."
            )
        pmcid = pmcid_identifier.identifier_value

    backend = _build_llm_backend_cached()
    extraction_version = f"{PROMPT_VERSION}:{backend.label}"
    source_name = "local_pdf_fulltext" if local_pdf_path else "europe_pmc_fulltext"
    external_identifier = str(Path(local_pdf_path).resolve()) if local_pdf_path else pmcid
    already_recorded = (
        session.query(Source)
        .filter_by(
            study_id=study.study_id,
            source_name=source_name,
            external_identifier=external_identifier,
            source_version=extraction_version,
        )
        .first()
        is not None
    )
    if already_recorded:
        logger.info(
            "full text for %s already processed for study %s with %s",
            external_identifier,
            study.study_id,
            extraction_version,
        )
        return

    fulltext_xml: str | None = None
    source_text_for_metadata: str | None = None
    if local_pdf_path:
        pdf_path = Path(local_pdf_path)
        if not pdf_path.exists():
            raise SourceRecordNotFoundError(f"local PDF does not exist: {pdf_path}")
        sections = extract_pdf_sections(pdf_path)
        source_text_for_metadata = extract_pdf_text(pdf_path)
        logger.info(
            "using local PDF full text for study %s: %s (%d relevant section(s))",
            study.study_id,
            pdf_path,
            len(sections),
        )
    else:
        europe_pmc = _build_enabled_adapters().get("europe_pmc")
        if europe_pmc is None:
            raise RuntimeError("europe_pmc adapter is not enabled in config/sources.yaml")
        try:
            fulltext_xml = europe_pmc.fetch_fulltext_xml(pmcid)
        except SourceRecordNotFoundError:
            logger.info("no open-access full text available for %s", pmcid)
            return
        source_text_for_metadata = fulltext_xml
        sections = select_relevant_sections(fulltext_xml)
    if not sections:
        logger.info("no relevant sections found in full text for %s", external_identifier)
        return

    source = create_source(
        session,
        Source(
            study_id=study.study_id,
            source_type=SourceType.ARTICLE_FULLTEXT.value,
            source_name=source_name,
            external_identifier=external_identifier,
            access_status=AccessStatus.OPEN.value,
            retrieved_at=utcnow(),
            content_hash=hash_payload(
                {
                    "source_name": source_name,
                    "external_identifier": external_identifier,
                    "section_titles": [s["title"] for s in sections],
                }
            ),
            source_version=extraction_version,
            inspection_status=InspectionStatus.INSPECTED.value,
            inspection_level=InspectionLevel.FULL.value,
        ),
    )
    old_fulltext_source_ids = [
        source_id
        for (source_id,) in session.execute(
            select(Source.source_id).where(
                Source.study_id == study.study_id,
                Source.source_name == source_name,
                Source.external_identifier == external_identifier,
                Source.source_version != extraction_version,
            )
        )
    ]
    if old_fulltext_source_ids:
        session.execute(
            update(RawFact)
            .where(
                RawFact.study_id == study.study_id,
                RawFact.source_id.in_(old_fulltext_source_ids),
                RawFact.extraction_method.in_(
                    (
                        "deterministic_text_search_flagging",
                        "llm_generated_funding_source",
                        "llm_generated_rights_holder",
                        "llm_generated_study_factor",
                        "llm_generated_study_target_taxonomic_scope",
                        "llm_judged_quote_search",
                        "llm_text_extraction",
                        "controlled_sample_prep_refinement",
                        "downstream_analysis_technique_detection",
                        "nucl_acid_ext_lysis_refinement",
                    )
                ),
            )
            .values(review_status=ReviewStatus.REJECTED.value)
        )

    # Rebuild mapping immediately before extraction so this ordering does
    # not depend on a caller remembering to run a separate MAP_FAIRE task.
    # FAIRe fields already resolved from structured sources
    # (NCBI/ENA/PANGAEA/...) are dropped from every section's checklist --
    # less prompt, fewer questions, no chance of the LLM contradicting an
    # already-resolved structured value.
    map_study_to_faire(session, study.study_id)
    session.flush()
    already_resolved = suppress_resolved_faire_hints_for_text(
        resolved_faire_fields_for_study(session, study.study_id)
    )
    llm_config = load_config().llm

    section_texts = tuple((section["title"], section["text"]) for section in sections)
    locator_prefix = f"{source_name}:{external_identifier}"
    # Narrowly-scoped structured-value verification (per an explicit user
    # request): only ever checks elev-vs-depth mislabeling on a soil/
    # sediment sample whose depth is otherwise blank -- see
    # extraction/api_verification.py's own docstring for the real audit
    # that motivated this. Corrects the mismatch and logs it to
    # api_paper_corrections when confirmed; a no-op for every other study.
    detect_and_correct_elev_depth_mislabeling(
        backend,
        session,
        study.study_id,
        list(section_texts),
        source_id=source.source_id,
        locator_prefix=locator_prefix,
    )
    section_category_facts = detect_section_categories_present(
        list(section_texts), locator_prefix=locator_prefix
    )
    flag_facts = detect_text_search_flags(section_texts, locator_prefix=locator_prefix)
    flag_facts = [*flag_facts, *detect_phix_percentage_facts(section_texts, locator_prefix=locator_prefix)]
    pcr_0_1_fact = derive_pcr_0_1_from_category_detection(list(section_texts))
    if pcr_0_1_fact is not None:
        flag_facts = [*flag_facts, pcr_0_1_fact]
    active_flags = frozenset(fact.fact_type_candidate for fact in flag_facts)
    controlled_search_facts = detect_controlled_search_facts(
        section_texts,
        locator_prefix=locator_prefix,
        active_flags=active_flags,
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        [*flag_facts, *controlled_search_facts, *section_category_facts],
        extraction_method="deterministic_text_search_flagging",
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    study_factor_facts = generate_study_factor(backend, source_text_for_metadata, locator_prefix=locator_prefix)
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        study_factor_facts,
        extraction_method="llm_generated_study_factor",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    study_target_scope_facts = generate_study_target_taxonomic_scope(
        backend, source_text_for_metadata, locator_prefix=locator_prefix
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        study_target_scope_facts,
        extraction_method="llm_generated_study_target_taxonomic_scope",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    rights_holder_facts = generate_rights_holder(
        backend,
        source_text_for_metadata,
        locator_prefix=locator_prefix,
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        rights_holder_facts,
        extraction_method="llm_generated_rights_holder",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    funding_source_facts = generate_funding_source(
        backend,
        source_text_for_metadata,
        locator_prefix=locator_prefix,
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        funding_source_facts,
        extraction_method="llm_generated_funding_source",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    # New categorize-then-extract pipeline (extraction/section_category_
    # extraction.py) for the PCR/library-prep/bioinformatics term
    # families -- deliberately additive alongside every mechanism below,
    # not a replacement, per an explicit user request to compare this
    # new pipeline's own output against the existing ones before
    # retiring anything. Runs BEFORE detect_llm_judged_search_facts (and
    # feeds its own resolved field names into that call's
    # exclude_field_names) so this pipeline's quote-grounded answer always
    # wins over the older mechanism's broader LLM judgement or its
    # keyword-default fallback (e.g. barcoding_pcr_appr's "one-step PCR"
    # default firing even when this pipeline already found real
    # "two-step" evidence) -- confirmed as a real bug against a live
    # paper audit.
    section_category_term_facts = extract_section_category_facts(
        backend, list(section_texts), locator_prefix=locator_prefix
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        section_category_term_facts,
        extraction_method="section_category_term_extraction",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    # Second pass over x_env_var_block's own quotes only (never re-reads
    # the paper) -- per an explicit user request to keep x_env_var_block
    # exactly as-is while adding a filtered, structured companion field.
    pulled_env_var_facts = extract_pulled_env_var_facts(
        backend, section_category_term_facts, locator_prefix=locator_prefix
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        pulled_env_var_facts,
        extraction_method="x_pulled_env_var_refinement",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    downstream_analysis_facts = detect_downstream_analysis_techniques(section_texts, locator_prefix=locator_prefix)
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        downstream_analysis_facts,
        extraction_method="downstream_analysis_technique_detection",
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )
    llm_judged_search_facts = detect_llm_judged_search_facts(
        backend,
        section_texts,
        locator_prefix=locator_prefix,
        max_output_tokens=MIN_LLM_MAX_OUTPUT_TOKENS,
        active_flags=active_flags,
        exclude_field_names=already_resolved
        | frozenset(fact.fact_type_candidate for fact in section_category_term_facts),
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        llm_judged_search_facts,
        extraction_method="llm_judged_quote_search",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )

    broad_text_facts: list[RawFactCandidate] = []
    for section in sections:
        try:
            facts, response = extract_facts_from_section(
                backend,
                section["title"],
                section["text"],
                exclude_faire_hints=already_resolved,
                max_section_chars_per_call=llm_config.extraction_max_chars_per_call,
                max_output_tokens=llm_config.max_output_tokens,
                active_flags=active_flags,
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
        broad_text_facts.extend(facts)
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

    normalized_sample_prep_facts = normalize_controlled_sample_prep_facts(
        backend,
        [*section_category_term_facts, *broad_text_facts],
        locator_prefix=locator_prefix,
    )
    _persist_candidate_facts(
        session,
        study.study_id,
        source.source_id,
        normalized_sample_prep_facts,
        extraction_method="controlled_sample_prep_refinement",
        model_name=backend.label,
        prompt_version=PROMPT_VERSION,
        review_status=ReviewStatus.ACCEPTED.value,
    )

    _resolve_and_seed_primer_references(session, study, source, fulltext_xml, locator_prefix=locator_prefix)

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
TASK_HANDLERS[TaskType.DISCOVER_CITING_STUDIES] = handle_discover_citing_studies
TASK_HANDLERS[TaskType.DISCOVER_PRIMER_REFERENCE_STUDIES] = handle_discover_primer_reference_study
TASK_HANDLERS[TaskType.EXTRACT_TEXT_FACTS] = handle_extract_text_facts
