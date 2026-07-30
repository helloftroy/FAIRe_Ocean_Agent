"""Task handlers for supplementary-material discovery and retrieval
(supplementary-material and structured-table retrieval layer).

Two stages, matching every other pipeline stage's shape (separate,
manually-triggered CLI backfill commands -- no handler auto-chains into the
next stage, so this stays opt-in exactly like EXTRACT_TEXT_FACTS already
is):

- DISCOVER_SUPPLEMENTS: parses <supplementary-material> tags out of the
  (cached) main article XML -- no new network call. Creates one `Source`
  row (source_type=SUPPLEMENT, created once with a terminal status, never
  mutated again -- same as every other Source-creation call site) plus one
  companion `DataAsset` row per referenced file. Also surfaces any
  externally-hosted repository supplement DOIs as RelatedIdentifiers, so
  the *existing* DISCOVER_IDENTIFIERS -> structured-adapter pipeline can
  pick them up on its own next pass.
- RETRIEVE_SUPPLEMENTS: for a study's not-yet-parsed supplement
  `DataAsset` rows, checks the summed *already-known* size (from discovery)
  against a config cap before ever downloading Europe PMC's single
  all-files-bundled zip, then routes each member by extension.

The mutable, evolving per-file retrieval state (the user's 6 named states:
referenced/available/retrieved/parsed/inaccessible/parse_failed) lives
entirely on the companion `DataAsset` row's existing `access_status`/
`inspection_level` columns (+ `description` for the retrieved/parse_failed
distinction) -- never on `Source`, which every other call site in this
codebase sets once at creation and never mutates again.
"""
from __future__ import annotations

import io
import posixpath
import zipfile

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.config import load_config
from fair_ocean_agent.database.enums import (
    AccessStatus,
    AssetType,
    EntityLevel,
    IdentifierType,
    InspectionLevel,
    InspectionStatus,
    RawOrProcessed,
    SourceType,
    SupportType,
    TaskType,
)
from fair_ocean_agent.database.models import DataAsset, ExternalIdentifier, RawFact, Source, Study, Task
from fair_ocean_agent.extraction.text import PROMPT_VERSION, extract_facts_from_section, resolved_faire_fields_for_study
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import RawFactCandidate, SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import discover_supplementary_materials
from fair_ocean_agent.sources.supplement_parsing import (
    ParsedTableResult,
    ZipMemberTooLargeError,
    extract_pdf_text,
    parse_delimited_table,
    parse_json_supplement,
    parse_xls_table,
    parse_xlsx_table,
    parse_xml_supplement,
    safe_read_zip_member,
)
from fair_ocean_agent.workflow.handlers import (
    _apply_related_identifiers,
    _build_enabled_adapters,
    _build_llm_backend_cached,
    _get_or_create_entity,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)

_TABULAR_EXTENSIONS = {"csv", "tsv", "xlsx", "xls"}


def _study_pmcid(session: Session, study: Study) -> str | None:
    identifier = next(
        (ei for ei in study.external_identifiers if ei.identifier_type == IdentifierType.PMCID.value), None
    )
    return identifier.identifier_value if identifier else None


def _extension(file_name: str) -> str:
    return file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""


def _asset_type_for(extension: str) -> str:
    return AssetType.SUPPLEMENTARY_SPREADSHEET.value if extension in _TABULAR_EXTENSIONS else AssetType.OTHER.value


def handle_discover_supplements(session: Session, task: Task) -> None:
    """Requires a PMCID (same requirement as EXTRACT_TEXT_FACTS -- Europe
    PMC's supplementary-material references only ever come from its own
    JATS full text). No-ops (not an error) if the study has no open-access
    full text at all, same convention as handle_extract_text_facts."""
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    pmcid = _study_pmcid(session, study)
    if pmcid is None:
        raise NotImplementedError(
            "DISCOVER_SUPPLEMENTS requires a PMCID (discovered via Europe PMC during "
            "DISCOVER_IDENTIFIERS) -- this study has none."
        )

    europe_pmc = _build_enabled_adapters().get("europe_pmc")
    if europe_pmc is None:
        raise RuntimeError("europe_pmc adapter is not enabled in config/sources.yaml")

    try:
        fulltext_xml = europe_pmc.fetch_fulltext_xml(pmcid)
    except SourceRecordNotFoundError:
        logger.info("no open-access full text available for %s -- nothing to discover", pmcid)
        return

    references = discover_supplementary_materials(fulltext_xml)

    related = europe_pmc.find_related_from_fulltext(fulltext_xml)
    if related:
        study = _apply_related_identifiers(session, study, related, source_name="europe_pmc_supplement_extlink")

    if not references:
        logger.info("no supplementary materials referenced for %s", pmcid)
        return

    existing_file_names = set(
        session.scalars(
            select(Source.external_identifier).where(
                Source.study_id == study.study_id,
                Source.source_name == "europe_pmc_supplement",
            )
        ).all()
    )

    created = 0
    for reference in references:
        if reference.file_name in existing_file_names:
            continue  # idempotent: already discovered on a prior run

        source = Source(
            study_id=study.study_id,
            source_type=SourceType.SUPPLEMENT.value,
            source_name="europe_pmc_supplement",
            external_identifier=reference.file_name,
            url=None,
            retrieved_at=utcnow(),
            content_hash=None,
            # Created once, with a terminal status -- "we know this file is
            # referenced and what it is" -- never mutated again, same as
            # every other Source-creation call site in this codebase.
            inspection_status=InspectionStatus.INSPECTED.value,
            inspection_level=InspectionLevel.METADATA_ONLY.value,
        )
        session.add(source)
        session.flush()

        extension = _extension(reference.file_name)
        session.add(
            DataAsset(
                study_id=study.study_id,
                asset_type=_asset_type_for(extension),
                repository="europe_pmc",
                identifier=reference.supplement_id,
                file_name=reference.file_name,
                format=extension or None,
                size_bytes=reference.size_bytes,
                # supplement_referenced/supplement_available (see module
                # docstring): Europe PMC's XML gives filename+mimetype+size
                # for free at discovery time, so "referenced" and
                # "available" are simultaneous here -- access_status stays
                # UNKNOWN (not yet confirmed fetchable) until RETRIEVE_SUPPLEMENTS
                # actually asks for the bytes.
                access_status=AccessStatus.UNKNOWN.value,
                raw_or_processed=RawOrProcessed.RAW.value,
                inspection_level=InspectionLevel.METADATA_ONLY.value,
                source_id=source.source_id,
            )
        )
        created += 1

    logger.info("discovered %d new supplementary file(s) for study %s (%s)", created, study.study_id, pmcid)


def enqueue_supplement_discovery_backfill(session: Session) -> int:
    """Queues DISCOVER_SUPPLEMENTS for every study with a PMCID, same
    pattern as enqueue_text_extraction_backfill."""
    study_ids = session.scalars(
        select(ExternalIdentifier.study_id).where(ExternalIdentifier.identifier_type == IdentifierType.PMCID.value)
    ).all()
    for study_id in study_ids:
        enqueue_task(session, TaskType.DISCOVER_SUPPLEMENTS, study_id=study_id)
    return len(study_ids)


def _parsed_result_summary(results: list[ParsedTableResult]) -> tuple[list[RawFactCandidate], str]:
    facts: list[RawFactCandidate] = []
    summaries: list[str] = []
    for result in results:
        facts.extend(result.facts)
        summaries.append(result.summary())
    return facts, " | ".join(summaries)


def _persist_supplement_facts(
    session: Session,
    study: Study,
    source: Source,
    facts: list[RawFactCandidate],
    *,
    extraction_method: str = "supplement_table_parsing",
    model_name: str | None = None,
    prompt_version: str | None = None,
) -> None:
    for fact in facts:
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
                evidence_quote=fact.evidence_quote,
                confidence_metadata=fact.confidence_metadata,
                fact_type_candidate=fact.fact_type_candidate,
                entity_level=fact.entity_level.value,
                support_type=fact.support_type.value,
                extraction_method=extraction_method,
                model_name=model_name,
                prompt_version=prompt_version,
            )
        )


def _mark(asset: DataAsset, *, access_status: str | None = None, inspection_level: str | None = None, description: str | None = None) -> None:
    if access_status is not None:
        asset.access_status = access_status
    if inspection_level is not None:
        asset.inspection_level = inspection_level
    if description is not None:
        asset.description = description


def _process_member(
    session: Session,
    study: Study,
    source: Source,
    asset: DataAsset,
    content: bytes,
    file_name: str,
    llm_exclude_faire_hints: frozenset[str],
) -> None:
    """Routes one already-retrieved supplement's bytes by extension. Never
    lets one file's parse failure raise out to the caller -- caught and
    recorded on that file's own DataAsset row (supplement_parse_failed),
    exactly like handle_extract_text_facts never lets one section's LLM
    failure abort the whole study."""
    extension = _extension(file_name)
    llm_config = load_config().llm
    via_llm = extension in ("txt", "md", "pdf")
    backend = _build_llm_backend_cached() if via_llm else None

    try:
        if extension == "csv":
            facts, summary = _parsed_result_summary([parse_delimited_table(content, file_name, delimiter=",")])
        elif extension == "tsv":
            facts, summary = _parsed_result_summary([parse_delimited_table(content, file_name, delimiter="\t")])
        elif extension == "xlsx":
            facts, summary = _parsed_result_summary(parse_xlsx_table(content, file_name))
        elif extension == "xls":
            facts, summary = _parsed_result_summary(parse_xls_table(content, file_name))
        elif extension == "json":
            facts, summary = _parsed_result_summary([parse_json_supplement(content, file_name)])
        elif extension == "xml":
            facts, summary = _parsed_result_summary([parse_xml_supplement(content, file_name)])
        elif extension in ("txt", "md"):
            text = content.decode("utf-8", errors="replace")
            facts, _response = extract_facts_from_section(
                backend,
                f"Supplementary Methods ({file_name})",
                text,
                exclude_faire_hints=llm_exclude_faire_hints,
                max_section_chars_per_call=llm_config.extraction_max_chars_per_call,
                max_output_tokens=llm_config.max_output_tokens,
            )
            summary = f"{len(text)} chars of supplementary text run through LLM extraction"
        elif extension == "pdf":
            text = extract_pdf_text(content)
            facts, _response = extract_facts_from_section(
                backend,
                f"Supplementary Methods ({file_name})",
                text,
                exclude_faire_hints=llm_exclude_faire_hints,
                max_section_chars_per_call=llm_config.extraction_max_chars_per_call,
                max_output_tokens=llm_config.max_output_tokens,
            )
            summary = f"{len(text)} chars extracted from PDF, run through LLM extraction"
        else:
            _mark(
                asset,
                access_status=AccessStatus.OPEN.value,
                inspection_level=InspectionLevel.LIGHTWEIGHT.value,
                description=f"retrieved: unsupported file type .{extension or '?'}, not parsed",
            )
            return
    except LLMBackendError as exc:
        _mark(
            asset,
            access_status=AccessStatus.OPEN.value,
            inspection_level=InspectionLevel.LIGHTWEIGHT.value,
            description=f"parse_failed: LLM extraction error: {exc}",
        )
        return
    except Exception as exc:  # noqa: BLE001 -- one bad file must never abort the whole task
        logger.warning("failed to parse supplement %s for study %s: %s", file_name, study.study_id, exc)
        _mark(
            asset,
            access_status=AccessStatus.OPEN.value,
            inspection_level=InspectionLevel.LIGHTWEIGHT.value,
            description=f"parse_failed: {exc}",
        )
        return

    if via_llm:
        _persist_supplement_facts(
            session, study, source, facts,
            extraction_method="supplement_llm_extraction",
            model_name=backend.label,
            prompt_version=PROMPT_VERSION,
        )
    else:
        _persist_supplement_facts(session, study, source, facts)
    _mark(asset, access_status=AccessStatus.OPEN.value, inspection_level=InspectionLevel.FULL.value, description=summary)


def handle_retrieve_supplements(session: Session, task: Task) -> None:
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    pmcid = _study_pmcid(session, study)
    if pmcid is None:
        raise NotImplementedError(
            "RETRIEVE_SUPPLEMENTS requires a PMCID -- this study has none."
        )

    supplement_source_ids = session.scalars(
        select(Source.source_id).where(
            Source.study_id == study.study_id,
            Source.source_type == SourceType.SUPPLEMENT.value,
        )
    ).all()
    if not supplement_source_ids:
        logger.info("no discovered supplements for study %s -- run DISCOVER_SUPPLEMENTS first", study.study_id)
        return

    assets = session.scalars(
        select(DataAsset).where(DataAsset.source_id.in_(supplement_source_ids))
    ).all()
    pending = [asset for asset in assets if asset.inspection_level != InspectionLevel.FULL.value]
    if not pending:
        logger.info("all discovered supplements for study %s already parsed", study.study_id)
        return

    supplement_config = load_config().supplements
    if not supplement_config.enabled:
        logger.info("supplement retrieval disabled (config.supplements.enabled=false)")
        return

    # Europe PMC has no per-file fetch -- the *whole* bundle must be sized
    # up before ever downloading it, using the byte sizes already known
    # from discovery (never a guess).
    total_known_bytes = sum(asset.size_bytes or 0 for asset in assets)
    if total_known_bytes > supplement_config.max_bundle_bytes:
        for asset in pending:
            _mark(
                asset,
                access_status=AccessStatus.NOT_ACCESSIBLE.value,
                description=(
                    f"supplement_inaccessible: whole-article bundle is {total_known_bytes} bytes, "
                    f"exceeds max_bundle_bytes={supplement_config.max_bundle_bytes} -- not downloaded"
                ),
            )
        logger.info(
            "study %s supplement bundle too large (%d bytes) -- marked inaccessible, proceeding without it",
            study.study_id, total_known_bytes,
        )
        return

    europe_pmc = _build_enabled_adapters().get("europe_pmc")
    if europe_pmc is None:
        raise RuntimeError("europe_pmc adapter is not enabled in config/sources.yaml")

    try:
        bundle_bytes = europe_pmc.fetch_supplementary_bundle(pmcid)
    except SourceRecordNotFoundError:
        for asset in pending:
            _mark(
                asset,
                access_status=AccessStatus.NOT_ACCESSIBLE.value,
                description="supplement_inaccessible: no supplementary-files bundle at Europe PMC",
            )
        return

    already_resolved = resolved_faire_fields_for_study(session, study.study_id)
    asset_by_file_name = {asset.file_name: asset for asset in pending}
    source_by_id = {source.source_id: source for source in session.scalars(
        select(Source).where(Source.source_id.in_(supplement_source_ids))
    ).all()}

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zip_file:
        zip_members_by_basename = {posixpath.basename(info.filename): info for info in zip_file.infolist()}

        for file_name, asset in asset_by_file_name.items():
            info = zip_members_by_basename.get(file_name)
            if info is None:
                _mark(
                    asset,
                    access_status=AccessStatus.NOT_ACCESSIBLE.value,
                    description="supplement_inaccessible: referenced file not found in the retrieved bundle",
                )
                continue

            if info.file_size > supplement_config.max_member_bytes:
                _mark(
                    asset,
                    access_status=AccessStatus.OPEN.value,
                    description=(
                        f"supplement_available: {info.file_size} bytes exceeds "
                        f"max_member_bytes={supplement_config.max_member_bytes} -- not retrieved"
                    ),
                )
                continue

            try:
                content = safe_read_zip_member(zip_file, info, supplement_config.max_member_bytes)
            except ZipMemberTooLargeError as exc:
                _mark(
                    asset,
                    access_status=AccessStatus.OPEN.value,
                    description=f"supplement_available: {exc} -- not retrieved",
                )
                continue

            source = source_by_id[asset.source_id]
            _process_member(session, study, source, asset, content, file_name, already_resolved)

    session.flush()
    logger.info("processed %d pending supplement(s) for study %s", len(pending), study.study_id)


def enqueue_supplement_retrieval_backfill(session: Session) -> int:
    """Queues RETRIEVE_SUPPLEMENTS for every study with at least one
    discovered (source_type=SUPPLEMENT) Source row."""
    study_ids = session.scalars(
        select(Source.study_id).where(Source.source_type == SourceType.SUPPLEMENT.value).distinct()
    ).all()
    for study_id in study_ids:
        enqueue_task(session, TaskType.RETRIEVE_SUPPLEMENTS, study_id=study_id)
    return len(study_ids)


TASK_HANDLERS[TaskType.DISCOVER_SUPPLEMENTS] = handle_discover_supplements
TASK_HANDLERS[TaskType.RETRIEVE_SUPPLEMENTS] = handle_retrieve_supplements
