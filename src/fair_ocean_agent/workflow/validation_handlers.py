"""Task handlers for Milestone 5: INVENTORY_DATA_ASSETS, VALIDATE_EVIDENCE,
VALIDATE_LOGIC, VALIDATE_CROSS_SOURCE. Split from workflow/handlers.py
(which covers Milestones 2-4's discovery/extraction handlers) since this
is a genuinely distinct concern -- auditing and materializing from facts
already persisted, never fetching anything new from a network source.

Importing this module registers its handlers as a side effect, same as
handlers.py; cli.py imports both.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.assets.inventory import inventory_ena_run_assets
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, MissingnessStatus, SourceType, TaskType
from fair_ocean_agent.database.models import (
    Entity,
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    Study,
    Task,
    ValidationResult,
)
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA as FAIRE_TARGET_SCHEMA
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA_VERSION as FAIRE_TARGET_SCHEMA_VERSION
from fair_ocean_agent.mapping.faire import resolve_project_id
from fair_ocean_agent.validation.cross_source import compare_field_across_sources
from fair_ocean_agent.validation.evidence import check_raw_fact_evidence_consistency
from fair_ocean_agent.validation.faire_completeness import unconditionally_mandatory_faire_fields
from fair_ocean_agent.validation.logical import (
    ValidationOutcome,
    validate_accession_format,
    validate_collection_before_publication,
    validate_coordinates,
    validate_depth,
    validate_primer_sequence,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)

VALIDATOR_VERSION = "v1"


def handle_inventory_data_assets(session: Session, task: Task) -> None:
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")
    created = inventory_ena_run_assets(session, study.study_id)
    logger.info("inventoried %d data asset(s) for study %s", created, study.study_id)


def _persist_fact_validation_result(
    session: Session, study_id: str, fact_id: str, validator_name: str, outcome: ValidationOutcome
) -> bool:
    """Idempotent on (study_id, fact_id, validator_name) -- a fact's own
    primary key makes this exact, unlike identifier-format checks below."""
    already = (
        session.query(ValidationResult)
        .filter_by(study_id=study_id, fact_id=fact_id, validator_name=validator_name)
        .first()
    )
    if already is not None:
        return False
    session.add(
        ValidationResult(
            study_id=study_id,
            fact_id=fact_id,
            validator_name=validator_name,
            validator_version=VALIDATOR_VERSION,
            severity=outcome.severity,
            status=outcome.status,
            message=outcome.message,
            compared_values=outcome.compared_values,
        )
    )
    return True


# --- VALIDATE_EVIDENCE -----------------------------------------------------


def handle_validate_evidence(session: Session, task: Task) -> None:
    """Only persists a ValidationResult for facts that FAIL the check --
    section 16's evidence validation is meant to surface problems, and a
    study can have thousands of facts; recording every passing check would
    bloat the table for no benefit (a passing fact's evidence bookkeeping
    is already visible on the raw_facts row itself)."""
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    facts = session.scalars(select(RawFact).where(RawFact.study_id == study.study_id)).all()
    flagged = 0
    for fact in facts:
        result = check_raw_fact_evidence_consistency(fact.support_type, fact.evidence_quote, fact.source_locator)
        if result.ok:
            continue
        created = _persist_fact_validation_result(
            session,
            study.study_id,
            fact.fact_id,
            "evidence_consistency",
            ValidationOutcome("unsupported", "error", result.message, {"fact_id": fact.fact_id}),
        )
        if created:
            flagged += 1

    logger.info("evidence-validated %d fact(s) for study %s, %d flagged", len(facts), study.study_id, flagged)


# --- VALIDATE_LOGIC ---------------------------------------------------------

_COORDINATE_FIELD_NAMES = {"lat_lon", "coordinates", "coordinate"}
_DEPTH_FIELD_NAMES = {"depth", "depths", "sampling_depth"}


def _publication_year_for_study(session: Session, study_id: str) -> int | None:
    """Publication bibliographic fields live per-adapter on each study's
    own `sources` rows now, not a separately-merged Publication row (see
    Source's class docstring) -- this picks the first non-null
    publication_year across a study's publication-type sources, the same
    "first non-null wins" semantic the old merge used, just computed at
    query time instead of stored redundantly."""
    return session.scalars(
        select(Source.publication_year).where(
            Source.study_id == study_id,
            Source.source_type == SourceType.PUBLICATION_API.value,
            Source.publication_year.is_not(None),
        )
    ).first()


def handle_validate_logic(session: Session, task: Task) -> None:
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    publication_year = _publication_year_for_study(session, study.study_id)

    facts = session.scalars(select(RawFact).where(RawFact.study_id == study.study_id)).all()
    checked = 0
    for fact in facts:
        field_name = (fact.fact_type_candidate or "").lower()
        if not fact.raw_value:
            continue

        outcome = None
        if field_name in _COORDINATE_FIELD_NAMES:
            outcome = validate_coordinates(fact.raw_value)
        elif field_name in _DEPTH_FIELD_NAMES:
            outcome = validate_depth(fact.raw_value)
        elif field_name == "collection_date" and publication_year:
            outcome = validate_collection_before_publication(fact.raw_value, publication_year)
        elif "primer" in field_name:
            outcome = validate_primer_sequence(fact.raw_value)

        if outcome is not None:
            if _persist_fact_validation_result(session, study.study_id, fact.fact_id, "logical_validator", outcome):
                checked += 1

    checked += _validate_identifier_formats(session, study)
    logger.info("logic-validated %d fact(s)/identifier(s) for study %s", checked, study.study_id)


def _validate_identifier_formats(session: Session, study: Study) -> int:
    """ExternalIdentifier has no fact_id to key idempotency on, so this
    fetches this study's already-recorded accession_format results once and
    matches by (identifier_type, raw_value) embedded in compared_values,
    rather than re-querying per identifier."""
    already_checked = {
        (r.compared_values or {}).get("raw_value")
        for r in session.query(ValidationResult)
        .filter_by(study_id=study.study_id, validator_name="accession_format")
        .all()
    }

    checked = 0
    for identifier in session.scalars(
        select(ExternalIdentifier).where(ExternalIdentifier.study_id == study.study_id)
    ).all():
        if identifier.identifier_value in already_checked:
            continue
        try:
            identifier_type = IdentifierType(identifier.identifier_type)
        except ValueError:
            continue

        outcome = validate_accession_format(identifier_type, identifier.identifier_value)
        session.add(
            ValidationResult(
                study_id=study.study_id,
                fact_id=None,
                validator_name="accession_format",
                validator_version=VALIDATOR_VERSION,
                severity=outcome.severity,
                status=outcome.status,
                message=outcome.message,
                compared_values=outcome.compared_values,
            )
        )
        checked += 1
    return checked


# --- VALIDATE_CROSS_SOURCE --------------------------------------------------

# Fields compared across sources: title is the one concept named identically
# by all three publication adapters (crossref/europe_pmc/openalex all use
# fact_type_candidate="title"). Journal/venue names differ per adapter
# ("container-title" vs "journalTitle" vs none for OpenAlex) -- harmonizing
# those is a mapping concern (Milestone 6), not validation.
_CROSS_SOURCE_FIELDS = ("title",)


def handle_validate_cross_source(session: Session, task: Task) -> None:
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")

    already_validated_fields = {
        (r.compared_values or {}).get("fact_type_candidate")
        for r in session.query(ValidationResult)
        .filter_by(study_id=study.study_id, validator_name="cross_source_agreement")
        .all()
    }

    for field_name in _CROSS_SOURCE_FIELDS:
        if field_name in already_validated_fields:
            continue
        comparison = compare_field_across_sources(session, study.study_id, field_name)
        session.add(
            ValidationResult(
                study_id=study.study_id,
                fact_id=None,
                validator_name="cross_source_agreement",
                validator_version=VALIDATOR_VERSION,
                severity="warning" if comparison.status == "conflicting" else "info",
                status=comparison.status,
                message=comparison.message,
                compared_values={"fact_type_candidate": field_name, "values_by_source": comparison.values_by_source},
            )
        )

    logger.info("cross-source-validated %d field(s) for study %s", len(_CROSS_SOURCE_FIELDS), study.study_id)


# --- Missingness (folded into standardized_values -- see that model's
# class docstring) ------------------------------------------------------

# A small, deliberately narrow set of core sampling-metadata fields --
# extending this to the full FAIRe/MIxS field list is a Milestone 6 concern
# (once there's a standards mapping to check completeness against); this
# is just "did we find the basics anywhere". Kept as its own target_schema
# ("core_sampling_metadata"), distinct from "faire", since it predates the
# FAIRe mapping layer and uses this pipeline's own field vocabulary
# (lat_lon, depth) rather than FAIRe's (decimalLatitude, minimumDepthInMeters).
CORE_SAMPLING_FIELDS = ("collection_date", "lat_lon", "depth")
MISSINGNESS_TARGET_SCHEMA = "core_sampling_metadata"


def populate_missingness_for_study(session: Session, study_id: str) -> int:
    """Presence-only check -- unlike a mapped StandardizedValue,
    `standardized_value` here is always null (this predates any real
    value extraction for this schema); only `missingness_status` carries
    information, same as it did in the old, now-removed `missingness`
    table."""
    existing_fields = {
        sv.target_field
        for sv in session.query(StandardizedValue)
        .filter_by(study_id=study_id, entity_id=None, target_schema=MISSINGNESS_TARGET_SCHEMA)
        .all()
    }
    present_fields = {
        f
        for f in session.scalars(
            select(RawFact.fact_type_candidate).where(
                RawFact.study_id == study_id, RawFact.fact_type_candidate.in_(CORE_SAMPLING_FIELDS)
            )
        ).all()
    }
    sources_inspected = list(
        session.scalars(select(Source.source_name).where(Source.study_id == study_id).distinct()).all()
    )

    created = 0
    for field_name in CORE_SAMPLING_FIELDS:
        if field_name in existing_fields:
            continue
        if field_name in present_fields:
            status = MissingnessStatus.PRESENT.value
            reason = f"Found at least one {field_name!r} raw_fact."
        elif sources_inspected:
            status = MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value
            reason = f"{field_name!r} not reported by any of the sources inspected for this study."
        else:
            status = MissingnessStatus.RELEVANT_SOURCE_NOT_INSPECTED.value
            reason = "No sources have been inspected for this study yet."

        session.add(
            StandardizedValue(
                study_id=study_id,
                entity_id=None,
                target_schema=MISSINGNESS_TARGET_SCHEMA,
                target_schema_version="",
                target_field=field_name,
                standardized_value=None,
                missingness_status=status,
                sources_inspected=sources_inspected,
                reason=reason,
            )
        )
        created += 1
    return created


# --- FAIRe Mandatory-field completeness (Milestone 6 follow-up) ----------

FAIRE_MISSINGNESS_TARGET_SCHEMA = "faire"


def _project_field_present(session: Session, study_id: str, target_field: str) -> bool:
    if target_field == "project_id":
        return resolve_project_id(session, study_id) is not None
    return (
        session.query(StandardizedValue)
        .filter_by(study_id=study_id, entity_id=None, target_schema=FAIRE_TARGET_SCHEMA, target_field=target_field)
        .count()
        > 0
    )


def _sample_field_present(session: Session, entity: Entity, target_field: str) -> bool:
    if target_field == "samp_name":
        return entity.external_identifier is not None
    return (
        session.query(StandardizedValue)
        .filter_by(entity_id=entity.entity_id, target_schema=FAIRE_TARGET_SCHEMA, target_field=target_field)
        .count()
        > 0
    )


def populate_faire_missingness_for_study(session: Session, study_id: str) -> int:
    """Checks coverage of every unconditionally-Mandatory FAIRe field in
    projectMetadata (one check per study) and sampleMetadata (one check
    per sample Entity) -- see validation/faire_completeness.py for why
    conditionally-Mandatory fields (most of FAIRe's 32
    requirement_level_condition fields) are excluded rather than
    guessed at.

    Missingness lives in the same `standardized_values` table
    `map_study_to_faire` writes real mapped values to (see that model's
    class docstring), so a field `map_study_to_faire` already mapped
    already has a row here -- this only tags its `missingness_status` as
    "present" rather than inserting a second, duplicate row for the same
    (study, entity, target_field). A field nothing mapped gets a fresh
    placeholder row (`standardized_value=None`). Idempotent: skips
    (entity_id, target_field) combinations that already carry a
    `missingness_status` from a prior run of this same function."""
    existing_by_key: dict[tuple[str | None, str], StandardizedValue] = {
        (sv.entity_id, sv.target_field): sv
        for sv in session.query(StandardizedValue)
        .filter_by(study_id=study_id, target_schema=FAIRE_MISSINGNESS_TARGET_SCHEMA)
        .all()
    }
    mandatory_by_class = unconditionally_mandatory_faire_fields()
    created = 0

    def upsert(entity_id: str | None, target_field: str, present: bool, table_name: str, not_found_reason: str) -> None:
        nonlocal created
        existing = existing_by_key.get((entity_id, target_field))
        status = MissingnessStatus.PRESENT.value if present else MissingnessStatus.NOT_FOUND_IN_INSPECTED_SOURCES.value
        if existing is not None:
            if existing.missingness_status is None:
                existing.missingness_status = status
            return
        session.add(
            StandardizedValue(
                study_id=study_id,
                entity_id=entity_id,
                target_schema=FAIRE_MISSINGNESS_TARGET_SCHEMA,
                target_schema_version=FAIRE_TARGET_SCHEMA_VERSION,
                target_field=target_field,
                standardized_value=None,
                missingness_status=status,
                reason=(
                    f"{target_field!r} is unconditionally Mandatory in FAIRe's {table_name}; "
                    + ("found." if present else not_found_reason)
                ),
            )
        )
        created += 1

    for term in mandatory_by_class["projectMetadata"]:
        target_field = term["upstream_field_name"]
        present = _project_field_present(session, study_id, target_field)
        upsert(None, target_field, present, "projectMetadata", "no standardized value or resolvable identifier found.")

    sample_terms = mandatory_by_class["sampleMetadata"]
    if sample_terms:
        sample_entities = session.scalars(
            select(Entity).where(Entity.study_id == study_id, Entity.entity_level == EntityLevel.SAMPLE.value)
        ).all()
        for entity in sample_entities:
            for term in sample_terms:
                target_field = term["upstream_field_name"]
                present = _sample_field_present(session, entity, target_field)
                upsert(entity.entity_id, target_field, present, "sampleMetadata", "no standardized value found for this sample.")

    return created


def handle_validate_faire_completeness(session: Session, task: Task) -> None:
    created = populate_faire_missingness_for_study(session, task.study_id)
    logger.info("recorded %d FAIRe Mandatory-field missingness row(s) for study %s", created, task.study_id)


def enqueue_faire_completeness_backfill(session: Session) -> int:
    """Queues VALIDATE_FAIRE_COMPLETENESS for every study with raw facts.
    Mapping may legitimately create zero real FAIRe values for a study, but
    those blank standardized values still need explainable missingness
    statuses after sources have been inspected."""
    study_ids = list(session.scalars(select(RawFact.study_id).distinct()))
    for study_id in study_ids:
        enqueue_task(session, TaskType.VALIDATE_FAIRE_COMPLETENESS, study_id=study_id)
    return len(study_ids)


def handle_validate_logic_and_missingness(session: Session, task: Task) -> None:
    """VALIDATE_LOGIC also populates missingness -- both are "what does this
    study's already-extracted data look like" passes over the same raw_facts,
    and splitting them into two task types would mean two nearly-identical
    scans. (Kept as a distinct function, not folded into handle_validate_logic,
    so each concern is independently testable.)"""
    handle_validate_logic(session, task)
    study_id = task.study_id
    created = populate_missingness_for_study(session, study_id)
    logger.info("recorded %d missingness row(s) for study %s", created, study_id)


def enqueue_data_asset_and_validation_backfill(session: Session) -> dict[str, int]:
    """Queues INVENTORY_DATA_ASSETS, VALIDATE_LOGIC (+ missingness),
    VALIDATE_EVIDENCE, and VALIDATE_CROSS_SOURCE for every study that has at
    least one raw_fact -- idempotent via enqueue_task's default key, same
    as enqueue_seed_backfill/enqueue_text_extraction_backfill."""
    study_ids = list(session.scalars(select(RawFact.study_id).distinct()).all())

    counts = {"inventory": 0, "logic": 0, "evidence": 0, "cross_source": 0}
    for study_id in study_ids:
        enqueue_task(session, TaskType.INVENTORY_DATA_ASSETS, study_id=study_id)
        counts["inventory"] += 1
        enqueue_task(session, TaskType.VALIDATE_LOGIC, study_id=study_id)
        counts["logic"] += 1
        enqueue_task(session, TaskType.VALIDATE_EVIDENCE, study_id=study_id)
        counts["evidence"] += 1
        enqueue_task(session, TaskType.VALIDATE_CROSS_SOURCE, study_id=study_id)
        counts["cross_source"] += 1
    return counts


TASK_HANDLERS[TaskType.INVENTORY_DATA_ASSETS] = handle_inventory_data_assets
TASK_HANDLERS[TaskType.VALIDATE_EVIDENCE] = handle_validate_evidence
TASK_HANDLERS[TaskType.VALIDATE_LOGIC] = handle_validate_logic_and_missingness
TASK_HANDLERS[TaskType.VALIDATE_CROSS_SOURCE] = handle_validate_cross_source
TASK_HANDLERS[TaskType.VALIDATE_FAIRE_COMPLETENESS] = handle_validate_faire_completeness
