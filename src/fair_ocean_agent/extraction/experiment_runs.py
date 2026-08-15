"""Compatibility materialization for legacy run-bound library facts.

Earlier pipeline versions attached every experimentRunMetadata candidate to
the physical sequencing_run entity. Preserve those raw facts, but derive a
separate experiment_run entity and copy only row-relevant facts onto it so
mapping/export obey FAIRe's library-level cardinality.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRelationshipType,
    SupportType,
)
from fair_ocean_agent.database.models import Entity, EntityRelationship, RawFact
from fair_ocean_agent.identity.entity_linking import get_or_create_entity as _get_or_create_entity

DERIVATION_METHOD = "derived:legacy_sequencing_run_to_experiment_run"

_EXPERIMENT_FACT_TYPES = frozenset(
    {
        "assay_name",
        "pcr_plate_id",
        "lib_id",
        "library_name",
        "experiment_accession",
        "seq_run_id",
        "run_accession",
        "sample_accession",
        "samp_name",
        "phix_perc",
        "filename",
        "filename2",
        "checksum_filename",
        "checksum_filename2",
        "associatedSequences",
        "input_read_count",
        "read_count",
        "fastq_ftp",
        "fastq_md5",
        "fastq_access_status",
        "fastq_access_checked_urls",
        "instrument_platform",
        "instrument_model",
        "library_construction_protocol",
    }
)

_ROW_ANCHOR_FACT_TYPES = frozenset(
    {
        "lib_id",
        "library_name",
        "experiment_accession",
        "run_accession",
        "seq_run_id",
        "sample_accession",
        "samp_name",
        "fastq_ftp",
        "filename",
        "filename2",
        "read_count",
        "input_read_count",
        "assay_name",
        "pcr_plate_id",
    }
)


def _first_value(facts: list[RawFact], *fact_types: str) -> str | None:
    for fact_type in fact_types:
        for fact in facts:
            if fact.fact_type_candidate == fact_type and fact.raw_value:
                return fact.raw_value
    return None


def _link(
    session: Session,
    study_id: str,
    experiment: Entity,
    target: Entity,
    relationship_type: EntityRelationshipType,
) -> None:
    existing = session.scalar(
        select(EntityRelationship.entity_relationship_id).where(
            EntityRelationship.from_entity_id == experiment.entity_id,
            EntityRelationship.to_entity_id == target.entity_id,
            EntityRelationship.relationship_type == relationship_type.value,
        )
    )
    if existing is None:
        session.add(
            EntityRelationship(
                study_id=study_id,
                from_entity_id=experiment.entity_id,
                to_entity_id=target.entity_id,
                relationship_type=relationship_type.value,
            )
        )


def materialize_legacy_experiment_runs(session: Session, study_id: str) -> int:
    """Create missing experiment entities from legacy sequencing-run facts.

    No synthetic FAIRe `lib_id` fact is created. The internal entity key is
    deliberately prefixed `internal:` when no source supplied a library or
    experiment identifier, allowing the row to exist while missingness for
    lib_id remains honest.
    """
    created = 0
    runs = list(
        session.scalars(
            select(Entity).where(
                Entity.study_id == study_id,
                Entity.entity_level == EntityLevel.SEQUENCING_RUN.value,
            )
        )
    )
    for run in runs:
        linked_experiment = session.scalar(
            select(EntityRelationship.entity_relationship_id).where(
                EntityRelationship.study_id == study_id,
                EntityRelationship.to_entity_id == run.entity_id,
                EntityRelationship.relationship_type == EntityRelationshipType.SEQUENCED_IN_RUN.value,
            )
        )
        if linked_experiment is not None:
            continue

        facts = list(session.scalars(select(RawFact).where(RawFact.entity_id == run.entity_id)))
        relevant = [fact for fact in facts if fact.fact_type_candidate in _EXPERIMENT_FACT_TYPES]
        if not relevant or not any(fact.fact_type_candidate in _ROW_ANCHOR_FACT_TYPES for fact in relevant):
            continue

        explicit_lib_id = _first_value(relevant, "lib_id", "library_name", "experiment_accession")
        experiment_external_id = explicit_lib_id or f"internal:legacy:{run.entity_id}"
        experiment = _get_or_create_entity(
            session,
            study_id,
            EntityLevel.EXPERIMENT_RUN,
            experiment_external_id,
            explicit_lib_id or f"Library linked to {run.external_identifier or run.entity_id}",
        )
        _link(session, study_id, experiment, run, EntityRelationshipType.SEQUENCED_IN_RUN)

        sample_id = _first_value(relevant, "sample_accession", "samp_name")
        if sample_id:
            sample = _get_or_create_entity(session, study_id, EntityLevel.SAMPLE, sample_id, sample_id)
            _link(session, study_id, experiment, sample, EntityRelationshipType.DERIVED_FROM_SAMPLE)
            if experiment.parent_entity_id is None:
                experiment.parent_entity_id = sample.entity_id

        assay_name = _first_value(relevant, "assay_name")
        if assay_name:
            assay = _get_or_create_entity(session, study_id, EntityLevel.ASSAY, assay_name, assay_name)
            _link(session, study_id, experiment, assay, EntityRelationshipType.USES_ASSAY)

        for fact in relevant:
            duplicate = session.scalar(
                select(RawFact.fact_id).where(
                    RawFact.entity_id == experiment.entity_id,
                    RawFact.source_id == fact.source_id,
                    RawFact.fact_type_candidate == fact.fact_type_candidate,
                    RawFact.raw_value == fact.raw_value,
                    RawFact.source_locator == fact.source_locator,
                )
            )
            if duplicate is not None:
                continue
            session.add(
                RawFact(
                    study_id=study_id,
                    entity_id=experiment.entity_id,
                    source_id=fact.source_id,
                    source_locator=fact.source_locator,
                    raw_field_name=fact.raw_field_name,
                    raw_value=fact.raw_value,
                    normalized_text_value=fact.normalized_text_value,
                    evidence_quote=fact.evidence_quote,
                    fact_type_candidate=fact.fact_type_candidate,
                    entity_level=EntityLevel.EXPERIMENT_RUN.value,
                    support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
                    extraction_method=DERIVATION_METHOD,
                    extractor_version=fact.extractor_version,
                    model_name=fact.model_name,
                    prompt_version=fact.prompt_version,
                    confidence_metadata={"derived_from_fact_id": fact.fact_id},
                    review_status=fact.review_status,
                )
            )
        created += 1
    session.flush()
    return created
