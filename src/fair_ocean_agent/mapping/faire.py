"""Applies mapping/rules.py's RULES table against a study's raw_facts to
produce StandardizedValue (+ StandardizedValueEvidence) rows targeting
FAIRe (target_schema="faire").

Two design decisions worth stating explicitly:

1. **entity_id resolution is table-shaped, not fact-shaped.**
   `projectMetadata` is a study-wide singleton table in FAIRe -- it doesn't
   have a per-sample or per-run row -- so every projectMetadata-targeted
   StandardizedValue gets `entity_id=None` regardless of which entity the
   source RawFact was attached to (e.g. `instrument_platform` facts live on
   `sequencing_run` entities in this pipeline, but 500 identical per-run
   facts must collapse to one project-wide `platform` value, not 500 rows).
   Similarly, a study-level LLM-extracted fact (`DNA_extraction_method`,
   `storage_conditions`, ...) mapped to a `sampleMetadata` field has no
   sample-specific entity_id to inherit -- `entity_id=None` there means
   "applies to every sample row as a broadcast default", which
   `exports/faire.py` interprets accordingly.

2. **`sample_accession` -> `materialSampleID` is a targeted redirect, not a
   broadcast.** A run's BioSample accession belongs to exactly one sample,
   so this rule looks up the `sample` Entity whose `external_identifier`
   matches the fact's value and attaches the StandardizedValue there --
   never `entity_id=None` (which would incorrectly broadcast one sample's
   accession onto every sample in the study).

Because multiple raw_facts (up to 500 sequencing-run facts per study) can
target the same study-wide (entity_id=None) field, this module de-
duplicates by (target_table, target_field, entity_id): the first value
wins; if a later fact disagrees, the existing row is flagged
`review_required=True` rather than silently overwritten or duplicated --
see `_upsert_study_wide` below.
"""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel
from fair_ocean_agent.database.models import (
    Entity,
    ExternalIdentifier,
    RawFact,
    StandardizedValue,
    StandardizedValueEvidence,
)
from fair_ocean_agent.mapping import vocabularies
from fair_ocean_agent.mapping.rules import MappingRule, rules_for

TARGET_SCHEMA = "faire"
TARGET_SCHEMA_VERSION = "1.0.2"


def _clear_existing_faire_mappings(session: Session, study_id: str) -> None:
    existing_ids = list(
        session.scalars(
            select(StandardizedValue.standardized_value_id).where(
                StandardizedValue.study_id == study_id,
                StandardizedValue.target_schema == TARGET_SCHEMA,
            )
        )
    )
    if not existing_ids:
        return
    session.execute(
        delete(StandardizedValueEvidence).where(
            StandardizedValueEvidence.standardized_value_id.in_(existing_ids)
        )
    )
    session.execute(
        delete(StandardizedValue).where(StandardizedValue.standardized_value_id.in_(existing_ids))
    )
    session.flush()


def _find_sample_entity_by_external_id(session: Session, study_id: str, external_id: str) -> Entity | None:
    return session.scalar(
        select(Entity).where(
            Entity.study_id == study_id,
            Entity.entity_level == EntityLevel.SAMPLE.value,
            Entity.external_identifier == external_id,
        )
    )


def _resolve_entity_id(session: Session, study_id: str, fact: RawFact, rule: MappingRule) -> str | None:
    if rule.target_field == "materialSampleID":
        sample = _find_sample_entity_by_external_id(session, study_id, fact.raw_value)
        return sample.entity_id if sample else None
    if rule.target_table == "projectMetadata":
        return None
    if fact.entity_level in (EntityLevel.SAMPLE.value, EntityLevel.SEQUENCING_RUN.value):
        # A run-level fact (e.g. ENA's per-run fastq_ftp/fastq_md5/read_count,
        # mapped onto experimentRunMetadata's filename/checksum_filename/
        # input_read_count) genuinely differs run to run -- unlike
        # instrument_platform/instrument_model, which map onto
        # projectMetadata and are expected to agree across a study's runs
        # (see the target_table check above), so those still correctly
        # collapse to one project-wide row. Before this, every
        # SEQUENCING_RUN-level rule targeting a *per-entity* table (not
        # projectMetadata) fell through to the study-wide broadcast default
        # below, silently discarding all but one run's value the first time
        # a real per-run field (rather than platform/instrument) was mapped.
        return fact.entity_id
    return None  # study-wide fact mapped onto a sample-scoped field: broadcast default


def map_study_to_faire(session: Session, study_id: str) -> int:
    """Idempotent: re-derives every FAIRe StandardizedValue for a study
    from scratch each time it's called (delete-then-recreate), so it's safe
    to call again after new raw_facts arrive or after a rules.py change."""
    _clear_existing_faire_mappings(session, study_id)

    facts = list(session.scalars(select(RawFact).where(RawFact.study_id == study_id)))
    created = 0
    seen: dict[tuple[str, str, str | None], StandardizedValue] = {}

    for fact in facts:
        if fact.fact_type_candidate is None or fact.raw_value is None:
            continue
        for rule in rules_for(fact.fact_type_candidate, fact.entity_level):
            if rule.target_field == "materialSampleID":
                entity_id = _resolve_entity_id(session, study_id, fact, rule)
                if entity_id is None:
                    continue  # no matching sample entity -- don't fabricate one
            else:
                entity_id = _resolve_entity_id(session, study_id, fact, rule)

            value = rule.transform(fact.raw_value)
            if value is None:
                continue

            review_required = rule.review_required
            if rule.enum_name:
                check = vocabularies.check_value(rule.enum_name, value)
                if not check.is_valid:
                    review_required = True

            key = (rule.target_table, rule.target_field, entity_id)
            existing = seen.get(key)
            if existing is not None:
                if existing.standardized_value != value:
                    existing.review_required = True
                continue

            standardized_value = StandardizedValue(
                study_id=study_id,
                entity_id=entity_id,
                target_schema=TARGET_SCHEMA,
                target_schema_version=TARGET_SCHEMA_VERSION,
                target_field=rule.target_field,
                standardized_value=value,
                mapping_method=rule.mapping_method,
                review_required=review_required,
            )
            session.add(standardized_value)
            session.flush()
            session.add(
                StandardizedValueEvidence(
                    standardized_value_id=standardized_value.standardized_value_id,
                    fact_id=fact.fact_id,
                )
            )
            seen[key] = standardized_value
            created += 1

    return created


def resolve_project_id(session: Session, study_id: str) -> str | None:
    """The FAIRe `project_id` join key. Not a mapped raw_fact -- it's read
    directly from ExternalIdentifier (an accession the study was resolved
    against, not something an extractor "found"), so it never gets a
    StandardizedValue/evidence row of its own. Preference order: a
    BioProject accession is the most FAIRe-idiomatic project identifier;
    fall back to other repository-level accessions, then DOI, so a study
    reached only through a publication still gets a usable project_id."""
    from fair_ocean_agent.database.enums import IdentifierType

    for identifier_type in (
        IdentifierType.BIOPROJECT_ACCESSION,
        IdentifierType.ENA_STUDY_ACCESSION,
        IdentifierType.SRA_STUDY_ACCESSION,
        IdentifierType.DOI,
    ):
        value = session.scalar(
            select(ExternalIdentifier.identifier_value).where(
                ExternalIdentifier.study_id == study_id,
                ExternalIdentifier.identifier_type == identifier_type.value,
            )
        )
        if value:
            return value
    return None
