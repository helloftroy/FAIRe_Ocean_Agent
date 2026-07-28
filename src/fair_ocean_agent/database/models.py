"""SQLAlchemy 2.x declarative models for the full schema (section 4 of the
design brief). All tables are created in Milestone 1; most are only
populated starting in later milestones (extraction, mapping, validation,
scheduling). Written to be SQLite-compatible now and PostgreSQL-compatible
later without migration-breaking changes (no SQLite-only types, JSON column
works as SQLite JSON and PostgreSQL JSONB).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from fair_ocean_agent.database.enums import (
    AccessStatus,
    AssetType,
    CandidateMatchMethod,
    CandidateMatchReviewStatus,
    CanonicalStatus,
    EntityLevel,
    IdentifierType,
    InspectionLevel,
    InspectionStatus,
    MappingMethod,
    RawOrProcessed,
    RelationshipType,
    RelevanceStatus,
    ReviewStatus,
    SourceType,
    SupportType,
    TaskStatus,
    TaskType,
    ValidationSeverity,
    ValidationStatus,
    WorkflowRunStatus,
    WorkflowRunType,
)
from fair_ocean_agent.database.ids import new_id
from fair_ocean_agent.clock import utcnow as _utcnow


class Base(DeclarativeBase):
    pass


JsonDocument = JSON().with_variant(JSONB, "postgresql")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Study(Base, TimestampMixin):
    __tablename__ = "studies"

    study_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("STUDY")
    )
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    study_type: Mapped[str | None] = mapped_column(String)
    marine_relevance_status: Mapped[str] = mapped_column(
        String, default=RelevanceStatus.UNKNOWN.value
    )
    molecular_relevance_status: Mapped[str] = mapped_column(
        String, default=RelevanceStatus.UNKNOWN.value
    )
    canonical_status: Mapped[str] = mapped_column(
        String, default=CanonicalStatus.CANDIDATE.value
    )
    review_status: Mapped[str] = mapped_column(
        String, default=ReviewStatus.UNREVIEWED.value
    )

    external_identifiers: Mapped[list["ExternalIdentifier"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )
    entities: Mapped[list["Entity"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )
    sources: Mapped[list["Source"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )


class ExternalIdentifier(Base, TimestampMixin):
    __tablename__ = "external_identifiers"
    __table_args__ = (
        UniqueConstraint(
            "identifier_type", "identifier_value", "study_id",
            name="uq_identifier_type_value_study",
        ),
    )

    identifier_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("EXTID")
    )
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    identifier_type: Mapped[str] = mapped_column(String)
    identifier_value: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str | None] = mapped_column(String)
    relationship_type: Mapped[str | None] = mapped_column(String)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)

    study: Mapped["Study"] = relationship(back_populates="external_identifiers")


class Source(Base, TimestampMixin):
    """A publication is just one flavor of source (source_type=
    "publication_api"), not a separate table -- `authors`/`journal`/
    `publication_year`/`fulltext_available` below are only ever populated
    for that flavor, left null otherwise. This replaces the old
    `publications` table (folded in as part of the Milestone 7 schema
    simplification): that table kept a single row per study merged
    ("first non-null wins") across crossref/europe_pmc/openalex, which was
    two tables that both had to stay in sync for the same DOI -- exactly
    the kind of bug magnet the rest of this schema avoids by treating
    facts as an append-only evidence log (raw_facts, standardized_values)
    rather than a mutable "current value" cache. Now each adapter's own
    Source row carries its own bibliographic fields as that adapter
    reported them; a consumer that wants one merged answer (e.g. "what
    publication year should validate this collection date") queries across
    a study's source_type="publication_api" rows directly (see
    workflow/validation_handlers.py's `_publication_year_for_study`)
    instead of reading a separately-synced aggregate.

    `doi`/`pmid`/`pmcid`/`openalex_id` are deliberately NOT columns here --
    `external_identifier` already holds the DOI for every publication-type
    source row (every publication adapter is queried by DOI, see
    workflow/handlers.py's `_resolve_publication_sources`), and any other
    cross-reference (PMID, PMCID, OpenAlex ID) belongs on
    `ExternalIdentifier`, which is already the one authoritative place for
    a study's identifiers -- duplicating them here would recreate the same
    sync problem this fold removes. `title` isn't a column here either --
    `Study.title` is already the canonical merged title.
    """

    __tablename__ = "sources"

    source_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("SRC")
    )
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    source_type: Mapped[str] = mapped_column(String)
    source_name: Mapped[str] = mapped_column(String)
    external_identifier: Mapped[str | None] = mapped_column(String, index=True)
    url: Mapped[str | None] = mapped_column(String)
    access_status: Mapped[str] = mapped_column(String, default=AccessStatus.UNKNOWN.value)
    license: Mapped[str | None] = mapped_column(String)
    is_mirror: Mapped[bool] = mapped_column(Boolean, default=False)
    mirror_group: Mapped[str | None] = mapped_column(String, index=True)
    source_version: Mapped[str | None] = mapped_column(String)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str | None] = mapped_column(String)
    inspection_status: Mapped[str] = mapped_column(
        String, default=InspectionStatus.NOT_INSPECTED.value
    )
    inspection_level: Mapped[str] = mapped_column(
        String, default=InspectionLevel.NONE.value
    )
    parent_source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.source_id"))
    # Publication-flavored fields (source_type="publication_api" only; see
    # class docstring). open_access_status folds onto access_status above
    # -- same AccessStatus enum, same meaning, no separate column needed.
    authors: Mapped[list | None] = mapped_column(JsonDocument)
    journal: Mapped[str | None] = mapped_column(String)
    publication_year: Mapped[int | None] = mapped_column(Integer)
    fulltext_available: Mapped[bool] = mapped_column(Boolean, default=False)

    study: Mapped["Study"] = relationship(back_populates="sources")


class SourceRelationship(Base, TimestampMixin):
    __tablename__ = "source_relationships"

    source_relationship_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("SRCREL")
    )
    from_source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    to_source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    relationship_type: Mapped[str] = mapped_column(String)


class Entity(Base, TimestampMixin):
    """Generic study sub-entity: project, sampling_event, sample, assay,
    sequencing_run, protocol, bioinformatics_workflow, data_asset (see
    EntityLevel). Kept generic (rather than one table per level) so new
    entity levels don't require a migration; entity-level-specific
    structured attributes live in raw_facts/standardized_values keyed by
    entity_id.

    external_identifier (added Milestone 3) is the entity's own accession
    when a source assigns one below the study level -- a BioSample
    accession for a `sample` entity, an ENA/SRA run accession for a
    `sequencing_run` entity, a BioProject accession for a `project` entity.
    Nullable because most entity levels (sampling_event, assay, protocol,
    ...) aren't independently accessioned anywhere. The unique constraint
    is what makes get-or-create-by-accession idempotent across retries;
    NULL values are exempt from a UNIQUE constraint in both SQLite and
    PostgreSQL (NULL <> NULL), so any number of entities without an
    external identifier is fine.
    """

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint(
            "study_id", "entity_level", "external_identifier",
            name="uq_entity_study_level_external_id",
        ),
    )

    entity_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("ENT")
    )
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    entity_level: Mapped[str] = mapped_column(String)
    label: Mapped[str | None] = mapped_column(String)
    external_identifier: Mapped[str | None] = mapped_column(String, index=True)
    parent_entity_id: Mapped[str | None] = mapped_column(ForeignKey("entities.entity_id"))

    study: Mapped["Study"] = relationship(back_populates="entities")


class RawFact(Base, TimestampMixin):
    __tablename__ = "raw_facts"
    __table_args__ = (
        Index("ix_raw_facts_study_fact_type_entity", "study_id", "fact_type_candidate", "entity_id"),
    )

    fact_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("FACT")
    )
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    entity_id: Mapped[str | None] = mapped_column(ForeignKey("entities.entity_id"), index=True)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.source_id"), index=True)
    source_locator: Mapped[str | None] = mapped_column(String)
    raw_field_name: Mapped[str | None] = mapped_column(String)
    raw_value: Mapped[str | None] = mapped_column(Text)
    normalized_text_value: Mapped[str | None] = mapped_column(Text)
    evidence_quote: Mapped[str | None] = mapped_column(Text)
    fact_type_candidate: Mapped[str | None] = mapped_column(String, index=True)
    entity_level: Mapped[str | None] = mapped_column(String)
    support_type: Mapped[str] = mapped_column(String)
    extraction_method: Mapped[str | None] = mapped_column(String)
    extractor_version: Mapped[str | None] = mapped_column(String)
    model_name: Mapped[str | None] = mapped_column(String)
    prompt_version: Mapped[str | None] = mapped_column(String)
    confidence_metadata: Mapped[dict | None] = mapped_column(JsonDocument)
    review_status: Mapped[str] = mapped_column(String, default=ReviewStatus.UNREVIEWED.value)


class StandardizedValue(Base, TimestampMixin):
    """Missingness tracking is folded into this table (Milestone 7 schema
    simplification) rather than kept in a separate `missingness` table:
    every (study, entity, target_field) this pipeline checks for gets
    exactly one row here, whether or not a value was actually found.
    `standardized_value` is null and `missingness_status` carries why
    (`not_found_in_inspected_sources`, `relevant_source_not_inspected`,
    etc. -- see MissingnessStatus) when nothing was found; when a value
    *was* found, `missingness_status` is simply "present" alongside the
    real `standardized_value`. Same tracking, one table instead of two
    that both had to agree on which (study, field) pairs existed.

    `sources_inspected`/`reason` are only ever meaningful for missingness
    rows (nothing to explain when a real value is sitting right there),
    left null otherwise.
    """

    __tablename__ = "standardized_values"
    __table_args__ = (
        Index("ix_standardized_values_schema_study_field", "target_schema", "study_id", "target_field"),
    )

    standardized_value_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("STDVAL")
    )
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    entity_id: Mapped[str | None] = mapped_column(ForeignKey("entities.entity_id"), index=True)
    target_schema: Mapped[str] = mapped_column(String)
    target_schema_version: Mapped[str] = mapped_column(String)
    target_field: Mapped[str] = mapped_column(String)
    standardized_value: Mapped[str | None] = mapped_column(Text)
    controlled_term_id: Mapped[str | None] = mapped_column(String)
    mapping_method: Mapped[str] = mapped_column(String, default=MappingMethod.UNRESOLVED.value)
    validation_status: Mapped[str] = mapped_column(
        String, default=ValidationStatus.NOT_ASSESSED.value
    )
    review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    missingness_status: Mapped[str | None] = mapped_column(String)
    sources_inspected: Mapped[list | None] = mapped_column(JsonDocument)
    reason: Mapped[str | None] = mapped_column(Text)


class StandardizedValueEvidence(Base):
    __tablename__ = "standardized_value_evidence"

    standardized_value_id: Mapped[str] = mapped_column(
        ForeignKey("standardized_values.standardized_value_id"), primary_key=True
    )
    fact_id: Mapped[str] = mapped_column(
        ForeignKey("raw_facts.fact_id"), primary_key=True
    )


class DataAsset(Base, TimestampMixin):
    __tablename__ = "data_assets"

    asset_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("ASSET")
    )
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    entity_id: Mapped[str | None] = mapped_column(ForeignKey("entities.entity_id"), index=True)
    asset_type: Mapped[str] = mapped_column(String)
    repository: Mapped[str | None] = mapped_column(String)
    identifier: Mapped[str | None] = mapped_column(String)
    file_name: Mapped[str | None] = mapped_column(String)
    format: Mapped[str | None] = mapped_column(String)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    access_status: Mapped[str] = mapped_column(String, default=AccessStatus.UNKNOWN.value)
    license: Mapped[str | None] = mapped_column(String)
    raw_or_processed: Mapped[str] = mapped_column(String, default=RawOrProcessed.UNKNOWN.value)
    description: Mapped[str | None] = mapped_column(Text)
    n_samples_reported: Mapped[int | None] = mapped_column(Integer)
    n_samples_observed: Mapped[int | None] = mapped_column(Integer)
    n_features_reported: Mapped[int | None] = mapped_column(Integer)
    n_features_observed: Mapped[int | None] = mapped_column(Integer)
    feature_type: Mapped[str | None] = mapped_column(String)
    matrix_orientation: Mapped[str | None] = mapped_column(String)
    taxonomy_available: Mapped[bool | None] = mapped_column(Boolean)
    environmental_metadata_available: Mapped[bool | None] = mapped_column(Boolean)
    sample_metadata_available: Mapped[bool | None] = mapped_column(Boolean)
    inspection_level: Mapped[str] = mapped_column(String, default=InspectionLevel.NONE.value)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.source_id"))


class ValidationResult(Base):
    __tablename__ = "validation_results"

    validation_result_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("VALID")
    )
    study_id: Mapped[str | None] = mapped_column(ForeignKey("studies.study_id"), index=True)
    entity_id: Mapped[str | None] = mapped_column(ForeignKey("entities.entity_id"))
    fact_id: Mapped[str | None] = mapped_column(ForeignKey("raw_facts.fact_id"))
    standardized_value_id: Mapped[str | None] = mapped_column(
        ForeignKey("standardized_values.standardized_value_id")
    )
    validator_name: Mapped[str] = mapped_column(String)
    validator_version: Mapped[str | None] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String, default=ValidationSeverity.INFO.value)
    status: Mapped[str] = mapped_column(String, default=ValidationStatus.NOT_ASSESSED.value)
    message: Mapped[str | None] = mapped_column(Text)
    compared_values: Mapped[dict | None] = mapped_column(JsonDocument)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CandidateMatch(Base, TimestampMixin):
    __tablename__ = "candidate_matches"

    candidate_match_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("CANDM")
    )
    study_a_id: Mapped[str] = mapped_column(ForeignKey("studies.study_id"), index=True)
    study_b_id: Mapped[str | None] = mapped_column(ForeignKey("studies.study_id"), index=True)
    candidate_record_ref: Mapped[str | None] = mapped_column(String)
    match_score: Mapped[float | None] = mapped_column(Float)
    supporting_features: Mapped[dict | None] = mapped_column(JsonDocument)
    match_method: Mapped[str] = mapped_column(String, default=CandidateMatchMethod.COMPOSITE_SCORE.value)
    review_status: Mapped[str] = mapped_column(
        String, default=CandidateMatchReviewStatus.PENDING.value
    )
    reviewer_decision: Mapped[str | None] = mapped_column(Text)


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_claimable", "status", "available_after", "priority", "created_at"),
        Index("ix_tasks_type_status_available", "task_type", "status", "available_after"),
        UniqueConstraint("idempotency_key", name="uq_task_idempotency_key"),
    )

    task_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("TASK")
    )
    task_type: Mapped[str] = mapped_column(String, index=True)
    study_id: Mapped[str | None] = mapped_column(ForeignKey("studies.study_id"), index=True)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.source_id"))
    payload: Mapped[dict | None] = mapped_column(JsonDocument)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    status: Mapped[str] = mapped_column(String, default=TaskStatus.PENDING.value, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    available_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    claimed_by: Mapped[str | None] = mapped_column(String)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String, index=True)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    run_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("RUN")
    )
    run_type: Mapped[str] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default=WorkflowRunStatus.RUNNING.value)
    code_version: Mapped[str | None] = mapped_column(String)
    model_config_snapshot: Mapped[dict | None] = mapped_column(JsonDocument)
    prompt_version: Mapped[str | None] = mapped_column(String)
    schema_versions: Mapped[dict | None] = mapped_column(JsonDocument)
    sources_queried: Mapped[dict | None] = mapped_column(JsonDocument)
    candidates_found: Mapped[int] = mapped_column(Integer, default=0)
    new_studies: Mapped[int] = mapped_column(Integer, default=0)
    updated_studies: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    manual_review_items: Mapped[int] = mapped_column(Integer, default=0)


class SourceWatermark(Base, TimestampMixin):
    __tablename__ = "source_watermarks"
    __table_args__ = (
        UniqueConstraint("source_name", "query_identifier", name="uq_source_watermark_query"),
    )

    watermark_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("WMARK")
    )
    source_name: Mapped[str] = mapped_column(String)
    query_identifier: Mapped[str] = mapped_column(String)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_cursor: Mapped[str | None] = mapped_column(String)
    last_run_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_runs.run_id"))
    overlap_window_days: Mapped[int] = mapped_column(Integer, default=14)
    last_status: Mapped[str | None] = mapped_column(String)
