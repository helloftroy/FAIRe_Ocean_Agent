"""Seed-study ingestion (section 5). Loads a CSV or JSONL file of seed
studies, validates/normalizes whatever identifiers are present, merges into
an existing canonical study on an exact identifier match, otherwise creates
a new candidate study -- and always preserves the original seed row as a
raw_fact for provenance.

This module does NOT enqueue discovery tasks; that is a separate, explicit
step (`enqueue_seed_backfill`, wired to the `enqueue-seed-backfill` CLI
command) so re-running ingestion is safe without re-queuing work.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    EntityLevel,
    IdentifierType,
    RelevanceStatus,
    ReviewStatus,
    SupportType,
    TaskType,
)
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Study
from fair_ocean_agent.identity.deduplication import find_existing_study_by_any_identifier
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier
from fair_ocean_agent.workflow.task_queue import enqueue_task

SEED_COLUMNS = (
    "seed_id",
    "title",
    "doi",
    "pmid",
    "pmcid",
    "bioproject_accession",
    "ena_study_accession",
    "dataset_id",
    "repository",
    "url",
    "notes",
)

_COLUMN_TO_IDENTIFIER_TYPE = {
    "doi": IdentifierType.DOI,
    "pmid": IdentifierType.PMID,
    "pmcid": IdentifierType.PMCID,
    "bioproject_accession": IdentifierType.BIOPROJECT_ACCESSION,
    "ena_study_accession": IdentifierType.ENA_STUDY_ACCESSION,
    "url": IdentifierType.URL,
}


class SeedRow(BaseModel):
    seed_id: str | None = None
    title: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    bioproject_accession: str | None = None
    ena_study_accession: str | None = None
    dataset_id: str | None = None
    repository: str | None = None
    url: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _blank_strings_to_none(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        return {k: (v if v not in ("", None) else None) for k, v in data.items()}

    def identifier_candidates(self) -> list[tuple[IdentifierType, str]]:
        candidates: list[tuple[IdentifierType, str]] = []
        for column, identifier_type in _COLUMN_TO_IDENTIFIER_TYPE.items():
            value = getattr(self, column)
            if value:
                candidates.append((identifier_type, value))
        if self.dataset_id:
            candidates.append((IdentifierType.OTHER, self.dataset_id))
        return candidates


@dataclass
class SeedIngestResult:
    seed_id: str | None
    study_id: str
    created: bool
    identifier_errors: list[str] = field(default_factory=list)


def load_seed_rows(path: str | Path) -> list[SeedRow]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(SeedRow.model_validate(json.loads(line)))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            return [SeedRow.model_validate(row) for row in reader]
    raise ValueError(f"Unsupported seed file type: {path.suffix} (expected .csv or .jsonl)")


def ingest_seed_row(session: Session, row: SeedRow) -> SeedIngestResult:
    candidates = row.identifier_candidates()

    identifier_errors: list[str] = []
    normalized: list[tuple[IdentifierType, str]] = []
    for identifier_type, raw_value in candidates:
        try:
            normalized.append((identifier_type, normalize_identifier(identifier_type, raw_value)))
        except IdentifierError as exc:
            identifier_errors.append(str(exc))

    existing_study = find_existing_study_by_any_identifier(session, normalized)
    created = existing_study is None

    if existing_study is not None:
        study = existing_study
    else:
        study = Study(
            title=row.title,
            marine_relevance_status=RelevanceStatus.UNKNOWN.value,
            molecular_relevance_status=RelevanceStatus.UNKNOWN.value,
            canonical_status=CanonicalStatus.CANDIDATE.value,
            review_status=ReviewStatus.UNREVIEWED.value,
        )
        session.add(study)
        session.flush()  # populate study.study_id before FK use below

    existing_values = {
        (ei.identifier_type, ei.identifier_value) for ei in study.external_identifiers
    }
    for identifier_type, value in normalized:
        key = (identifier_type.value, value)
        if key in existing_values:
            continue
        session.add(
            ExternalIdentifier(
                study_id=study.study_id,
                identifier_type=identifier_type.value,
                identifier_value=value,
                source=row.repository,
                is_primary=(identifier_type in (IdentifierType.DOI, IdentifierType.BIOPROJECT_ACCESSION)),
                verified=False,
            )
        )
        existing_values.add(key)

    session.add(
        RawFact(
            study_id=study.study_id,
            entity_id=None,
            source_id=None,
            source_locator="seed_file",
            raw_field_name="seed_row",
            raw_value=row.model_dump_json(),
            evidence_quote=row.model_dump_json(),
            fact_type_candidate="seed_record",
            entity_level=EntityLevel.STUDY.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method="seed_loader",
            review_status=ReviewStatus.ACCEPTED.value,
        )
    )

    return SeedIngestResult(
        seed_id=row.seed_id,
        study_id=study.study_id,
        created=created,
        identifier_errors=identifier_errors,
    )


def ingest_seed_file(session: Session, path: str | Path) -> list[SeedIngestResult]:
    rows = load_seed_rows(path)
    return [ingest_seed_row(session, row) for row in rows]


def enqueue_seed_backfill(session: Session) -> int:
    """Queue a DISCOVER_IDENTIFIERS task for every candidate study that
    doesn't already have one (idempotent via enqueue_task's default
    idempotency key). Returns the number of tasks enqueued or already
    present."""
    study_ids = session.scalars(
        select(Study.study_id).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
    ).all()
    for study_id in study_ids:
        enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=study_id)
    return len(study_ids)
