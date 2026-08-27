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
    DataAvailabilityStatus,
    EntityLevel,
    IdentifierType,
    RelevanceStatus,
    ReviewStatus,
    SupportType,
    TaskType,
)
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Study
from fair_ocean_agent.discovery.exclusions import load_excluded_dois
from fair_ocean_agent.identity.deduplication import find_existing_study_by_any_identifier
from fair_ocean_agent.identity.identifiers import (
    CNCB_EXPERIMENT_PATTERN,
    CNCB_PROJECT_PATTERN,
    CNCB_RUN_PATTERN,
    CNCB_STUDY_PATTERN,
    IdentifierError,
    normalize_identifier,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task

SEED_COLUMNS = (
    "seed_id",
    "title",
    "doi",
    "pmid",
    "pmcid",
    "bioproject_accession",
    "ena_study_accession",
    "sra_study_accession",
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
    # ena_study_accession's own normalizer only accepts ERP.../PRJEB... --
    # a DDBJ-native study is cited as DRP..., which never matches it (a
    # real seed source found live: MGnify/ENA-first seed discovery's own
    # secondary_study_accession field is frequently DRP-format). Rather
    # than silently dropping those or stretching ena_study_accession's own
    # normalizer to accept a prefix its name no longer matches,
    # sra_study_accession is its own column mapping to the identifier type
    # that already normalizes the full SRP/ERP/DRP family correctly
    # (identity/identifiers.py's SRA_STUDY_PATTERN).
    "ena_study_accession": IdentifierType.ENA_STUDY_ACCESSION,
    "sra_study_accession": IdentifierType.SRA_STUDY_ACCESSION,
    "url": IdentifierType.URL,
}


def _dataset_identifier_type(repository: str | None, dataset_id: str) -> IdentifierType:
    repo = (repository or "").lower()
    if "bco" in repo or "bcodmo" in repo or "bco-dmo" in repo:
        return IdentifierType.BCODMO_DATASET_ID
    if "pangaea" in repo:
        return IdentifierType.PANGAEA_ID
    if "obis" in repo:
        return IdentifierType.OBIS_DATASET_UUID
    if "gbif" in repo:
        return IdentifierType.GBIF_DATASET_KEY
    if "datacite" in repo:
        return IdentifierType.DATASET_DOI
    if "qiita" in repo:
        return IdentifierType.QIITA_STUDY_ID
    if "mgrast" in repo or "mg-rast" in repo:
        return IdentifierType.MGRAST_PROJECT_ID
    if "cncb" in repo or "gsa" in repo or "ngdc" in repo:
        value = dataset_id.strip()
        if CNCB_PROJECT_PATTERN.match(value):
            return IdentifierType.CNCB_PROJECT_ACCESSION
        if CNCB_STUDY_PATTERN.match(value):
            return IdentifierType.CNCB_STUDY_ACCESSION
        if CNCB_EXPERIMENT_PATTERN.match(value):
            return IdentifierType.CNCB_EXPERIMENT_ACCESSION
        if CNCB_RUN_PATTERN.match(value):
            return IdentifierType.CNCB_RUN_ACCESSION
    if dataset_id.strip().lower().startswith(("10.", "doi:", "https://doi.org/")):
        return IdentifierType.DATASET_DOI
    return IdentifierType.OTHER


class SeedRow(BaseModel):
    seed_id: str | None = None
    title: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    bioproject_accession: str | None = None
    ena_study_accession: str | None = None
    sra_study_accession: str | None = None
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
            candidates.append((_dataset_identifier_type(self.repository, self.dataset_id), self.dataset_id))
        return candidates


@dataclass
class SeedIngestResult:
    seed_id: str | None
    study_id: str | None
    created: bool
    identifier_errors: list[str] = field(default_factory=list)
    skipped: bool = False


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
    if row.doi:
        try:
            normalized_doi = normalize_identifier(IdentifierType.DOI, row.doi)
        except IdentifierError:
            normalized_doi = None
        if normalized_doi is not None and normalized_doi in load_excluded_dois():
            return SeedIngestResult(seed_id=row.seed_id, study_id=None, created=False, skipped=True)

    candidates = row.identifier_candidates()
    seed_payload = row.model_dump_json()

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

    existing_seed_fact = session.scalars(
        select(RawFact).where(
            RawFact.study_id == study.study_id,
            RawFact.source_locator == "seed_file",
            RawFact.raw_field_name == "seed_row",
            RawFact.fact_type_candidate == "seed_record",
            RawFact.raw_value == seed_payload,
        )
    ).first()
    if existing_seed_fact is None:
        session.add(
            RawFact(
                study_id=study.study_id,
                entity_id=None,
                source_id=None,
                source_locator="seed_file",
                raw_field_name="seed_row",
                raw_value=seed_payload,
                evidence_quote=seed_payload,
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
    present.

    Excludes a study already marked NOT_ACCESSIBLE (workflow/handlers.py's
    _has_accessible_sequence_data_signal, checked at the end of a prior
    DISCOVER_IDENTIFIERS run) -- per an explicit user request, once the
    staged repository search has found nothing accessible for a study, a
    plain re-run of this same backfill shouldn't keep re-queueing it for
    the identical search."""
    study_ids = session.scalars(
        select(Study.study_id).where(
            Study.canonical_status == CanonicalStatus.CANDIDATE.value,
            Study.data_availability_status != DataAvailabilityStatus.NOT_ACCESSIBLE.value,
        )
    ).all()
    for study_id in study_ids:
        enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=study_id)
    return len(study_ids)
