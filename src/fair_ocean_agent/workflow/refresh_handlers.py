"""REFRESH_STUDY_SOURCES (Milestone 7): re-checks an already-known study's
existing DOI/BioProject/ENA identifiers against their sources, catching
real-world changes a one-time DISCOVER_IDENTIFIERS never revisits -- a
BioProject genuinely gaining samples past a prior
`MAX_SAMPLES_PER_PROJECT` cap, a citation count updating, previously
embargoed data becoming available.

This deliberately reuses handlers.py's `_resolve_publication_sources` /
`_resolve_repository_sources` (adapter traversal, study.title merging,
related-identifier discovery) via their `persist_fn` parameter, rather
than duplicating that logic -- only *how a fetched record gets persisted*
differs between discovery and refresh:

- Discovery (`_persist_source_and_facts`): skip entirely if a Source row
  for this (study, adapter, identifier) already exists at all -- an
  existing record means "already handled," full stop.
- Refresh (`_persist_refreshed_source_and_facts` below): always fetch
  fresh (the caller clears the adapter's HTTP cache first -- see
  scheduling/weekly.py), then compare `content_hash` against the most
  recent prior Source row for that (study, adapter, identifier). Identical
  content -> nothing to do but record the check. Different content -> a
  new Source row is created (linked via `parent_source_id` to the one it
  supersedes) with its own fresh RawFacts and, for publication-type
  sources, its own fresh bibliographic fields (authors/journal/
  publication_year/fulltext_available -- same `_apply_publication_fields`
  helper the discovery path uses) -- rather than mutating or deleting the
  old snapshot -- consistent with raw_facts being an append-only evidence
  log everywhere else in this pipeline (Milestone 5's validators already
  assume a fact can be compared across sources, not just trusted as "the
  current value").

Known limitation, not fixed here: `mapping/faire.py` doesn't yet prefer
the most recent Source's facts when a field has been recorded by more
than one Source snapshot for the same entity (its dedup is "first
raw_fact encountered wins", by RawFact iteration order) -- so a refresh
that changes a value doesn't automatically flow through to a re-mapped
StandardizedValue on its own. Re-running `enqueue-mapping-backfill` after
a refresh with real changes is still required. Fixing mapping/faire.py to
prefer the newest Source is a follow-up, not attempted here to keep this
milestone's actual scope (detecting and recording change) separate from
Milestone 6's mapping-layer concern.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import (
    AccessStatus,
    IdentifierType,
    InspectionLevel,
    InspectionStatus,
    ReviewStatus,
    SourceType,
    TaskType,
)
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Source, Study, Task
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.scheduling.watermarks import record_check
from fair_ocean_agent.sources.base import SourceAdapter, SourceRecord
from fair_ocean_agent.workflow.handlers import (
    _apply_publication_fields,
    _get_or_create_entity,
    _resolve_publication_sources,
    _resolve_repository_sources,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)


class RefreshOutcome:
    """Accumulates what happened during one study's refresh, for the
    caller (scheduling/weekly.py) to fold into a WorkflowRun's tallies."""

    def __init__(self) -> None:
        self.checked = 0
        self.changed = 0

    def record(self, changed: bool) -> None:
        self.checked += 1
        if changed:
            self.changed += 1


def _persist_refreshed_source_and_facts(
    session: Session,
    study: Study,
    adapter: SourceAdapter,
    source_type: SourceType,
    query_identifier: str,
    record: SourceRecord,
    *,
    run_id: str | None,
    outcome: RefreshOutcome,
) -> bool:
    latest_source = (
        session.query(Source)
        .filter_by(study_id=study.study_id, source_name=adapter.name, external_identifier=query_identifier)
        .order_by(Source.created_at.desc())
        .first()
    )
    changed = latest_source is None or latest_source.content_hash != record.content_hash
    record_check(
        session, adapter.name, query_identifier,
        status="changed" if changed else "unchanged", run_id=run_id,
    )
    outcome.record(changed)

    if not changed:
        return False

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
        parent_source_id=latest_source.source_id if latest_source else None,
    )
    session.add(source)
    session.flush()

    if source_type == SourceType.PUBLICATION_API:
        _apply_publication_fields(source, adapter.parse_publication_fields(record))

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
                confidence_metadata=fact.confidence_metadata,
            )
        )

    logger.info(
        "refresh found a change for study %s / %s %s (new source %s, prior %s)",
        study.study_id, adapter.name, query_identifier, source.source_id,
        latest_source.source_id if latest_source else None,
    )
    return True


def refresh_study(session: Session, study: Study, *, run_id: str | None = None) -> RefreshOutcome:
    """Re-checks every source this study already has a DOI/BioProject/ENA
    identifier for. Caller is responsible for clearing each enabled
    adapter's HTTP cache beforehand (once per weekly-update run, not once
    per study -- see scheduling/weekly.py) so fetch_record() below actually
    hits the network instead of replaying an old cached response."""

    def identifier_value(identifier_type: IdentifierType) -> str | None:
        ei = next((e for e in study.external_identifiers if e.identifier_type == identifier_type.value), None)
        return ei.identifier_value if ei else None

    doi = identifier_value(IdentifierType.DOI)
    bioproject_accession = identifier_value(IdentifierType.BIOPROJECT_ACCESSION)
    ena_accession = identifier_value(IdentifierType.ENA_STUDY_ACCESSION)

    outcome = RefreshOutcome()

    def persist_fn(session, study, adapter, source_type, query_identifier, record):
        return _persist_refreshed_source_and_facts(
            session, study, adapter, source_type, query_identifier, record, run_id=run_id, outcome=outcome
        )

    if doi:
        study = _resolve_publication_sources(session, study, doi, persist_fn=persist_fn)
    if bioproject_accession or ena_accession:
        study = _resolve_repository_sources(session, study, bioproject_accession, ena_accession, persist_fn=persist_fn)

    return outcome


def handle_refresh_study_sources(session: Session, task: Task) -> None:
    study = session.get(Study, task.study_id)
    if study is None:
        raise ValueError(f"Study {task.study_id} not found")
    run_id = (task.payload or {}).get("run_id")
    outcome = refresh_study(session, study, run_id=run_id)
    logger.info(
        "refreshed study %s: checked %d source(s), %d changed",
        study.study_id, outcome.checked, outcome.changed,
    )


def enqueue_refresh_backfill(session: Session, *, run_id: str | None = None) -> int:
    """Queues REFRESH_STUDY_SOURCES for every study that has at least one
    DOI/BioProject/ENA identifier (the only ones any adapter can refresh
    today)."""
    refreshable_types = (
        IdentifierType.DOI.value,
        IdentifierType.BIOPROJECT_ACCESSION.value,
        IdentifierType.ENA_STUDY_ACCESSION.value,
    )
    study_ids = list(
        session.scalars(
            select(ExternalIdentifier.study_id)
            .where(ExternalIdentifier.identifier_type.in_(refreshable_types))
            .distinct()
        )
    )
    for study_id in study_ids:
        enqueue_task(
            session, TaskType.REFRESH_STUDY_SOURCES, study_id=study_id,
            payload={"run_id": run_id} if run_id else None,
        )
    return len(study_ids)


TASK_HANDLERS[TaskType.REFRESH_STUDY_SOURCES] = handle_refresh_study_sources
