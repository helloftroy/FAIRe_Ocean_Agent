"""`SchedulingConfig.quarterly_full_rediscovery`: periodically re-run
DISCOVER_IDENTIFIERS for every candidate study, not just ones with a
failed/manual-review task -- the concrete case this catches is a
newly-enabled adapter in config/sources.yaml (or a newly-supported
identifier type) that existing studies were never checked against, since
`_resolve_publication_sources`/`_resolve_repository_sources` iterate
`_build_enabled_adapters()` fresh on every call and only skip a
(study, adapter, identifier) combination that already has a Source row --
a study checked before openalex was enabled, for example, would happily
pick up an openalex Source the next time DISCOVER_IDENTIFIERS runs for it.

This needs a genuinely new Task row each time it fires, not a plain
`enqueue_task` call -- the default idempotency key
(task_type, study_id, source_id, payload) already exists forever once a
study's first DISCOVER_IDENTIFIERS task was created, so a second call
with the same key is a no-op (see enqueue_task's docstring in
workflow/task_queue.py). Passing an explicit `idempotency_key` that
includes the triggering run's id sidesteps that -- see
scheduling/weekly.py for how run_id is threaded through.

Cadence ("quarterly") isn't tracked with a new table -- it's read directly
from WorkflowRun history (has a completed run_type="quarterly_full_rediscovery"
row already fired within the window?), since that's exactly what
WorkflowRun already exists to record.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import as_aware_utc, utcnow
from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    DataAvailabilityStatus,
    IdentifierType,
    TaskType,
    WorkflowRunStatus,
)
from fair_ocean_agent.database.models import ExternalIdentifier, Study, WorkflowRun
from fair_ocean_agent.workflow.task_queue import enqueue_task

RUN_TYPE = "quarterly_full_rediscovery"
DEFAULT_INTERVAL_DAYS = 90

CITATION_RUN_TYPE = "citation_rediscovery"
CITATION_DEFAULT_INTERVAL_DAYS = 90


def is_rediscovery_due(session: Session, interval_days: int = DEFAULT_INTERVAL_DAYS) -> bool:
    last_run = session.scalars(
        select(WorkflowRun)
        .where(WorkflowRun.run_type == RUN_TYPE, WorkflowRun.status == WorkflowRunStatus.COMPLETED.value)
        .order_by(WorkflowRun.started_at.desc())
    ).first()
    if last_run is None:
        return True
    return utcnow() - as_aware_utc(last_run.started_at) >= timedelta(days=interval_days)


def enqueue_full_rediscovery(session: Session, run_id: str) -> int:
    """Unconditional -- callers check is_rediscovery_due first. Targets
    every non-merged study (canonical_status == CANDIDATE, same scope as
    discovery/seed_loader.py's enqueue_seed_backfill -- CANONICAL/REJECTED
    are never actually set anywhere in this codebase yet, but the filter
    is here for when they are), excluding one already marked
    NOT_ACCESSIBLE (workflow/handlers.py's _has_accessible_sequence_data_signal)
    -- this function's whole purpose is bypassing normal task idempotency
    via a run-scoped key specifically to force periodic re-processing, so
    without this filter it would keep re-running the identical staged
    repository search against a study already confirmed to have nothing,
    every single rediscovery cycle, forever."""
    study_ids = session.scalars(
        select(Study.study_id).where(
            Study.canonical_status == CanonicalStatus.CANDIDATE.value,
            Study.data_availability_status != DataAvailabilityStatus.NOT_ACCESSIBLE.value,
        )
    ).all()
    for study_id in study_ids:
        enqueue_task(
            session,
            TaskType.DISCOVER_IDENTIFIERS,
            study_id=study_id,
            idempotency_key=f"{RUN_TYPE}:{study_id}:{run_id}",
        )
    return len(study_ids)


def is_citation_rediscovery_due(session: Session, interval_days: int = CITATION_DEFAULT_INTERVAL_DAYS) -> bool:
    last_run = session.scalars(
        select(WorkflowRun)
        .where(WorkflowRun.run_type == CITATION_RUN_TYPE, WorkflowRun.status == WorkflowRunStatus.COMPLETED.value)
        .order_by(WorkflowRun.started_at.desc())
    ).first()
    if last_run is None:
        return True
    return utcnow() - as_aware_utc(last_run.started_at) >= timedelta(days=interval_days)


def enqueue_citation_rediscovery_backfill(session: Session, run_id: str) -> int:
    """The "node-adding" recurring pass: workflow/handlers.py's
    handle_discover_citing_studies is normally triggered exactly once per
    BioProject accession, right after that accession is first resolved
    (see handle_discover_identifiers's enqueue hook) -- its idempotency key
    is the accession alone, by design, so two studies resolving the same
    accession only trigger it once for free. That default key means a
    paper published/indexed AFTER a study's first resolution (the whole
    "new research linking to old samples" case the user asked this
    mechanism to eventually cover) would never get picked up again on its
    own. This function re-enqueues DISCOVER_CITING_STUDIES for every
    already-known BioProject accession with a run-scoped idempotency key
    (`f"{CITATION_RUN_TYPE}:{accession}:{run_id}"`), so a periodic re-run
    is never permanently no-op'd by that first key.

    Targets distinct PROJECT-level accession *values* (not Study rows,
    unlike enqueue_full_rediscovery) -- two studies sharing one accession
    only need one re-check, and handle_discover_citing_studies itself
    already resolves which Study to treat as the citation-discovery
    parent's own EntityStudy links, not which Study happened to trigger
    this particular re-enqueue. The same depth/fan-out safety valves
    (DiscoveryConfig.citation_expansion_max_depth/
    max_citing_papers_per_bioproject) apply unconditionally on this path
    too, since they're read from config inside the handler itself, not
    passed in here."""
    accessions = sorted(
        set(
            session.scalars(
                select(ExternalIdentifier.identifier_value).where(
                    ExternalIdentifier.identifier_type == IdentifierType.BIOPROJECT_ACCESSION.value
                )
            )
        )
    )
    # Deterministically pick whichever study claimed this accession FIRST
    # (oldest ExternalIdentifier row) as the citation-discovery "parent" --
    # handle_discover_citing_studies reads that study's discovery_depth/
    # discovery_root_study_id for cap checks and provenance, so this should
    # be stable across repeated runs, not an arbitrary pick.
    study_id_by_accession = {
        accession: session.scalars(
            select(ExternalIdentifier.study_id)
            .where(
                ExternalIdentifier.identifier_type == IdentifierType.BIOPROJECT_ACCESSION.value,
                ExternalIdentifier.identifier_value == accession,
            )
            .order_by(ExternalIdentifier.created_at)
            .limit(1)
        ).first()
        for accession in accessions
    }
    for accession in accessions:
        study_id = study_id_by_accession[accession]
        if study_id is None:
            continue
        enqueue_task(
            session,
            TaskType.DISCOVER_CITING_STUDIES,
            study_id=study_id,
            payload={"bioproject_accession": accession},
            idempotency_key=f"{CITATION_RUN_TYPE}:{accession}:{run_id}",
        )
    return len(accessions)
