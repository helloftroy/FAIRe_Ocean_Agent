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
from fair_ocean_agent.database.enums import CanonicalStatus, TaskType, WorkflowRunStatus
from fair_ocean_agent.database.models import Study, WorkflowRun
from fair_ocean_agent.workflow.task_queue import enqueue_task

RUN_TYPE = "quarterly_full_rediscovery"
DEFAULT_INTERVAL_DAYS = 90


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
    is here for when they are)."""
    study_ids = session.scalars(
        select(Study.study_id).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
    ).all()
    for study_id in study_ids:
        enqueue_task(
            session,
            TaskType.DISCOVER_IDENTIFIERS,
            study_id=study_id,
            idempotency_key=f"{RUN_TYPE}:{study_id}:{run_id}",
        )
    return len(study_ids)
