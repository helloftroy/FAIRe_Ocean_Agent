"""`fair-ocean weekly-update`'s orchestration: refresh known studies'
sources, sweep long-stale failed/manual-review tasks back to PENDING, and
(quarterly) give every study one more DISCOVER_IDENTIFIERS pass in case a
newly-enabled adapter now applies to it. Meant to be invoked periodically
by an external scheduler (cron/systemd -- see README's "Server
deployment" section); this module doesn't track or enforce "weekly" as a
cadence itself (see scheduling/watermarks.py's docstring for why).

**This is an enqueue-only operation, like every other `enqueue-*` command
in this pipeline** -- it queues REFRESH_STUDY_SOURCES tasks and resets
stale FAILED/MANUAL_REVIEW_REQUIRED tasks back to PENDING, but does not
itself run the worker loop. `fair-ocean worker --until-empty` (run
separately, e.g. under its own supervisor loop) is what actually processes
whatever ends up in the queue, regardless of which command put it there.
Because of this, the WorkflowRun row this produces records *what was
queued* (in `sources_queried`, a free-form JSON summary), not task
outcomes -- `completed_tasks`/`new_studies`/`updated_studies` stay 0 here
since nothing has actually run yet from this call's perspective;
`failed_tasks`/`manual_review_items` are a snapshot of current queue
totals (a meaningful signal even though this run didn't cause them).

`--dry-run` reports the same counts this would enqueue/reset without
writing anything, by construction sharing exactly the same selection
queries as the real path (see `_plan`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.config import load_config
from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType, TaskStatus, WorkflowRunStatus
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task, WorkflowRun
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.scheduling.rediscovery import enqueue_full_rediscovery, is_rediscovery_due
from fair_ocean_agent.scheduling.retry_policies import retry_stale_failed_tasks, retry_stale_manual_review_tasks
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.refresh_handlers import enqueue_refresh_backfill
from fair_ocean_agent.workflow.task_queue import release_stale_claims

logger = get_logger(__name__)

RUN_TYPE = "weekly_update"
STALE_CLAIM_MINUTES = 30  # matches the `release-stale-claims` CLI command's own default


@dataclass
class WeeklyUpdatePlan:
    refreshable_study_count: int
    stale_failed_count: int
    stale_manual_review_count: int
    quarterly_rediscovery_due: bool
    quarterly_rediscovery_candidate_count: int


def _refreshable_study_count(session: Session) -> int:
    refreshable_types = (
        IdentifierType.DOI.value,
        IdentifierType.BIOPROJECT_ACCESSION.value,
        IdentifierType.ENA_STUDY_ACCESSION.value,
    )
    return len(
        set(
            session.scalars(
                select(ExternalIdentifier.study_id).where(ExternalIdentifier.identifier_type.in_(refreshable_types))
            )
        )
    )


def _count_stale(session: Session, status: TaskStatus, older_than_hours: int) -> int:
    cutoff = utcnow() - timedelta(hours=older_than_hours)
    return session.query(Task).filter(Task.status == status.value, Task.updated_at <= cutoff).count()


def plan_weekly_update(session: Session) -> WeeklyUpdatePlan:
    """Read-only: the exact same selection logic the real run uses, so
    --dry-run can never drift from what a real run would actually do."""
    scheduling_config = load_config().scheduling
    quarterly_due = is_rediscovery_due(session)
    quarterly_candidates = (
        session.query(Study).filter(Study.canonical_status == CanonicalStatus.CANDIDATE.value).count()
        if quarterly_due and scheduling_config.quarterly_full_rediscovery
        else 0
    )
    return WeeklyUpdatePlan(
        refreshable_study_count=_refreshable_study_count(session),
        stale_failed_count=_count_stale(session, TaskStatus.FAILED, scheduling_config.retry_failed_after_hours),
        stale_manual_review_count=(
            _count_stale(session, TaskStatus.MANUAL_REVIEW_REQUIRED, 24 * 30)
            if scheduling_config.monthly_unresolved_retry
            else 0
        ),
        quarterly_rediscovery_due=quarterly_due and scheduling_config.quarterly_full_rediscovery,
        quarterly_rediscovery_candidate_count=quarterly_candidates,
    )


def run_weekly_update(session: Session, *, dry_run: bool = False) -> WorkflowRun:
    scheduling_config = load_config().scheduling
    run = WorkflowRun(run_type=RUN_TYPE, status=WorkflowRunStatus.RUNNING.value)
    session.add(run)
    session.flush()

    try:
        plan = plan_weekly_update(session)

        if dry_run:
            summary = {
                "dry_run": True,
                "refreshable_studies": plan.refreshable_study_count,
                "stale_failed_tasks_would_retry": plan.stale_failed_count,
                "stale_manual_review_tasks_would_retry": plan.stale_manual_review_count,
                "quarterly_rediscovery_would_run": plan.quarterly_rediscovery_due,
                "quarterly_rediscovery_study_count": plan.quarterly_rediscovery_candidate_count,
            }
        else:
            released = release_stale_claims(session, STALE_CLAIM_MINUTES)

            for adapter in handlers._build_enabled_adapters().values():
                adapter.http.clear_cache()

            refresh_enqueued = enqueue_refresh_backfill(session, run_id=run.run_id)
            failed_retried = retry_stale_failed_tasks(session, scheduling_config.retry_failed_after_hours)
            manual_review_retried = (
                retry_stale_manual_review_tasks(session)
                if scheduling_config.monthly_unresolved_retry
                else 0
            )

            rediscovery_enqueued = 0
            rediscovery_ran = False
            if plan.quarterly_rediscovery_due:
                rediscovery_enqueued = enqueue_full_rediscovery(session, run.run_id)
                rediscovery_ran = True
                rediscovery_run = WorkflowRun(
                    run_type="quarterly_full_rediscovery",
                    status=WorkflowRunStatus.COMPLETED.value,
                    ended_at=utcnow(),
                    candidates_found=rediscovery_enqueued,
                    sources_queried={"triggered_by_run_id": run.run_id},
                )
                session.add(rediscovery_run)

            summary = {
                "dry_run": False,
                "released_stale_claims": released,
                "refresh_tasks_enqueued": refresh_enqueued,
                "stale_failed_tasks_retried": failed_retried,
                "stale_manual_review_tasks_retried": manual_review_retried,
                "quarterly_rediscovery_ran": rediscovery_ran,
                "quarterly_rediscovery_tasks_enqueued": rediscovery_enqueued,
            }

        run.sources_queried = summary
        run.failed_tasks = session.query(Task).filter(Task.status == TaskStatus.FAILED.value).count()
        run.manual_review_items = (
            session.query(Task).filter(Task.status == TaskStatus.MANUAL_REVIEW_REQUIRED.value).count()
        )
        run.status = WorkflowRunStatus.COMPLETED.value
    except Exception:
        run.status = WorkflowRunStatus.FAILED.value
        raise
    finally:
        run.ended_at = utcnow()
        session.flush()

    return run
