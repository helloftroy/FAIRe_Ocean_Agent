"""Database-backed task queue (section 17). Supports both SQLite
(single-worker-safe) and PostgreSQL (`FOR UPDATE SKIP LOCKED`, multi-worker)
without a code branch in callers -- `claim_next_task` detects the dialect.

Idempotency: `enqueue_task` derives a deterministic idempotency_key from
(task_type, study_id, source_id, payload) when the caller doesn't supply
one, and returns the existing row instead of inserting a duplicate. This is
what makes re-running `enqueue-seed-backfill` or a discovery step safe.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.workflow.retries import compute_backoff_seconds

_CLAIMABLE_STATUSES = (TaskStatus.PENDING.value, TaskStatus.RETRY_PENDING.value)
_IN_FLIGHT_STATUSES = (TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value)


def _default_idempotency_key(
    task_type: TaskType,
    study_id: str | None,
    source_id: str | None,
    payload: dict | None,
) -> str:
    basis = json.dumps(
        {
            "task_type": task_type.value,
            "study_id": study_id,
            "source_id": source_id,
            "payload": payload or {},
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]
    return f"{task_type.value}:{digest}"


def enqueue_task(
    session: Session,
    task_type: TaskType,
    study_id: str | None = None,
    source_id: str | None = None,
    payload: dict | None = None,
    priority: int = 100,
    max_attempts: int = 3,
    idempotency_key: str | None = None,
    available_after: datetime | None = None,
) -> Task:
    key = idempotency_key or _default_idempotency_key(task_type, study_id, source_id, payload)

    existing = session.scalars(select(Task).where(Task.idempotency_key == key)).first()
    if existing is not None:
        return existing

    try:
        with session.begin_nested():
            task = Task(
                task_type=task_type.value,
                study_id=study_id,
                source_id=source_id,
                payload=payload,
                priority=priority,
                status=TaskStatus.PENDING.value,
                max_attempts=max_attempts,
                available_after=available_after or utcnow(),
                idempotency_key=key,
            )
            session.add(task)
            session.flush()
            return task
    except IntegrityError:
        # Another worker inserted the same idempotency key concurrently.
        # The SAVEPOINT rollback keeps the outer transaction usable.
        existing = session.scalars(select(Task).where(Task.idempotency_key == key)).one()
        return existing


def build_claim_next_task_statement(
    dialect_name: str,
    task_types: list[TaskType] | None = None,
) -> Select[tuple[Task]]:
    stmt = (
        select(Task)
        .where(Task.status.in_(_CLAIMABLE_STATUSES))
        .where(Task.available_after <= utcnow())
        .order_by(Task.priority.asc(), Task.created_at.asc())
        .limit(1)
    )
    if task_types is not None:
        stmt = stmt.where(Task.task_type.in_([t.value for t in task_types]))

    if dialect_name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)

    return stmt


def claim_next_task(
    session: Session,
    worker_id: str,
    task_types: list[TaskType] | None = None,
) -> Task | None:
    """Atomically claim the next available task, or None if the queue is
    empty. On PostgreSQL this uses SELECT ... FOR UPDATE SKIP LOCKED so
    multiple worker processes can poll concurrently without contention. On
    SQLite (no row-level locking) this remains single-worker-only -- see
    docs/architecture.md."""
    dialect_name = session.get_bind().dialect.name
    stmt = build_claim_next_task_statement(dialect_name, task_types=task_types)
    task = session.scalars(stmt).first()
    if task is None:
        return None

    task.status = TaskStatus.CLAIMED.value
    task.claimed_by = worker_id
    task.claimed_at = utcnow()
    task.attempt_count += 1
    session.flush()
    return task


def mark_running(session: Session, task: Task) -> None:
    task.status = TaskStatus.RUNNING.value
    task.started_at = utcnow()
    session.flush()


def complete_task(session: Session, task: Task) -> None:
    task.status = TaskStatus.COMPLETED.value
    task.completed_at = utcnow()
    session.flush()


def fail_task(
    session: Session,
    task: Task,
    error: str,
    manual_review_after_max_attempts: bool = True,
) -> None:
    """Retry with exponential backoff while attempts remain, otherwise mark
    failed (or manual_review_required, the safer default per section 16 --
    silent permanent failure is worse than a human noticing)."""
    task.last_error = error[:4000]

    if task.attempt_count >= task.max_attempts:
        task.status = (
            TaskStatus.MANUAL_REVIEW_REQUIRED.value
            if manual_review_after_max_attempts
            else TaskStatus.FAILED.value
        )
        session.flush()
        return

    backoff = compute_backoff_seconds(task.attempt_count)
    task.status = TaskStatus.RETRY_PENDING.value
    task.available_after = utcnow() + timedelta(seconds=backoff)
    session.flush()


def cancel_task(session: Session, task: Task) -> None:
    task.status = TaskStatus.CANCELLED.value
    session.flush()


def release_stale_claims(session: Session, stale_after_minutes: int) -> int:
    """Recover tasks orphaned by a crashed/killed worker: any task still
    `claimed` or `running` past the staleness window goes back through
    fail_task's retry/manual-review logic. Returns the number released."""
    cutoff = utcnow() - timedelta(minutes=stale_after_minutes)
    stmt = select(Task).where(
        Task.status.in_(_IN_FLIGHT_STATUSES),
        Task.claimed_at.is_not(None),
        Task.claimed_at <= cutoff,
    )
    stale_tasks = session.scalars(stmt).all()
    for task in stale_tasks:
        fail_task(session, task, error="stale claim released (worker did not report completion)")
    return len(stale_tasks)


def count_tasks_by_status(session: Session) -> dict[str, int]:
    rows = session.execute(select(Task.status, func.count(Task.task_id)).group_by(Task.status)).all()
    counts = {status.value: 0 for status in TaskStatus}
    counts.update({status: count for status, count in rows})
    return counts
