"""Long-cadence retry sweeps -- distinct from `fail_task`'s existing
short-term exponential backoff (workflow/task_queue.py), which only
retries a task while `attempt_count < max_attempts` and then leaves it in
a terminal state (FAILED or MANUAL_REVIEW_REQUIRED) that the normal
worker loop never revisits.

`SchedulingConfig.retry_failed_after_hours` and `.monthly_unresolved_retry`
give tasks that gave up a second chance on a much longer clock, on the
theory that whatever caused the failure (a transient API outage, a bug
since fixed, a data issue since corrected upstream) may no longer apply.
This resets the task back to PENDING with a fresh attempt budget --
`enqueue_task`'s idempotency key would otherwise make a plain re-enqueue
call a no-op, since a Task row for that (task_type, study_id, ...) already
exists (see scheduling/weekly.py's module docstring)."""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import TaskStatus
from fair_ocean_agent.database.models import Task


def _reset_stale_terminal_tasks(session: Session, status: TaskStatus, older_than: timedelta) -> int:
    cutoff = utcnow() - older_than
    tasks = session.scalars(
        select(Task).where(Task.status == status.value, Task.updated_at <= cutoff)
    ).all()
    for task in tasks:
        task.status = TaskStatus.PENDING.value
        task.attempt_count = 0
        task.available_after = utcnow()
        task.claimed_by = None
        task.claimed_at = None
        task.started_at = None
        task.completed_at = None
    return len(tasks)


def retry_stale_failed_tasks(session: Session, older_than_hours: int) -> int:
    return _reset_stale_terminal_tasks(session, TaskStatus.FAILED, timedelta(hours=older_than_hours))


def retry_stale_manual_review_tasks(session: Session, older_than_days: int = 30) -> int:
    return _reset_stale_terminal_tasks(session, TaskStatus.MANUAL_REVIEW_REQUIRED, timedelta(days=older_than_days))
