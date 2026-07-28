"""Worker loop: claim a task, dispatch to a registered handler, mark
complete/fail. TASK_HANDLERS is intentionally empty in Milestone 1 -- no
source adapters or extraction/mapping/validation logic exist yet. Claiming a
task with no registered handler is treated as a real failure (goes through
the normal retry -> manual_review_required path) rather than silently
succeeding, so the queue mechanics are exercised honestly end-to-end ahead
of Milestone 2+ wiring in real handlers.
"""
from __future__ import annotations

from typing import Callable

from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.logging_setup import get_logger, log_context
from fair_ocean_agent.workflow.task_queue import claim_next_task, complete_task, fail_task, mark_running

logger = get_logger(__name__)

TaskHandler = Callable[[Session, Task], None]

TASK_HANDLERS: dict[TaskType, TaskHandler] = {}


def run_worker(
    session: Session,
    worker_id: str,
    max_tasks: int | None = None,
    until_empty: bool = False,
) -> dict[str, int]:
    """Process tasks until `max_tasks` is reached, the queue is empty (if
    until_empty), or neither limit is set (processes exactly one task).
    Returns a summary count. Each task is committed individually so a crash
    mid-loop only loses at most the in-flight task's progress, not prior
    completions."""
    processed = 0
    completed = 0
    failed = 0

    while True:
        if max_tasks is not None and processed >= max_tasks:
            break

        task = claim_next_task(session, worker_id=worker_id)
        session.commit()
        if task is None:
            if until_empty or max_tasks is None:
                break
            break

        with log_context(task_id=task.task_id, study_id=task.study_id):
            processed += 1
            mark_running(session, task)
            session.commit()

            handler = TASK_HANDLERS.get(TaskType(task.task_type))
            try:
                if handler is None:
                    raise NotImplementedError(
                        f"No handler registered for task type {task.task_type} "
                        "(source adapters / extraction land in later milestones)"
                    )
                handler(session, task)
                complete_task(session, task)
                completed += 1
                logger.info("task completed")
            except Exception as exc:  # noqa: BLE001 - task-level isolation is intentional
                # A failure partway through a flush (e.g. an IntegrityError)
                # leaves the session's transaction in a "pending rollback"
                # state -- any further operation on it, including fail_task's
                # own attribute writes, raises PendingRollbackError unless
                # rolled back first. This also correctly discards whatever
                # partial, uncommitted work the handler did before failing;
                # the idempotency guards in workflow/handlers.py are what
                # make redoing the safe parts on retry non-duplicating.
                session.rollback()
                fail_task(session, task, error=str(exc))
                failed += 1
                logger.warning("task failed: %s", exc)
            session.commit()

        if not until_empty and max_tasks is None:
            break

    return {"processed": processed, "completed": completed, "failed": failed}
