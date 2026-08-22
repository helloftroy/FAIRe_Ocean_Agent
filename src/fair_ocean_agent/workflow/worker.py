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

import httpx
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.logging_setup import get_logger, log_context
from fair_ocean_agent.workflow.task_queue import claim_next_task, complete_task, fail_task, mark_running

logger = get_logger(__name__)

TaskHandler = Callable[[Session, Task], None]

TASK_HANDLERS: dict[TaskType, TaskHandler] = {}

# Per an explicit, urgent user request: RateLimitedClient already retries
# a single request on 429 with backoff (sources/base.py), but that's
# entirely per-call -- nothing tracked repeated 429s ACROSS separate
# tasks, so a genuinely rate-limited/blocked source (confirmed live:
# OpenAlex, 182 real DISCOVER_IDENTIFIERS failures in one run, all the
# same "429 Too Many Requests") just kept getting hit again on every
# subsequent task pulled from a queue thousands deep, for as long as
# --until-empty kept going -- exactly the sustained hammering that risks
# an actual IP ban, not just wasted time. 5 was chosen as "clearly not
# a coincidental blip mixed in with real successes" while still stopping
# fast -- a handful of scattered 429s recovered by retry are normal and
# reset this counter; five in a row means the block is real and ongoing.
DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES = 5


def is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def run_worker(
    session: Session,
    worker_id: str,
    max_tasks: int | None = None,
    until_empty: bool = False,
    task_types: list[TaskType] | None = None,
    max_consecutive_rate_limit_failures: int | None = DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
) -> dict[str, int | str | None]:
    """Process tasks until `max_tasks` is reached, the queue is empty (if
    until_empty), or neither limit is set (processes exactly one task).
    Returns a summary count. Each task is committed individually so a crash
    mid-loop only loses at most the in-flight task's progress, not prior
    completions.

    `task_types` restricts claiming to just those types (see
    claim_next_task) -- per an explicit user request: a machine that can
    only run the CPU-only discovery stage well (no local GPU worth
    running EXTRACT_TEXT_FACTS's LLM calls on) needs a way to run a
    worker that will never claim one, rather than relying on the queue
    happening to be empty of them.

    `max_consecutive_rate_limit_failures` stops the whole loop (not just
    the one task) once that many task failures IN A ROW were caused by a
    429 -- reset to zero by any completed task or any failure NOT caused
    by a 429, so this only fires on a genuinely sustained block, never on
    a few real successes with occasional retried-and-recovered 429s mixed
    in. None disables this (e.g. for tests that want the old unconditional
    behavior)."""
    processed = 0
    completed = 0
    failed = 0
    consecutive_rate_limit_failures = 0
    stopped_reason: str | None = None

    while True:
        if max_tasks is not None and processed >= max_tasks:
            break

        task = claim_next_task(session, worker_id=worker_id, task_types=task_types)
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
                consecutive_rate_limit_failures = 0
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
                if is_rate_limit_error(exc):
                    consecutive_rate_limit_failures += 1
                else:
                    consecutive_rate_limit_failures = 0
            session.commit()

        if (
            max_consecutive_rate_limit_failures is not None
            and consecutive_rate_limit_failures >= max_consecutive_rate_limit_failures
        ):
            stopped_reason = (
                f"{consecutive_rate_limit_failures} consecutive tasks failed with 429 Too Many Requests -- "
                "stopping to avoid hammering a source that's actively rate-limiting/blocking this machine. "
                "Wait a while before retrying."
            )
            logger.warning(stopped_reason)
            break

        if not until_empty and max_tasks is None:
            break

    return {"processed": processed, "completed": completed, "failed": failed, "stopped_reason": stopped_reason}
