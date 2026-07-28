"""Regression test for a real bug found during Milestone 3 live validation:
a handler failure that leaves the session in SQLAlchemy's "pending
rollback" state (e.g. an IntegrityError mid-flush) used to crash the whole
worker loop, because fail_task() itself needs to write to the session and
that write raised PendingRollbackError before session.rollback() ran.
run_worker must roll back before calling fail_task, so one bad task doesn't
take down the rest of a batch."""
from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS, run_worker


def test_run_worker_recovers_from_integrity_error_without_crashing(db_session):
    def handler_that_violates_a_unique_constraint(session, task):
        # Real DB-level failure (duplicate idempotency_key), not a mocked
        # exception -- this is what actually happens if a handler's flush
        # hits a constraint violation partway through.
        session.add(Task(task_type=TaskType.FETCH_SOURCE.value, idempotency_key=task.idempotency_key))
        session.flush()

    TASK_HANDLERS[TaskType.FETCH_SOURCE] = handler_that_violates_a_unique_constraint
    try:
        task = enqueue_task(db_session, TaskType.FETCH_SOURCE, study_id=None, max_attempts=2)
        db_session.commit()

        summary = run_worker(db_session, worker_id="w1", max_tasks=1)  # must not raise

        assert summary == {"processed": 1, "completed": 0, "failed": 1}
        refreshed = db_session.get(Task, task.task_id)
        assert refreshed.status == TaskStatus.RETRY_PENDING.value
        assert refreshed.last_error  # populated, not lost to the crash

        # the session itself must still be usable afterward
        assert db_session.query(Task).count() == 1
    finally:
        del TASK_HANDLERS[TaskType.FETCH_SOURCE]
