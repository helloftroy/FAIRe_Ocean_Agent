"""Regression test for a real bug found during Milestone 3 live validation:
a handler failure that leaves the session in SQLAlchemy's "pending
rollback" state (e.g. an IntegrityError mid-flush) used to crash the whole
worker loop, because fail_task() itself needs to write to the session and
that write raised PendingRollbackError before session.rollback() ran.
run_worker must roll back before calling fail_task, so one bad task doesn't
take down the rest of a batch."""
import httpx

from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS, run_worker


def _rate_limit_error() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://api.openalex.org/works/x")
    response = httpx.Response(429, request=request)
    return httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)


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

        assert summary == {"processed": 1, "completed": 0, "failed": 1, "stopped_reason": None}
        refreshed = db_session.get(Task, task.task_id)
        assert refreshed.status == TaskStatus.RETRY_PENDING.value
        assert refreshed.last_error  # populated, not lost to the crash

        # the session itself must still be usable afterward
        assert db_session.query(Task).count() == 1
    finally:
        del TASK_HANDLERS[TaskType.FETCH_SOURCE]


def test_run_worker_task_types_filter_skips_other_queued_types(db_session):
    """Per an explicit user request: a machine with no GPU worth running
    local LLM extraction on needs a way to run a worker that only ever
    processes specific task types -- confirmed here against a queue that
    has BOTH an allowed and a disallowed type queued, not just an empty
    queue of the disallowed type (which would pass even with no filter
    applied at all)."""
    processed_types = []

    def record(session, task):
        processed_types.append(task.task_type)

    TASK_HANDLERS[TaskType.FETCH_SOURCE] = record
    try:
        enqueue_task(db_session, TaskType.EXTRACT_TEXT_FACTS, study_id=None)
        enqueue_task(db_session, TaskType.FETCH_SOURCE, study_id=None)
        db_session.commit()

        summary = run_worker(db_session, worker_id="w1", until_empty=True, task_types=[TaskType.FETCH_SOURCE])

        assert summary["processed"] == 1
        assert processed_types == [TaskType.FETCH_SOURCE.value]
        remaining = db_session.query(Task).filter_by(task_type=TaskType.EXTRACT_TEXT_FACTS.value).one()
        assert remaining.status == TaskStatus.PENDING.value
    finally:
        del TASK_HANDLERS[TaskType.FETCH_SOURCE]


def test_run_worker_stops_after_consecutive_rate_limit_failures(db_session):
    """Real live incident: OpenAlex actively rate-limiting/blocking this
    machine produced 182 real DISCOVER_IDENTIFIERS failures, all 429, in
    one --until-empty run against a queue thousands deep -- nothing
    stopped it from continuing to hammer the same blocked source task
    after task. 10 queued, all failing with 429: must stop after exactly
    5 (the default threshold), leaving the rest untouched and pending for
    a later, separated retry once the block has passed."""

    def always_rate_limited(session, task):
        raise _rate_limit_error()

    TASK_HANDLERS[TaskType.FETCH_SOURCE] = always_rate_limited
    try:
        for i in range(10):
            enqueue_task(db_session, TaskType.FETCH_SOURCE, study_id=None, max_attempts=99, idempotency_key=f"rate-limit-test-{i}")
        db_session.commit()

        summary = run_worker(db_session, worker_id="w1", until_empty=True)

        assert summary["processed"] == 5
        assert summary["failed"] == 5
        assert summary["stopped_reason"] is not None
        assert "429" in summary["stopped_reason"]
        still_pending = db_session.query(Task).filter_by(status=TaskStatus.PENDING.value).count()
        assert still_pending == 5
    finally:
        del TASK_HANDLERS[TaskType.FETCH_SOURCE]


def test_run_worker_rate_limit_counter_resets_on_success(db_session):
    """A handful of scattered 429s recovered by RateLimitedClient's own
    per-request retry, mixed in with real successes, is normal operation
    -- must NOT trip the circuit breaker just because 429s happened at
    all somewhere in the run."""
    calls = {"n": 0}

    def mostly_succeeds_with_occasional_429(session, task):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise _rate_limit_error()

    TASK_HANDLERS[TaskType.FETCH_SOURCE] = mostly_succeeds_with_occasional_429
    try:
        for i in range(9):
            enqueue_task(db_session, TaskType.FETCH_SOURCE, study_id=None, max_attempts=99, idempotency_key=f"rate-limit-reset-test-{i}")
        db_session.commit()

        summary = run_worker(db_session, worker_id="w1", until_empty=True)

        assert summary["processed"] == 9  # never tripped -- ran the whole queue
        assert summary["stopped_reason"] is None
        assert summary["completed"] == 6
        assert summary["failed"] == 3
    finally:
        del TASK_HANDLERS[TaskType.FETCH_SOURCE]


def test_run_worker_max_consecutive_rate_limit_failures_none_disables_breaker(db_session):
    def always_rate_limited(session, task):
        raise _rate_limit_error()

    TASK_HANDLERS[TaskType.FETCH_SOURCE] = always_rate_limited
    try:
        for i in range(7):
            enqueue_task(db_session, TaskType.FETCH_SOURCE, study_id=None, max_attempts=99, idempotency_key=f"rate-limit-disabled-test-{i}")
        db_session.commit()

        summary = run_worker(
            db_session, worker_id="w1", until_empty=True, max_consecutive_rate_limit_failures=None
        )

        assert summary["processed"] == 7
        assert summary["stopped_reason"] is None
    finally:
        del TASK_HANDLERS[TaskType.FETCH_SOURCE]
