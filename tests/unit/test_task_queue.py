from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.workflow.retries import compute_backoff_seconds
from fair_ocean_agent.workflow.task_queue import (
    build_claim_next_task_statement,
    claim_next_task,
    complete_task,
    enqueue_task,
    fail_task,
    release_stale_claims,
)


def test_enqueue_task_is_idempotent(db_session):
    t1 = enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-1")
    t2 = enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-1")
    db_session.commit()

    assert t1.task_id == t2.task_id
    assert db_session.query(t1.__class__).count() == 1


def test_enqueue_task_distinct_studies_are_distinct_tasks(db_session):
    t1 = enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-1")
    t2 = enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-2")
    db_session.commit()

    assert t1.task_id != t2.task_id


def test_explicit_idempotency_key_overrides_default(db_session):
    t1 = enqueue_task(
        db_session, TaskType.FETCH_SOURCE, study_id="STUDY-1", idempotency_key="custom-key"
    )
    t2 = enqueue_task(
        db_session, TaskType.FETCH_SOURCE, study_id="STUDY-2", idempotency_key="custom-key"
    )
    db_session.commit()

    assert t1.task_id == t2.task_id


def test_claim_next_task_orders_by_priority_then_created(db_session):
    enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-LOW", priority=200)
    enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-HIGH", priority=1)
    db_session.commit()

    claimed = claim_next_task(db_session, worker_id="w1")
    db_session.commit()

    assert claimed.study_id == "STUDY-HIGH"
    assert claimed.status == TaskStatus.CLAIMED.value
    assert claimed.claimed_by == "w1"
    assert claimed.attempt_count == 1


def test_claim_next_task_returns_none_when_empty(db_session):
    assert claim_next_task(db_session, worker_id="w1") is None


def test_postgres_claim_statement_uses_skip_locked():
    from sqlalchemy.dialects import postgresql

    stmt = build_claim_next_task_statement("postgresql")
    compiled = str(stmt.compile(dialect=postgresql.dialect())).upper()

    assert "FOR UPDATE SKIP LOCKED" in compiled


def test_sqlite_claim_statement_does_not_emit_row_locking():
    from sqlalchemy.dialects import sqlite

    stmt = build_claim_next_task_statement("sqlite")
    compiled = str(stmt.compile(dialect=sqlite.dialect())).upper()

    assert "FOR UPDATE" not in compiled
    assert "SKIP LOCKED" not in compiled


def test_task_table_has_multi_worker_claim_indexes():
    index_names = {index.name for index in Task.__table__.indexes}

    assert "ix_tasks_claimable" in index_names
    assert "ix_tasks_type_status_available" in index_names


def test_complete_task(db_session):
    task = enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-1")
    db_session.commit()
    claimed = claim_next_task(db_session, worker_id="w1")
    complete_task(db_session, claimed)
    db_session.commit()

    assert claimed.status == TaskStatus.COMPLETED.value
    assert claimed.completed_at is not None


def test_fail_task_retries_until_max_attempts_then_manual_review(db_session):
    task = enqueue_task(
        db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-1", max_attempts=2
    )
    db_session.commit()

    claimed = claim_next_task(db_session, worker_id="w1")  # attempt 1
    fail_task(db_session, claimed, error="boom")
    db_session.commit()
    assert claimed.status == TaskStatus.RETRY_PENDING.value

    # simulate the backoff window having passed
    claimed.available_after = claimed.available_after.replace(year=2000)
    db_session.commit()

    reclaimed = claim_next_task(db_session, worker_id="w1")  # attempt 2 == max_attempts
    assert reclaimed.task_id == claimed.task_id
    fail_task(db_session, reclaimed, error="boom again")
    db_session.commit()

    assert reclaimed.status == TaskStatus.MANUAL_REVIEW_REQUIRED.value
    assert reclaimed.last_error == "boom again"


def test_release_stale_claims_moves_orphaned_tasks_back_to_queue(db_session):
    task = enqueue_task(db_session, TaskType.DISCOVER_IDENTIFIERS, study_id="STUDY-1")
    db_session.commit()
    claimed = claim_next_task(db_session, worker_id="dead-worker")
    db_session.commit()

    # simulate a claim from 60 minutes ago
    claimed.claimed_at = claimed.claimed_at.replace(year=2000)
    db_session.commit()

    released = release_stale_claims(db_session, stale_after_minutes=30)
    db_session.commit()

    assert released == 1
    assert claimed.status == TaskStatus.RETRY_PENDING.value


def test_compute_backoff_seconds_grows_and_caps():
    assert compute_backoff_seconds(1, base_seconds=30, max_seconds=3600) == 30
    assert compute_backoff_seconds(2, base_seconds=30, max_seconds=3600) == 60
    assert compute_backoff_seconds(3, base_seconds=30, max_seconds=3600) == 120
    assert compute_backoff_seconds(20, base_seconds=30, max_seconds=3600) == 3600
