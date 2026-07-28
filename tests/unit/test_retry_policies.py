from datetime import timedelta

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.scheduling.retry_policies import retry_stale_failed_tasks, retry_stale_manual_review_tasks


def _task(session, *, status, updated_at=None, **kwargs) -> Task:
    task = Task(
        task_type=TaskType.DISCOVER_IDENTIFIERS.value,
        status=status,
        idempotency_key=f"k-{id(kwargs)}-{status}-{updated_at}",
        attempt_count=3,
        claimed_by="some-worker",
        **kwargs,
    )
    session.add(task)
    session.commit()
    if updated_at is not None:
        task.updated_at = updated_at
        session.commit()
    return task


def test_retry_stale_failed_tasks_resets_old_failures(db_session):
    stale = _task(db_session, status=TaskStatus.FAILED.value, updated_at=utcnow() - timedelta(hours=48))
    fresh = _task(db_session, status=TaskStatus.FAILED.value, updated_at=utcnow() - timedelta(hours=1))

    count = retry_stale_failed_tasks(db_session, older_than_hours=24)
    db_session.commit()

    assert count == 1
    db_session.refresh(stale)
    db_session.refresh(fresh)
    assert stale.status == TaskStatus.PENDING.value
    assert stale.attempt_count == 0
    assert stale.claimed_by is None
    assert fresh.status == TaskStatus.FAILED.value  # untouched -- not stale enough yet


def test_retry_stale_manual_review_tasks_resets_old_ones(db_session):
    stale = _task(db_session, status=TaskStatus.MANUAL_REVIEW_REQUIRED.value, updated_at=utcnow() - timedelta(days=45))
    fresh = _task(db_session, status=TaskStatus.MANUAL_REVIEW_REQUIRED.value, updated_at=utcnow() - timedelta(days=2))

    count = retry_stale_manual_review_tasks(db_session, older_than_days=30)
    db_session.commit()

    assert count == 1
    db_session.refresh(stale)
    db_session.refresh(fresh)
    assert stale.status == TaskStatus.PENDING.value
    assert fresh.status == TaskStatus.MANUAL_REVIEW_REQUIRED.value


def test_retry_policies_never_touch_completed_or_pending_tasks(db_session):
    completed = _task(db_session, status=TaskStatus.COMPLETED.value, updated_at=utcnow() - timedelta(days=999))
    pending = _task(db_session, status=TaskStatus.PENDING.value, updated_at=utcnow() - timedelta(days=999))

    retry_stale_failed_tasks(db_session, older_than_hours=1)
    retry_stale_manual_review_tasks(db_session, older_than_days=1)
    db_session.commit()

    db_session.refresh(completed)
    db_session.refresh(pending)
    assert completed.status == TaskStatus.COMPLETED.value
    assert pending.status == TaskStatus.PENDING.value
