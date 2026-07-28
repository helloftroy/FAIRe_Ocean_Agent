from datetime import timedelta

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import CanonicalStatus, TaskType, WorkflowRunStatus
from fair_ocean_agent.database.models import Study, Task, WorkflowRun
from fair_ocean_agent.scheduling.rediscovery import enqueue_full_rediscovery, is_rediscovery_due


def test_is_rediscovery_due_true_when_never_run(db_session):
    assert is_rediscovery_due(db_session) is True


def test_is_rediscovery_due_false_right_after_a_run(db_session):
    run = WorkflowRun(run_type="quarterly_full_rediscovery", status=WorkflowRunStatus.COMPLETED.value)
    db_session.add(run)
    db_session.commit()

    assert is_rediscovery_due(db_session, interval_days=90) is False


def test_is_rediscovery_due_true_after_interval_elapsed(db_session):
    run = WorkflowRun(run_type="quarterly_full_rediscovery", status=WorkflowRunStatus.COMPLETED.value)
    db_session.add(run)
    db_session.commit()
    run.started_at = utcnow() - timedelta(days=91)
    db_session.commit()

    assert is_rediscovery_due(db_session, interval_days=90) is True


def test_is_rediscovery_due_ignores_non_completed_runs(db_session):
    run = WorkflowRun(run_type="quarterly_full_rediscovery", status=WorkflowRunStatus.FAILED.value)
    db_session.add(run)
    db_session.commit()

    assert is_rediscovery_due(db_session) is True


def test_enqueue_full_rediscovery_targets_candidate_studies_with_fresh_idempotency_key(db_session):
    study = Study(title="Candidate", canonical_status=CanonicalStatus.CANDIDATE.value)
    merged = Study(title="Already merged", canonical_status=CanonicalStatus.MERGED.value)
    db_session.add_all([study, merged])
    db_session.commit()

    # Simulate a prior, already-completed DISCOVER_IDENTIFIERS task for `study`
    db_session.add(
        Task(
            task_type=TaskType.DISCOVER_IDENTIFIERS.value, study_id=study.study_id,
            status="completed", idempotency_key=f"old-key-{study.study_id}",
        )
    )
    db_session.commit()

    count = enqueue_full_rediscovery(db_session, run_id="RUN-1")
    db_session.commit()

    assert count == 1  # only the candidate study, not the merged one
    new_tasks = db_session.query(Task).filter_by(study_id=study.study_id, task_type=TaskType.DISCOVER_IDENTIFIERS.value).all()
    assert len(new_tasks) == 2  # the old completed one, plus a genuinely new one this call created
    assert any(t.idempotency_key == f"quarterly_full_rediscovery:{study.study_id}:RUN-1" for t in new_tasks)
