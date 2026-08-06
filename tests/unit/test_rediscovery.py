from datetime import timedelta

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType, TaskType, WorkflowRunStatus
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task, WorkflowRun
from fair_ocean_agent.scheduling.rediscovery import (
    enqueue_citation_rediscovery_backfill,
    enqueue_full_rediscovery,
    is_citation_rediscovery_due,
    is_rediscovery_due,
)


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


def test_is_citation_rediscovery_due_true_when_never_run(db_session):
    assert is_citation_rediscovery_due(db_session) is True


def test_is_citation_rediscovery_due_false_right_after_a_run(db_session):
    run = WorkflowRun(run_type="citation_rediscovery", status=WorkflowRunStatus.COMPLETED.value)
    db_session.add(run)
    db_session.commit()

    assert is_citation_rediscovery_due(db_session, interval_days=90) is False


def test_is_citation_rediscovery_due_true_after_interval_elapsed(db_session):
    run = WorkflowRun(run_type="citation_rediscovery", status=WorkflowRunStatus.COMPLETED.value)
    db_session.add(run)
    db_session.commit()
    run.started_at = utcnow() - timedelta(days=91)
    db_session.commit()

    assert is_citation_rediscovery_due(db_session, interval_days=90) is True


def test_enqueue_citation_rediscovery_backfill_targets_distinct_accessions_with_fresh_idempotency_key(db_session):
    """Two different studies sharing one BioProject accession only need one
    re-check -- and a prior, already-processed DISCOVER_CITING_STUDIES task
    (from the accession's original first-resolution trigger, or an earlier
    rediscovery run) must not block a fresh one with a new run-scoped key."""
    study_a = Study(title="Original")
    study_b = Study(title="Second paper, same accession")
    db_session.add_all([study_a, study_b])
    db_session.flush()
    db_session.add_all(
        [
            ExternalIdentifier(
                study_id=study_a.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value,
                identifier_value="PRJNA1", created_at=utcnow() - timedelta(days=10),
            ),
            ExternalIdentifier(
                study_id=study_b.study_id, identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value,
                identifier_value="PRJNA1", created_at=utcnow(),
            ),
        ]
    )
    db_session.commit()
    db_session.add(
        Task(
            task_type=TaskType.DISCOVER_CITING_STUDIES.value, study_id=study_a.study_id,
            status="completed", idempotency_key="DISCOVER_CITING_STUDIES:bioproject:PRJNA1",
            payload={"bioproject_accession": "PRJNA1"},
        )
    )
    db_session.commit()

    count = enqueue_citation_rediscovery_backfill(db_session, run_id="RUN-1")
    db_session.commit()

    assert count == 1  # one distinct accession, even though two studies claim it
    new_tasks = db_session.query(Task).filter_by(task_type=TaskType.DISCOVER_CITING_STUDIES.value).all()
    assert len(new_tasks) == 2  # the old completed one, plus a genuinely new one this call created
    fresh_task = next(t for t in new_tasks if t.idempotency_key == "citation_rediscovery:PRJNA1:RUN-1")
    assert fresh_task.study_id == study_a.study_id  # deterministically the FIRST study to claim this accession
    assert fresh_task.payload == {"bioproject_accession": "PRJNA1"}
