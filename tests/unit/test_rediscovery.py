from datetime import timedelta

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    DataAvailabilityStatus,
    IdentifierType,
    TaskType,
    WorkflowRunStatus,
)
from fair_ocean_agent.database.models import (
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    StandardizedValueEvidence,
    Study,
    Task,
    WorkflowRun,
)
from fair_ocean_agent.scheduling.rediscovery import (
    enqueue_citation_rediscovery_backfill,
    enqueue_full_publication_metadata_backfill,
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


def test_enqueue_full_rediscovery_excludes_not_accessible_studies(db_session):
    """Give-up tracking, per an explicit user request: this function
    deliberately bypasses normal task idempotency via a run-scoped key to
    force periodic re-processing -- without this filter it would keep
    re-running the identical staged repository search against a study
    already confirmed to have nothing, every single rediscovery cycle,
    forever."""
    accessible = Study(
        title="Has real data", canonical_status=CanonicalStatus.CANDIDATE.value,
        data_availability_status=DataAvailabilityStatus.ACCESSIBLE.value,
    )
    not_accessible = Study(
        title="Confirmed nothing accessible", canonical_status=CanonicalStatus.CANDIDATE.value,
        data_availability_status=DataAvailabilityStatus.NOT_ACCESSIBLE.value,
    )
    db_session.add_all([accessible, not_accessible])
    db_session.commit()

    count = enqueue_full_rediscovery(db_session, run_id="RUN-2")
    db_session.commit()

    assert count == 1
    assert db_session.query(Task).filter_by(study_id=not_accessible.study_id).count() == 0
    assert db_session.query(Task).filter_by(study_id=accessible.study_id).count() == 1


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


def test_enqueue_full_publication_metadata_backfill_clears_the_stale_source_and_reenqueues(db_session):
    """Real gap found live (STUDY-0161dd80b492, 10.7717/peerj.17091):
    _discover_publication_metadata_from_sources guards itself with a bare
    "does a publication_metadata_extraction Source already exist for this
    DOI" check, with no version-awareness of its own -- unlike
    EXTRACT_TEXT_FACTS's own handler, simply re-enqueuing DISCOVER_IDENTIFIERS
    (even with a fresh idempotency key) does nothing while that stale
    Source row still exists. This backfill must delete it (and its
    RawFacts/StandardizedValueEvidence) before re-enqueuing, so the next
    DISCOVER_IDENTIFIERS run genuinely redoes the extraction."""
    study = Study(title="Stale publication metadata")
    other_study = Study(title="No publication metadata source at all")
    db_session.add_all([study, other_study])
    db_session.flush()

    source = Source(
        study_id=study.study_id, source_type="publication_api",
        source_name="publication_metadata_extraction", external_identifier="10.7717/peerj.17091",
    )
    db_session.add(source)
    db_session.flush()
    fact = RawFact(
        study_id=study.study_id, source_id=source.source_id, raw_field_name="code_repo",
        raw_value="no code published", fact_type_candidate="code_repo", entity_level="study",
        support_type="deterministically_derived",
    )
    db_session.add(fact)
    db_session.flush()
    standardized_value = StandardizedValue(
        study_id=study.study_id, target_schema="faire", target_schema_version="v1",
        target_field="code_repo", standardized_value="no code published",
    )
    db_session.add(standardized_value)
    db_session.flush()
    db_session.add(
        StandardizedValueEvidence(standardized_value_id=standardized_value.standardized_value_id, fact_id=fact.fact_id)
    )
    db_session.commit()

    count = enqueue_full_publication_metadata_backfill(db_session, run_id="RUN-1")
    db_session.commit()

    assert count == 1  # only the study with a stale Source, not other_study
    assert db_session.get(Source, source.source_id) is None
    assert db_session.get(RawFact, fact.fact_id) is None
    assert db_session.query(StandardizedValueEvidence).filter_by(fact_id=fact.fact_id).first() is None
    new_task = db_session.query(Task).filter_by(
        study_id=study.study_id, task_type=TaskType.DISCOVER_IDENTIFIERS.value,
    ).one()
    assert new_task.idempotency_key == f"full_publication_metadata_rediscovery:{study.study_id}:RUN-1"
