"""Tests for scheduling/weekly.py -- the fair-ocean weekly-update
orchestrator. Config is monkeypatched rather than relying on real
config/default.yaml + config/local.yaml (see Milestone 4's
test_load_config_defaults regression: a test silently depending on real
on-disk config broke the moment a real config/local.yaml existed)."""
from types import SimpleNamespace

from fair_ocean_agent.database.enums import IdentifierType, TaskStatus, TaskType, WorkflowRunStatus
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task, WorkflowRun
from fair_ocean_agent.identity.identifiers import normalize_doi
from fair_ocean_agent.scheduling import weekly
from fair_ocean_agent.workflow import handlers


def _fake_config(**scheduling_overrides):
    defaults = dict(
        weekly_overlap_days=14,
        retry_failed_after_hours=24,
        monthly_unresolved_retry=True,
        quarterly_full_rediscovery=False,  # off by default in tests -- exercised in its own test
    )
    defaults.update(scheduling_overrides)
    return SimpleNamespace(scheduling=SimpleNamespace(**defaults))


def _study_with_doi(session, doi="10.1234/a") -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    session.add(
        ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=normalize_doi(doi))
    )
    session.flush()
    return study


def test_dry_run_creates_a_workflow_run_but_no_tasks(db_session, monkeypatch):
    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config())
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})
    _study_with_doi(db_session)
    db_session.commit()

    run = weekly.run_weekly_update(db_session, dry_run=True)
    db_session.commit()

    assert run.status == WorkflowRunStatus.COMPLETED.value
    assert run.sources_queried["dry_run"] is True
    assert run.sources_queried["refreshable_studies"] == 1
    assert db_session.query(Task).count() == 0


def test_live_run_enqueues_refresh_tasks_and_records_summary(db_session, monkeypatch):
    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config())
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})
    _study_with_doi(db_session)
    db_session.commit()

    run = weekly.run_weekly_update(db_session, dry_run=False)
    db_session.commit()

    assert run.status == WorkflowRunStatus.COMPLETED.value
    assert run.sources_queried["dry_run"] is False
    assert run.sources_queried["refresh_tasks_enqueued"] == 1
    assert db_session.query(Task).filter_by(task_type=TaskType.REFRESH_STUDY_SOURCES.value).count() == 1


def test_live_run_retries_stale_failed_tasks(db_session, monkeypatch):
    from datetime import timedelta

    from fair_ocean_agent.clock import utcnow

    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config(retry_failed_after_hours=24))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})
    stale = Task(
        task_type=TaskType.EXTRACT_TEXT_FACTS.value, status=TaskStatus.FAILED.value,
        idempotency_key="stale-1", attempt_count=3,
    )
    db_session.add(stale)
    db_session.commit()
    stale.updated_at = utcnow() - timedelta(hours=48)
    db_session.commit()

    run = weekly.run_weekly_update(db_session, dry_run=False)
    db_session.commit()

    assert run.sources_queried["stale_failed_tasks_retried"] == 1
    db_session.refresh(stale)
    assert stale.status == TaskStatus.PENDING.value


def test_live_run_skips_quarterly_rediscovery_when_disabled(db_session, monkeypatch):
    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config(quarterly_full_rediscovery=False))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})

    run = weekly.run_weekly_update(db_session, dry_run=False)
    db_session.commit()

    assert run.sources_queried["quarterly_rediscovery_ran"] is False
    assert db_session.query(WorkflowRun).filter_by(run_type="quarterly_full_rediscovery").count() == 0


def test_live_run_fires_quarterly_rediscovery_when_enabled_and_due(db_session, monkeypatch):
    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config(quarterly_full_rediscovery=True))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})
    study = Study(title="Candidate")
    db_session.add(study)
    db_session.commit()

    run = weekly.run_weekly_update(db_session, dry_run=False)
    db_session.commit()

    assert run.sources_queried["quarterly_rediscovery_ran"] is True
    assert run.sources_queried["quarterly_rediscovery_tasks_enqueued"] == 1
    rediscovery_run = db_session.query(WorkflowRun).filter_by(run_type="quarterly_full_rediscovery").one()
    assert rediscovery_run.status == WorkflowRunStatus.COMPLETED.value
    assert rediscovery_run.sources_queried["triggered_by_run_id"] == run.run_id


def test_second_weekly_run_does_not_refire_quarterly_rediscovery(db_session, monkeypatch):
    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config(quarterly_full_rediscovery=True))
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})

    weekly.run_weekly_update(db_session, dry_run=False)
    db_session.commit()
    second = weekly.run_weekly_update(db_session, dry_run=False)
    db_session.commit()

    assert second.sources_queried["quarterly_rediscovery_ran"] is False
    assert db_session.query(WorkflowRun).filter_by(run_type="quarterly_full_rediscovery").count() == 1


def test_failed_run_marks_workflow_run_failed_and_reraises(db_session, monkeypatch):
    monkeypatch.setattr(weekly, "load_config", lambda: _fake_config())

    def boom():
        raise RuntimeError("adapter build blew up")

    monkeypatch.setattr(handlers, "_build_enabled_adapters", boom)

    try:
        weekly.run_weekly_update(db_session, dry_run=False)
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
    db_session.commit()

    run = db_session.query(WorkflowRun).filter_by(run_type="weekly_update").one()
    assert run.status == WorkflowRunStatus.FAILED.value
    assert run.ended_at is not None
