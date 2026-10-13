"""The worker must not claim DISCOVER_CITING_STUDIES while
discovery.citation_expansion_enabled is False -- the tasks stay pending
(resumable), they are just skipped. See config.py's own comment."""
from contextlib import contextmanager

from typer.testing import CliRunner

from fair_ocean_agent import cli
from fair_ocean_agent.config import AppConfig, DiscoveryConfig
from fair_ocean_agent.database.enums import TaskType


def _run_worker_capturing_task_types(monkeypatch, enabled: bool, args: list[str]):
    captured = {}

    @contextmanager
    def _fake_session_scope():
        yield object()

    def _fake_run_worker(session, **kwargs):
        captured.update(kwargs)
        return {"processed": 0, "completed": 0, "failed": 0, "stopped_reason": None}

    monkeypatch.setattr(cli, "session_scope", _fake_session_scope)
    monkeypatch.setattr(cli, "run_worker", _fake_run_worker)
    monkeypatch.setattr(
        cli, "load_config", lambda: AppConfig(discovery=DiscoveryConfig(citation_expansion_enabled=enabled))
    )
    result = CliRunner().invoke(cli.app, ["worker", *args])
    assert result.exit_code == 0, result.output
    return captured["task_types"]


def test_worker_claims_everything_by_default(monkeypatch):
    assert _run_worker_capturing_task_types(monkeypatch, True, ["--until-empty"]) is None


def test_paused_worker_skips_citing_studies_but_keeps_everything_else(monkeypatch):
    task_types = _run_worker_capturing_task_types(monkeypatch, False, ["--until-empty", "--no-llm"])
    assert TaskType.DISCOVER_CITING_STUDIES not in task_types
    assert TaskType.DISCOVER_IDENTIFIERS in task_types
    assert TaskType.CHECK_COMPONENT_SETTLED in task_types
    assert TaskType.EXTRACT_TEXT_FACTS not in task_types  # --no-llm still applies


def test_explicit_task_type_still_wins_while_paused(monkeypatch):
    task_types = _run_worker_capturing_task_types(
        monkeypatch, False, ["--until-empty", "--task-type", "DISCOVER_CITING_STUDIES"]
    )
    assert task_types == [TaskType.DISCOVER_CITING_STUDIES]
