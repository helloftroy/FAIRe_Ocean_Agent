"""Regression test for a real bug found during live Milestone 4 testing: a
relative sqlite:/// URL was passed straight to create_engine(), which
resolves it against the process's *current working directory* at connect
time -- not against REPO_ROOT, unlike the directory-creation step that ran
just before it. They only ever coincided because every prior command in
this project had been run with the repo as cwd. A one-off analysis script
run with cwd one level up hit "unable to open database file" -- and a cron
job or systemd unit (Milestone 7) invoked from a different working
directory would hit the exact same failure silently connecting to (or
creating) the wrong database file entirely, if the directory happened to
exist."""
import os
import sqlite3

from fair_ocean_agent.config import REPO_ROOT
import fair_ocean_agent.database.session as session_module
from fair_ocean_agent.database.session import _apply_sqlite_pragmas, _resolve_sqlite_url


def test_relative_sqlite_url_resolves_to_absolute_path_anchored_at_repo_root():
    resolved = _resolve_sqlite_url("sqlite:///data/fair_ocean.db")
    assert resolved == f"sqlite:///{REPO_ROOT}/data/fair_ocean.db"


def test_resolution_is_independent_of_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # simulates the exact failure mode found live
    resolved = _resolve_sqlite_url("sqlite:///data/fair_ocean.db")
    assert resolved == f"sqlite:///{REPO_ROOT}/data/fair_ocean.db"
    assert os.getcwd() == str(tmp_path)  # sanity: cwd really did change


def test_already_absolute_url_is_left_alone():
    resolved = _resolve_sqlite_url("sqlite:////tmp/somewhere/fair_ocean.db")
    assert resolved == "sqlite:////tmp/somewhere/fair_ocean.db"


def test_creates_parent_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("fair_ocean_agent.database.session.REPO_ROOT", tmp_path)
    _resolve_sqlite_url("sqlite:///nested/dir/fair_ocean.db")
    assert (tmp_path / "nested" / "dir").is_dir()


class _FakeCursor:
    """Records every PRAGMA sent, optionally raising sqlite3.OperationalError
    for one specific statement -- simulates a filesystem that rejects WAL
    mode without needing a real network/parallel filesystem to test against."""

    def __init__(self, raise_for: str | None = None):
        self.raise_for = raise_for
        self.executed: list[str] = []

    def execute(self, statement: str) -> None:
        self.executed.append(statement)
        if self.raise_for is not None and statement == self.raise_for:
            raise sqlite3.OperationalError("locking protocol")


def test_apply_sqlite_pragmas_sets_wal_and_busy_timeout_when_wal_is_supported(monkeypatch):
    monkeypatch.setattr(session_module, "_wal_mode_unavailable_warned", False)
    cursor = _FakeCursor()
    _apply_sqlite_pragmas(cursor)
    assert cursor.executed == ["PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=30000"]


def test_apply_sqlite_pragmas_falls_back_to_default_journal_when_wal_unsupported(monkeypatch):
    """Real gap found live (a real SLURM job array on a Lustre-backed
    cluster scratch mount): every single array task crashed at its very
    first DB connection with "OperationalError: locking protocol" trying
    to enable WAL mode -- confirmed this is a known WAL/network-filesystem
    incompatibility, not a bug in the claim logic itself (see
    workflow/task_queue.py's own atomic UPDATE...RETURNING comment, which
    is correct under any journal mode). busy_timeout must still get set
    even when WAL fails, and the failure must not propagate."""
    monkeypatch.setattr(session_module, "_wal_mode_unavailable_warned", False)
    cursor = _FakeCursor(raise_for="PRAGMA journal_mode=WAL")
    _apply_sqlite_pragmas(cursor)  # must not raise
    assert cursor.executed == ["PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=30000"]


def test_apply_sqlite_pragmas_only_warns_once_per_process(monkeypatch, caplog):
    monkeypatch.setattr(session_module, "_wal_mode_unavailable_warned", False)
    with caplog.at_level("WARNING", logger=session_module.logger.name):
        _apply_sqlite_pragmas(_FakeCursor(raise_for="PRAGMA journal_mode=WAL"))
        _apply_sqlite_pragmas(_FakeCursor(raise_for="PRAGMA journal_mode=WAL"))
    wal_warnings = [r for r in caplog.records if "WAL mode is not supported" in r.message]
    assert len(wal_warnings) == 1
    assert session_module._wal_mode_unavailable_warned is True
