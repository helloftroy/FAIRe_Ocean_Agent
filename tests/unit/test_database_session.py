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

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.session import _resolve_sqlite_url


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
