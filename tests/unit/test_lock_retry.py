"""Tests for _extraction_sharding.with_lock_retry -- the retry mitigation
for a real gap found live: a real cluster's Lustre-backed scratch mount
raised sqlite3.OperationalError: locking protocol even from a single,
solitary process (no concurrent array tasks at all), almost certainly
because some OTHER process was concurrently touching the same shared
database file. See cluster/README.md's "Speeding up extraction" section.
"""
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _extraction_sharding import with_lock_retry  # noqa: E402


def test_succeeds_immediately_when_no_error_occurs():
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2

    assert with_lock_retry(fn, 21) == 42
    assert calls == [21]


def test_retries_on_a_locking_protocol_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = {"count": 0}

    def fn():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise sqlite3.OperationalError("locking protocol")
        return "ok"

    assert with_lock_retry(fn) == "ok"
    assert attempts["count"] == 3


def test_raises_after_exhausting_all_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = {"count": 0}

    def fn():
        attempts["count"] += 1
        raise sqlite3.OperationalError("locking protocol")

    with pytest.raises(sqlite3.OperationalError, match="locking protocol"):
        with_lock_retry(fn, attempts=3)
    assert attempts["count"] == 3


def test_does_not_retry_an_unrelated_operational_error():
    """Only the specific "locking protocol" message gets retried -- a
    genuine SQL error (bad statement, real constraint violation, etc.)
    must fail immediately, not get silently masked by 5 retries."""
    calls = {"count": 0}

    def fn():
        calls["count"] += 1
        raise sqlite3.OperationalError("no such table: bogus")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        with_lock_retry(fn)
    assert calls["count"] == 1


def test_retries_a_sqlalchemy_wrapped_locking_protocol_error_too(monkeypatch):
    """ORM-layer calls (e.g. through session_scope) raise SQLAlchemy's own
    OperationalError wrapping the raw sqlite3 one, not the raw exception
    directly -- confirmed live from the real traceback this was built to
    fix (workflow/handlers.py's enqueue_text_extraction_backfill, called
    through a plain SQLAlchemy Session)."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    attempts = {"count": 0}

    def fn():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise SQLAlchemyOperationalError("SELECT 1", {}, sqlite3.OperationalError("locking protocol"))
        return "ok"

    assert with_lock_retry(fn) == "ok"
    assert attempts["count"] == 2


def test_passes_through_args_and_kwargs():
    def fn(a, b, *, c):
        return a + b + c

    assert with_lock_retry(fn, 1, 2, c=3) == 6
