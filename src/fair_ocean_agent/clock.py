"""Single source of truth for "now" so tests can monkeypatch one function
instead of every call site."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_aware_utc(value: datetime) -> datetime:
    """SQLite (used in dev/tests) has no real timezone-aware storage --
    a `DateTime(timezone=True)` column round-trips as a naive datetime
    even though every writer in this codebase is `utcnow()` (always UTC).
    Subtracting that naive value directly from `utcnow()` raises
    TypeError ("can't subtract offset-naive and offset-aware datetimes");
    this makes it comparable again without assuming which dialect wrote
    it. A value that's already aware is returned unchanged."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
