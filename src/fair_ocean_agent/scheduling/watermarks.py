"""Thin read/write helpers over the `source_watermarks` table (modeled
since Milestone 1, unused until now).

Deliberately NOT a cadence gate: `weekly-update` re-checks every known
study's every known source identifier on every invocation, and relies on
whatever external scheduler invokes it (cron/systemd -- see README's
"Server deployment" section) to decide how often that is. A watermark
record is bookkeeping/audit trail (when was this last checked, did it
change, which run recorded it), not a "skip if checked recently" filter --
adding a second, internal cadence policy on top of an external one would
just be two sources of truth for the same question.

`overlap_window_days` (on the SourceWatermark row) stays unused by this
module: it exists for a genuinely different mechanism -- querying an
adapter's `search()` with a "since this date, with an N-day safety buffer"
filter -- which Milestone 7's refresh path doesn't use (it re-fetches each
known identifier's current full record via `fetch_record` and diffs by
content_hash, rather than asking a source API for a delta). See
scheduling/weekly.py's module docstring for why that's the current
design; the column stays in the schema for whenever real
search()-with-date-filter support is added.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.models import SourceWatermark


def get_or_create_watermark(session: Session, source_name: str, query_identifier: str) -> SourceWatermark:
    watermark = (
        session.query(SourceWatermark)
        .filter_by(source_name=source_name, query_identifier=query_identifier)
        .one_or_none()
    )
    if watermark is None:
        watermark = SourceWatermark(source_name=source_name, query_identifier=query_identifier)
        session.add(watermark)
        session.flush()
    return watermark


def record_check(
    session: Session,
    source_name: str,
    query_identifier: str,
    *,
    status: str,
    run_id: str | None = None,
) -> SourceWatermark:
    """Updates a watermark after a refresh check actually happened
    (successfully or not) -- `status` is a short free-text summary
    ("changed", "unchanged", "not_found", "error: ..."), not a controlled
    vocabulary; this is an audit trail field, not something downstream
    logic branches on."""
    watermark = get_or_create_watermark(session, source_name, query_identifier)
    watermark.last_success_at = utcnow()
    watermark.last_status = status
    watermark.last_run_id = run_id
    return watermark
