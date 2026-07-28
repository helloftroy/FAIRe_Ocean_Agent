"""Unit/format normalization for mapping raw_facts into FAIRe's expected
value shapes. Coordinate and depth parsing are NOT reimplemented here --
`validation/logical.py` (Milestone 5) already has real-data-tested parsers
(`parse_lat_lon`, `parse_depth_meters`) for the exact MIxS/free-text formats
this pipeline observes, and validation must agree with mapping on what a
value means. This module only adds the FAIRe-specific step those validators
don't need: turning a parsed value into the plain-number string FAIRe's
`schema.yaml` expects (range: string, but numeric-shaped) for
decimalLatitude/decimalLongitude/minimumDepthInMeters/maximumDepthInMeters.
"""
from __future__ import annotations

from datetime import datetime

from dateutil import parser as date_parser

from fair_ocean_agent.validation.logical import parse_depth_meters, parse_lat_lon

# Two distinct, deliberately-wrong anchors: whichever date fields agree
# between parses against both anchors were actually present in the input;
# whichever differ were defaulted by dateutil. This lets us emit an
# ISO 8601 string truncated to the precision the source actually reported
# ("March 2021" -> "2021-03", not a falsely-precise "2021-03-01") --
# `dates.try_parse_date`'s single fixed anchor is right for *comparing*
# two dates (Milestone 4 benchmark scoring, Milestone 5 date-ordering
# validation) but would silently fabricate day-level precision here.
_ANCHOR_A = datetime(1111, 1, 1)
_ANCHOR_B = datetime(2222, 2, 2)


def to_decimal_lat_lon(value: str) -> tuple[str, str] | None:
    """Returns (decimalLatitude, decimalLongitude) as FAIRe-ready decimal
    strings, or None if `value` isn't a parseable coordinate pair."""
    parsed = parse_lat_lon(value)
    if parsed is None:
        return None
    lat, lon = parsed
    return f"{lat:.6f}", f"{lon:.6f}"


def to_meters(value: str) -> str | None:
    """Returns a plain meters value as a FAIRe-ready decimal string, or
    None if `value` isn't a parseable depth."""
    meters = parse_depth_meters(value)
    if meters is None:
        return None
    return f"{meters:g}"


def to_iso_event_date(value: str) -> str | None:
    """Returns `value` reformatted as ISO 8601, truncated to whatever
    precision (year / year-month / year-month-day) was actually present in
    the source string, or None if unparseable. FAIRe's `eventDate` slot
    requires ISO 8601 but explicitly allows right-truncated precision --
    see the schema.yaml description quoted in mapping/rules.py."""
    try:
        parsed_a = date_parser.parse(value, fuzzy=False, default=_ANCHOR_A)
        parsed_b = date_parser.parse(value, fuzzy=False, default=_ANCHOR_B)
    except (ValueError, OverflowError, TypeError):
        return None
    if parsed_a.year != parsed_b.year:
        return None  # year itself wasn't specified -- not usable as a date at all
    if parsed_a.month != parsed_b.month:
        return f"{parsed_a.year:04d}"
    if parsed_a.day != parsed_b.day:
        return f"{parsed_a.year:04d}-{parsed_a.month:02d}"
    return f"{parsed_a.year:04d}-{parsed_a.month:02d}-{parsed_a.day:02d}"
