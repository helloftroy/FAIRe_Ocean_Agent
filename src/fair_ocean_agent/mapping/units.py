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

import re
from datetime import datetime

from dateutil import parser as date_parser

from fair_ocean_agent.validation.logical import parse_depth_meters, parse_depth_range_meters, parse_lat_lon

_SINGLE_COORDINATE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([NSEW])?\s*$", re.IGNORECASE)

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


def to_decimal_latitude(value: str) -> str | None:
    """Returns a single latitude value (e.g. a supplementary table's own
    "Latitude" column, reported separately from longitude, unlike MIxS's
    combined lat_lon convention) as a FAIRe-ready signed decimal string, or
    None if unparseable. Accepts a plain signed decimal or one with a
    trailing N/S hemisphere letter."""
    match = _SINGLE_COORDINATE_PATTERN.match(value)
    if not match:
        return None
    magnitude, hemisphere = match.groups()
    lat = float(magnitude)
    if hemisphere and hemisphere.upper() == "S":
        lat = -abs(lat)
    return f"{lat:.6f}"


def to_decimal_longitude(value: str) -> str | None:
    """Same contract as to_decimal_latitude, for a standalone longitude
    value with an optional trailing E/W hemisphere letter."""
    match = _SINGLE_COORDINATE_PATTERN.match(value)
    if not match:
        return None
    magnitude, hemisphere = match.groups()
    lon = float(magnitude)
    if hemisphere and hemisphere.upper() == "W":
        lon = -abs(lon)
    return f"{lon:.6f}"


def to_meters(value: str) -> str | None:
    """Returns a plain meters value as a FAIRe-ready decimal string, or
    None if `value` isn't a parseable depth."""
    meters = parse_depth_meters(value)
    if meters is None:
        return None
    return f"{meters:g}"


def to_min_meters(value: str) -> str | None:
    """minimumDepthInMeters's own transform: the lesser end of a genuine
    depth range (e.g. "0-20 m" -> "0"), or the single value itself when
    there's no range -- see parse_depth_range_meters."""
    parsed = parse_depth_range_meters(value)
    if parsed is None:
        return None
    return f"{parsed[0]:g}"


def to_max_meters(value: str) -> str | None:
    """maximumDepthInMeters's own transform -- the greater end of a
    genuine depth range (e.g. "0-20 m" -> "20"), or the single value
    itself when there's no range -- see parse_depth_range_meters."""
    parsed = parse_depth_range_meters(value)
    if parsed is None:
        return None
    return f"{parsed[1]:g}"


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
