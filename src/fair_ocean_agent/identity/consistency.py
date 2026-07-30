"""Date/location consistency check for identity/resolution.py's
resolve_or_create_study(): does newly-discovered evidence's own sampling
date range and geographic extent plausibly belong to a pre-existing Study
that shares an ambiguous identifier with it, or does it look like a
different investigation entirely?

Pure-ish (reads via session, never writes) and mirrors
validation/logical.py's own style: an unparseable value is NOT_ASSESSED,
never a false conflict -- "unknown or blank is preferable to guessing"
applies here exactly as it does to every other validator in this codebase.

Explicit non-goal, stated once here rather than re-derived by a future
reader: this module never reads assay/marker-gene/primer/target-gene
`fact_type_candidate` values as a splitting signal, anywhere, full stop.
FAIRe's own data model treats a single project running multiple assays
(e.g. 16S and 18S on the same samples) as normal single-project structure,
not evidence of two investigations -- that's handled at the assay/entity
level, never at the Study-boundary level. Do not add it back "for more
signal" without re-reading this reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.models import RawFact, Source, Study
from fair_ocean_agent.dates import try_parse_date
from fair_ocean_agent.validation.logical import parse_lat_lon

# A single field season/cruise commonly straddles a calendar-year boundary
# (e.g. a winter cruise sampling Dec-Feb) -- a year of slack on each side of
# an existing study's recorded date range avoids flagging a normal
# single-campaign span as a conflict while still catching genuinely
# different multi-year projects.
PLAUSIBLE_ADJACENCY_DAYS = 365

# Most single-paper marine eDNA studies describe a one-to-two-year sampling
# window; a registered span beyond this looks more like an umbrella
# BioProject/ENA study accumulating multiple unrelated deposits over time
# than one paper's actual field work. Used only for the no-signal-either-way
# default-conservative case (new evidence has no parseable date at all).
SINGLE_STUDY_MAX_SPAN_DAYS = 730

# A generous, round-number margin: two degrees comfortably covers normal
# within-region sampling-site spread (e.g. a coastal transect or an
# estuary-to-shelf gradient) without so loosely bounding the box that a
# genuinely different ocean basin would pass.
GEO_ADJACENCY_DEGREES = 2.0

_DATE_FACT_TYPES = ("collection_date",)


@dataclass
class ConsistencyOutcome:
    status: str  # "consistent" | "inconsistent" | "not_assessed"
    message: str


def _dates_for(session: Session, study_id: str | None = None, source_id: str | None = None) -> list[date]:
    stmt = select(RawFact.raw_value).where(RawFact.fact_type_candidate.in_(_DATE_FACT_TYPES))
    if study_id is not None:
        stmt = stmt.where(RawFact.study_id == study_id)
    if source_id is not None:
        stmt = stmt.where(RawFact.source_id == source_id)
    parsed = []
    for raw_value in session.scalars(stmt):
        if raw_value is None:
            continue
        parsed_date = try_parse_date(raw_value)
        if parsed_date is not None:
            parsed.append(parsed_date)
    return parsed


def _coordinates_for(session: Session, study_id: str | None = None, source_id: str | None = None) -> list[tuple[float, float]]:
    stmt = select(RawFact.raw_value).where(RawFact.fact_type_candidate == "lat_lon")
    if study_id is not None:
        stmt = stmt.where(RawFact.study_id == study_id)
    if source_id is not None:
        stmt = stmt.where(RawFact.source_id == source_id)
    parsed = []
    for raw_value in session.scalars(stmt):
        if raw_value is None:
            continue
        coordinate = parse_lat_lon(raw_value)
        if coordinate is not None:
            parsed.append(coordinate)
    return parsed


def _check_dates(existing_dates: list[date], new_dates: list[date]) -> str:
    """Returns "consistent" | "inconsistent" | "not_assessed" for the date
    dimension alone."""
    if not existing_dates or not new_dates:
        if not new_dates and existing_dates:
            existing_min, existing_max = min(existing_dates), max(existing_dates)
            if (existing_max - existing_min).days > SINGLE_STUDY_MAX_SPAN_DAYS:
                return "inconsistent"  # default-conservative: no positive signal, broad accession-level range
        return "not_assessed"

    existing_min, existing_max = min(existing_dates), max(existing_dates)
    new_min, new_max = min(new_dates), max(new_dates)
    window_start = existing_min.toordinal() - PLAUSIBLE_ADJACENCY_DAYS
    window_end = existing_max.toordinal() + PLAUSIBLE_ADJACENCY_DAYS
    if new_min.toordinal() <= window_end and new_max.toordinal() >= window_start:
        return "consistent"
    return "inconsistent"


def _check_geography(
    existing_coordinates: list[tuple[float, float]], new_coordinates: list[tuple[float, float]]
) -> str:
    """Returns "consistent" | "inconsistent" | "not_assessed" for the
    geographic dimension alone. No broad-range default-conservative case
    here (unlike dates): a large-area survey isn't on its face implausible
    for a single expedition the way a multi-year date span is for a single
    sampling campaign -- a new-evidence side with no coordinates just
    contributes not_assessed."""
    if not existing_coordinates or not new_coordinates:
        return "not_assessed"

    existing_lats = [lat for lat, _lon in existing_coordinates]
    existing_lons = [lon for _lat, lon in existing_coordinates]
    box = (
        min(existing_lats) - GEO_ADJACENCY_DEGREES, max(existing_lats) + GEO_ADJACENCY_DEGREES,
        min(existing_lons) - GEO_ADJACENCY_DEGREES, max(existing_lons) + GEO_ADJACENCY_DEGREES,
    )
    lat_min, lat_max, lon_min, lon_max = box
    for lat, lon in new_coordinates:
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            return "inconsistent"
    return "consistent"


def check_study_consistency(
    session: Session, *, existing_study: Study, new_source: Source | None
) -> ConsistencyOutcome:
    """Compares a pre-existing Study's recorded sampling date range and
    geographic extent against newly-discovered evidence's own (`new_source`
    is None when the caller has no single Source to compare against --
    e.g. fulltext-mined identifiers not tied to one adapter fetch -- in
    which case this always returns not_assessed, the conservative default).
    """
    if new_source is None:
        return ConsistencyOutcome("not_assessed", "no single Source to compare evidence against")

    existing_dates = _dates_for(session, study_id=existing_study.study_id)
    new_dates = _dates_for(session, source_id=new_source.source_id)
    date_status = _check_dates(existing_dates, new_dates)

    existing_coordinates = _coordinates_for(session, study_id=existing_study.study_id)
    new_coordinates = _coordinates_for(session, source_id=new_source.source_id)
    geo_status = _check_geography(existing_coordinates, new_coordinates)

    if date_status == "inconsistent" or geo_status == "inconsistent":
        return ConsistencyOutcome(
            "inconsistent",
            f"date={date_status}, geography={geo_status} against study {existing_study.study_id}",
        )
    if date_status == "consistent" or geo_status == "consistent":
        return ConsistencyOutcome(
            "consistent",
            f"date={date_status}, geography={geo_status} against study {existing_study.study_id}",
        )
    return ConsistencyOutcome(
        "not_assessed",
        f"date={date_status}, geography={geo_status} against study {existing_study.study_id} -- no signal either way",
    )
