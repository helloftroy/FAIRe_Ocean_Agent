"""Deterministic logical validators (section 16): coordinate bounds,
depth plausibility, date ordering, primer-sequence character validity,
identifier format consistency. Every function here is pure -- takes raw
string(s), returns a ValidationOutcome -- and never touches the database;
workflow/handlers.py is what persists these as ValidationResult rows.

Parsers here are deliberately tolerant of the real formats observed in
this pipeline's own data: MIxS-style structured facts from NCBI BioSample
("38.03 N 122.151667 W", bare "1" for a 1-meter depth, ISO
"2023-12-06") and free-text LLM-extracted facts ("14 March 2021", "5
meters", "up to 5,000 m"). An unparseable value is NOT_ASSESSED, never a
false ERROR -- section 2's "unknown or blank is preferable to guessing"
applies to validation too: we'd rather decline to judge than misjudge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from fair_ocean_agent.database.enums import IdentifierType, ValidationSeverity, ValidationStatus
from fair_ocean_agent.dates import try_parse_date
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier


@dataclass
class ValidationOutcome:
    status: str
    severity: str
    message: str
    compared_values: dict | None = None


# --- Coordinates -------------------------------------------------------

_LAT_LON_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([NSns])?\s+(-?\d+(?:\.\d+)?)\s*([EWew])?\s*$")


def parse_lat_lon(value: str) -> tuple[float, float] | None:
    """Parses the MIxS lat_lon convention ("38.03 N 122.151667 W") or a
    signed decimal pair ("38.03, -122.15"). Returns None (not an error) if
    unparseable -- plenty of environmental-context values aren't
    coordinates at all."""
    match = _LAT_LON_PATTERN.match(value.strip().replace(",", " "))
    if not match:
        return None
    lat_raw, lat_hem, lon_raw, lon_hem = match.groups()
    lat = float(lat_raw)
    lon = float(lon_raw)
    if lat_hem and lat_hem.upper() == "S":
        lat = -abs(lat)
    if lon_hem and lon_hem.upper() == "W":
        lon = -abs(lon)
    return lat, lon


def validate_coordinates(value: str) -> ValidationOutcome:
    parsed = parse_lat_lon(value)
    if parsed is None:
        return ValidationOutcome(
            ValidationStatus.NOT_ASSESSED.value,
            ValidationSeverity.INFO.value,
            f"Could not parse {value!r} as coordinates",
            {"raw_value": value},
        )
    lat, lon = parsed
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return ValidationOutcome(
            ValidationStatus.UNSUPPORTED.value,
            ValidationSeverity.ERROR.value,
            f"Coordinates out of range: lat={lat}, lon={lon}",
            {"lat": lat, "lon": lon},
        )
    return ValidationOutcome(
        ValidationStatus.CONFIRMED.value,
        ValidationSeverity.INFO.value,
        f"Coordinates in range: lat={lat}, lon={lon}",
        {"lat": lat, "lon": lon},
    )


# --- Depth ---------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_DEPTH_UNIT_PATTERN = re.compile(r"\b(km|kilometers?|kilometres?|m|meters?|metres?|cm|centimeters?)\b", re.IGNORECASE)
_DEPTH_UNIT_TO_METERS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
}
DEEPEST_OCEAN_POINT_METERS = 11000  # Challenger Deep, ~10,935 m -- generous round-number ceiling


def parse_depth_meters(value: str) -> float | None:
    """Extracts the first number in `value` and converts to meters using
    whatever unit token appears anywhere in the string (handles ranges
    like "0-5 cm" where the unit trails the second number, and bare MIxS
    values like "1", which are meters by convention). Returns None if no
    number is found."""
    cleaned = value.replace(",", "")
    number_match = _NUMBER_PATTERN.search(cleaned)
    if not number_match:
        return None
    number = float(number_match.group())
    unit_match = _DEPTH_UNIT_PATTERN.search(cleaned)
    unit = unit_match.group().lower() if unit_match else None
    factor = _DEPTH_UNIT_TO_METERS.get(unit, 1.0 if unit is None else None)
    if factor is None:
        return None
    return number * factor


def validate_depth(value: str) -> ValidationOutcome:
    depth_m = parse_depth_meters(value)
    if depth_m is None:
        return ValidationOutcome(
            ValidationStatus.NOT_ASSESSED.value, ValidationSeverity.INFO.value,
            f"Could not parse {value!r} as a depth", {"raw_value": value},
        )
    if depth_m < 0 or depth_m > DEEPEST_OCEAN_POINT_METERS:
        return ValidationOutcome(
            ValidationStatus.UNSUPPORTED.value, ValidationSeverity.ERROR.value,
            f"Depth {depth_m}m is outside the plausible ocean range (0-{DEEPEST_OCEAN_POINT_METERS}m)",
            {"depth_m": depth_m},
        )
    return ValidationOutcome(
        ValidationStatus.CONFIRMED.value, ValidationSeverity.INFO.value,
        f"Depth {depth_m}m is in the plausible ocean range", {"depth_m": depth_m},
    )


# --- Dates -----------------------------------------------------------------


def validate_collection_before_publication(collection_date_value: str, publication_year: int) -> ValidationOutcome:
    """A sample can't have been collected after the year the paper describing
    it was published. Publication date is only known to year precision, so
    this only flags a collection date whose *year* exceeds the publication
    year -- it cannot detect a same-year ordering problem, which would need
    the publication's exact date (not currently extracted)."""
    collection = try_parse_date(collection_date_value)
    if collection is None:
        return ValidationOutcome(
            ValidationStatus.NOT_ASSESSED.value, ValidationSeverity.INFO.value,
            f"Could not parse {collection_date_value!r} as a collection date",
            {"collection_date": collection_date_value},
        )
    if collection.year > publication_year:
        return ValidationOutcome(
            ValidationStatus.UNSUPPORTED.value, ValidationSeverity.WARNING.value,
            f"Collection date {collection} is after publication year {publication_year}",
            {"collection_date": str(collection), "publication_year": publication_year},
        )
    return ValidationOutcome(
        ValidationStatus.CONFIRMED.value, ValidationSeverity.INFO.value,
        f"Collection date {collection} does not postdate publication year {publication_year}",
        {"collection_date": str(collection), "publication_year": publication_year},
    )


# --- Primer sequences --------------------------------------------------

_IUPAC_NUCLEOTIDE_PATTERN = re.compile(r"^[ACGTURYSWKMBDHVN]+$", re.IGNORECASE)
_MIN_PLAUSIBLE_PRIMER_LENGTH = 6


def validate_primer_sequence(value: str) -> ValidationOutcome:
    """Only meaningful when `value` actually looks like a nucleotide
    sequence (IUPAC letters only). Many "primer" facts report a primer's
    *name* (e.g. "TAReuk454FWD1") rather than its base sequence, since
    papers often cite a primer by name without spelling it out -- a name is
    NOT_ASSESSED, not an invalid sequence."""
    candidate = re.sub(r"\s", "", value)
    if len(candidate) < _MIN_PLAUSIBLE_PRIMER_LENGTH:
        return ValidationOutcome(
            ValidationStatus.NOT_ASSESSED.value, ValidationSeverity.INFO.value,
            f"Too short to assess as a sequence: {value!r}", {"raw_value": value},
        )
    if not _IUPAC_NUCLEOTIDE_PATTERN.match(candidate):
        return ValidationOutcome(
            ValidationStatus.NOT_ASSESSED.value, ValidationSeverity.INFO.value,
            f"Does not look like a nucleotide sequence (may be a primer name): {value!r}",
            {"raw_value": value},
        )
    return ValidationOutcome(
        ValidationStatus.CONFIRMED.value, ValidationSeverity.INFO.value,
        f"Valid IUPAC nucleotide sequence ({len(candidate)} bases)",
        {"length": len(candidate)},
    )


# --- Identifier formats --------------------------------------------------


def validate_accession_format(identifier_type: IdentifierType, value: str) -> ValidationOutcome:
    """Re-validates an already-stored identifier's format -- an audit/
    consistency check, since malformed identifiers should already have been
    rejected at ingestion (seed_loader, workflow/handlers.py). A failure
    here would indicate a normalization bug elsewhere, not bad input."""
    try:
        normalized = normalize_identifier(identifier_type, value)
    except IdentifierError as exc:
        return ValidationOutcome(
            ValidationStatus.UNSUPPORTED.value, ValidationSeverity.ERROR.value, str(exc), {"raw_value": value},
        )
    return ValidationOutcome(
        ValidationStatus.CONFIRMED.value, ValidationSeverity.INFO.value,
        f"Valid {identifier_type.value} format", {"raw_value": value, "normalized": normalized},
    )
