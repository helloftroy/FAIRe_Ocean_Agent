import pytest

from fair_ocean_agent.database.enums import IdentifierType, ValidationStatus
from fair_ocean_agent.validation.logical import (
    parse_depth_meters,
    parse_lat_lon,
    validate_accession_format,
    validate_collection_before_publication,
    validate_coordinates,
    validate_depth,
    validate_primer_sequence,
)

# --- Coordinates: all real-data formats observed live ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("38.03 N 122.151667 W", (38.03, -122.151667)),
        ("69.00203 N, 33.0201 E", (69.00203, 33.0201)),
        ("38.03, -122.15", (38.03, -122.15)),
    ],
)
def test_parse_lat_lon_real_formats(value, expected):
    assert parse_lat_lon(value) == pytest.approx(expected)


def test_parse_lat_lon_unparseable_returns_none():
    assert parse_lat_lon("San Francisco Bay") is None


def test_validate_coordinates_in_range():
    outcome = validate_coordinates("38.03 N 122.151667 W")
    assert outcome.status == ValidationStatus.CONFIRMED.value


def test_validate_coordinates_out_of_range():
    outcome = validate_coordinates("95 N 200 W")
    assert outcome.status == ValidationStatus.UNSUPPORTED.value


def test_validate_coordinates_unparseable_is_not_assessed_not_error():
    outcome = validate_coordinates("intertidal zone")
    assert outcome.status == ValidationStatus.NOT_ASSESSED.value


# --- Depth: all real-data formats observed live ---


@pytest.mark.parametrize(
    "value,expected_meters",
    [
        ("1", 1.0),
        ("5 meters", 5.0),
        ("0–5 cm", 0.0),
        ("up to 5,000 m", 5000.0),
        ("2 km", 2000.0),
    ],
)
def test_parse_depth_meters_real_formats(value, expected_meters):
    assert parse_depth_meters(value) == pytest.approx(expected_meters)


def test_parse_depth_unparseable_returns_none():
    assert parse_depth_meters("surface") is None


def test_validate_depth_plausible():
    assert validate_depth("100 m").status == ValidationStatus.CONFIRMED.value


def test_validate_depth_implausible():
    assert validate_depth("50000 m").status == ValidationStatus.UNSUPPORTED.value


def test_validate_depth_negative_is_unsupported():
    assert validate_depth("-5 m").status == ValidationStatus.UNSUPPORTED.value


# --- Dates ---


def test_collection_before_publication_confirmed():
    outcome = validate_collection_before_publication("2023-12-06", 2025)
    assert outcome.status == ValidationStatus.CONFIRMED.value


def test_collection_after_publication_is_unsupported():
    outcome = validate_collection_before_publication("2027-01-01", 2025)
    assert outcome.status == ValidationStatus.UNSUPPORTED.value


def test_collection_same_year_as_publication_is_confirmed():
    outcome = validate_collection_before_publication("2025-06-01", 2025)
    assert outcome.status == ValidationStatus.CONFIRMED.value


def test_unparseable_collection_date_is_not_assessed():
    outcome = validate_collection_before_publication("during the wet season", 2025)
    assert outcome.status == ValidationStatus.NOT_ASSESSED.value


# --- Primer sequences ---


def test_primer_name_is_not_assessed_not_invalid():
    """Regression: many "primer" facts report a primer's NAME (real
    example: "TAReuk454FWD1"), not its base sequence -- must not be
    flagged as an invalid sequence."""
    outcome = validate_primer_sequence("TAReuk454FWD1")
    assert outcome.status == ValidationStatus.NOT_ASSESSED.value


def test_valid_nucleotide_sequence_confirmed():
    outcome = validate_primer_sequence("GGWACWGGWTGAACWGTWTAYCCYCC")
    assert outcome.status == ValidationStatus.CONFIRMED.value


def test_short_value_is_not_assessed():
    outcome = validate_primer_sequence("AC")
    assert outcome.status == ValidationStatus.NOT_ASSESSED.value


# --- Accession format re-validation ---


def test_valid_accession_confirmed():
    outcome = validate_accession_format(IdentifierType.BIOPROJECT_ACCESSION, "PRJNA123456")
    assert outcome.status == ValidationStatus.CONFIRMED.value
    assert outcome.compared_values["raw_value"] == "PRJNA123456"


def test_invalid_accession_is_unsupported():
    outcome = validate_accession_format(IdentifierType.BIOPROJECT_ACCESSION, "not-an-accession")
    assert outcome.status == ValidationStatus.UNSUPPORTED.value
