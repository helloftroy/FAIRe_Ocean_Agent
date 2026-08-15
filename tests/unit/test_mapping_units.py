from fair_ocean_agent.mapping.rules import _volume_value
from fair_ocean_agent.mapping.units import to_decimal_lat_lon, to_iso_event_date, to_max_meters, to_meters, to_min_meters


def test_to_iso_event_date_full_precision():
    assert to_iso_event_date("2023-12-06") == "2023-12-06"
    assert to_iso_event_date("14 March 2021") == "2021-03-14"


def test_to_iso_event_date_month_precision_not_falsely_precise():
    """Regression guard: a month-only source value must not silently gain a
    fabricated day-of-month in the ISO output."""
    assert to_iso_event_date("March 2021") == "2021-03"
    assert to_iso_event_date("2021-03") == "2021-03"


def test_to_iso_event_date_year_only():
    assert to_iso_event_date("2021") == "2021"


def test_to_iso_event_date_unparseable_returns_none():
    assert to_iso_event_date("Illumina MiSeq") is None


def test_volume_value_accepts_per_volume_concentration_units():
    assert _volume_value("RNA samples were normalized to 10 ng μL−1") == (
        "RNA samples were normalized to 10 ng μL−1"
    )
    assert _volume_value("template normalized to 10 ng uL-1") == "template normalized to 10 ng uL-1"


def test_to_meters_plain_and_unitful():
    assert to_meters("5") == "5"
    assert to_meters("5 meters") == "5"
    assert to_meters("1.5 km") == "1500"


def test_to_meters_unparseable_returns_none():
    assert to_meters("not a depth") is None


def test_to_min_max_meters_split_a_real_depth_range():
    """Regression guard for a real bug found live (10.1371/journal.pone.0303937):
    "epilimnion (0-20 m)" is a genuine range, but minimumDepthInMeters and
    maximumDepthInMeters both used to call plain to_meters on the same raw
    fact, which only ever returns the FIRST number -- silently producing
    the identical (wrong) value for both ends of the range."""
    assert to_min_meters("Integrated samples of the epilimnion (0–20 m) were taken") == "0"
    assert to_max_meters("Integrated samples of the epilimnion (0–20 m) were taken") == "20"


def test_to_min_max_meters_single_value_are_the_same():
    """FAIRe's own stated convention: no range means min and max are the
    same value."""
    assert to_min_meters("5 m") == to_max_meters("5 m") == "5"


def test_to_min_max_meters_unparseable_returns_none():
    assert to_min_meters("upper few millimeters") is None
    assert to_max_meters("upper few millimeters") is None


def test_to_decimal_lat_lon_mixs_convention():
    lat, lon = to_decimal_lat_lon("38.03 N 122.151667 W")
    assert lat == "38.030000"
    assert lon == "-122.151667"


def test_to_decimal_lat_lon_unparseable_returns_none():
    assert to_decimal_lat_lon("marine pelagic zone") is None
