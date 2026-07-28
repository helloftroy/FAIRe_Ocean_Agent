from fair_ocean_agent.mapping.units import to_decimal_lat_lon, to_iso_event_date, to_meters


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


def test_to_meters_plain_and_unitful():
    assert to_meters("5") == "5"
    assert to_meters("5 meters") == "5"
    assert to_meters("1.5 km") == "1500"


def test_to_meters_unparseable_returns_none():
    assert to_meters("not a depth") is None


def test_to_decimal_lat_lon_mixs_convention():
    lat, lon = to_decimal_lat_lon("38.03 N 122.151667 W")
    assert lat == "38.030000"
    assert lon == "-122.151667"


def test_to_decimal_lat_lon_unparseable_returns_none():
    assert to_decimal_lat_lon("marine pelagic zone") is None
