"""scripts/faire_strict/env_vars.py (R25), with values seen in the export."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "faire_strict"))
from env_vars import env_values, parse_value  # noqa: E402


@pytest.mark.parametrize("term,raw,want", [
    ("temp", "6.5 ◦C", ("6.5", None)),
    ("temp", "15.9 C", ("15.9", None)),
    ("temp", "26.0°C ± 0.5°C", ("26.0", None)),
    ("temp", "81.1 °C", ("81.1", None)),
    ("temp", "3.0817", ("3.0817", None)),
    ("salinity", "33.1 ppt", ("33.1", None)),
    ("salinity", "6.4 PSU", ("6.4", None)),
    ("ph", "8.07", ("8.07", None)),
    ("chlorophyll", "0.727", ("0.727", None)),
    ("nitrite", "0.08 uM", ("0.08", "µM")),
    ("nitrate", "10.9", ("10.9", "missing: not provided")),
    ("diss_oxygen", "330.1 _mol/kg", ("330.1", "µmol/kg")),
    ("diss_oxygen", "8.766667 mg/L", ("8.766667", "mg/L")),
    ("tot_org_carb", "16.5349 %", ("16.5349", "other: %")),
    ("nitrate", "1,5 µmol/L", ("1.5", "µmol/L")),
])
def test_usable_values(term, raw, want):
    assert parse_value(term, raw) == want


@pytest.mark.parametrize("term,raw", [
    ("salinity", ">30 PSU"),
    ("temp", "in situ bottom water temperature"),
    ("temp", "room temperature"),
    ("temp", "+2 -4°C"),
    ("temp", "250 °C"),
    ("ph", "not applicable"),
    ("nitrite", "bdl"),
    ("salinity", "salinity"),
    ("temp", "10-12 °C"),
    ("ph", "Missing"),
    ("temp", "26 °F"),
    ("tot_carb", "not collected"),
])
def test_unusable_values_are_left_alone(term, raw):
    assert parse_value(term, raw) is None


def test_block_wins_and_pulled_fills_the_rest():
    block = "dissolved oxygen: 330.1 _mol/kg|salinity: 32.3228|temperature: 3.0817|Water temperature (°C) was measured"
    pulled = "salinity = 35 | pH = 8.1 | temperature = room temperature"
    got = env_values(block, pulled)
    assert got["salinity"][:3] == ("32.3228", None, "R25a_env_var_from_block")
    assert got["temp"][:2] == ("3.0817", None)
    assert got["diss_oxygen"][:2] == ("330.1", "µmol/kg")
    assert got["ph"][:3] == ("8.1", None, "R25b_env_var_from_pulled")
    assert set(got) == {"salinity", "temp", "diss_oxygen", "ph"}


def test_prose_is_ignored():
    block = "Samples intended for sequencing were stored at -80 °C | The pH values were measured on the mini-cores"
    assert env_values(block, "") == {}
