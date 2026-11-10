"""FAIRe CSV writing must survive NUL characters in values (found live: 109
values from PDF text extraction made csv raise "need to escape, but no
escapechar set" and stopped the export after the first class)."""
import csv

from fair_ocean_agent.exports.faire import _write_csv


def test_write_csv_drops_nul_characters(tmp_path):
    path = tmp_path / "out.csv"
    _write_csv(path, ["a", "b"], [{"a": "x\x00y", "b": "plain, with comma"}])
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "xy", "b": "plain, with comma"}]
