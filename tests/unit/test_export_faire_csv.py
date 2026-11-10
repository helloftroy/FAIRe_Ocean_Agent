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


def test_export_faire_writes_only_the_selected_tables(db_session, tmp_path):
    import pytest

    from fair_ocean_agent.database.models import Study
    from fair_ocean_agent.exports.faire import export_faire

    db_session.add(Study(title="One study"))
    db_session.commit()
    counts = export_faire(db_session, tmp_path, tables=["projectMetadata"])
    assert list(counts) == ["projectMetadata"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["projectMetadata.csv"]
    with pytest.raises(ValueError, match="unknown FAIRe export table"):
        export_faire(db_session, tmp_path, tables=["sampleMetadta"])
