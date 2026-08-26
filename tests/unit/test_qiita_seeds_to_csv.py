from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qiita_seeds_to_csv import convert  # noqa: E402


def _qiita_db(tmp_path: Path, studies: list[dict]) -> Path:
    db_path = tmp_path / "qiita.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE qiita_studies (
            qiita_study_id TEXT PRIMARY KEY, title TEXT, primary_doi TEXT, pmids_json TEXT,
            primary_bioproject TEXT, ena_study_accessions_json TEXT, marine_confidence TEXT,
            overlaps_mgnify INTEGER, accession_resolution_status TEXT, sequence_accessibility_status TEXT
        )
        """
    )
    for study in studies:
        conn.execute(
            """
            INSERT INTO qiita_studies(
                qiita_study_id, title, primary_doi, pmids_json, primary_bioproject,
                ena_study_accessions_json, marine_confidence, overlaps_mgnify,
                accession_resolution_status, sequence_accessibility_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study["qiita_study_id"], study.get("title"), study.get("primary_doi"),
                json.dumps(study.get("pmids", [])), study.get("primary_bioproject"),
                json.dumps(study.get("ena_study_accessions", [])), study.get("marine_confidence", "high"),
                int(study.get("overlaps_mgnify", 0)), study.get("accession_resolution_status", "unresolved_sequence_accession"),
                study.get("sequence_accessibility_status", "no_sequence_locator_found"),
            ),
        )
    conn.commit()
    conn.close()
    return db_path


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_bioproject_takes_priority(tmp_path):
    db = _qiita_db(tmp_path, [{
        "qiita_study_id": "1", "primary_bioproject": "PRJNA100",
        "ena_study_accessions": ["PRJEB200"], "primary_doi": "10.1/x",
    }])
    written, bioproject_seeded, ena_seeded, qiita_only = convert(db, tmp_path / "out.csv")
    assert (written, bioproject_seeded, ena_seeded, qiita_only) == (1, 1, 0, 0)
    row = _read_rows(tmp_path / "out.csv")[0]
    assert row["bioproject_accession"] == "PRJNA100"
    assert row["ena_study_accession"] == ""


def test_valid_ena_accession_used_when_no_bioproject(tmp_path):
    db = _qiita_db(tmp_path, [{
        "qiita_study_id": "2", "ena_study_accessions": ["PRJEB200"], "primary_doi": "10.1/y",
    }])
    written, bioproject_seeded, ena_seeded, qiita_only = convert(db, tmp_path / "out.csv")
    assert (bioproject_seeded, ena_seeded, qiita_only) == (0, 1, 0)
    assert _read_rows(tmp_path / "out.csv")[0]["ena_study_accession"] == "PRJEB200"


def test_ncbi_shaped_ena_value_rejected_same_as_mgnify_exporter(tmp_path):
    """PRJNA... in ena_study_accessions_json is an NCBI-mirrored value, not
    ENA-native -- same real bug scripts/mgnify_seeds_to_csv.py already had
    to fix for the same reason (ENA's own strict accession normalizer
    rejects it outright)."""
    db = _qiita_db(tmp_path, [{
        "qiita_study_id": "3", "ena_study_accessions": ["PRJNA555"], "primary_doi": "10.1/z",
    }])
    written, bioproject_seeded, ena_seeded, qiita_only = convert(db, tmp_path / "out.csv")
    assert (bioproject_seeded, ena_seeded, qiita_only) == (0, 0, 1)
    row = _read_rows(tmp_path / "out.csv")[0]
    assert row["ena_study_accession"] == ""
    assert row["repository"] == "qiita"


def test_qiita_only_study_preserved_with_repository_marker(tmp_path):
    """The case the user explicitly cares about: real DOI + sequence data
    that was never submitted to ENA/SRA at all -- must not be silently
    dropped just because there's no BioProject/ENA accession to seed."""
    db = _qiita_db(tmp_path, [{
        "qiita_study_id": "4", "title": "Never-in-ENA study", "primary_doi": "10.1/new-data",
        "sequence_accessibility_status": "raw_artifact_present_download_unverified",
    }])
    written, bioproject_seeded, ena_seeded, qiita_only = convert(db, tmp_path / "out.csv")
    assert (written, qiita_only) == (1, 1)
    row = _read_rows(tmp_path / "out.csv")[0]
    assert row["doi"] == "10.1/new-data"
    assert row["dataset_id"] == "4"
    assert row["repository"] == "qiita"
    assert row["url"] == "https://qiita.ucsd.edu/study/description/4"


def test_not_marine_studies_are_excluded(tmp_path):
    db = _qiita_db(tmp_path, [{"qiita_study_id": "5", "marine_confidence": "not_marine"}])
    written, *_ = convert(db, tmp_path / "out.csv")
    assert written == 0
