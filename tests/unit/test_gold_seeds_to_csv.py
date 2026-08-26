from __future__ import annotations

import csv
import sys
from pathlib import Path

from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB, utc_iso

# scripts/ isn't a package on sys.path by default (these are standalone
# CLI entry points, not part of the fair_ocean_agent package) -- add it
# just for this test module rather than introducing a new global
# sys.path convention.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gold_seeds_to_csv import convert  # noqa: E402


def _db(tmp_path: Path) -> SeedDiscoveryDB:
    db = SeedDiscoveryDB(tmp_path / "gold.sqlite")
    db.initialize()
    return db


def _insert_study(db: SeedDiscoveryDB, gold_study_id: str, study_name: str, primary_doi: str | None, status: str | None, fanout: int | None) -> None:
    now = utc_iso()
    db.conn.execute(
        """
        INSERT INTO gold_studies(gold_study_id, study_name, primary_doi, primary_doi_status,
                                  primary_doi_bioproject_fanout, source_metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, '{}', ?, ?)
        """,
        (gold_study_id, study_name, primary_doi, status, fanout, now, now),
    )
    db.conn.commit()


def _insert_project(db: SeedDiscoveryDB, gold_project_id: str, gold_study_id: str, bioproject: str) -> None:
    now = utc_iso()
    db.conn.execute(
        """
        INSERT INTO gold_sequencing_projects(gold_project_id, gold_study_id, ncbi_bioproject_accession, source_metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, '{}', ?, ?)
        """,
        (gold_project_id, gold_study_id, bioproject, now, now),
    )
    db.conn.commit()


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_resolved_study_seeds_all_its_bioprojects_with_the_same_doi(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_1", "Some GOLD Study", "10.1/paper", "resolved", 2)
    _insert_project(db, "Gp_1a", "Gs_1", "PRJNA100")
    _insert_project(db, "Gp_1b", "Gs_1", "PRJNA101")
    db.close()

    written, no_doi = convert(tmp_path / "gold.sqlite", tmp_path / "out.csv")

    assert written == 2
    assert no_doi == 0
    rows = {r["bioproject_accession"]: r for r in _read_rows(tmp_path / "out.csv")}
    assert rows["PRJNA100"]["doi"] == "10.1/paper"
    assert rows["PRJNA101"]["doi"] == "10.1/paper"
    assert rows["PRJNA100"]["seed_id"] == "gold-PRJNA100"


def test_unresolved_study_excluded_by_default_but_included_with_flag(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_unresolved", "No Paper Found", None, "likely_reanalysis_only", None)
    _insert_project(db, "Gp_u", "Gs_unresolved", "PRJNA200")
    db.close()

    written, _ = convert(tmp_path / "gold.sqlite", tmp_path / "out.csv")
    assert written == 0

    written, no_doi = convert(tmp_path / "gold.sqlite", tmp_path / "out.csv", include_unresolved=True)
    assert written == 1
    assert no_doi == 1


def test_bioproject_reachable_from_two_studies_prefers_clean_resolution_over_ambiguous(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_ambiguous", "Ambiguous Study", "10.2/tie", "resolved_ambiguous_primary", 1)
    _insert_study(db, "Gs_clean", "Clean Study", "10.3/clear-winner", "resolved", 1)
    _insert_project(db, "Gp_shared_a", "Gs_ambiguous", "PRJNA300")
    _insert_project(db, "Gp_shared_b", "Gs_clean", "PRJNA300")
    db.close()

    written, _ = convert(tmp_path / "gold.sqlite", tmp_path / "out.csv")

    assert written == 1
    row = _read_rows(tmp_path / "out.csv")[0]
    assert row["doi"] == "10.3/clear-winner"
