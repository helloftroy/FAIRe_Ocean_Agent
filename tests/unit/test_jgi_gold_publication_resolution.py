from __future__ import annotations

from pathlib import Path

from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB, utc_iso
from fair_ocean_agent.seed_discovery.jgi_gold import resolve_gold_primary_publications
from fair_ocean_agent.seed_discovery.models import ResolutionStatus


def _db(tmp_path: Path) -> SeedDiscoveryDB:
    db = SeedDiscoveryDB(tmp_path / "gold.sqlite")
    db.initialize()
    return db


def _insert_study(db: SeedDiscoveryDB, gold_study_id: str) -> None:
    now = utc_iso()
    db.conn.execute(
        """
        INSERT INTO gold_studies(gold_study_id, study_name, source_metadata_json, created_at, updated_at)
        VALUES (?, ?, '{}', ?, ?)
        """,
        (gold_study_id, gold_study_id, now, now),
    )


def _insert_project(db: SeedDiscoveryDB, gold_project_id: str, gold_study_id: str, ncbi_bioproject_accession: str | None) -> None:
    now = utc_iso()
    db.conn.execute(
        """
        INSERT INTO gold_sequencing_projects(gold_project_id, gold_study_id, ncbi_bioproject_accession, source_metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, '{}', ?, ?)
        """,
        (gold_project_id, gold_study_id, ncbi_bioproject_accession, now, now),
    )


def _insert_publication(
    db: SeedDiscoveryDB,
    *,
    gold_study_id: str | None,
    gold_project_id: str | None,
    doi: str,
) -> None:
    now = utc_iso()
    db.conn.execute(
        """
        INSERT INTO gold_study_publications(
            gold_study_id, gold_project_id, doi, match_method, matched_identifier,
            match_confidence, match_score, is_primary, created_at, updated_at
        )
        VALUES (?, ?, ?, 'gold_bulk_publication_field', ?, 'high', 1.0, 1, ?, ?)
        """,
        (gold_study_id, gold_project_id, doi, doi, now, now),
    )


def _primary(db: SeedDiscoveryDB, gold_study_id: str) -> tuple[str | None, str | None, int | None]:
    row = db.conn.execute(
        "SELECT primary_doi, primary_doi_status, primary_doi_bioproject_fanout FROM gold_studies WHERE gold_study_id = ?",
        (gold_study_id,),
    ).fetchone()
    return row["primary_doi"], row["primary_doi_status"], row["primary_doi_bioproject_fanout"]


def test_resolve_gold_primary_publications_picks_lowest_fanout_candidate(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_source")
    _insert_project(db, "Gp_source", "Gs_source", "PRJNA_SOURCE")
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_source", doi="10.1/source-paper")

    # A reanalysis paper reused across many other unrelated bioprojects.
    for i in range(40):
        study_id = f"Gs_other_{i}"
        project_id = f"Gp_other_{i}"
        _insert_study(db, study_id)
        _insert_project(db, project_id, study_id, f"PRJNA_OTHER_{i}")
        _insert_publication(db, gold_study_id=None, gold_project_id=project_id, doi="10.2/reanalysis-paper")
    # The reanalysis paper is also (weakly) linked to our study of interest.
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_source", doi="10.2/reanalysis-paper")
    db.conn.commit()

    counts = resolve_gold_primary_publications(db, low_fanout_threshold=5, apply=True)

    assert counts["resolved"] == 1  # only Gs_source has a candidate within the threshold
    assert counts["likely_reanalysis_only"] == 40  # every Gs_other_* study's only candidate is the fanout-41 paper
    primary_doi, status, fanout = _primary(db, "Gs_source")
    assert primary_doi == "10.1/source-paper"
    assert status == ResolutionStatus.RESOLVED.value
    assert fanout == 1


def test_resolve_gold_primary_publications_flags_ambiguous_tie(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_tied")
    _insert_project(db, "Gp_tied_a", "Gs_tied", "PRJNA_A")
    _insert_project(db, "Gp_tied_b", "Gs_tied", "PRJNA_B")
    # Two candidates, each linked to exactly one bioproject -- a genuine tie.
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_tied_a", doi="10.3/paper-b")
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_tied_b", doi="10.3/paper-a")
    db.conn.commit()

    counts = resolve_gold_primary_publications(db, low_fanout_threshold=5, apply=True)

    assert counts["resolved_ambiguous"] == 1
    primary_doi, status, fanout = _primary(db, "Gs_tied")
    assert primary_doi == "10.3/paper-a"  # lexicographically first, deterministic
    assert status == ResolutionStatus.RESOLVED_AMBIGUOUS.value
    assert fanout == 1


def test_resolve_gold_primary_publications_leaves_reanalysis_only_study_unresolved(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_lonely")
    for i in range(10):
        study_id = f"Gs_x_{i}"
        project_id = f"Gp_x_{i}"
        _insert_study(db, study_id)
        _insert_project(db, project_id, study_id, f"PRJNA_X_{i}")
        _insert_publication(db, gold_study_id=None, gold_project_id=project_id, doi="10.4/survey-paper")
    # Gs_lonely's only candidate is that same high-fanout survey paper.
    _insert_project(db, "Gp_lonely", "Gs_lonely", "PRJNA_LONELY")
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_lonely", doi="10.4/survey-paper")
    db.conn.commit()

    counts = resolve_gold_primary_publications(db, low_fanout_threshold=5, apply=True)

    primary_doi, status, fanout = _primary(db, "Gs_lonely")
    assert primary_doi is None
    assert status == ResolutionStatus.LIKELY_REANALYSIS_ONLY.value
    assert fanout is None
    assert counts["likely_reanalysis_only"] >= 1


def test_resolve_gold_primary_publications_study_level_row_covers_all_its_projects(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_multi")
    _insert_project(db, "Gp_multi_1", "Gs_multi", "PRJNA_M1")
    _insert_project(db, "Gp_multi_2", "Gs_multi", "PRJNA_M2")
    _insert_project(db, "Gp_multi_3", "Gs_multi", "PRJNA_M3")
    # A study-level row (no gold_project_id) implicitly claims all 3 of this study's bioprojects.
    _insert_publication(db, gold_study_id="Gs_multi", gold_project_id=None, doi="10.5/multi-site-paper")
    db.conn.commit()

    counts = resolve_gold_primary_publications(db, low_fanout_threshold=5, apply=True)

    primary_doi, status, fanout = _primary(db, "Gs_multi")
    assert fanout == 3
    assert primary_doi == "10.5/multi-site-paper"
    assert status == ResolutionStatus.RESOLVED.value
    assert counts["resolved"] == 1


def test_resolve_gold_primary_publications_unscoreable_candidate_is_not_treated_as_safe(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_unscoreable")
    # No ncbi_bioproject_accession recorded at all -- fanout can't be computed.
    _insert_project(db, "Gp_unscoreable", "Gs_unscoreable", None)
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_unscoreable", doi="10.6/unverifiable-paper")
    db.conn.commit()

    resolve_gold_primary_publications(db, low_fanout_threshold=5, apply=True)

    primary_doi, status, fanout = _primary(db, "Gs_unscoreable")
    assert primary_doi is None
    assert status == ResolutionStatus.LIKELY_REANALYSIS_ONLY.value
    assert fanout is None


def test_resolve_gold_primary_publications_dry_run_does_not_write(tmp_path):
    db = _db(tmp_path)
    _insert_study(db, "Gs_dry")
    _insert_project(db, "Gp_dry", "Gs_dry", "PRJNA_DRY")
    _insert_publication(db, gold_study_id=None, gold_project_id="Gp_dry", doi="10.7/dry-run-paper")
    db.conn.commit()

    counts = resolve_gold_primary_publications(db, low_fanout_threshold=5, apply=False)

    assert counts["resolved"] == 1
    primary_doi, status, fanout = _primary(db, "Gs_dry")
    assert primary_doi is None
    assert status is None
    assert fanout is None
