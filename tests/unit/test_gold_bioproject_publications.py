from __future__ import annotations

from pathlib import Path

import httpx

from fair_ocean_agent.seed_discovery.clients.europepmc import EuropePmcSeedClient
from fair_ocean_agent.seed_discovery.clients.ncbi import NcbiPublicationClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB, utc_iso
from fair_ocean_agent.seed_discovery.gold_bioproject_publications import GoldBioprojectPublicationSearchRunner
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


def _db(tmp_path: Path) -> SeedDiscoveryDB:
    db = SeedDiscoveryDB(tmp_path / "gold.sqlite")
    db.initialize()
    return db


def _insert_project(db: SeedDiscoveryDB, gold_project_id: str, gold_study_id: str, ncbi_bioproject_accession: str) -> None:
    now = utc_iso()
    db.conn.execute(
        """
        INSERT INTO gold_sequencing_projects(gold_project_id, gold_study_id, ncbi_bioproject_accession, source_metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, '{}', ?, ?)
        """,
        (gold_project_id, gold_study_id, ncbi_bioproject_accession, now, now),
    )
    db.conn.commit()


def _runner(tmp_path: Path, db: SeedDiscoveryDB) -> GoldBioprojectPublicationSearchRunner:
    return GoldBioprojectPublicationSearchRunner(SeedDiscoveryConfig(db_path=db.path), db)


def test_search_stores_candidate_against_the_owning_study_and_project(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _insert_project(db, "Gp_1", "Gs_1", "PRJNA100")

    monkeypatch.setattr(
        EuropePmcSeedClient, "accession_search",
        lambda self, accession: [
            PublicationCandidate(
                doi="10.1/found-it", title="A real paper", match_method="europepmc_accession",
                matched_identifier=accession, match_confidence=MatchConfidence.HIGH, match_score=90.0,
            )
        ],
    )
    monkeypatch.setattr(NcbiPublicationClient, "pubmed_for_bioproject", lambda self, accession: [])

    runner = _runner(tmp_path, db)
    counts = runner.run(max_consecutive_rate_limit_failures=5)
    runner.close()

    assert counts["accessions_checked"] == 1
    assert counts["accessions_with_candidates"] == 1
    row = db.conn.execute(
        "SELECT gold_study_id, gold_project_id, doi, match_method FROM gold_study_publications WHERE doi = '10.1/found-it'"
    ).fetchone()
    assert row["gold_study_id"] == "Gs_1"
    assert row["gold_project_id"] == "Gp_1"
    assert row["match_method"] == "europepmc_accession"


def test_ncbi_pmid_only_candidate_gets_doi_filled_in_from_europepmc(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _insert_project(db, "Gp_2", "Gs_2", "PRJNA200")

    monkeypatch.setattr(EuropePmcSeedClient, "accession_search", lambda self, accession: [])
    monkeypatch.setattr(
        NcbiPublicationClient, "pubmed_for_bioproject",
        lambda self, accession: [
            PublicationCandidate(pmid="12345", match_method="ncbi_link", matched_identifier=accession,
                                  match_confidence=MatchConfidence.VERY_HIGH, match_score=95.0)
        ],
    )
    monkeypatch.setattr(
        EuropePmcSeedClient, "resolve_pmid",
        lambda self, pmid: PublicationCandidate(doi="10.2/from-pmid", pmid=pmid, title="Resolved via PMID"),
    )

    runner = _runner(tmp_path, db)
    runner.run(max_consecutive_rate_limit_failures=5)
    runner.close()

    row = db.conn.execute("SELECT doi, pmid FROM gold_study_publications WHERE gold_study_id = 'Gs_2'").fetchone()
    assert row["doi"] == "10.2/from-pmid"
    assert row["pmid"] == "12345"


def test_second_run_skips_already_checked_accessions_unless_refreshed(tmp_path, monkeypatch):
    db = _db(tmp_path)
    _insert_project(db, "Gp_3", "Gs_3", "PRJNA300")
    calls = {"count": 0}

    def fake_search(self, accession):
        calls["count"] += 1
        return []

    monkeypatch.setattr(EuropePmcSeedClient, "accession_search", fake_search)
    monkeypatch.setattr(NcbiPublicationClient, "pubmed_for_bioproject", lambda self, accession: [])

    runner = _runner(tmp_path, db)
    runner.run()
    runner.close()
    assert calls["count"] == 1

    runner2 = _runner(tmp_path, db)
    counts = runner2.run()
    runner2.close()
    assert calls["count"] == 1  # not called again -- already checked
    assert counts["accessions_to_check"] == 0

    runner3 = _runner(tmp_path, db)
    runner3.run(refresh=True)
    runner3.close()
    assert calls["count"] == 2  # refresh re-checks it


def test_stops_after_consecutive_rate_limit_failures(tmp_path, monkeypatch):
    db = _db(tmp_path)
    for i in range(10):
        _insert_project(db, f"Gp_rl_{i}", f"Gs_rl_{i}", f"PRJNA_RL_{i}")

    def failing_search(self, accession):
        request = httpx.Request("GET", "https://www.ebi.ac.uk/europepmc/webservices/rest/search")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("429", request=request, response=response)

    monkeypatch.setattr(EuropePmcSeedClient, "accession_search", failing_search)
    monkeypatch.setattr(NcbiPublicationClient, "pubmed_for_bioproject", lambda self, accession: [])

    runner = _runner(tmp_path, db)
    counts = runner.run(max_consecutive_rate_limit_failures=3)
    runner.close()

    assert counts["errored"] == 3
    assert counts["stopped_reason"] is not None
    remaining_unchecked = db.conn.execute(
        "SELECT COUNT(*) FROM gold_sequencing_projects sp LEFT JOIN gold_bioproject_publication_search s "
        "ON s.bioproject_accession = sp.ncbi_bioproject_accession WHERE s.bioproject_accession IS NULL"
    ).fetchone()[0]
    assert remaining_unchecked == 7
