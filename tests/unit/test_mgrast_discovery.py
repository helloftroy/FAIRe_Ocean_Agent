from __future__ import annotations

import csv
import sqlite3

import httpx
import pytest

from fair_ocean_agent.seed_discovery.mgrast_discovery import (
    MgrastClient,
    MgrastConfig,
    MgrastDB,
    MgrastDiscoveryRunner,
    classify_marine,
    dataset_rows,
    metadata_sample_rows,
    project_row_from_metadata,
    project_seed_from_item,
    sequence_accessibility_status,
    update_overlap,
)
from scripts.mgrast_seeds_to_csv import convert


def _fast_config(tmp_path) -> MgrastConfig:
    return MgrastConfig(
        db_path=tmp_path / "mgrast.sqlite",
        data_dir=tmp_path / "mgrast_data",
        mgnify_db_path=tmp_path / "mgnify.sqlite",
        qiita_db_path=tmp_path / "qiita.sqlite",
        gold_db_path=tmp_path / "gold.sqlite",
        cncb_db_path=tmp_path / "cncb.sqlite",
        min_request_interval_seconds=0.0,
        max_retries=2,
        retry_base_seconds=0.001,
    )


def test_project_seed_from_public_item():
    row = project_seed_from_item({"id": "mgp3", "name": "Marine bathypelagic", "status": "public", "pi": "A. PI"})
    assert row is not None
    assert row["mgrast_project_id"] == "mgp3"
    assert row["project_public"] is True


def test_marine_filter_uses_structured_environment_metadata():
    confidence, evidence = classify_marine({"biome": "marine habitat", "material": "seawater"})
    assert confidence == "high"
    assert any(item["term"] == "marine" for item in evidence)


def test_sample_metadata_parser_promotes_environment_fields():
    project = {
        "metagenomes": [
            {
                "sample": "mgs1",
                "coordinates": "36.51, 15.68",
                "biome": "marine habitat",
                "feature": "marine habitat",
                "material": "seawater",
            }
        ]
    }
    export = {
        "samples": [
            {
                "id": "mgs1",
                "envPackage": {
                    "data": {
                        "temperature": {"value": "12", "unit": "degree Celsius"},
                        "salinity": {"value": "38", "unit": "PSU"},
                        "nitrate": {"value": "4", "unit": "micromole per liter"},
                    }
                },
            }
        ]
    }
    rows = metadata_sample_rows("mgp3", project, export)
    assert rows[0]["mgrast_sample_id"] == "mgs1"
    assert rows[0]["latitude"] == "36.51"
    assert rows[0]["longitude"] == "15.68"
    assert rows[0]["env_medium"] == "seawater"
    assert rows[0]["temperature"] == "12 degree Celsius"


def test_dataset_parser_preserves_download_and_insdc_identifiers():
    project = {
        "metagenomes": [
            {
                "metagenome_id": "mgm4441025.3",
                "sample": "mgs1",
                "library": "mgl1",
                "name": "16S V4 seawater",
                "sequence_type": "Amplicon",
                "attributes": {"ebi_id": "ERR2192273"},
            }
        ]
    }
    export = {
        "samples": [
            {
                "id": "mgs1",
                "libraries": [
                    {
                        "id": "mgl1",
                        "data": {
                            "investigation_type": {"value": "mimarks-survey", "unit": ""},
                            "sequencing_method": {"value": "Illumina MiSeq", "unit": ""},
                        },
                    }
                ],
            }
        ]
    }
    downloads = {
        "mgm4441025.3": {
            "data": [
                {
                    "file_name": "sample_R1.fastq.gz",
                    "data_type": "sequence",
                    "file_format": "fastq",
                    "file_size": 10,
                    "file_md5": "abc",
                    "url": "https://api.mg-rast.org/download/mgm4441025.3?file=050.1",
                },
                {"file_name": "sample_R2.fastq.gz", "data_type": "sequence", "file_format": "fastq"},
            ]
        }
    }
    rows = dataset_rows("mgp3", project, export, downloads)
    assert rows[0]["mgrast_dataset_id"] == "mgm4441025.3"
    assert rows[0]["raw_sequence_available"] is True
    assert rows[0]["paired_end"] == "paired end"
    assert rows[0]["target_gene"] == "16S rRNA"
    assert rows[0]["target_subfragment"] == "V4"
    assert rows[0]["checksum"] == "abc"


def test_project_row_allows_mgrast_only_public_data_with_doi():
    project = {
        "id": "mgp3",
        "status": "public",
        "name": "Marine project",
        "pi": "A. PI",
        "description": "Publication DOI: 10.1234/example.1",
        "metagenomes": [],
    }
    export = {"id": "mgp3", "data": {"publication": {"value": "A real marine paper title", "unit": ""}}}
    datasets = [{"raw_sequence_available": True, "insdc_study_accessions_json": "[]", "run_accessions_json": "[]"}]
    row = project_row_from_metadata("mgp3", project, export, [], datasets)
    assert row["primary_doi"] == "10.1234/example.1"
    assert row["publication_resolution_status"] == "resolved"
    assert row["bioprojects_json"] == "[]"
    assert row["sequence_accessibility_status"] == "mgrast_raw_reads_confirmed"


def test_sequence_access_status_distinguishes_mgrast_plus_insdc():
    assert sequence_accessibility_status([{"raw_sequence_available": True, "run_accessions_json": "[\"ERR1\"]", "insdc_study_accessions_json": "[]"}]) == "mgrast_and_insdc_raw_reads"
    assert sequence_accessibility_status([{"raw_sequence_available": True, "run_accessions_json": "[]", "insdc_study_accessions_json": "[]"}]) == "mgrast_raw_reads_confirmed"


def test_mgrast_db_three_tables_and_export_without_bioproject(tmp_path):
    db_path = tmp_path / "mgrast.sqlite"
    db = MgrastDB(db_path)
    db.initialize()
    db.upsert_project(
        {
            "mgrast_project_id": "mgp3",
            "title": "Marine project",
            "primary_paper_title": "A paper title",
            "publication_resolution_status": "title_known_doi_missing",
            "sequence_accessibility_status": "mgrast_raw_reads_confirmed",
            "marine_confidence": "high",
            "bioprojects_json": "[]",
            "publication_dois_json": "[]",
            "pmids_json": "[]",
            "paper_titles_json": "[\"A paper title\"]",
            "publication_urls_json": "[]",
            "contacts_json": "[]",
            "marine_match_methods_json": "[]",
            "overlap_sources_json": "{}",
            "source_metadata_json": "{}",
            "first_seen_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    tables = {row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"mgrast_projects", "mgrast_samples", "mgrast_datasets"} <= tables
    assert db.conn.execute("SELECT count(*) FROM paper_seeds").fetchone()[0] == 1
    db.close()

    out = tmp_path / "seeds.csv"
    written, mgrast_only, title_only = convert(db_path, out)
    assert (written, mgrast_only, title_only) == (1, 1, 1)
    with out.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["dataset_id"] == "mgp3"
    assert row["repository"] == "mgrast"
    assert row["bioproject_accession"] == ""


def test_mgrast_client_retries_transient_timeout_then_succeeds(tmp_path):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={"ok": True})

    client = MgrastClient(_fast_config(tmp_path), transport=httpx.MockTransport(handler))
    payload = client.get_json("project")
    assert payload == {"ok": True}
    assert calls["count"] == 2


def test_mgrast_client_raises_after_exhausting_retries(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated persistent timeout", request=request)

    client = MgrastClient(_fast_config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        client.get_json("project")


def test_mgrast_client_does_not_retry_a_client_error(tmp_path):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(404, json={"error": "not found"})

    client = MgrastClient(_fast_config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("project/mgpDOES_NOT_EXIST")
    assert calls["count"] == 1


def test_discovery_stops_gracefully_on_persistent_listing_failure(tmp_path):
    """Regression test mirroring the real live crashes hit in Qiita and
    CNCB discovery: before this fix, _discovery had no try/except around
    its project-listing call at all, so a persistent failure (after
    MgrastClient's own retries -- which also didn't exist before this fix
    -- were exhausted) would propagate straight out of run() and kill the
    whole job, discarding whatever had already been discovered up to that
    point from the caller's perspective even though the DB rows themselves
    were already safely committed."""

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(dict(request.url.params).get("offset", "0"))
        if offset == 0:
            return httpx.Response(
                200,
                json={"data": [{"id": "mgp1", "name": "t", "status": "public"}], "next": "https://api.mg-rast.org/project?offset=100", "total_count": 2},
            )
        raise httpx.ReadTimeout("simulated persistent timeout", request=request)

    config = _fast_config(tmp_path)
    runner = MgrastDiscoveryRunner(config, transport=httpx.MockTransport(handler))
    try:
        counts = runner.run("discovery")
    finally:
        runner.close()

    assert counts["projects_seen"] == 1
    assert counts["discovery_errors"] == 1

    conn = sqlite3.connect(config.db_path)
    stored = {row[0] for row in conn.execute("SELECT mgrast_project_id FROM mgrast_projects")}
    conn.close()
    assert stored == {"mgp1"}


def test_update_overlap_finds_ena_doi_match_via_publication_candidates_join(tmp_path):
    """Regression test for a real bug found live: ena_studies has no
    primary_doi column at all (confirmed against the real, already-
    populated data/seed_discovery/mgnify_paper_seeds.sqlite on disk -- an
    ENA study's DOI lives in the shared publication_candidates table,
    joined via ena_study_id, matching the shape cncb_gsa_discovery.py's
    own load_mgnify_overlap already uses for mgnify_study_id). The
    original flat "SELECT primary_doi FROM ena_studies" query raised
    sqlite3.OperationalError on every real run, silently swallowed by
    _doi_set's own try/except -- ENA DOI overlap was always reported as
    zero, never actually checked."""
    mgnify_db = tmp_path / "mgnify.sqlite"
    conn = sqlite3.connect(mgnify_db)
    conn.executescript(
        """
        CREATE TABLE mgnify_studies (id INTEGER PRIMARY KEY, primary_doi TEXT, bioproject_accession TEXT);
        CREATE TABLE ena_studies (id INTEGER PRIMARY KEY, bioproject_accession TEXT);
        CREATE TABLE publication_candidates (
            id INTEGER PRIMARY KEY, mgnify_study_id INTEGER, ena_study_id INTEGER, normalized_doi TEXT
        );
        """
    )
    conn.execute("INSERT INTO ena_studies (id) VALUES (1)")
    conn.execute("INSERT INTO publication_candidates (ena_study_id, normalized_doi) VALUES (1, '10.1234/shared.doi')")
    conn.commit()
    conn.close()

    config = _fast_config(tmp_path)
    config.mgnify_db_path = mgnify_db
    db = MgrastDB(config.db_path)
    db.initialize()
    db.upsert_project(
        {
            "mgrast_project_id": "mgp9",
            "primary_doi": "10.1234/shared.doi",
            "sequence_accessibility_status": "mgrast_raw_reads_confirmed",
            "bioprojects_json": "[]",
        }
    )

    update_overlap(db, config)

    row = db.conn.execute("SELECT overlap_status, overlap_sources_json FROM mgrast_projects WHERE mgrast_project_id = 'mgp9'").fetchone()
    db.close()
    assert row[0] == "known_project_with_new_metadata"
    assert "ena" in row[1]
