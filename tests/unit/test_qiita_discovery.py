import json
import sqlite3

from fair_ocean_agent.seed_discovery.qiita_discovery import (
    QiitaDB,
    build_study_db_row,
    classify_marine,
    extract_public_study_ids_from_stats_html,
    parse_study_html,
    prep_rows_to_db,
    sample_row_to_db,
    update_mgnify_overlap,
)


def test_qiita_public_study_enumeration_from_stats_html():
    html = """
    <script>
    _generate_iconFeature(42, 1.0, 2.0);
    _generate_iconFeature(7, 3.0, 4.0);
    _generate_iconFeature(42, 1.0, 2.0);
    </script>
    """

    assert extract_public_study_ids_from_stats_html(html) == ["7", "42"]


def test_qiita_study_parser_extracts_publication_accessions_and_prep_ids():
    html = """
    <html>
      <head><title>Qiita</title></head>
      <body>
        <h1>Marine sediment amplicon study</h1>
        <p>DOI: 10.1234/Marine.Study. PMID: 12345678.</p>
        <p>Raw reads are in PRJNA704967 and ERP123456.</p>
        <a href="/public_download/?data=prep_information&prep_id=777">prep</a>
      </body>
    </html>
    """

    parsed = parse_study_html("99", html)

    assert parsed["title"] == "Marine sediment amplicon study"
    assert parsed["publication_dois"] == ["10.1234/marine.study"]
    assert parsed["pmids"] == ["12345678"]
    assert parsed["primary_bioproject"] == "PRJNA704967"
    assert parsed["ena_study_accessions"] == ["ERP123456", "PRJNA704967"]
    assert parsed["prep_ids"] == ["777"]


def test_qiita_marine_filter_keeps_broad_environment_and_rejects_human_context():
    high, high_methods = classify_marine({"title": "Coral reef seawater microbiome"})
    medium, medium_methods = classify_marine({"title": "Freshwater sediment microbial community"})
    rejected, reject_methods = classify_marine({"title": "Human gut environmental microbiome"})

    assert high == "high"
    assert "marine_term:seawater" in high_methods
    assert medium == "medium"
    assert any(method.startswith("environment_term:") for method in medium_methods)
    assert rejected == "not_marine"
    assert any(method.startswith("reject_context:") for method in reject_methods)


def test_qiita_db_uses_three_physical_tables_and_expected_views(tmp_path):
    db = QiitaDB(tmp_path / "qiita.sqlite")
    try:
        db.initialize()
        tables = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'qiita_%'"
            )
        }
        views = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'view' AND name IN "
                "('paper_seeds', 'qiita_faire_sample_enrichment', 'qiita_faire_experiment_enrichment')"
            )
        }

        assert tables == {"qiita_studies", "qiita_samples", "qiita_preparations"}
        assert views == {"paper_seeds", "qiita_faire_sample_enrichment", "qiita_faire_experiment_enrichment"}
    finally:
        db.close()


def test_qiita_sample_and_prep_ingestion_preserve_source_metadata():
    sample = sample_row_to_db(
        "12",
        {
            "sample_name": "S1",
            "latitude": "32.1",
            "longitude": "-117.2",
            "environment material": "seawater",
            "unexpected cruise field": "R/V Test",
        },
    )
    prep = prep_rows_to_db(
        "12",
        "34",
        [
            {
                "target_gene": "16S",
                "target_subfragment": "V4",
                "forward_primer": "515F",
                "reverse_primer": "806R",
                "instrument_model": "Illumina MiSeq",
            }
        ],
        {"download_status": "locator_recorded_not_downloaded", "raw_data_url": "https://qiita.ucsd.edu/public_download/?data=raw&prep_id=34"},
    )

    assert sample["qiita_sample_id"] == "S1"
    assert sample["env_medium"] == "seawater"
    assert json.loads(sample["source_metadata_json"])["unexpected cruise field"] == "R/V Test"
    assert prep["target_gene"] == "16S"
    assert prep["target_subfragment"] == "V4"
    assert prep["primer_forward"] == "515F"
    assert prep["primer_reverse"] == "806R"
    assert prep["platform"] == "Illumina MiSeq"
    assert prep["raw_sequence_available"] == 0


def test_qiita_study_upsert_is_idempotent_and_paper_seed_view(tmp_path):
    db = QiitaDB(tmp_path / "qiita.sqlite")
    try:
        db.initialize()
        study = {
            "qiita_study_id": "1",
            "title": "Marine sediment study",
            "publication_dois": ["10.1000/example"],
            "pmids": [],
            "all_sequence_accessions": ["PRJEB1"],
            "ena_study_accessions": ["PRJEB1"],
            "primary_bioproject": "PRJEB1",
        }
        db.upsert_study(build_study_db_row(study))
        db.upsert_study(build_study_db_row(study | {"title": "Marine sediment study updated"}))
        db.commit()

        assert db.count("qiita_studies") == 1
        seed = db.conn.execute("SELECT * FROM paper_seeds WHERE source_study_id = '1'").fetchone()
        assert seed["seed_source"] == "qiita"
        assert seed["primary_doi"] == "10.1000/example"
        assert seed["bioproject_accession"] == "PRJEB1"
    finally:
        db.close()


def test_qiita_mgnify_overlap_by_bioproject_and_doi(tmp_path):
    mgnify_path = tmp_path / "mgnify.sqlite"
    mgnify = sqlite3.connect(mgnify_path)
    mgnify.executescript(
        """
        CREATE TABLE mgnify_studies (
            id INTEGER PRIMARY KEY,
            mgnify_accession TEXT,
            bioproject_accession TEXT
        );
        CREATE TABLE publication_candidates (
            mgnify_study_id INTEGER,
            doi TEXT,
            normalized_doi TEXT
        );
        INSERT INTO mgnify_studies(id, mgnify_accession, bioproject_accession)
        VALUES (1, 'MGYS1', 'PRJNA1'), (2, 'MGYS2', 'PRJNA2');
        INSERT INTO publication_candidates(mgnify_study_id, doi, normalized_doi)
        VALUES (2, '10.2000/example', '10.2000/example');
        """
    )
    mgnify.commit()
    mgnify.close()

    db = QiitaDB(tmp_path / "qiita.sqlite")
    try:
        db.initialize()
        db.upsert_study(
            build_study_db_row(
                {
                    "qiita_study_id": "10",
                    "title": "Marine study",
                    "publication_dois": [],
                    "pmids": [],
                    "all_sequence_accessions": ["PRJNA1"],
                    "ena_study_accessions": ["PRJNA1"],
                    "primary_bioproject": "PRJNA1",
                }
            )
        )
        db.upsert_study(
            build_study_db_row(
                {
                    "qiita_study_id": "11",
                    "title": "Marine study",
                    "publication_dois": ["10.2000/example"],
                    "pmids": [],
                    "all_sequence_accessions": [],
                    "ena_study_accessions": [],
                }
            )
        )
        db.commit()

        assert update_mgnify_overlap(db, mgnify_path) == 2
        matches = {
            row["qiita_study_id"]: json.loads(row["matched_mgnify_ids_json"])
            for row in db.conn.execute("SELECT qiita_study_id, matched_mgnify_ids_json FROM qiita_studies")
        }
        assert matches == {"10": ["MGYS1"], "11": ["MGYS2"]}
    finally:
        db.close()
