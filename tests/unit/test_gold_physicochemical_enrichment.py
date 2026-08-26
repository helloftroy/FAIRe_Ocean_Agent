from __future__ import annotations

import sys
from pathlib import Path

from fair_ocean_agent.database.enums import CanonicalStatus, EntityLevel, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, RawFact, Study, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from apply_gold_physicochemical_enrichment import apply_enrichment  # noqa: E402


def _gold_db(tmp_path: Path, rows: list[dict]) -> Path:
    import sqlite3

    db_path = tmp_path / "gold.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE gold_sequencing_projects (gold_project_id TEXT, gold_biosample_id TEXT, ncbi_biosample_accession TEXT)"
    )
    conn.execute(
        """
        CREATE TABLE gold_faire_enrichment (
            gold_biosample_id TEXT, decimalLatitude TEXT, decimalLongitude TEXT, eventDate TEXT,
            depth TEXT, env_broad_scale TEXT, env_local_scale TEXT, env_medium TEXT,
            sample_collection_method TEXT, size_fraction TEXT, temperature TEXT, salinity TEXT,
            ph TEXT, oxygen TEXT, chlorophyll TEXT
        )
        """
    )
    for row in rows:
        conn.execute(
            "INSERT INTO gold_sequencing_projects(gold_project_id, gold_biosample_id, ncbi_biosample_accession) VALUES (?, ?, ?)",
            (row.get("gold_project_id", "Gp_1"), row["gold_biosample_id"], row["ncbi_biosample_accession"]),
        )
        conn.execute(
            """
            INSERT INTO gold_faire_enrichment(
                gold_biosample_id, decimalLatitude, decimalLongitude, eventDate, depth,
                env_broad_scale, env_local_scale, env_medium, sample_collection_method,
                size_fraction, temperature, salinity, ph, oxygen, chlorophyll
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["gold_biosample_id"],
                row.get("decimalLatitude"), row.get("decimalLongitude"), row.get("eventDate"), row.get("depth"),
                row.get("env_broad_scale"), row.get("env_local_scale"), row.get("env_medium"),
                row.get("sample_collection_method"), row.get("size_fraction"),
                row.get("temperature"), row.get("salinity"), row.get("ph"), row.get("oxygen"), row.get("chlorophyll"),
            ),
        )
    conn.commit()
    conn.close()
    return db_path


def _sample_entity(session, study_id: str, external_identifier: str) -> Entity:
    study = session.get(Study, study_id)
    if study is None:
        study = Study(study_id=study_id, canonical_status=CanonicalStatus.CANDIDATE.value)
        session.add(study)
        session.flush()
    entity = Entity(study_id=study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=external_identifier)
    session.add(entity)
    session.flush()
    return entity


def test_matched_biosample_gets_real_facts(db_session, tmp_path):
    entity = _sample_entity(db_session, "STUDY-1", "SAMN00622972")
    gold_db = _gold_db(tmp_path, [{
        "gold_biosample_id": "Gb1", "ncbi_biosample_accession": "SAMN00622972",
        "temperature": "4.2", "salinity": "35.1", "ph": "7.9", "chlorophyll": "0.3", "oxygen": "250",
        "decimalLatitude": "10.5", "decimalLongitude": "-140.2",
    }])

    result = apply_enrichment(db_session, gold_db, apply=True)
    db_session.commit()

    assert result["samples_matched"] == 1
    assert result["facts_created"] == 7  # temp, salinity, ph, chlorophyll, diss_oxygen, latitude, longitude
    facts = {f.fact_type_candidate: f.raw_value for f in db_session.query(RawFact).filter_by(entity_id=entity.entity_id).all()}
    assert facts["temp"] == "4.2"
    assert facts["salinity"] == "35.1"
    assert facts["diss_oxygen"] == "250"
    assert facts["chlorophyll"] == "0.3"
    assert facts["latitude"] == "10.5"


def test_unmatched_biosample_accession_is_skipped(db_session, tmp_path):
    _sample_entity(db_session, "STUDY-2", "SAMN99999999")
    gold_db = _gold_db(tmp_path, [{"gold_biosample_id": "Gb2", "ncbi_biosample_accession": "SAMN00000001", "temperature": "1.0"}])

    result = apply_enrichment(db_session, gold_db, apply=True)

    assert result["samples_matched"] == 0
    assert result["facts_created"] == 0


def test_dry_run_reports_but_does_not_write(db_session, tmp_path):
    entity = _sample_entity(db_session, "STUDY-3", "SAMN00622972")
    gold_db = _gold_db(tmp_path, [{"gold_biosample_id": "Gb1", "ncbi_biosample_accession": "SAMN00622972", "temperature": "4.2"}])

    result = apply_enrichment(db_session, gold_db, apply=False)

    assert result["facts_created"] == 1
    assert db_session.query(RawFact).filter_by(entity_id=entity.entity_id).count() == 0


def test_rerun_is_idempotent_and_does_not_duplicate_facts(db_session, tmp_path):
    entity = _sample_entity(db_session, "STUDY-4", "SAMN00622972")
    gold_db = _gold_db(tmp_path, [{"gold_biosample_id": "Gb1", "ncbi_biosample_accession": "SAMN00622972", "temperature": "4.2"}])

    apply_enrichment(db_session, gold_db, apply=True)
    db_session.commit()
    result_second_run = apply_enrichment(db_session, gold_db, apply=True)
    db_session.commit()

    assert result_second_run["facts_created"] == 0
    assert result_second_run["facts_already_present"] == 1
    assert db_session.query(RawFact).filter_by(entity_id=entity.entity_id, fact_type_candidate="temp").count() == 1


def test_reports_studies_that_already_completed_map_faire(db_session, tmp_path):
    entity = _sample_entity(db_session, "STUDY-5", "SAMN00622972")
    db_session.add(Task(task_type=TaskType.MAP_FAIRE.value, study_id="STUDY-5", status=TaskStatus.COMPLETED.value, idempotency_key="k1"))
    db_session.flush()
    gold_db = _gold_db(tmp_path, [{"gold_biosample_id": "Gb1", "ncbi_biosample_accession": "SAMN00622972", "temperature": "4.2"}])

    result = apply_enrichment(db_session, gold_db, apply=True)

    assert result["studies_needing_remap"] == ["STUDY-5"]
