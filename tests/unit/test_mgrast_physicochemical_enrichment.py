from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from fair_ocean_agent.database.enums import CanonicalStatus, EntityLevel, IdentifierType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Study, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from apply_mgrast_physicochemical_enrichment import apply_enrichment  # noqa: E402

_SAMPLE_COLUMNS = (
    "mgrast_project_id", "mgrast_sample_id", "sample_name", "biosample_accession", "latitude", "longitude",
    "collection_date", "depth", "env_broad_scale", "env_local_scale", "env_medium", "size_fraction",
    "temperature", "salinity", "ph", "dissolved_oxygen", "oxygen", "chlorophyll",
)


def _mgrast_db(tmp_path: Path, samples: list[dict]) -> Path:
    db_path = tmp_path / "mgrast.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(f"CREATE TABLE mgrast_samples ({', '.join(f'{c} TEXT' for c in _SAMPLE_COLUMNS)})")
    for sample in samples:
        conn.execute(
            f"INSERT INTO mgrast_samples({', '.join(_SAMPLE_COLUMNS)}) VALUES ({', '.join('?' for _ in _SAMPLE_COLUMNS)})",
            tuple(sample.get(c) for c in _SAMPLE_COLUMNS),
        )
    conn.commit()
    conn.close()
    return db_path


def _study(session, study_id: str) -> Study:
    study = session.get(Study, study_id)
    if study is None:
        study = Study(study_id=study_id, canonical_status=CanonicalStatus.CANDIDATE.value)
        session.add(study)
        session.flush()
    return study


def _mgrast_identifier(session, study_id: str, project_id: str) -> None:
    _study(session, study_id)
    session.add(ExternalIdentifier(study_id=study_id, identifier_type=IdentifierType.MGRAST_PROJECT_ID.value, identifier_value=project_id))
    session.flush()


def test_matched_project_creates_sample_entities_and_facts(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-1", "mgp3")
    mgrast_db = _mgrast_db(tmp_path, [{
        "mgrast_project_id": "mgp3", "mgrast_sample_id": "mgs1", "sample_name": "Station A",
        "temperature": "4.2", "salinity": "35.1", "ph": "7.9", "chlorophyll": "0.3",
        "dissolved_oxygen": "250", "latitude": "10.5", "longitude": "-140.2",
    }])

    result = apply_enrichment(db_session, mgrast_db, apply=True)
    db_session.commit()

    assert result["studies_matched_to_mgrast_samples"] == 1
    assert result["sample_entities_new"] == 1
    assert result["facts_created"] == 7  # temp, salinity, ph, chlorophyll, diss_oxygen, latitude, longitude

    entity = db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier="mgrast:mgs1").one()
    assert entity.study_id == "STUDY-1"
    assert entity.label == "Station A"
    facts = {f.fact_type_candidate: f.raw_value for f in db_session.query(RawFact).filter_by(entity_id=entity.entity_id).all()}
    assert facts["temp"] == "4.2"
    assert facts["diss_oxygen"] == "250"
    assert facts["latitude"] == "10.5"


def test_prefers_real_biosample_accession_as_external_identifier(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-2", "mgp4")
    mgrast_db = _mgrast_db(tmp_path, [{
        "mgrast_project_id": "mgp4", "mgrast_sample_id": "mgs2", "biosample_accession": "SAMN00622972", "temperature": "1.0",
    }])

    apply_enrichment(db_session, mgrast_db, apply=True)
    db_session.commit()

    entity = db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value).one()
    assert entity.external_identifier == "SAMN00622972"


def test_falls_back_to_namespaced_identifier_without_biosample_accession(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-3", "mgp5")
    mgrast_db = _mgrast_db(tmp_path, [{"mgrast_project_id": "mgp5", "mgrast_sample_id": "mgs3", "temperature": "1.0"}])

    apply_enrichment(db_session, mgrast_db, apply=True)
    db_session.commit()

    entity = db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value).one()
    assert entity.external_identifier == "mgrast:mgs3"


def test_unmatched_project_is_skipped(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-4", "mgp999")
    mgrast_db = _mgrast_db(tmp_path, [{"mgrast_project_id": "mgp1", "mgrast_sample_id": "mgs1", "temperature": "1.0"}])

    result = apply_enrichment(db_session, mgrast_db, apply=True)

    assert result["studies_matched_to_mgrast_samples"] == 0
    assert result["facts_created"] == 0
    assert db_session.query(Entity).count() == 0


def test_dry_run_reports_but_does_not_create_entities_or_facts(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-5", "mgp3")
    mgrast_db = _mgrast_db(tmp_path, [{"mgrast_project_id": "mgp3", "mgrast_sample_id": "mgs1", "temperature": "1.0"}])

    result = apply_enrichment(db_session, mgrast_db, apply=False)

    assert result["sample_entities_new"] == 1
    assert result["facts_created"] == 1
    assert db_session.query(Entity).count() == 0
    assert db_session.query(RawFact).count() == 0


def test_rerun_is_idempotent_and_does_not_duplicate_entities_or_facts(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-6", "mgp3")
    mgrast_db = _mgrast_db(tmp_path, [{"mgrast_project_id": "mgp3", "mgrast_sample_id": "mgs1", "temperature": "1.0"}])

    apply_enrichment(db_session, mgrast_db, apply=True)
    db_session.commit()
    result_second_run = apply_enrichment(db_session, mgrast_db, apply=True)
    db_session.commit()

    assert result_second_run["facts_created"] == 0
    assert result_second_run["facts_already_present"] == 1
    assert result_second_run["sample_entities_already_existed"] == 1
    assert db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier="mgrast:mgs1").count() == 1
    assert db_session.query(RawFact).filter_by(fact_type_candidate="temp").count() == 1


def test_existing_entity_from_another_source_is_reused_not_duplicated(db_session, tmp_path):
    _study(db_session, "STUDY-7")
    existing = Entity(study_id="STUDY-7", entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN00622972")
    db_session.add(existing)
    db_session.flush()
    existing_id = existing.entity_id

    _mgrast_identifier(db_session, "STUDY-7", "mgp3")
    mgrast_db = _mgrast_db(tmp_path, [{
        "mgrast_project_id": "mgp3", "mgrast_sample_id": "mgs1", "biosample_accession": "SAMN00622972", "temperature": "1.0",
    }])

    result = apply_enrichment(db_session, mgrast_db, apply=True)
    db_session.commit()

    assert result["sample_entities_already_existed"] == 1
    assert result["sample_entities_new"] == 0
    assert db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN00622972").count() == 1
    fact = db_session.query(RawFact).filter_by(fact_type_candidate="temp").one()
    assert fact.entity_id == existing_id


def test_reports_studies_that_already_completed_map_faire(db_session, tmp_path):
    _mgrast_identifier(db_session, "STUDY-8", "mgp3")
    db_session.add(Task(task_type=TaskType.MAP_FAIRE.value, study_id="STUDY-8", status=TaskStatus.COMPLETED.value, idempotency_key="k1"))
    db_session.flush()
    mgrast_db = _mgrast_db(tmp_path, [{"mgrast_project_id": "mgp3", "mgrast_sample_id": "mgs1", "temperature": "1.0"}])

    result = apply_enrichment(db_session, mgrast_db, apply=True)

    assert result["studies_needing_remap"] == ["STUDY-8"]
