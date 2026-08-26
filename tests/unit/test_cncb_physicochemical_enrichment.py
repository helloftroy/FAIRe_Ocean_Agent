from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from fair_ocean_agent.database.enums import CanonicalStatus, EntityLevel, IdentifierType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Study, Task

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from apply_cncb_physicochemical_enrichment import apply_enrichment  # noqa: E402

_SAMPLE_COLUMNS = (
    "cncb_bioproject", "samc_accession", "sample_name", "latitude", "longitude", "collection_date",
    "depth", "env_broad_scale", "env_local_scale", "env_medium", "size_fraction", "temperature",
    "salinity", "ph", "dissolved_oxygen", "oxygen", "chlorophyll",
)


def _cncb_db(tmp_path: Path, *, projects: list[dict], samples: list[dict]) -> Path:
    db_path = tmp_path / "cncb.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE cncb_projects (cncb_bioproject TEXT, cra_accessions_json TEXT)")
    conn.execute(f"CREATE TABLE cncb_samples ({', '.join(f'{c} TEXT' for c in _SAMPLE_COLUMNS)})")
    for project in projects:
        conn.execute(
            "INSERT INTO cncb_projects(cncb_bioproject, cra_accessions_json) VALUES (?, ?)",
            (project["cncb_bioproject"], project.get("cra_accessions_json", "[]")),
        )
    for sample in samples:
        conn.execute(
            f"INSERT INTO cncb_samples({', '.join(_SAMPLE_COLUMNS)}) VALUES ({', '.join('?' for _ in _SAMPLE_COLUMNS)})",
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


def _cncb_identifier(session, study_id: str, identifier_type: IdentifierType, value: str) -> None:
    _study(session, study_id)
    session.add(ExternalIdentifier(study_id=study_id, identifier_type=identifier_type.value, identifier_value=value))
    session.flush()


def test_matched_project_creates_sample_entities_and_facts(db_session, tmp_path):
    _cncb_identifier(db_session, "STUDY-1", IdentifierType.CNCB_PROJECT_ACCESSION, "PRJCA0001")
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0001", "cra_accessions_json": '["CRA0001"]'}],
        samples=[{
            "cncb_bioproject": "PRJCA0001", "samc_accession": "SAMC0001", "sample_name": "Station A",
            "temperature": "4.2", "salinity": "35.1", "ph": "7.9", "chlorophyll": "0.3",
            "dissolved_oxygen": "250", "latitude": "10.5", "longitude": "-140.2",
        }],
    )

    result = apply_enrichment(db_session, cncb_db, apply=True)
    db_session.commit()

    assert result["studies_matched_to_cncb_samples"] == 1
    assert result["sample_entities_new"] == 1
    assert result["facts_created"] == 7  # temp, salinity, ph, chlorophyll, diss_oxygen, latitude, longitude

    entity = db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMC0001").one()
    assert entity.study_id == "STUDY-1"
    assert entity.label == "Station A"
    facts = {f.fact_type_candidate: f.raw_value for f in db_session.query(RawFact).filter_by(entity_id=entity.entity_id).all()}
    assert facts["temp"] == "4.2"
    assert facts["diss_oxygen"] == "250"
    assert facts["latitude"] == "10.5"


def test_matched_via_cncb_study_accession_resolves_parent_bioproject(db_session, tmp_path):
    _cncb_identifier(db_session, "STUDY-2", IdentifierType.CNCB_STUDY_ACCESSION, "CRA0002")
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0002", "cra_accessions_json": '["CRA0002"]'}],
        samples=[{"cncb_bioproject": "PRJCA0002", "samc_accession": "SAMC0002", "sample_name": "S2", "temperature": "1.0"}],
    )

    result = apply_enrichment(db_session, cncb_db, apply=True)

    assert result["studies_matched_to_cncb_samples"] == 1
    assert result["facts_created"] == 1


def test_unmatched_cncb_identifier_is_skipped(db_session, tmp_path):
    _cncb_identifier(db_session, "STUDY-3", IdentifierType.CNCB_PROJECT_ACCESSION, "PRJCA9999")
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0001", "cra_accessions_json": "[]"}],
        samples=[{"cncb_bioproject": "PRJCA0001", "samc_accession": "SAMC0001", "sample_name": "S1", "temperature": "1.0"}],
    )

    result = apply_enrichment(db_session, cncb_db, apply=True)

    assert result["studies_matched_to_cncb_samples"] == 0
    assert result["facts_created"] == 0
    assert db_session.query(Entity).count() == 0


def test_dry_run_reports_but_does_not_create_entities_or_facts(db_session, tmp_path):
    _cncb_identifier(db_session, "STUDY-4", IdentifierType.CNCB_PROJECT_ACCESSION, "PRJCA0001")
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0001", "cra_accessions_json": "[]"}],
        samples=[{"cncb_bioproject": "PRJCA0001", "samc_accession": "SAMC0001", "sample_name": "S1", "temperature": "1.0"}],
    )

    result = apply_enrichment(db_session, cncb_db, apply=False)

    assert result["sample_entities_new"] == 1
    assert result["facts_created"] == 1
    assert db_session.query(Entity).count() == 0
    assert db_session.query(RawFact).count() == 0


def test_rerun_is_idempotent_and_does_not_duplicate_entities_or_facts(db_session, tmp_path):
    _cncb_identifier(db_session, "STUDY-5", IdentifierType.CNCB_PROJECT_ACCESSION, "PRJCA0001")
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0001", "cra_accessions_json": "[]"}],
        samples=[{"cncb_bioproject": "PRJCA0001", "samc_accession": "SAMC0001", "sample_name": "S1", "temperature": "1.0"}],
    )

    apply_enrichment(db_session, cncb_db, apply=True)
    db_session.commit()
    result_second_run = apply_enrichment(db_session, cncb_db, apply=True)
    db_session.commit()

    assert result_second_run["facts_created"] == 0
    assert result_second_run["facts_already_present"] == 1
    assert result_second_run["sample_entities_already_existed"] == 1
    assert db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMC0001").count() == 1
    assert db_session.query(RawFact).filter_by(fact_type_candidate="temp").count() == 1


def test_existing_entity_from_another_source_is_reused_not_duplicated(db_session, tmp_path):
    """A SAMC accession independently discovered another way (e.g. free-text
    mining of a paper that cites individual BioSample accessions) should
    merge into the same Entity via get_or_create_entity's own established
    accession-based lookup, not create a second, duplicate sample."""
    _study(db_session, "STUDY-6")
    existing = Entity(study_id="STUDY-6", entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMC0001")
    db_session.add(existing)
    db_session.flush()
    existing_id = existing.entity_id

    _cncb_identifier(db_session, "STUDY-6", IdentifierType.CNCB_PROJECT_ACCESSION, "PRJCA0001")
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0001", "cra_accessions_json": "[]"}],
        samples=[{"cncb_bioproject": "PRJCA0001", "samc_accession": "SAMC0001", "sample_name": "S1", "temperature": "1.0"}],
    )

    result = apply_enrichment(db_session, cncb_db, apply=True)
    db_session.commit()

    assert result["sample_entities_already_existed"] == 1
    assert result["sample_entities_new"] == 0
    assert db_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMC0001").count() == 1
    fact = db_session.query(RawFact).filter_by(fact_type_candidate="temp").one()
    assert fact.entity_id == existing_id


def test_reports_studies_that_already_completed_map_faire(db_session, tmp_path):
    _cncb_identifier(db_session, "STUDY-7", IdentifierType.CNCB_PROJECT_ACCESSION, "PRJCA0001")
    db_session.add(Task(task_type=TaskType.MAP_FAIRE.value, study_id="STUDY-7", status=TaskStatus.COMPLETED.value, idempotency_key="k1"))
    db_session.flush()
    cncb_db = _cncb_db(
        tmp_path,
        projects=[{"cncb_bioproject": "PRJCA0001", "cra_accessions_json": "[]"}],
        samples=[{"cncb_bioproject": "PRJCA0001", "samc_accession": "SAMC0001", "sample_name": "S1", "temperature": "1.0"}],
    )

    result = apply_enrichment(db_session, cncb_db, apply=True)

    assert result["studies_needing_remap"] == ["STUDY-7"]
