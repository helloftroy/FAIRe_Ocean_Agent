import json

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Study
from fair_ocean_agent.discovery.seed_loader import (
    SeedRow,
    enqueue_seed_backfill,
    ingest_seed_file,
    ingest_seed_row,
    load_seed_rows,
)

TEMPLATE_PATH = REPO_ROOT / "data" / "seeds" / "studies_template.csv"


def test_load_seed_rows_from_template_csv():
    rows = load_seed_rows(TEMPLATE_PATH)
    assert len(rows) == 3
    assert rows[0].seed_id == "seed-001"
    assert rows[0].doi == "10.1234/example.doi"
    assert rows[1].title is None  # blank cell normalized to None


def test_load_seed_rows_from_jsonl(tmp_path):
    path = tmp_path / "seeds.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"seed_id": "a", "doi": "10.1/one"}),
                json.dumps({"seed_id": "b", "bioproject_accession": "PRJNA1"}),
            ]
        )
    )
    rows = load_seed_rows(path)
    assert len(rows) == 2
    assert rows[0].seed_id == "a"
    assert rows[1].bioproject_accession == "PRJNA1"


def test_ingest_seed_file_creates_studies_and_identifiers(db_session):
    results = ingest_seed_file(db_session, TEMPLATE_PATH)
    db_session.commit()

    assert len(results) == 3
    assert all(r.created for r in results)
    assert db_session.query(Study).count() == 3
    assert db_session.query(ExternalIdentifier).count() >= 3  # at least one id per row
    assert db_session.query(RawFact).filter_by(fact_type_candidate="seed_record").count() == 3


def test_ingest_seed_file_is_idempotent_on_rerun(db_session):
    ingest_seed_file(db_session, TEMPLATE_PATH)
    db_session.commit()
    first_study_count = db_session.query(Study).count()

    results = ingest_seed_file(db_session, TEMPLATE_PATH)
    db_session.commit()

    assert db_session.query(Study).count() == first_study_count
    assert all(not r.created for r in results)  # every row merged into existing study


def test_ingest_seed_row_reports_invalid_identifier_but_still_creates_study(db_session):
    row = SeedRow(seed_id="bad-doi", title="Broken DOI study", doi="not-a-real-doi")
    result = ingest_seed_row(db_session, row)
    db_session.commit()

    assert result.created is True
    assert len(result.identifier_errors) == 1
    study = db_session.get(Study, result.study_id)
    assert study.title == "Broken DOI study"
    assert db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).count() == 0


def test_dataset_id_uses_repository_specific_identifier_type(db_session):
    row = SeedRow(seed_id="pangaea", dataset_id="923577", repository="PANGAEA")
    result = ingest_seed_row(db_session, row)
    db_session.commit()

    identifier = db_session.query(ExternalIdentifier).filter_by(study_id=result.study_id).one()
    assert identifier.identifier_type == IdentifierType.PANGAEA_ID.value
    assert identifier.identifier_value == "923577"


def test_dataset_doi_without_repository_is_typed_as_dataset_doi(db_session):
    row = SeedRow(seed_id="dataset-doi", dataset_id="https://doi.org/10.1594/PANGAEA.923577")
    result = ingest_seed_row(db_session, row)
    db_session.commit()

    identifier = db_session.query(ExternalIdentifier).filter_by(study_id=result.study_id).one()
    assert identifier.identifier_type == IdentifierType.DATASET_DOI.value
    assert identifier.identifier_value == "10.1594/pangaea.923577"


def test_enqueue_seed_backfill_queues_one_task_per_candidate_study(db_session):
    ingest_seed_file(db_session, TEMPLATE_PATH)
    db_session.commit()

    n = enqueue_seed_backfill(db_session)
    db_session.commit()
    assert n == 3

    # re-running is idempotent, not additive
    n2 = enqueue_seed_backfill(db_session)
    db_session.commit()
    assert n2 == 3

    from fair_ocean_agent.database.models import Task

    assert db_session.query(Task).count() == 3
