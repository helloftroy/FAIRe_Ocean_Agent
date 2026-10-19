"""enqueue_text_extraction_backfill must queue every study the extraction
handler can actually read -- including a paper whose only full text is a
local PDF. Real gap found live: it used to queue PMCID studies only, so
PDF-only papers in FAIR_OCEAN_LOCAL_PDF_DIR were never extracted."""
from fair_ocean_agent.database.enums import IdentifierType, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, Study, Task
from fair_ocean_agent.workflow import handlers


def _study(session, **identifiers) -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    for identifier_type, value in identifiers.items():
        session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=identifier_type, identifier_value=value))
    session.flush()
    return study


def _queued(session) -> set[str]:
    return {t.study_id for t in session.query(Task).filter_by(task_type=TaskType.EXTRACT_TEXT_FACTS.value)}


def test_backfill_queues_pmcid_and_pdf_only_studies_but_not_studies_without_full_text(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("FAIR_OCEAN_LOCAL_PDF_DIR", str(tmp_path))
    monkeypatch.delenv("FAIR_OCEAN_LOCAL_PDF_PATH", raising=False)
    pmcid_only = _study(db_session, **{IdentifierType.PMCID.value: "PMC123"})
    pdf_only = _study(db_session, **{IdentifierType.DOI.value: "10.1234/abc.5"})
    no_text = _study(db_session, **{IdentifierType.DOI.value: "10.1234/missing"})
    (tmp_path / "10.1234_abc.5.pdf").write_bytes(b"%PDF-1.4")

    assert handlers.enqueue_text_extraction_backfill(db_session) == 2
    db_session.commit()

    assert _queued(db_session) == {pmcid_only.study_id, pdf_only.study_id}
    assert no_text.study_id not in _queued(db_session)
    # Exactly what the extraction handler itself will look up.
    assert handlers._local_pdf_path_for_study(db_session, pdf_only) is not None
    assert handlers._local_pdf_path_for_study(db_session, no_text) is None


def test_studies_with_local_pdf_matches_the_handler_on_the_oldest_doi_only(db_session, tmp_path, monkeypatch):
    """A study with two recorded DOIs is matched on the one the handler
    actually uses (the oldest) -- matching on any DOI would queue tasks the
    handler then cannot read a PDF for."""
    monkeypatch.setenv("FAIR_OCEAN_LOCAL_PDF_DIR", str(tmp_path))
    monkeypatch.delenv("FAIR_OCEAN_LOCAL_PDF_PATH", raising=False)
    study = _study(db_session, **{IdentifierType.DOI.value: "10.1234/first"})
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1234/second"))
    db_session.flush()
    (tmp_path / "10.1234_second.pdf").write_bytes(b"%PDF-1.4")

    assert study.study_id not in handlers.studies_with_local_pdf(db_session)
    assert handlers._local_pdf_path_for_study(db_session, study) is None


def test_backfill_is_idempotent(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("FAIR_OCEAN_LOCAL_PDF_DIR", str(tmp_path))
    _study(db_session, **{IdentifierType.PMCID.value: "PMC1"})
    handlers.enqueue_text_extraction_backfill(db_session)
    handlers.enqueue_text_extraction_backfill(db_session)
    db_session.commit()
    assert db_session.query(Task).filter_by(task_type=TaskType.EXTRACT_TEXT_FACTS.value).count() == 1
