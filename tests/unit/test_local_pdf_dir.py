"""Tests for the per-study local-PDF directory lookup
(_local_pdf_path_for_study / FAIR_OCEAN_LOCAL_PDF_DIR) -- per an explicit
user request to test PDF-supplied and non-PDF-supplied papers together in
the same batch run, which the older single-path FAIR_OCEAN_LOCAL_PDF_PATH
override can't do (one PDF per whole process invocation)."""
from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.database.models import ExternalIdentifier, Study
from fair_ocean_agent.identity.identifiers import normalize_doi
from fair_ocean_agent.sources.base import SourceRecordNotFoundError
from fair_ocean_agent.workflow import handlers
import pytest


def _study_with_doi(session, doi) -> Study:
    study = Study(title="A study")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=normalize_doi(doi)))
    session.flush()
    return study


def test_doi_pdf_filename_replaces_slashes_and_lowercases():
    assert handlers._doi_pdf_filename("10.1002/2015JG003300") == "10.1002_2015jg003300.pdf"


def test_local_pdf_path_for_study_finds_matching_file_in_dir(db_session, monkeypatch, tmp_path):
    study = _study_with_doi(db_session, "10.1002/2015JG003300")
    pdf_path = tmp_path / "10.1002_2015jg003300.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))

    found = handlers._local_pdf_path_for_study(db_session, study)

    assert found == pdf_path


def test_local_pdf_path_for_study_returns_none_when_no_matching_file(db_session, monkeypatch, tmp_path):
    """The ordinary, expected case for a paper with no supplied PDF --
    must not raise, callers fall back to the normal Europe PMC fetch."""
    study = _study_with_doi(db_session, "10.1234/not-supplied")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))

    assert handlers._local_pdf_path_for_study(db_session, study) is None


def test_local_pdf_path_for_study_returns_none_without_a_doi(db_session, monkeypatch, tmp_path):
    study = Study(title="No DOI at all")
    db_session.add(study)
    db_session.flush()
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))

    assert handlers._local_pdf_path_for_study(db_session, study) is None


def test_local_pdf_path_env_still_wins_over_dir(db_session, monkeypatch, tmp_path):
    """The single-file global override keeps its original one-paper-per-run
    behavior even when a directory is also configured."""
    study = _study_with_doi(db_session, "10.1002/2015JG003300")
    dir_pdf = tmp_path / "10.1002_2015jg003300.pdf"
    dir_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    override_pdf = tmp_path / "override.pdf"
    override_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(handlers.LOCAL_PDF_PATH_ENV, str(override_pdf))

    assert handlers._local_pdf_path_for_study(db_session, study) == override_pdf


def test_local_pdf_path_env_raises_when_explicitly_set_but_missing(db_session, monkeypatch):
    """A deliberately-set override that doesn't exist is a real
    configuration mistake, not "no PDF for this study" -- still raises,
    matching the original single-path behavior exactly."""
    study = _study_with_doi(db_session, "10.1002/2015JG003300")
    monkeypatch.setenv(handlers.LOCAL_PDF_PATH_ENV, "/nonexistent/path.pdf")

    with pytest.raises(SourceRecordNotFoundError):
        handlers._local_pdf_path_for_study(db_session, study)


def test_local_pdf_path_for_study_matches_case_insensitively(db_session, monkeypatch, tmp_path):
    """A DOI's own casing (often uppercase from a citation) shouldn't have
    to match the saved filename's casing exactly."""
    study = _study_with_doi(db_session, "10.1002/2015JG003300")
    pdf_path = tmp_path / "10.1002_2015jg003300.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    monkeypatch.setenv(handlers.LOCAL_PDF_DIR_ENV, str(tmp_path))

    assert handlers._local_pdf_path_for_study(db_session, study) == pdf_path
