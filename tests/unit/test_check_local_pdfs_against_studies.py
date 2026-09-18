"""Tests for scripts/check_local_pdfs_against_studies.py -- a real,
concrete question a batch of newly-uploaded PDFs raises: which of them
correspond to a study the pipeline already knows about (and has it already
used the PDF, or not yet) vs. which are orphaned (no matching study at all,
so nothing will ever look at them until one is seeded)."""
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_local_pdfs_against_studies import build_report  # noqa: E402

from fair_ocean_agent.database.enums import IdentifierType, SourceType
from fair_ocean_agent.database.models import Base, ExternalIdentifier, Source, Study


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def _add_doi_study(session, doi: str) -> Study:
    study = Study(title=f"paper for {doi}")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=doi))
    session.commit()
    return study


def test_orphaned_pdf_with_no_matching_study(tmp_path):
    session = _session(tmp_path)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "10.1234_never-seeded.pdf").write_bytes(b"%PDF-1.4")

    report = build_report(session, pdf_dir)

    assert report["orphaned_no_study"] == ["10.1234_never-seeded.pdf"]
    assert report["matched_and_confirmed"] == []
    assert report["matched_not_yet_used"] == []


def test_matched_pdf_not_yet_used_when_no_fulltext_source_exists(tmp_path):
    session = _session(tmp_path)
    study = _add_doi_study(session, "10.1234/already-seeded")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "10.1234_already-seeded.pdf").write_bytes(b"%PDF-1.4")

    report = build_report(session, pdf_dir)

    assert report["matched_not_yet_used"] == [("10.1234_already-seeded.pdf", study.study_id)]
    assert report["matched_and_confirmed"] == []
    assert report["orphaned_no_study"] == []


def test_matched_pdf_already_confirmed_when_fulltext_source_exists(tmp_path):
    session = _session(tmp_path)
    study = _add_doi_study(session, "10.1234/already-extracted")
    session.add(Source(study_id=study.study_id, source_type=SourceType.ARTICLE_FULLTEXT.value, source_name="local_pdf"))
    session.commit()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "10.1234_already-extracted.pdf").write_bytes(b"%PDF-1.4")

    report = build_report(session, pdf_dir)

    assert report["matched_and_confirmed"] == [("10.1234_already-extracted.pdf", study.study_id)]
    assert report["matched_not_yet_used"] == []


def test_filename_matching_is_case_insensitive(tmp_path):
    session = _session(tmp_path)
    study = _add_doi_study(session, "10.1234/Mixed-Case-DOI")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "10.1234_MIXED-CASE-DOI.PDF".lower()).write_bytes(b"%PDF-1.4")

    report = build_report(session, pdf_dir)

    assert report["matched_not_yet_used"] == [("10.1234_mixed-case-doi.pdf", study.study_id)]


def test_missing_directory_reports_zero_pdfs_without_raising(tmp_path):
    session = _session(tmp_path)
    report = build_report(session, tmp_path / "does_not_exist")
    assert report["total_pdfs_on_disk"] == 0
    assert report["orphaned_no_study"] == []
