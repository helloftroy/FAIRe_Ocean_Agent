#!/usr/bin/env python3
"""Answers a real, concrete question a pile of newly-uploaded PDFs raises:
which of them actually correspond to a study this pipeline already knows
about, and which don't?

A PDF sitting in FAIR_OCEAN_LOCAL_PDF_DIR is invisible to the pipeline
unless a Study row with the matching DOI already exists --
_local_pdf_path_for_study (workflow/handlers.py) looks up a study's own
DOI and checks for a file named after it; it never scans the directory the
other way around. A study only gets created by ingest-seeds (from a seed
CSV) or by discovery finding it via some other paper's citation -- so a
freshly-uploaded PDF for a paper nobody has seeded yet will just sit there
doing nothing, indefinitely, with no error or warning anywhere, until a
matching Study row shows up some other way.

This is read-only and DB-only (no network calls, no PDF text extraction):
1. matched_and_confirmed   -- study exists, PDF is on file, extraction
                               already produced an article_fulltext Source
                               row for it. Nothing to do.
2. matched_not_yet_used    -- study exists, PDF is on file, but discovery/
                               extraction hasn't picked it up yet. Just
                               needs run_discovery.sbatch / extraction to
                               run (or re-run) with FAIR_OCEAN_LOCAL_PDF_DIR
                               set to this directory.
3. orphaned_no_study       -- a PDF whose filename implies a DOI (or
                               doesn't) that has NO matching study in the
                               database at all. These need ingest-seeds
                               run first (a seed CSV with at least their
                               DOI) before anything else can ever see them.

Usage:
    python scripts/check_local_pdfs_against_studies.py
    python scripts/check_local_pdfs_against_studies.py --dir data/local_pdfs
    python scripts/check_local_pdfs_against_studies.py --export-csv data/paper_classification
    FAIR_OCEAN_DATABASE_URL=sqlite:////path/to/other.db python scripts/check_local_pdfs_against_studies.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import IdentifierType, SourceType
from fair_ocean_agent.database.models import ExternalIdentifier, Source
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.workflow.handlers import _doi_pdf_filename, _pdf_lookup_dir

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "paper_classification"


def build_report(session, pdf_dir: Path) -> dict:
    pdf_files = {p.name.lower(): p for p in pdf_dir.glob("*.pdf")} if pdf_dir.is_dir() else {}

    doi_rows = session.execute(
        select(ExternalIdentifier.study_id, ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.identifier_type == IdentifierType.DOI.value)
    ).all()
    # A study can (rarely) carry more than one recorded DOI variant -- every
    # one of them is a legitimate filename this PDF could have been saved
    # under, so every one is checked, not just the first.
    expected_filename_to_study: dict[str, str] = {}
    for study_id, doi in doi_rows:
        expected_filename_to_study[_doi_pdf_filename(doi).lower()] = study_id

    studies_with_fulltext = set(
        session.scalars(
            select(Source.study_id).where(Source.source_type == SourceType.ARTICLE_FULLTEXT.value)
        ).all()
    )

    matched_and_confirmed: list[tuple[str, str]] = []
    matched_not_yet_used: list[tuple[str, str]] = []
    orphaned_no_study: list[str] = []

    for filename, path in sorted(pdf_files.items()):
        study_id = expected_filename_to_study.get(filename)
        if study_id is None:
            orphaned_no_study.append(path.name)
        elif study_id in studies_with_fulltext:
            matched_and_confirmed.append((path.name, study_id))
        else:
            matched_not_yet_used.append((path.name, study_id))

    return {
        "pdf_dir": str(pdf_dir),
        "total_pdfs_on_disk": len(pdf_files),
        "matched_and_confirmed": matched_and_confirmed,
        "matched_not_yet_used": matched_not_yet_used,
        "orphaned_no_study": orphaned_no_study,
    }


def _write_csv(rows: list[tuple], path: Path, header: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="directory of DOI-named PDFs to check (default: FAIR_OCEAN_LOCAL_PDF_DIR, or data/auto_fetched_pdfs/ if unset -- same resolution handlers.py itself uses)",
    )
    parser.add_argument("--export-csv", type=Path, default=None, help="directory to write orphaned_no_study.csv into")
    args = parser.parse_args()

    pdf_dir = args.dir or _pdf_lookup_dir()

    with session_scope() as session:
        report = build_report(session, pdf_dir)

    print(f"PDF directory: {report['pdf_dir']}")
    print(f"Total PDFs on disk: {report['total_pdfs_on_disk']}")
    print(f"  matched, full text already extracted: {len(report['matched_and_confirmed'])}")
    print(f"  matched, waiting on discovery/extraction: {len(report['matched_not_yet_used'])}")
    print(f"  orphaned -- no matching study in the database: {len(report['orphaned_no_study'])}")

    if report["orphaned_no_study"]:
        print("\nOrphaned filenames (first 20) -- these need ingest-seeds run for them first:")
        for name in report["orphaned_no_study"][:20]:
            print(f"  {name}")
        if len(report["orphaned_no_study"]) > 20:
            print(f"  ... and {len(report['orphaned_no_study']) - 20} more")

    if args.export_csv:
        out_path = args.export_csv / "orphaned_local_pdfs.csv"
        _write_csv([(name,) for name in report["orphaned_no_study"]], out_path, header=("filename",))
        print(f"\nWrote {len(report['orphaned_no_study'])} orphaned filename(s) to {out_path}")

        pending_path = args.export_csv / "local_pdfs_pending_extraction.csv"
        _write_csv(report["matched_not_yet_used"], pending_path, header=("filename", "study_id"))
        print(f"Wrote {len(report['matched_not_yet_used'])} pending filename(s) to {pending_path}")


if __name__ == "__main__":
    main()
