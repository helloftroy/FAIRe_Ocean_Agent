#!/usr/bin/env python3
"""Reports how many candidate studies actually have real, reachable paper
full text -- vs. ones that are effectively paywalled/inaccessible right
now and would need a manually-supplied PDF. Read-only, DB-only (no
network calls), so it's fast and safe to run any time.

A Study row existing does NOT mean its paper's full text is readable --
Study rows get created as soon as ANY identifier is found (DOI, BioProject
accession, ...), long before anyone checks whether the actual paper text
is reachable. The ground truth for "we actually got this paper's real
text" is a Source row with source_type=ARTICLE_FULLTEXT
(workflow/handlers.py's handle_extract_text_facts only ever creates one
after successfully obtaining real text, either from Europe PMC's
open-access full text or a manually-supplied local PDF) -- a bibliographic
API's own "fulltext_available" metadata flag is not used here since it
just reflects what Crossref/OpenAlex/Europe PMC claims, not whether this
pipeline actually obtained and used the text.

Four buckets per candidate study:
  confirmed_fulltext       -- has an ARTICLE_FULLTEXT Source row already.
  has_pmcid_not_yet_fetched -- a PMCID exists (a strong, though not
                                guaranteed, signal of open access) but no
                                ARTICLE_FULLTEXT row yet -- text extraction
                                for this study just hasn't reached/finished
                                yet, not necessarily paywalled.
  local_pdf_supplied_not_yet_used -- a local PDF already sits in
                                FAIR_OCEAN_LOCAL_PDF_DIR (or the default
                                data/auto_fetched_pdfs/) named for this
                                study's DOI, but text extraction hasn't
                                picked it up yet.
  no_confirmed_path_to_fulltext -- none of the above: no PMCID, no local
                                PDF on file, no confirmed full text --
                                this is the "likely paywalled, or just
                                needs a manually-supplied PDF" bucket.

Repository-only studies (no DOI at all -- found purely via a BioProject/
BioSample accession) are reported separately: they never need a paper's
own full text for their sample data, so lumping them into "no confirmed
path to fulltext" would overstate how many papers are actually stuck.

--export-csv writes the FULL no_confirmed_path_to_fulltext and
repository_only lists (not just a truncated sample) to CSV files, one row
per study with its title and whatever identifiers it does have (DOI,
PMCID, BioProject accession, PMID) -- enough to actually go search for
each paper by hand.

Usage:
    python scripts/check_fulltext_access.py
    python scripts/check_fulltext_access.py --json
    python scripts/check_fulltext_access.py --export-csv data/paper_classification
    FAIR_OCEAN_DATABASE_URL=sqlite:////path/to/other.db python scripts/check_fulltext_access.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType, SourceType
from fair_ocean_agent.database.models import ExternalIdentifier, Source, Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.workflow.handlers import _local_pdf_path_for_study

_EXPORT_IDENTIFIER_TYPES = (
    IdentifierType.DOI,
    IdentifierType.PMCID,
    IdentifierType.PMID,
    IdentifierType.BIOPROJECT_ACCESSION,
)


def _identifiers_by_study(session: Session, identifier_type: IdentifierType) -> dict[str, str]:
    rows = session.execute(
        select(ExternalIdentifier.study_id, ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.identifier_type == identifier_type.value)
        .order_by(ExternalIdentifier.created_at)
    ).all()
    result: dict[str, str] = {}
    for study_id, value in rows:
        result.setdefault(study_id, value)
    return result


def build_report(session: Session) -> dict:
    studies = session.scalars(
        select(Study).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
    ).all()

    doi_study_ids = {
        row[0]
        for row in session.execute(
            select(ExternalIdentifier.study_id).where(ExternalIdentifier.identifier_type == IdentifierType.DOI.value)
        ).all()
    }
    pmcid_study_ids = {
        row[0]
        for row in session.execute(
            select(ExternalIdentifier.study_id).where(
                ExternalIdentifier.identifier_type == IdentifierType.PMCID.value
            )
        ).all()
    }
    fulltext_study_ids = {
        row[0]
        for row in session.execute(
            select(Source.study_id).where(Source.source_type == SourceType.ARTICLE_FULLTEXT.value)
        ).all()
    }

    repository_only: list[str] = []
    confirmed_fulltext: list[str] = []
    has_pmcid_not_yet_fetched: list[str] = []
    local_pdf_supplied_not_yet_used: list[str] = []
    no_confirmed_path: list[str] = []

    for study in studies:
        if study.study_id in fulltext_study_ids:
            confirmed_fulltext.append(study.study_id)
            continue
        if study.study_id not in doi_study_ids:
            repository_only.append(study.study_id)
            continue
        if study.study_id in pmcid_study_ids:
            has_pmcid_not_yet_fetched.append(study.study_id)
            continue
        if _local_pdf_path_for_study(session, study) is not None:
            local_pdf_supplied_not_yet_used.append(study.study_id)
            continue
        no_confirmed_path.append(study.study_id)

    return {
        "total_candidate_studies": len(studies),
        "confirmed_fulltext": len(confirmed_fulltext),
        "has_pmcid_not_yet_fetched": len(has_pmcid_not_yet_fetched),
        "local_pdf_supplied_not_yet_used": len(local_pdf_supplied_not_yet_used),
        "no_confirmed_path_to_fulltext": len(no_confirmed_path),
        "repository_only_no_doi_no_paper_needed": len(repository_only),
        "no_confirmed_path_study_ids_sample": no_confirmed_path[:25],
        "no_confirmed_path_study_ids": no_confirmed_path,
        "repository_only_study_ids": repository_only,
    }


def _export_csv(session: Session, study_ids: list[str], out_path: Path) -> None:
    titles = {study.study_id: study.title for study in session.scalars(select(Study).where(Study.study_id.in_(study_ids)))}
    identifiers_by_type = {
        identifier_type: _identifiers_by_study(session, identifier_type) for identifier_type in _EXPORT_IDENTIFIER_TYPES
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "title", *(t.value for t in _EXPORT_IDENTIFIER_TYPES)])
        for study_id in study_ids:
            writer.writerow(
                [
                    study_id,
                    titles.get(study_id) or "",
                    *(identifiers_by_type[t].get(study_id, "") for t in _EXPORT_IDENTIFIER_TYPES),
                ]
            )


def render_text(report: dict) -> str:
    lines = [
        f"Total candidate studies: {report['total_candidate_studies']}",
        "",
        f"  Confirmed full text already obtained: {report['confirmed_fulltext']}",
        f"  Has a PMCID, full text not fetched/processed yet: {report['has_pmcid_not_yet_fetched']}",
        f"  Local PDF supplied, not yet used: {report['local_pdf_supplied_not_yet_used']}",
        f"  No confirmed path to full text (likely paywalled, needs a manual PDF): {report['no_confirmed_path_to_fulltext']}",
        f"  Repository-only, no DOI, no paper full text needed: {report['repository_only_no_doi_no_paper_needed']}",
    ]
    if report["no_confirmed_path_study_ids_sample"]:
        lines.append("")
        lines.append("Sample of studies with no confirmed path to full text (first 25):")
        lines.extend(f"  {study_id}" for study_id in report["no_confirmed_path_study_ids_sample"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a text report")
    parser.add_argument(
        "--export-csv",
        type=Path,
        metavar="DIR",
        help="write the full no_confirmed_path_to_fulltext.csv and repository_only.csv lists to this directory",
    )
    args = parser.parse_args()

    with session_scope() as session:
        report = build_report(session)
        if args.export_csv:
            _export_csv(session, report["no_confirmed_path_study_ids"], args.export_csv / "no_confirmed_path_to_fulltext.csv")
            _export_csv(session, report["repository_only_study_ids"], args.export_csv / "repository_only.csv")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
        if args.export_csv:
            print(f"\nWrote {args.export_csv / 'no_confirmed_path_to_fulltext.csv'}")
            print(f"Wrote {args.export_csv / 'repository_only.csv'}")


if __name__ == "__main__":
    main()
