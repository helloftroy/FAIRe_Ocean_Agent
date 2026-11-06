#!/usr/bin/env python3
"""Write a CSV of candidate studies with no local PDF, to fetch PDFs for.

Read-only. Uses the same PDF matching the extraction jobs use (see
workflow/handlers.py's studies_with_local_pdf): the study's OLDEST DOI,
lowercased with "/" replaced by "_", plus ".pdf", in
FAIR_OCEAN_LOCAL_PDF_DIR. So save each fetched PDF under the
`pdf_filename` column's name, or it will not be found.

Rows are sorted so the papers that matter most come first:
`has_pmcid` = no means the LLM has no full text at all for that study
(no PDF and no PMCID); `has_pmcid` = yes ones can already be extracted
from Europe PMC, a PDF only adds a second source. Studies with no DOI
cannot be matched to a PDF by filename and are listed last.

Usage:
    FAIR_OCEAN_LOCAL_PDF_DIR=$(pwd)/data/local_pdfs \\
        python scripts/export_missing_pdfs.py --output data/missing_pdfs.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType
from fair_ocean_agent.database.models import ExternalIdentifier, Source, Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.workflow.handlers import _doi_pdf_filename, _pdf_lookup_dir, studies_with_local_pdf

COLUMNS = ("study_id", "has_pmcid", "doi", "pdf_filename", "other_dois", "pmcid", "pmid", "year", "journal", "title")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/missing_pdfs.csv"))
    args = parser.parse_args()

    with session_scope() as session:
        studies = {
            study_id: title
            for study_id, title in session.execute(
                select(Study.study_id, Study.title).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
            )
        }
        with_pdf = studies_with_local_pdf(session)

        ids: dict[str, dict[str, list[str]]] = {}
        for study_id, id_type, value in session.execute(
            select(ExternalIdentifier.study_id, ExternalIdentifier.identifier_type, ExternalIdentifier.identifier_value)
            .where(
                ExternalIdentifier.identifier_type.in_(
                    (IdentifierType.DOI.value, IdentifierType.PMCID.value, IdentifierType.PMID.value)
                )
            )
            .order_by(ExternalIdentifier.created_at)
        ):
            values = ids.setdefault(study_id, {}).setdefault(id_type, [])
            if value not in values:
                values.append(value)

        year_journal: dict[str, tuple] = {}
        for study_id, year, journal in session.execute(
            select(Source.study_id, Source.publication_year, Source.journal).where(
                (Source.publication_year.is_not(None)) | (Source.journal.is_not(None))
            )
        ):
            year_journal.setdefault(study_id, (year, journal))

    rows = []
    for study_id, title in studies.items():
        if study_id in with_pdf:
            continue
        study_ids = ids.get(study_id, {})
        dois = study_ids.get(IdentifierType.DOI.value, [])
        year, journal = year_journal.get(study_id, (None, None))
        rows.append(
            {
                "study_id": study_id,
                "has_pmcid": "yes" if study_ids.get(IdentifierType.PMCID.value) else "no",
                "doi": dois[0] if dois else "",
                "pdf_filename": _doi_pdf_filename(dois[0]) if dois else "",
                "other_dois": " | ".join(dois[1:]),
                "pmcid": " | ".join(study_ids.get(IdentifierType.PMCID.value, [])),
                "pmid": " | ".join(study_ids.get(IdentifierType.PMID.value, [])),
                "year": year or "",
                "journal": journal or "",
                "title": (title or "").replace("\n", " "),
            }
        )
    rows.sort(key=lambda r: (r["doi"] == "", r["has_pmcid"] == "yes", r["study_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    no_text = sum(r["has_pmcid"] == "no" for r in rows)
    no_doi = sum(r["doi"] == "" for r in rows)
    print(f"PDF directory checked: {_pdf_lookup_dir()}")
    print(f"Wrote {len(rows)} studies with no local PDF to {args.output}")
    print(f"  no PDF and no PMCID (no full text at all): {no_text}")
    print(f"  have a PMCID (already extractable):        {len(rows) - no_text}")
    print(f"  no DOI (cannot be matched to a PDF):       {no_doi}")


if __name__ == "__main__":
    main()
