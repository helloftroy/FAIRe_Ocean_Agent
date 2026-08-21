#!/usr/bin/env python3
"""Buckets already-discovered studies into three review lists so manual
effort (removing dead papers, chasing ambiguous ones, downloading PDFs)
goes to the right place instead of re-checking the same studies over and
over. Read-only against whatever FAIR_OCEAN_DATABASE_URL points at.

Deliberately does NOT trust Study.data_availability_status: that field is
only ever set at the end of a DISCOVER_IDENTIFIERS run, so a study that was
seeded/discovered before that feature existed (or before its most recent
code change) sits at "unknown" forever until someone re-runs full
rediscovery for it -- which is itself slow/network-bound and exactly the
kind of busywork this script exists to let you skip. Instead it recomputes
the same "does this study actually have real sample data" signal directly
from entities, and -- for studies that don't -- goes looking for a Data
Availability paragraph in whatever full text is already cached (or a local/
auto-fetched PDF), so the paragraph can be quoted for you to judge instead
of guessing.

Three CSVs, written to --out-dir (default: data/paper_classification/):

  no_sample_data.csv           -- no sample entities, and either no Data
                                   Availability statement was found at all,
                                   or the one that was found doesn't mention
                                   anything sequencing-related. Candidates
                                   for removal from seeds + the exclude list
                                   (see --write-exclude-list).
  ambiguous_data_availability.csv
                                -- no sample entities, but either (a) a Data
                                   Availability statement mentions
                                   sequencing/accession-like language that
                                   the pipeline still couldn't resolve into
                                   real facts, or (b) full text was never
                                   reachable at all so nothing could be
                                   checked. The actual quoted text (or a
                                   note explaining why there's no quote) is
                                   included so you don't have to re-open the
                                   paper to see what it says.
  needs_manual_pdf.csv         -- has real sample entities (confirmed good),
                                   but the pipeline has no PMCID and no
                                   local/auto-fetched PDF on file for it --
                                   meaning whatever found the sample data
                                   didn't come from reading this paper's own
                                   full text (e.g. a BioProject citation
                                   link), so other FAIRe fields normally
                                   pulled from the paper itself are still
                                   unfilled. These need a manually-supplied
                                   PDF (see cluster/README.md's
                                   FAIR_OCEAN_LOCAL_PDF_DIR section).

Usage:
    python scripts/classify_papers.py
    python scripts/classify_papers.py --write-exclude-list
    FAIR_OCEAN_DATABASE_URL=sqlite:////path/to/other.db python scripts/classify_papers.py
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.discovery.exclusions import append_excluded_doi
from fair_ocean_agent.discovery.text_identifiers import xml_to_text
from fair_ocean_agent.sources.base import SourceRecordNotFoundError
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.workflow.handlers import _build_enabled_adapters, _local_pdf_path_for_study

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "paper_classification"

_DA_HEADING_RE = re.compile(
    r"data\s+availability(?:\s+statement)?|availability\s+of\s+data(?:\s+and\s+materials?)?",
    re.IGNORECASE,
)
_NEXT_HEADING_RE = re.compile(
    r"\b(funding|acknowledg(?:e)?ments?|author\s+contributions?|conflicts?\s+of\s+interest|"
    r"competing\s+interests?|references|supplementary\s+material|ethics\s+approval|declarations?)\b",
    re.IGNORECASE,
)
# Deliberately broad -- false positives just land a study in the "ambiguous,
# needs a human look" bucket instead of "remove", which is the safe
# direction to be wrong in; false negatives would wrongly recommend
# deleting a study that actually has real sequence data cited.
_SEQUENCE_SIGNAL_RE = re.compile(
    r"\b(SRA|SRR\d|SRS\d|SRP\d|ERR\d|ERS\d|ERP\d|DRR\d|DRS\d|DRP\d|PRJNA|PRJEB|PRJDB|"
    r"BioProject|BioSample|GenBank|zenodo|dryad|figshare|osf\.io|DDBJ|NCBI|ENA|"
    r"deposited|accession|sequence\s+read\s+archive)\b",
    re.IGNORECASE,
)
MAX_QUOTE_CHARS = 500
MAX_SEARCH_WINDOW = 2500


def _extract_data_availability_quote(text: str) -> str | None:
    match = _DA_HEADING_RE.search(text)
    if match is None:
        return None
    window = text[match.end():match.end() + MAX_SEARCH_WINDOW]
    cutoff = _NEXT_HEADING_RE.search(window)
    quote = (window[: cutoff.start()] if cutoff else window).strip(" .:\n\t")
    if not quote:
        return None
    return quote[:MAX_QUOTE_CHARS]


def _fetch_cached_fulltext(session, study: Study, adapters: dict) -> tuple[str | None, bool]:
    """Returns (text, was_reachable). was_reachable is False only when
    there was genuinely no way to get full text at all (no PMCID, no
    local/auto-fetched PDF) -- distinct from a PMCID that resolves but
    Europe PMC has no open-access full text for it, which still counts as
    "we checked and there's nothing"."""
    pdf_path = _local_pdf_path_for_study(session, study)
    if pdf_path is not None:
        from fair_ocean_agent.extraction.pdf import extract_pdf_text

        return extract_pdf_text(pdf_path), True

    pmcid = session.scalars(
        select(ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.study_id == study.study_id)
        .where(ExternalIdentifier.identifier_type == IdentifierType.PMCID.value)
    ).first()
    if pmcid is None:
        return None, False

    europe_pmc = adapters.get("europe_pmc")
    if not isinstance(europe_pmc, EuropePmcAdapter):
        return None, False
    try:
        xml = europe_pmc.fetch_fulltext_xml(pmcid)
    except SourceRecordNotFoundError:
        return None, True
    return xml_to_text(xml), True


def _doi(session, study_id: str) -> str | None:
    return session.scalars(
        select(ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.study_id == study_id)
        .where(ExternalIdentifier.identifier_type == IdentifierType.DOI.value)
    ).first()


def _has_pmcid_or_pdf(session, study: Study) -> bool:
    if _local_pdf_path_for_study(session, study) is not None:
        return True
    pmcid = session.scalars(
        select(ExternalIdentifier.identifier_value)
        .where(ExternalIdentifier.study_id == study.study_id)
        .where(ExternalIdentifier.identifier_type == IdentifierType.PMCID.value)
    ).first()
    return pmcid is not None


def classify(session, adapters: dict) -> tuple[list[dict], list[dict], list[dict]]:
    no_sample_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    needs_pdf_rows: list[dict] = []

    studies = session.scalars(
        select(Study).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
    ).all()

    for study in studies:
        doi = _doi(session, study.study_id)
        n_samples = session.scalars(
            select(Entity.entity_id)
            .where(Entity.study_id == study.study_id)
            .where(Entity.entity_level == "sample")
        ).all()
        sample_count = len(n_samples)

        if sample_count > 0:
            if not _has_pmcid_or_pdf(session, study):
                needs_pdf_rows.append(
                    {
                        "study_id": study.study_id,
                        "doi": doi or "",
                        "title": study.title or "",
                        "sample_count": sample_count,
                        "reason": "has confirmed sample data but no PMCID and no local/auto-fetched PDF on file",
                    }
                )
            continue

        text, reachable = _fetch_cached_fulltext(session, study, adapters)
        if not reachable:
            ambiguous_rows.append(
                {
                    "study_id": study.study_id,
                    "doi": doi or "",
                    "title": study.title or "",
                    "data_availability_quote": "",
                    "note": "full text never reachable (no PMCID, no local/auto-fetched PDF) -- cannot determine",
                }
            )
            continue

        quote = _extract_data_availability_quote(text) if text else None
        if quote and _SEQUENCE_SIGNAL_RE.search(quote):
            ambiguous_rows.append(
                {
                    "study_id": study.study_id,
                    "doi": doi or "",
                    "title": study.title or "",
                    "data_availability_quote": quote,
                    "note": "mentions sequencing/repository language but the pipeline resolved no samples",
                }
            )
        elif quote:
            no_sample_rows.append(
                {
                    "study_id": study.study_id,
                    "doi": doi or "",
                    "title": study.title or "",
                    "data_availability_quote": quote,
                    "reason": "has a Data Availability statement but it doesn't mention sequence data",
                }
            )
        else:
            no_sample_rows.append(
                {
                    "study_id": study.study_id,
                    "doi": doi or "",
                    "title": study.title or "",
                    "data_availability_quote": "",
                    "reason": "no Data Availability statement found in full text",
                }
            )

    return no_sample_rows, ambiguous_rows, needs_pdf_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--write-exclude-list",
        action="store_true",
        help="Append every no_sample_data.csv DOI to cluster/excluded_dois.csv "
        "so future seed ingestion skips them permanently. Does not touch "
        "studies already in the database -- see the script's own docstring.",
    )
    args = parser.parse_args()

    adapters = _build_enabled_adapters()
    with session_scope() as session:
        no_sample_rows, ambiguous_rows, needs_pdf_rows = classify(session, adapters)

    _write_csv(
        args.out_dir / "no_sample_data.csv",
        no_sample_rows,
        ["study_id", "doi", "title", "data_availability_quote", "reason"],
    )
    _write_csv(
        args.out_dir / "ambiguous_data_availability.csv",
        ambiguous_rows,
        ["study_id", "doi", "title", "data_availability_quote", "note"],
    )
    _write_csv(
        args.out_dir / "needs_manual_pdf.csv",
        needs_pdf_rows,
        ["study_id", "doi", "title", "sample_count", "reason"],
    )

    print(f"no_sample_data:            {len(no_sample_rows):4d} -> {args.out_dir / 'no_sample_data.csv'}")
    print(f"ambiguous_data_availability: {len(ambiguous_rows):3d} -> {args.out_dir / 'ambiguous_data_availability.csv'}")
    print(f"needs_manual_pdf:           {len(needs_pdf_rows):4d} -> {args.out_dir / 'needs_manual_pdf.csv'}")

    if args.write_exclude_list:
        added = 0
        for row in no_sample_rows:
            if row["doi"]:
                if append_excluded_doi(row["doi"], reason=row["reason"]):
                    added += 1
        print(f"Added {added} new DOI(s) to cluster/excluded_dois.csv")


if __name__ == "__main__":
    main()
