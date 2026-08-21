#!/usr/bin/env python3
"""Exports only "HPC-quality" studies from the local database -- confirmed
real sample entities AND a readable full-text source (PMCID or a local/
auto-fetched PDF) -- as a fresh seed CSV, ready to git-commit and
`ingest-seeds` on the cluster. Reuses classify_papers.py's own sample/
PDF-availability logic directly, so "HPC-quality" always means the exact
same thing in both places.

Per an explicit user request: run the CPU-only discovery/classification/
auto-fetch work locally first (cheap, no GPU needed), and only ever hand
the cluster's GPU/LLM extraction stage studies already confirmed worth
the compute -- not the ones still needing a manual PDF or missing sample
data entirely.

Usage:
    python scripts/export_hpc_ready_seeds.py
    python scripts/export_hpc_ready_seeds.py --out cluster/seeds_hpc_ready.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, Study
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.discovery.seed_loader import SEED_COLUMNS
from fair_ocean_agent.workflow.handlers import _local_pdf_path_for_study

DEFAULT_OUT = REPO_ROOT / "cluster" / "seeds_hpc_ready.csv"


def _has_pmcid_or_pdf(session, study: Study) -> bool:
    if _local_pdf_path_for_study(session, study) is not None:
        return True
    return (
        session.scalars(
            select(ExternalIdentifier.identifier_value)
            .where(ExternalIdentifier.study_id == study.study_id)
            .where(ExternalIdentifier.identifier_type == IdentifierType.PMCID.value)
        ).first()
        is not None
    )


def _identifier(session, study_id: str, identifier_type: IdentifierType) -> str:
    return (
        session.scalars(
            select(ExternalIdentifier.identifier_value)
            .where(ExternalIdentifier.study_id == study_id)
            .where(ExternalIdentifier.identifier_type == identifier_type.value)
        ).first()
        or ""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[dict] = []
    skipped_no_samples = 0
    skipped_no_source = 0
    with session_scope() as session:
        studies = session.scalars(
            select(Study).where(Study.canonical_status == CanonicalStatus.CANDIDATE.value)
        ).all()
        for study in studies:
            has_samples = (
                session.scalars(
                    select(Entity.entity_id)
                    .where(Entity.study_id == study.study_id)
                    .where(Entity.entity_level == "sample")
                ).first()
                is not None
            )
            if not has_samples:
                skipped_no_samples += 1
                continue
            if not _has_pmcid_or_pdf(session, study):
                skipped_no_source += 1
                continue

            rows.append(
                {
                    "seed_id": study.study_id,
                    "title": study.title or "",
                    "doi": _identifier(session, study.study_id, IdentifierType.DOI),
                    "pmid": _identifier(session, study.study_id, IdentifierType.PMID),
                    "pmcid": _identifier(session, study.study_id, IdentifierType.PMCID),
                    "bioproject_accession": _identifier(session, study.study_id, IdentifierType.BIOPROJECT_ACCESSION),
                    "ena_study_accession": _identifier(session, study.study_id, IdentifierType.ENA_STUDY_ACCESSION),
                    "sra_study_accession": _identifier(session, study.study_id, IdentifierType.SRA_STUDY_ACCESSION),
                    "dataset_id": "",
                    "repository": "",
                    "url": "",
                    "notes": "hpc_ready_export: confirmed real samples + PMCID/local PDF",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"HPC-ready: {len(rows)} studies -> {args.out}")
    print(f"Skipped (no confirmed samples): {skipped_no_samples}")
    print(f"Skipped (samples but no PMCID/PDF yet): {skipped_no_source}")


if __name__ == "__main__":
    main()
