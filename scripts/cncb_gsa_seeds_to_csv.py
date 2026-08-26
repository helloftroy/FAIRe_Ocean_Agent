#!/usr/bin/env python3
"""Export CNCB/GSA paper_seeds rows to the standard ingest-seeds CSV shape.

The rich CNCB sample/experiment metadata stays in the standalone SQLite DB;
this CSV is only a paper/repository seed list for the main FAIRe workflow.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

SEED_COLUMNS = (
    "seed_id",
    "title",
    "doi",
    "pmid",
    "pmcid",
    "bioproject_accession",
    "ena_study_accession",
    "sra_study_accession",
    "dataset_id",
    "repository",
    "url",
    "notes",
)


def convert(db_path: Path, out_path: Path, *, include_unresolved: bool = True) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM paper_seeds ORDER BY source_study_id").fetchall()
    conn.close()

    seen: set[tuple[str, str, str]] = set()
    written = 0
    no_doi_repository_only = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        for row in rows:
            doi = row["primary_doi"] or ""
            if not include_unresolved and not doi:
                continue
            native_project = row["native_project_accession"] or ""
            source_study = row["source_study_id"] or native_project
            key = (doi, native_project, source_study)
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(
                {
                    "seed_id": f"cncb_gsa-{source_study}",
                    "title": row["study_title"] or "",
                    "doi": doi,
                    "pmid": row["primary_pmid"] or "",
                    "pmcid": "",
                    "bioproject_accession": row["bioproject_accession"] or "",
                    "ena_study_accession": "",
                    "sra_study_accession": "",
                    "dataset_id": source_study or native_project,
                    "repository": "cncb_gsa",
                    "url": f"https://ngdc.cncb.ac.cn/gsa/browse/{source_study}" if source_study.startswith("CRA") else "",
                    "notes": (
                        f"seed_source=cncb_gsa; native_project_accession={native_project}; "
                        f"sequence_accessibility_status={row['sequence_accessibility_status'] or 'none'}; "
                        f"marine_confidence={row['marine_confidence'] or 'none'}; "
                        f"overlap_status={row['overlap_status'] or 'none'}; "
                        f"publication_resolution_status={row['publication_resolution_status'] or 'none'}"
                    ),
                }
            )
            written += 1
            if not doi:
                no_doi_repository_only += 1
    return written, no_doi_repository_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/seed_discovery/cncb_gsa_paper_seeds.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("cluster/seeds_cncb_gsa.csv"))
    parser.add_argument("--resolved-only", action="store_true", help="Only export rows with a resolved primary DOI.")
    args = parser.parse_args()
    written, no_doi = convert(args.db, args.out, include_unresolved=not args.resolved_only)
    print(f"Wrote {written} CNCB/GSA seed rows to {args.out} ({no_doi} with no DOI, repository-only/native CNCB seeds)")


if __name__ == "__main__":
    main()
