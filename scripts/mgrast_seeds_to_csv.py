#!/usr/bin/env python3
"""Export MG-RAST paper_seeds rows to the standard ingest-seeds CSV shape.

MG-RAST-only projects are valid seeds: a BioProject is not required when
MG-RAST public sequence files and a paper identity are present.
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


def convert(db_path: Path, out_path: Path, *, include_title_only: bool = True) -> tuple[int, int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM paper_seeds ORDER BY source_study_id").fetchall()
    conn.close()

    written = mgrast_only = title_only = 0
    seen: set[tuple[str, str]] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        for row in rows:
            doi = row["primary_doi"] or ""
            paper_title = row["primary_paper_title"] or ""
            title = paper_title or row["study_title"] or ""
            if not doi and not include_title_only:
                continue
            if not doi and not row["primary_pmid"] and not paper_title:
                continue
            project_id = row["native_project_accession"] or row["source_project_id"] or row["source_study_id"]
            key = (doi, project_id)
            if key in seen:
                continue
            seen.add(key)
            if not row["bioproject_accession"]:
                mgrast_only += 1
            if paper_title and not doi:
                title_only += 1
            writer.writerow(
                {
                    "seed_id": f"mgrast-{project_id}",
                    "title": title,
                    "doi": doi,
                    "pmid": row["primary_pmid"] or "",
                    "pmcid": "",
                    "bioproject_accession": row["bioproject_accession"] or "",
                    "ena_study_accession": "",
                    "sra_study_accession": "",
                    "dataset_id": project_id,
                    "repository": "mgrast",
                    "url": f"https://mg-rast.org/linkin.cgi?project={project_id}",
                    "notes": (
                        f"seed_source=mgrast; native_project_accession={project_id}; "
                        f"sequence_accessibility_status={row['sequence_accessibility_status'] or 'none'}; "
                        f"marine_confidence={row['marine_confidence'] or 'none'}; "
                        f"overlap_status={row['overlap_status'] or 'none'}; "
                        f"publication_resolution_status={row['publication_resolution_status'] or 'none'}"
                    ),
                }
            )
            written += 1
    return written, mgrast_only, title_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/seed_discovery/mgrast_paper_seeds.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("cluster/seeds_mgrast.csv"))
    parser.add_argument("--doi-only", action="store_true", help="Only export rows with a resolved primary DOI.")
    args = parser.parse_args()
    written, mgrast_only, title_only = convert(args.db, args.out, include_title_only=not args.doi_only)
    print(f"Wrote {written} MG-RAST seed rows to {args.out}")
    print(f"  MG-RAST-only/no BioProject seeds: {mgrast_only}")
    print(f"  paper-title seeds with DOI missing: {title_only}")


if __name__ == "__main__":
    main()
