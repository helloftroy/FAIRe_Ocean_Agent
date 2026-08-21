#!/usr/bin/env python3
"""Converts the seed-discovery database's paper_seeds view -- MGnify- AND
ENA-sourced studies alike (the view is a UNION ALL over both, see
db.py's schema) -- into the plain CSV shape discovery/seed_loader.py's
ingest-seeds already consumes -- reuses all the existing seed-ingestion
machinery rather than teaching the main pipeline a second seed format.

Deliberately includes every row, not just ones with a resolved paper: a
row whose paper was never found still has a real, confirmed BioProject
with real deposited sequence data -- worth seeding as a repository-only
study (handle_discover_identifiers already supports a study with no DOI
at all, see workflow/handlers.py) even without a paper's own methods
text to extract from yet.

Usage:
    python scripts/mgnify_seeds_to_csv.py
    python scripts/mgnify_seeds_to_csv.py --db data/seed_discovery/mgnify_paper_seeds.sqlite --out cluster/seeds_mgnify.csv
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


def convert(db_path: Path, out_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM paper_seeds ORDER BY canonical_dataset_id").fetchall()
    conn.close()

    written = 0
    no_doi_repository_only = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        for row in rows:
            doi = row["primary_doi"] or ""
            source = row["seed_source"]  # 'mgnify' or 'ena' -- paper_seeds is a UNION ALL of both
            # secondary_study_accession is SRP/ERP/DRP -- sra_study_accession
            # (not ena_study_accession, whose normalizer only accepts
            # ERP.../PRJEB...) is the seed column that actually normalizes
            # the full family (discovery/seed_loader.py). ENA rows carry a
            # real ena_study_accession of their own (the view's ENA branch
            # reads it straight off ena_studies, unlike the MGnify branch,
            # which never has one) -- routed to that column directly rather
            # than through sra_study_accession's broader-but-less-specific
            # normalizer.
            writer.writerow(
                {
                    "seed_id": f"{source}-{row['mgnify_accession'] or row['canonical_dataset_id']}",
                    "title": row["primary_paper_title"] or row["study_title"] or "",
                    "doi": doi,
                    "pmid": row["primary_pmid"] or "",
                    "pmcid": row["primary_pmcid"] or "",
                    "bioproject_accession": row["bioproject_accession"] or "",
                    "ena_study_accession": (row["ena_study_accession"] or "") if source == "ena" else "",
                    "sra_study_accession": row["secondary_study_accession"] or "",
                    "dataset_id": row["mgnify_accession"] or "",
                    "repository": source,
                    "url": "",
                    "notes": (
                        f"seed_source={source}; seed_status={row['seed_status']}; "
                        f"publication_match_confidence={row['publication_match_confidence'] or 'none'}; "
                        f"publication_match_method={row['publication_match_method'] or 'none'}"
                    ),
                }
            )
            written += 1
            if not doi:
                no_doi_repository_only += 1
    return written, no_doi_repository_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/seed_discovery/mgnify_paper_seeds.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("cluster/seeds_mgnify.csv"))
    args = parser.parse_args()

    written, no_doi = convert(args.db, args.out)
    print(f"Wrote {written} seed rows to {args.out} ({no_doi} with no resolved DOI -- repository-only via BioProject)")


if __name__ == "__main__":
    main()
