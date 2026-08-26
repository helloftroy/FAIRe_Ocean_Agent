#!/usr/bin/env python3
"""Converts resolved GOLD BioProjects into the plain CSV shape
discovery/seed_loader.py's ingest-seeds already consumes -- same idea as
scripts/seed_discovery_to_csv.py for MGnify/ENA, reusing the existing
seed-ingestion machinery rather than teaching the main pipeline a third
seed format.

Deliberately resolved-only by default (a gold_studies row with a real
primary_doi from resolve_gold_primary_publications/
GoldBioprojectPublicationSearchRunner) -- unlike MGnify/ENA's own
exporter, which seeds every discovered study including repository-only
ones. GOLD's own marine-relevance signal is far looser than MGnify/ENA's
(most of GOLD's 63,803 studies sit at marine_confidence=low, which is
closer to "unclassified" than "confirmed marine" -- see
resolve_gold_primary_publications's own docstring for the fuller
picture), so dumping every GOLD BioProject in unfiltered would flood the
pipeline's candidate pool with mostly-irrelevant studies. Requiring a
resolved primary_doi is a real, independent quality bar on its own (it
took a genuine, low-fanout source paper to get there) -- pass
--include-unresolved to seed every BioProject regardless, if that's
ever wanted.

One CSV row per distinct real NCBI BioProject accession (not per
gold_study -- a study can own several BioProjects, each seeded
separately, all sharing that study's same resolved paper, matching how
resolve_gold_primary_publications itself treats a study-level paper as
covering every one of its projects). In the rare case the same
BioProject accession is reachable from more than one gold_study row, a
non-ambiguous ('resolved') resolution wins over an ambiguous tie,
broken deterministically by BioProject accession for reproducibility.

Usage:
    python scripts/gold_seeds_to_csv.py
    python scripts/gold_seeds_to_csv.py --db data/jgi_gold/gold_sharded.sqlite --out cluster/seeds_gold.csv
    python scripts/gold_seeds_to_csv.py --include-unresolved
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

_STATUS_RANK = {"resolved": 0, "resolved_ambiguous_primary": 1}


def convert(db_path: Path, out_path: Path, *, include_unresolved: bool = False) -> tuple[int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT sp.ncbi_bioproject_accession AS bioproject_accession,
               s.gold_study_id, s.study_name, s.primary_doi, s.primary_doi_status,
               s.primary_doi_bioproject_fanout
        FROM gold_sequencing_projects sp
        JOIN gold_studies s ON s.gold_study_id = sp.gold_study_id
        WHERE sp.ncbi_bioproject_accession IS NOT NULL AND sp.ncbi_bioproject_accession != ''
        """
    ).fetchall()
    conn.close()

    by_bioproject: dict[str, sqlite3.Row] = {}
    for row in rows:
        if not include_unresolved and not row["primary_doi"]:
            continue
        accession = row["bioproject_accession"]
        existing = by_bioproject.get(accession)
        if existing is None:
            by_bioproject[accession] = row
            continue
        # Same BioProject reachable from more than one gold_study row (rare) --
        # prefer a clean 'resolved' pick over an ambiguous tie; break further
        # ties by gold_study_id for a reproducible, deterministic choice.
        existing_rank = _STATUS_RANK.get(existing["primary_doi_status"], 2)
        new_rank = _STATUS_RANK.get(row["primary_doi_status"], 2)
        if (new_rank, row["gold_study_id"]) < (existing_rank, existing["gold_study_id"]):
            by_bioproject[accession] = row

    written = 0
    no_doi_repository_only = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        for accession in sorted(by_bioproject):
            row = by_bioproject[accession]
            doi = row["primary_doi"] or ""
            writer.writerow(
                {
                    "seed_id": f"gold-{accession}",
                    "title": row["study_name"] or "",
                    "doi": doi,
                    "pmid": "",
                    "pmcid": "",
                    "bioproject_accession": accession,
                    "ena_study_accession": "",
                    "sra_study_accession": "",
                    "dataset_id": row["gold_study_id"] or "",
                    "repository": "gold",
                    "url": "",
                    "notes": (
                        f"gold_study_id={row['gold_study_id']}; "
                        f"primary_doi_status={row['primary_doi_status'] or 'none'}; "
                        f"primary_doi_bioproject_fanout={row['primary_doi_bioproject_fanout']}"
                    ),
                }
            )
            written += 1
            if not doi:
                no_doi_repository_only += 1
    return written, no_doi_repository_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/jgi_gold/gold_sharded.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("cluster/seeds_gold.csv"))
    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Also seed BioProjects with no resolved primary_doi (repository-only, GOLD's own marine "
        "filter is much looser than MGnify/ENA's -- see this script's own docstring before using this).",
    )
    args = parser.parse_args()

    written, no_doi = convert(args.db, args.out, include_unresolved=args.include_unresolved)
    print(f"Wrote {written} seed rows to {args.out} ({no_doi} with no resolved DOI -- repository-only via BioProject)")


if __name__ == "__main__":
    main()
