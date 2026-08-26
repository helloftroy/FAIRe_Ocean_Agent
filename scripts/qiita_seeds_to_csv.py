#!/usr/bin/env python3
"""Converts discovered Qiita studies (data/seed_discovery/qiita_paper_seeds.sqlite)
into the plain CSV shape discovery/seed_loader.py's ingest-seeds already
consumes -- same idea as scripts/seed_discovery_to_csv.py (MGnify/ENA) and
scripts/gold_seeds_to_csv.py (GOLD), the missing piece that actually
connects qiita_discovery.py's own database to the main pipeline. Before
this script, nothing read qiita_discovery.py's own `paper_seeds` view at
all -- it sat there, disconnected.

Deliberately does NOT pre-filter on qiita_studies.overlaps_mgnify (which
also can't see ENA overlap at all -- it only ever compares against
mgnify_studies). ingest-seeds' own find_existing_study_by_any_identifier
already does real, identifier-based deduplication against every already-
ingested study regardless of source -- a Qiita study sharing a real
BioProject/DOI with an already-known ENA or MGnify study collapses into
the same Study record automatically at ingest time. Re-implementing that
check here would just be a second, less complete copy of it (it can't
see ENA at all) -- ingest order among Qiita/ENA/MGnify/GOLD doesn't
matter for correctness, only for which one happens to create the Study
row first.

Three cases per study, in priority order:
  1. Has a real BioProject accession (found in the study's own page
     text) -- seeded as bioproject_accession, resolved via the same
     NCBI BioProject/BioSample path as any other bioproject-sourced
     study. No new adapter needed.
  2. No BioProject, but a real ENA-native study accession (ERP.../
     PRJEB...) -- seeded as ena_study_accession, same existing ENA path.
  3. No BioProject, no ENA accession -- the case worth flagging: this is
     either just insufficiently discovered, OR genuinely new sequence
     data that was only ever deposited at Qiita and never mirrored to
     ENA/SRA at all. Seeded as a repository-only "qiita" dataset
     (dataset_id/repository/url) so the study and its DOI aren't lost.
     sources/qiita.py (added after this script was first written -- if
     you're reading an old copy of this docstring, it lied) now resolves
     this case too: one real SAMPLE + EXPERIMENT_RUN per actual Qiita
     sample name, marked as real data available for download at Qiita's
     own study page. Deliberately light (see that adapter's own
     docstring) -- it doesn't verify individual files or resolve a real
     BioSample/run accession the way the ENA/BioProject path does, but
     "how many samples, and is there real data" is answered.

Usage:
    python scripts/qiita_seeds_to_csv.py
    python scripts/qiita_seeds_to_csv.py --db data/seed_discovery/qiita_paper_seeds.sqlite --out cluster/seeds_qiita.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

_ENA_STUDY_ACCESSION_RE = re.compile(r"^(ERP\d+|PRJEB\d+)$")

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


def _first_valid_ena_accession(ena_study_accessions_json: str | None) -> str | None:
    for value in json.loads(ena_study_accessions_json or "[]"):
        if _ENA_STUDY_ACCESSION_RE.match(value):
            return value
    return None


def convert(db_path: Path, out_path: Path) -> tuple[int, int, int, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM qiita_studies WHERE marine_confidence IN ('high', 'medium') ORDER BY qiita_study_id"
    ).fetchall()
    conn.close()

    written = bioproject_seeded = ena_seeded = qiita_only_seeded = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        for row in rows:
            pmids = json.loads(row["pmids_json"] or "[]")
            doi = row["primary_doi"] or ""
            ena_accession = _first_valid_ena_accession(row["ena_study_accessions_json"])

            dataset_id = ""
            repository = ""
            url = ""
            if row["primary_bioproject"]:
                bioproject_accession = row["primary_bioproject"]
                ena_accession = ""
                bioproject_seeded += 1
            elif ena_accession:
                bioproject_accession = ""
                ena_seeded += 1
            else:
                bioproject_accession = ""
                ena_accession = ""
                dataset_id = row["qiita_study_id"]
                repository = "qiita"
                url = f"https://qiita.ucsd.edu/study/description/{row['qiita_study_id']}"
                qiita_only_seeded += 1

            writer.writerow(
                {
                    "seed_id": f"qiita-{row['qiita_study_id']}",
                    "title": row["title"] or "",
                    "doi": doi,
                    "pmid": pmids[0] if pmids else "",
                    "pmcid": "",
                    "bioproject_accession": bioproject_accession,
                    "ena_study_accession": ena_accession or "",
                    "sra_study_accession": "",
                    "dataset_id": dataset_id,
                    "repository": repository,
                    "url": url,
                    "notes": (
                        f"seed_source=qiita; marine_confidence={row['marine_confidence']}; "
                        f"overlaps_mgnify={bool(row['overlaps_mgnify'])}; "
                        f"accession_resolution_status={row['accession_resolution_status']}; "
                        f"sequence_accessibility_status={row['sequence_accessibility_status']}"
                    ),
                }
            )
            written += 1
    return written, bioproject_seeded, ena_seeded, qiita_only_seeded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/seed_discovery/qiita_paper_seeds.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("cluster/seeds_qiita.csv"))
    args = parser.parse_args()

    written, bioproject_seeded, ena_seeded, qiita_only_seeded = convert(args.db, args.out)
    print(f"Wrote {written} seed rows to {args.out}")
    print(f"  via BioProject accession: {bioproject_seeded}")
    print(f"  via ENA study accession:  {ena_seeded}")
    print(f"  Qiita-only (no BioProject/ENA found -- resolved via sources/qiita.py, real samples + data-availability marker): {qiita_only_seeded}")


if __name__ == "__main__":
    main()
