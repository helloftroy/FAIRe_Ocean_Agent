#!/usr/bin/env python3
"""Per-study paper access class, for the fill-summary figure.

  fulltext        the article full text was retrieved and read
                  (an article_fulltext source exists)
  no_fulltext     a paper is linked (article DOI / PMID / PMCID, or the
                  study is in missing_pdfs.csv) but its full text was never
                  retrieved: paywalled or otherwise not open
  no_publication  no linked paper (repository-only data)

Dataset DOIs (Zenodo, PANGAEA, Dryad, figshare, DataCite-only) do not
count as a paper.

Usage: python scripts/faire_strict/study_access.py DB MISSING_PDFS_CSV OUT_CSV
"""
from __future__ import annotations

import csv
import sqlite3
import sys

DATASET_DOI_PREFIXES = ("10.5281/", "10.1594/", "10.5061/", "10.6084/", "10.26008/", "10.17882/", "10.7910/")


def main() -> None:
    db_path, missing_pdfs, out = sys.argv[1:4]
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    fulltext = {s for (s,) in db.execute("select distinct study_id from sources where source_type='article_fulltext'")}
    paper = set()
    for study_id, kind, value in db.execute(
        "select study_id, identifier_type, identifier_value from external_identifiers "
        "where identifier_type in ('doi','pmid','pmcid') and (relationship_type is null or relationship_type='')"
    ):
        if kind == "doi" and value.lower().startswith(DATASET_DOI_PREFIXES):
            continue
        paper.add(study_id)
    with open(missing_pdfs, newline="") as handle:
        for r in csv.DictReader(handle):
            if r["doi"] and not r["doi"].lower().startswith(DATASET_DOI_PREFIXES):
                paper.add(r["study_id"])
    counts: dict[str, int] = {}
    with open(out, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["study_id", "access"])
        for (study_id,) in db.execute("select study_id from studies order by study_id"):
            access = "fulltext" if study_id in fulltext else ("no_fulltext" if study_id in paper else "no_publication")
            counts[access] = counts.get(access, 0) + 1
            w.writerow([study_id, access])
    print(counts)


if __name__ == "__main__":
    main()
