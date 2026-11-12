#!/usr/bin/env python3
"""Match a batch of new PDFs (prepare_new_pdfs.py's manifest.csv) to studies
in the live database, so extraction can use them (2026-11-12).

The pipeline finds a study's PDF only by the study's own DOI
(workflow/handlers.py _local_pdf_path_for_study), so per PDF:

  doi            a study already has this DOI -> nothing to change
  title          no study has the DOI, but exactly one non-merged study with
                 NO DOI has the same title (case and punctuation ignored):
                 created from a BioProject title, never given its paper's
                 DOI. --apply records the DOI on that study.
  title_conflict same title, but that study already has a different DOI
                 (preprint vs journal version?) -> left alone, listed
  ambiguous      several studies share the title -> left alone, listed
  no_study       not in the database: needs ingest-seeds + discovery first

Also notes, per matched study, whether text extraction already ran: a
completed EXTRACT_TEXT_FACTS task means its text was extracted before (from
PMC full text), so it is listed but not put on the extraction list.

Writes OUT_DIR/pdf_matches.csv, OUT_DIR/studies.txt (the studies to
extract: matched, no finished extraction) and OUT_DIR/no_study_seeds.csv
(an ingest-seeds file for the PDFs with no study).

Usage: python scripts/match_new_pdfs.py DB MANIFEST_CSV OUT_DIR [--apply]
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sqlite3
from pathlib import Path

SOURCE = "manual_pdf_title_match_2026-11-12"


def norm(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    dois_of: dict[str, set[str]] = collections.defaultdict(set)
    study_of_doi: dict[str, set[str]] = collections.defaultdict(set)
    for sid, v in conn.execute("select study_id, identifier_value from external_identifiers where identifier_type='doi'"):
        dois_of[sid].add(v.lower())
        study_of_doi[v.lower()].add(sid)
    by_title: dict[str, list[str]] = collections.defaultdict(list)
    for sid, title in conn.execute("select study_id, title from studies where canonical_status != 'merged'"):
        if norm(title):
            by_title[norm(title)].append(sid)
    extracted = {s for (s,) in conn.execute(
        "select distinct study_id from tasks where task_type='EXTRACT_TEXT_FACTS' and status='completed'")}

    rows, to_add = [], []
    for r in csv.DictReader(open(a.manifest)):
        doi = r["doi"].lower()
        if study_of_doi.get(doi):
            status, studies = "doi", sorted(study_of_doi[doi])
        else:
            hits = by_title.get(norm(r["title"]), []) if norm(r["title"]) else []
            if len(hits) == 1 and not dois_of.get(hits[0]):
                status, studies = "title", hits
                to_add.append((hits[0], doi))
            elif len(hits) == 1:
                status, studies = "title_conflict", hits
            elif hits:
                status, studies = "ambiguous", hits
            else:
                status, studies = "no_study", []
        rows.append({**r, "match": status, "study_ids": ";".join(studies),
                     "existing_dois": ";".join(sorted(set().union(*(dois_of.get(s, set()) for s in studies)))) if studies else "",
                     "already_extracted": "yes" if any(s in extracted for s in studies) else "no"})

    if a.apply and to_add:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from fair_ocean_agent.database.models import ExternalIdentifier

        with Session(create_engine(f"sqlite:///{a.db}")) as session:
            for sid, doi in to_add:
                session.add(ExternalIdentifier(study_id=sid, identifier_type="doi", identifier_value=doi,
                                               source=SOURCE, is_primary=True, verified=False))
            session.commit()

    fields = list(rows[0].keys()) if rows else []
    with open(out / "pdf_matches.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    todo = sorted({s for r in rows if r["match"] in ("doi", "title") and r["already_extracted"] == "no"
                   for s in r["study_ids"].split(";")})
    (out / "studies.txt").write_text("".join(f"{s}\n" for s in todo))
    with open(out / "no_study_seeds.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["seed_id", "title", "doi", "notes"])
        w.writerows((f"newpdf-{i:04d}", r["title"], r["doi"], "PDF supplied 2026-11-12")
                    for i, r in enumerate((r for r in rows if r["match"] == "no_study"), 1))

    c = collections.Counter(r["match"] for r in rows)
    print(f"{len(rows)} PDFs: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print(f"already extracted (listed, not queued): {sum(r['already_extracted'] == 'yes' for r in rows)}")
    print(f"studies to extract: {len(todo)} -> {out / 'studies.txt'}")
    print(("added" if a.apply else "would add (dry run; --apply to write)") + f" {len(to_add)} DOIs to title-matched studies")


if __name__ == "__main__":
    main()
