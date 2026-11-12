#!/usr/bin/env python3
"""Rename a batch of hand-downloaded PDFs to the pipeline's DOI file names
(2026-11-12).

Inputs, all in one folder:
  - Paperpile export zips ("Paperpile files/<Author> et al. <year> - <title>.pdf")
  - Paperpile reference CSVs (Attachments column -> DOI)
  - loose PDFs named after their DOI ("10.3390:fishes10100509.pdf",
    "doi.org:10.1371:journal.pone.0125844"; ':' stands for '/')

Output: OUT_DIR/<doi lower, '/' -> '_'>.pdf (workflow/handlers.py
_doi_pdf_filename) and OUT_DIR/manifest.csv. When a paper has several
attachments, the first one the CSV lists is used (Paperpile lists the main
article first); " (1)" / " (2)" duplicate downloads are folded into their
original. Files that are not PDFs (no %PDF header) are reported, not copied.

With --db, each DOI is also looked up in the database (read-only):
study_id, whether a PDF for it already exists in --existing dirs.

Usage:
  python scripts/prepare_new_pdfs.py IN_DIR OUT_DIR [--db DB] [--existing DIR ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sqlite3
import zipfile
from pathlib import Path

DUP = re.compile(r" \(\d+\)(?=\.pdf$)", re.I)


def pdf_name(doi: str) -> str:
    return doi.strip().lower().replace("/", "_") + ".pdf"


def doi_from_loose_name(name: str) -> str | None:
    stem = re.sub(r"\.pdf$", "", name, flags=re.I)
    stem = re.sub(r"^(https?:)?(doi\.org|dx\.doi\.org)[:/]", "", stem, flags=re.I)
    stem = stem.replace(":", "/")
    return stem if re.match(r"^10\.\d{4,9}/\S+$", stem) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--db")
    ap.add_argument("--existing", nargs="*", default=[])
    a = ap.parse_args()
    src, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # attachment basename -> (doi, rank within that reference, title, folders)
    attach: dict[str, tuple[str, int, str, str]] = {}
    for f in sorted(glob.glob(str(src / "Paperpile*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            doi = (r.get("DOI") or "").strip()
            for rank, x in enumerate(p.strip() for p in (r.get("Attachments") or "").split(";") if p.strip()):
                attach.setdefault(os.path.basename(x), (doi, rank, r.get("Title", ""), r.get("Folders filed in", "")))

    # doi -> best candidate (rank, origin, reader)
    chosen: dict[str, tuple[int, str, object, str]] = {}
    problems: list[tuple[str, str]] = []

    def offer(doi: str, rank: int, origin: str, read, title: str) -> None:
        cur = chosen.get(doi.lower())
        if cur is None or rank < cur[0]:
            chosen[doi.lower()] = (rank, origin, read, title)

    for z in sorted(glob.glob(str(src / "*.zip"))):
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            base = os.path.basename(n)
            key = base if base in attach else DUP.sub("", base)
            dup = key != base
            if key not in attach:
                problems.append((f"{Path(z).name}:{base}", "no matching Paperpile attachment"))
                continue
            doi, rank, title, _ = attach[key]
            if not doi:
                problems.append((f"{Path(z).name}:{base}", "reference has no DOI"))
                continue
            offer(doi, rank + (100 if dup else 0), f"{Path(z).name}:{base}", (lambda zf=zf, n=n: zf.read(n)), title)

    for f in sorted(src.iterdir()):
        if f.is_dir() or f.suffix.lower() in (".zip", ".csv") or f.name.startswith("."):
            continue
        doi = doi_from_loose_name(f.name)
        if not doi:
            problems.append((f.name, "file name is not a DOI"))
            continue
        offer(doi, -1, f.name, (lambda f=f: f.read_bytes()), "")

    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) if a.db else None
    existing = {p.name.lower() for d in a.existing for p in Path(d).glob("*.pdf")}
    rows = []
    for doi, (_, origin, read, title) in sorted(chosen.items()):
        data = read()
        if not data.startswith(b"%PDF"):
            problems.append((origin, "not a PDF (no %PDF header)"))
            continue
        name = pdf_name(doi)
        (out / name).write_bytes(data)
        studies = ""
        if db is not None:
            studies = ";".join(s for (s,) in db.execute(
                "select distinct study_id from external_identifiers where identifier_type='doi' and identifier_value in (?, ?)",
                (doi, doi.lower())))
        rows.append({"doi": doi, "pdf_filename": name, "source": origin, "title": title[:150],
                     "study_ids": studies, "already_had_pdf": "yes" if name in existing else "no"})

    with open(out / "manifest.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["doi", "pdf_filename", "source", "title", "study_ids", "already_had_pdf"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "problems.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["file", "problem"])
        w.writerows(problems)

    print(f"{len(rows)} PDFs written to {out} ({sum(r['already_had_pdf'] == 'yes' for r in rows)} already on file)")
    if db is not None:
        print(f"  in the database: {sum(bool(r['study_ids']) for r in rows)}; not a study: {sum(not r['study_ids'] for r in rows)}")
    print(f"{len(problems)} problems -> {out / 'problems.csv'}")


if __name__ == "__main__":
    main()
