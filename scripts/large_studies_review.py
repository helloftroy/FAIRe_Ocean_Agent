#!/usr/bin/env python3
"""Review the largest studies still waiting for MAP_FAIRE (2026-11-18):
are they marine, and are they new data or a reanalysis of other studies'
samples? Read-only.

Per study with >= --min-samples samples and a pending/running MAP_FAIRE task:
  samples, shared_pct (samples also linked to another study: a high share
  means the study reuses someone else's data), bioprojects, doi,
  marine_relevance_status, the 3 most common sample organism /
  env_broad_scale / isolation_source / host values, a reanalysis keyword
  hit in title/description, and the first 200 characters of the description.

Writes OUT_TSV (default data/audits/large_studies_review.tsv).

Usage: python scripts/large_studies_review.py [--min-samples 1000] [--out OUT_TSV]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path

REANALYSIS = re.compile(
    r"re-?analy|meta-?analy|third[- ]party|\bTPA\b|compendium|catalog(ue)?|atlas|database|"
    r"public(ly available)? (data|metagenom|sequenc)|reprocess|re-?assembl|benchmark|global survey of public",
    re.I,
)
FIELDS = ("organism", "env_broad_scale", "isolation_source", "host")


def db_path() -> str:
    url = os.environ.get("FAIR_OCEAN_DATABASE_URL", "")
    return url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else "data/fair_ocean.db"


def top(conn, study: str, field: str) -> str:
    rows = conn.execute(
        "select raw_value, count(*) n from raw_facts where study_id=? and fact_type_candidate=? "
        "and entity_level='sample' group by raw_value order by n desc limit 3", (study, field)).fetchall()
    return "; ".join(f"{(v or '')[:60]} ({n:,})" for v, n in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=1000)
    ap.add_argument("--out", default="data/audits/large_studies_review.tsv")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)

    waiting = {s for (s,) in conn.execute(
        "select study_id from tasks where task_type='MAP_FAIRE' and status in ('pending','retry_pending','running')")}
    counts = conn.execute(
        "select es.study_id, count(*) from entity_studies es join entities e using (entity_id) "
        "where e.entity_level='sample' group by es.study_id").fetchall()
    big = sorted(((n, s) for s, n in counts if s in waiting and n >= a.min_samples), reverse=True)
    print(f"{len(big)} waiting studies with >= {a.min_samples:,} samples")

    rows = []
    for n, s in big:
        title, desc, status = conn.execute(
            "select coalesce(title,''), coalesce(description,''), marine_relevance_status from studies where study_id=?",
            (s,)).fetchone()
        shared = conn.execute(
            "select count(*) from entity_studies es join entities e using (entity_id) where es.study_id=? "
            "and e.entity_level='sample' and exists (select 1 from entity_studies o "
            "where o.entity_id=es.entity_id and o.study_id != es.study_id)", (s,)).fetchone()[0]
        ids = conn.execute(
            "select identifier_type, identifier_value from external_identifiers where study_id=?", (s,)).fetchall()
        hit = REANALYSIS.search(f"{title} {desc}")
        rows.append({
            "study_id": s, "samples": n, "shared_pct": round(100 * shared / n), "marine_status": status,
            "reanalysis_keyword": hit.group(0) if hit else "",
            "bioprojects": ";".join(v for t, v in ids if t == "bioproject_accession")[:120],
            "doi": ";".join(v for t, v in ids if t == "doi")[:120],
            "title": title[:200],
            **{f"top_{f}": top(conn, s, f) for f in FIELDS},
            "description": re.sub(r"\s+", " ", desc)[:200],
        })
        print(f"  {s} {n:>7,} samples  shared {rows[-1]['shared_pct']:>3}%  {status:<14} {title[:70]}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]) if rows else ["study_id"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
