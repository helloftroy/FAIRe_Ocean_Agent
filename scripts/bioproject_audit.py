#!/usr/bin/env python3
"""Per-BioProject audit of every study with 2+ BioProjects (2026-11-10).

Read-only. For each (study, BioProject):
  arrival      rank by when the BioProject identifier was added (1 = first)
  via          the step(s) that added it (external_identifiers.source)
  samples      sample entities placed in it (curation_lib.placements)
  categories   marine / medical / terrestrial / freshwater / engineered /
               other_host / unclassified (curation_lib.sample_categories)
  collisions   SAMN digit collisions of a SAMEA/SAMD the study already knew
  verdict      see VERDICTS. Nothing is changed.

Usage: python scripts/bioproject_audit.py DB OUT_DIR
"""
from __future__ import annotations

import collections
import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_lib import (  # noqa: E402
    KEPT_KINDS,
    NON_MARINE,
    ORDER,
    bioproject_identifiers,
    digit_collisions,
    placements,
    sample_categories,
)

RESOLUTION = "external_identifier_sequence_accession_resolution"
PAPER_TEXT = {"europe_pmc_fulltext_identifier_scan", "local_pdf_identifier_scan"}
DROP_VERDICTS = ("foreign_bug_chain", "non_marine")

VERDICTS = """
  first_arrived        the study's first BioProject (kept as its own data)
  foreign_bug_chain    added only by sequence-accession resolution in a study
                       with digit-collision samples, and none of its samples
                       is marine or freshwater -> almost surely foreign
  non_marine           >= 90 % of classified samples non-marine (user rule:
                       drop per BioProject; freshwater counts as kept)
  paper_cited          the accession is written in the paper's own text
  jgi_gold             added from JGI GOLD (one BioProject per sequencing
                       project, so a GOLD study legitimately has hundreds)
  resolved_clean       added by sequence-accession resolution in a study
                       with no digit-collision samples
  review               anything else
"""


def audit(conn: sqlite3.Connection) -> list[dict]:
    bp_rows = bioproject_identifiers(conn)
    studies = {s for s, rows in bp_rows.items() if len({v for v, _, _ in rows}) >= 2}
    print(f"{len(studies):,} studies with 2+ BioProjects", flush=True)
    cats = sample_categories(conn)
    placed = placements(conn, studies)
    collisions = digit_collisions(conn, studies)
    print(f"{len(placed):,} (study, sample) placements", flush=True)

    per = collections.defaultdict(collections.Counter)
    for (s, e), bp in placed.items():
        c = per[(s, bp)]
        c["samples"] += 1
        c[cats.get(e) or "unclassified"] += 1
        if e in collisions:
            c["collisions"] += 1

    titles = {}
    for bp, title in conn.execute(
        "select src.external_identifier, f.raw_value from raw_facts f join sources src on src.source_id=f.source_id "
        "where src.source_name='ncbi_bioproject' and f.fact_type_candidate in ('title','project_title')"
    ):
        titles.setdefault(bp, (title or "")[:120])
    study_titles = dict(conn.execute("select study_id, title from studies"))

    rows = []
    for s in sorted(studies):
        first: dict[str, tuple[str, set]] = {}
        for v, src, created in sorted(bp_rows[s], key=lambda r: r[2]):
            first.setdefault(v, (created, set()))[1].add(src or "(seed)")
        ranked = sorted(first.items(), key=lambda kv: kv[1][0])
        study_collisions = sum(per[(s, bp)]["collisions"] for bp, _ in ranked)
        for rank, (bp, (created, srcs)) in enumerate(ranked, 1):
            c = per[(s, bp)]
            classified = sum(c[k] for k in ORDER)
            nonmar = sum(c[k] for k in NON_MARINE)
            kept_kind = sum(c[k] for k in KEPT_KINDS)
            if rank == 1:
                verdict = "first_arrived"
            elif srcs == {RESOLUTION} and study_collisions and kept_kind == 0:
                verdict = "foreign_bug_chain"
            elif classified and nonmar / classified >= 0.9:
                verdict = "non_marine"
            elif srcs & PAPER_TEXT:
                verdict = "paper_cited"
            elif "gold" in srcs:
                verdict = "jgi_gold"
            elif srcs == {RESOLUTION} and not study_collisions:
                verdict = "resolved_clean"
            else:
                verdict = "review"
            rows.append({
                "study_id": s, "study_title": (study_titles.get(s) or "")[:100],
                "bioproject": bp, "arrival": rank, "added": created[:19], "via": " ".join(sorted(srcs)),
                "samples": c["samples"], "collisions": c["collisions"],
                **{k: c[k] for k in ORDER}, "unclassified": c["unclassified"],
                "verdict": verdict, "bioproject_title": titles.get(bp, ""),
            })
    return rows


def main() -> None:
    conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = audit(conn)
    with (out / "bioproject_audit.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        summary[r["verdict"]][0] += 1
        summary[r["verdict"]][1] += r["samples"]
    print(VERDICTS)
    for k, (n, smp) in sorted(summary.items()):
        print(f"{k:20s} {n:>6,} BioProjects {smp:>9,} samples")


if __name__ == "__main__":
    main()
