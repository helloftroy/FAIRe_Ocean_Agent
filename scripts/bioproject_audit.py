#!/usr/bin/env python3
"""Per-BioProject audit of every study with 2+ BioProjects (2026-11-10).

Read-only. For each (study, BioProject):
  arrival      rank by when the BioProject identifier was added (1 = first)
  via          the step(s) that added it (external_identifiers.source)
  samples      sample entities placed in it: NCBI BioSample facts (the
               fetch's BioProject) or ENA run facts (samp_name accession,
               the ENA source's project)
  categories   marine / medical / terrestrial / freshwater / engineered /
               other_host / unclassified (scripts/marine_audit.categorize)
  collisions   samples that are SAMN digit collisions of a SAMEA/SAMD the
               study already knew (the efetch bug, see data_integrity_audit.md)
  verdict      see VERDICTS below. Nothing is changed.

Usage: python scripts/bioproject_audit.py DB OUT_DIR
"""
from __future__ import annotations

import collections
import csv
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marine_audit import ORDER, categorize  # noqa: E402

RESOLUTION = "external_identifier_sequence_accession_resolution"
NON_MARINE = ("medical", "terrestrial", "engineered", "other_host")

VERDICTS = """
  first_arrived        the study's first BioProject (kept as its own data)
  foreign_bug_chain    added only by sequence-accession resolution in a study
                       with digit-collision samples, and none of its samples
                       is marine or freshwater -> almost surely foreign
  non_marine           >= 90 % of classified samples non-marine (user rule:
                       drop per BioProject; freshwater counts as kept)
  jgi_gold             added from JGI GOLD (one BioProject per sequencing
                       project, so a GOLD study legitimately has hundreds)
  paper_cited          the accession is written in the paper's own text
                       (its own data, or a dataset it reanalysed)
  resolved_clean       added by sequence-accession resolution in a study
                       with no digit-collision samples
  review               anything else
"""
PAPER_TEXT = {"europe_pmc_fulltext_identifier_scan", "local_pdf_identifier_scan"}


def main() -> None:
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    bp_rows = collections.defaultdict(list)  # study -> [(bp, source, created)]
    for s, v, src, created in db.execute(
        "select study_id, identifier_value, coalesce(source,''), created_at from external_identifiers "
        "where identifier_type='bioproject_accession' and (relationship_type is null or relationship_type='')"
    ):
        bp_rows[s].append((v, src, created))
    studies = {s for s, rows in bp_rows.items() if len({v for v, _, _ in rows}) >= 2}
    print(f"{len(studies):,} studies with 2+ BioProjects", flush=True)
    sq = ",".join(f"'{s}'" for s in studies)

    # entity accession lookup and digit-collision set
    acc_to_entity = {}
    for e, x, s in db.execute(f"select entity_id, external_identifier, study_id from entities where entity_level='sample' and study_id in ({sq})"):
        if x:
            acc_to_entity[x] = e
    known = collections.defaultdict(set)
    for s, v in db.execute(
        f"select study_id, identifier_value from external_identifiers where identifier_type='biosample_accession' "
        f"and study_id in ({sq}) and (identifier_value like 'SAMEA%' or identifier_value like 'SAMD%')"
    ):
        known[s].add(int(re.sub(r"\D", "", v)))

    sample_bp: dict[tuple[str, str], str] = {}  # (study, entity) -> bioproject
    evidence: dict[str, dict[str, str]] = collections.defaultdict(dict)
    collision: set[str] = set()
    for s, e, x, field, value, bp in db.execute(
        f"select f.study_id, f.entity_id, en.external_identifier, f.fact_type_candidate, f.raw_value, src.external_identifier "
        f"from raw_facts f join sources src on src.source_id=f.source_id join entities en on en.entity_id=f.entity_id "
        f"where f.study_id in ({sq}) and f.entity_level='sample' and f.extraction_method='adapter:ncbi_biosample' "
        f"and f.fact_type_candidate in ('organism','host','isolation_source','env_broad_scale')"
    ):
        sample_bp.setdefault((s, e), bp)
        evidence[e].setdefault(field, value or "")
        if x and x.startswith("SAMN") and x[4:].isdigit() and int(x[4:]) in known[s]:
            collision.add(e)
    print(f"{len(sample_bp):,} samples placed from NCBI facts", flush=True)
    for s, acc, bp in db.execute(
        f"select f.study_id, f.raw_value, src.external_identifier from raw_facts f join sources src on src.source_id=f.source_id "
        f"where f.study_id in ({sq}) and f.fact_type_candidate='samp_name' and f.entity_level='experiment_run' "
        f"and src.source_name='ena'"
    ):
        e = acc_to_entity.get(acc or "")
        if e:
            sample_bp.setdefault((s, e), bp)
    print(f"{len(sample_bp):,} samples placed after ENA runs", flush=True)

    titles = {}
    for bp, title in db.execute(
        "select src.external_identifier, f.raw_value from raw_facts f join sources src on src.source_id=f.source_id "
        "where src.source_name='ncbi_bioproject' and f.fact_type_candidate in ('title','project_title')"
    ):
        titles.setdefault(bp, (title or "")[:120])

    per = collections.defaultdict(lambda: collections.Counter())
    for (s, e), bp in sample_bp.items():
        f = evidence.get(e, {})
        cat = None
        for field in ("organism", "host", "isolation_source", "env_broad_scale"):
            cat = categorize(f.get(field))
            if cat:
                break
        c = per[(s, bp)]
        c["samples"] += 1
        c[cat or "unclassified"] += 1
        if e in collision:
            c["collisions"] += 1

    study_titles = dict(db.execute(f"select study_id, title from studies where study_id in ({sq})"))
    rows = []
    for s in sorted(studies):
        first = {}
        for v, src, created in sorted(bp_rows[s], key=lambda r: r[2]):
            first.setdefault(v, (created, set()))[1].add(src or "(seed)")
        ranked = sorted(first.items(), key=lambda kv: kv[1][0])
        study_collisions = sum(per[(s, bp)]["collisions"] for bp, _ in ranked)
        for rank, (bp, (created, srcs)) in enumerate(ranked, 1):
            c = per[(s, bp)]
            classified = sum(c[k] for k in ORDER)
            nonmar = sum(c[k] for k in NON_MARINE)
            kept_kind = c["marine"] + c["freshwater"]
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
