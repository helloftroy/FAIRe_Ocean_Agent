#!/usr/bin/env python3
"""Spot-check a curation dry-run plan before applying it (2026-11-11).

Read-only. Writes three files into the plan folder:

  spotcheck_examples.tsv   40 random samples per deletion rule, with the
                           evidence the classifier saw and the study title
  spotcheck_organisms.tsv  the 40 most common organism values per
                           sample_non_marine category, with counts
  spotcheck_studies.tsv    studies NOT flagged not_relevant (kept as marine
                           or freshwater) that lose >= 25 % of their samples:
                           the place a classifier mistake would hurt

Usage: python scripts/curation_spotcheck.py DB PLAN_DIR
"""
from __future__ import annotations

import collections
import csv
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curation_lib import EVIDENCE_FIELDS  # noqa: E402

PER_RULE = 40
TOP_ORGANISMS = 40
LOSS_SHARE = 0.25


def main() -> None:
    db, plan_dir = sys.argv[1], Path(sys.argv[2])
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    random.seed(11)

    deleted: dict[str, str] = {}
    for r in csv.DictReader(open(plan_dir / "delete_entities.csv")):
        if r["level"] == "sample":
            deleted[r["entity_id"]] = r["rule"]
    flags = {r["study_id"]: r["marine_relevance_status"] for r in csv.DictReader(open(plan_dir / "study_flags.csv"))}
    print(f"{len(deleted):,} samples to delete; reading evidence")

    evidence: dict[str, dict[str, str]] = collections.defaultdict(dict)
    marks = ",".join("?" * len(EVIDENCE_FIELDS))
    for eid, field, value in conn.execute(
        "select entity_id, fact_type_candidate, raw_value from raw_facts indexed by ix_raw_facts_fact_type_candidate "
        f"where fact_type_candidate in ({marks}) and entity_level='sample' and entity_id is not null",
        EVIDENCE_FIELDS,
    ):
        if eid in deleted:
            evidence[eid].setdefault(field, (value or "").replace("\t", " ").replace("\n", " ")[:120])

    titles = dict(conn.execute("select study_id, coalesce(title,'') from studies"))
    total = collections.Counter()
    lost = collections.Counter()
    lost_by_rule: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    lost_orgs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    home: dict[str, str] = {}
    for eid, study in conn.execute("select es.entity_id, es.study_id from entity_studies es join entities e using (entity_id) "
                                   "where e.entity_level='sample'"):
        total[study] += 1
        if eid in deleted:
            lost[study] += 1
            lost_by_rule[study][deleted[eid]] += 1
            lost_orgs[study][evidence.get(eid, {}).get("organism", "")] += 1
            home.setdefault(eid, study)

    by_rule: dict[str, list[str]] = collections.defaultdict(list)
    for eid, rule in deleted.items():
        by_rule[rule].append(eid)
    with open(plan_dir / "spotcheck_examples.tsv", "w", newline="") as h:
        w = csv.writer(h, delimiter="\t")
        w.writerow(["rule", "entity_id", *EVIDENCE_FIELDS, "study_id", "study_title"])
        for rule in sorted(by_rule):
            for eid in random.sample(by_rule[rule], min(PER_RULE, len(by_rule[rule]))):
                ev = evidence.get(eid, {})
                s = home.get(eid, "")
                w.writerow([rule, eid, *(ev.get(f, "") for f in EVIDENCE_FIELDS), s, titles.get(s, "")[:100]])

    with open(plan_dir / "spotcheck_organisms.tsv", "w", newline="") as h:
        w = csv.writer(h, delimiter="\t")
        w.writerow(["rule", "organism", "samples"])
        for rule in sorted(r for r in by_rule if r.startswith("sample_non_marine")):
            counts = collections.Counter(evidence.get(e, {}).get("organism", "") for e in by_rule[rule])
            for org, n in counts.most_common(TOP_ORGANISMS):
                w.writerow([rule, org, n])

    rows = []
    for study, n in lost.items():
        if flags.get(study) == "not_relevant" or n < LOSS_SHARE * total[study]:
            continue
        rows.append((n, study, total[study], flags.get(study, ""),
                     "; ".join(f"{r} {c}" for r, c in lost_by_rule[study].most_common(3)),
                     "; ".join(f"{o or '(none)'} {c}" for o, c in lost_orgs[study].most_common(3)),
                     titles.get(study, "")[:100]))
    rows.sort(reverse=True)
    with open(plan_dir / "spotcheck_studies.tsv", "w", newline="") as h:
        w = csv.writer(h, delimiter="\t")
        w.writerow(["samples_lost", "study_id", "samples_total", "flag", "top_rules", "top_organisms_lost", "title"])
        w.writerows(rows)

    print(f"examples: {sum(min(PER_RULE, len(v)) for v in by_rule.values())} rows over {len(by_rule)} rules")
    print(f"kept studies losing >= {LOSS_SHARE:.0%} of samples: {len(rows):,}"
          f" ({sum(r[0] for r in rows):,} samples)")
    print(f"wrote spotcheck_*.tsv to {plan_dir}")


if __name__ == "__main__":
    main()
