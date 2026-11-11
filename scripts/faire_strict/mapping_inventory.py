#!/usr/bin/env python3
"""Inventory of raw fact names vs mapping rules (gap-2 audit).

For every (fact_type_candidate, entity_level) in raw_facts: number of
facts, distinct entities, distinct studies, a few example values, and the
FAIRe target(s) mapping/rules.py's rules_for() gives it (empty = unmapped).

Usage: python scripts/faire_strict/mapping_inventory.py DB OUT_CSV
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fair_ocean_agent.mapping.rules import rules_for  # noqa: E402


def main() -> None:
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    rows = db.execute(
        "select fact_type_candidate, entity_level, count(*), count(distinct entity_id), count(distinct study_id) "
        "from raw_facts group by 1, 2"
    ).fetchall()
    print(len(rows), "name/level pairs", flush=True)
    with open(sys.argv[2], "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["fact_type", "entity_level", "facts", "entities", "studies", "targets", "examples"])
        for name, level, n, ents, studies in sorted(rows, key=lambda r: -r[2]):
            targets = sorted({f"{r.target_table}.{r.target_field}" for r in rules_for(name or "", level)})
            examples = ""
            if n >= 50:
                vals = db.execute(
                    "select distinct raw_value from raw_facts indexed by ix_raw_facts_fact_type_candidate "
                    "where fact_type_candidate=? and entity_level=? limit 4", (name, level)
                ).fetchall()
                examples = " || ".join((v or "")[:60] for (v,) in vals)
            w.writerow([name, level, n, ents, studies, " ".join(targets), examples])


if __name__ == "__main__":
    main()
