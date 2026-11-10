#!/usr/bin/env python3
"""Per-term fill counts of the strict-FAIRe copy.

Each cell is classed as:
  value         a FAIRe-conforming value
  other         "other: ..." (present, but not in FAIRe's format/vocabulary)
  missing       a FAIRe missing-value term, or empty
  not_applicable  "not applicable[: ...]" (e.g. env_* for control samples)
A list cell counts as "value" if any item is a value, else "other" if any
item is other, else not_applicable / missing.

Usage: python scripts/faire_strict/fill_summary.py STRICT_DIR OUT_CSV
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checklist import TABLES, load_checklist  # noqa: E402

csv.field_size_limit(10**9)


def classify(value: str) -> str:
    if not value:
        return "missing"
    kinds = set()
    for item in (p.strip() for p in value.split("|")):
        if not item:
            continue
        low = item.lower()
        if low.startswith("other:"):
            kinds.add("other")
        elif low.startswith("not applicable"):
            kinds.add("not_applicable")
        elif low.startswith("missing"):
            kinds.add("missing")
        else:
            kinds.add("value")
    for k in ("value", "other", "not_applicable", "missing"):
        if k in kinds:
            return k
    return "missing"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    terms = load_checklist()
    rows = []
    for table in TABLES:
        with (src / f"{table}.csv").open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            counts = [collections.Counter() for _ in header]
            n = 0
            for row in reader:
                n += 1
                for i, v in enumerate(row):
                    counts[i][classify(v)] += 1
        for col, c in zip(header, counts):
            if col not in terms:
                continue
            t = terms[col]
            rows.append({
                "table": table, "term": col, "section": t.section, "requirement": t.requirement,
                "rows": n, "value": c["value"], "other": c["other"],
                "not_applicable": c["not_applicable"], "missing": c["missing"],
                "pct_value": round(100 * c["value"] / n, 2),
                "pct_present": round(100 * (c["value"] + c["other"]) / n, 2),
            })
        print(table, n, flush=True)
    with out.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
