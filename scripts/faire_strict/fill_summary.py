#!/usr/bin/env python3
"""Per-term fill counts of the strict-FAIRe copy.

Each cell is classed as:
  value         a FAIRe-conforming value
  other         "other: ..." (present, but not in FAIRe's format/vocabulary)
  missing       a FAIRe missing-value term, or empty
  not_applicable  "not applicable[: ...]" (e.g. env_* for control samples)
A list cell counts as "value" if any item is a value, else "other" if any
item is other, else not_applicable / missing.

With ACCESS_CSV (scripts/faire_strict/study_access.py), each row also gets
the paper access of its study (best over linked studies: fulltext >
no_fulltext > no_publication), and missing cells are split by it:
missing_no_fulltext / missing_no_publication, plus rows_no_fulltext /
rows_no_publication.

Usage: python scripts/faire_strict/fill_summary.py STRICT_DIR OUT_CSV [ACCESS_CSV]
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


_ACCESS_RANK = {"fulltext": 0, "no_fulltext": 1, "no_publication": 2}


def load_access(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {r["study_id"]: r["access"] for r in csv.DictReader(handle)}


def row_access(access: dict[str, str], study_ids: str) -> str:
    kinds = [access.get(s.strip(), "no_publication") for s in study_ids.split("|") if s.strip()]
    return min(kinds, key=_ACCESS_RANK.__getitem__) if kinds else "no_publication"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    access = load_access(Path(sys.argv[3])) if len(sys.argv) > 3 else None
    terms = load_checklist()
    rows = []
    for table in TABLES:
        with (src / f"{table}.csv").open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            counts = [collections.Counter() for _ in header]
            n = 0
            by_access = collections.Counter()
            sid = header.index("internal_study_id")
            for row in reader:
                n += 1
                acc = row_access(access, row[sid]) if access is not None else "fulltext"
                by_access[acc] += 1
                for i, v in enumerate(row):
                    kind = classify(v)
                    counts[i][kind] += 1
                    if kind == "missing" and acc != "fulltext":
                        counts[i]["missing_" + acc] += 1
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
                **({
                    "missing_no_fulltext": c["missing_no_fulltext"],
                    "missing_no_publication": c["missing_no_publication"],
                    "rows_no_fulltext": by_access["no_fulltext"],
                    "rows_no_publication": by_access["no_publication"],
                } if access is not None else {}),
            })
        print(table, n, flush=True)
    with out.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
