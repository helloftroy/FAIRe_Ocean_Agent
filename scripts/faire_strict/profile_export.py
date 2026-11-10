#!/usr/bin/env python3
"""Profile the existing FAIRe export against the pinned FAIRe v1.0.2
checklist: per column, how full it is and which values break the term's
type, controlled vocabulary or fixed format. Read-only.

Usage: python scripts/faire_strict/profile_export.py data/snapshots/faire_2026-11 OUT_DIR
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checklist import MISSING_VALUE_TERMS, TABLES, is_number, load_checklist  # noqa: E402
from geo import CURRENT, HISTORICAL  # noqa: E402

csv.field_size_limit(10**9)
MAX_DISTINCT = 20000

FIXED_FORMAT_PATTERNS = {
    "yyyy-mm-dd": re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$"),
    "yyyy-mm-ddThh:mm:ss+-hh:mm": re.compile(
        r"^\d{4}(-\d{2}(-\d{2}(T\d{2}(:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}(:?\d{2})?)?)?)?)?"
        r"(/\d{4}(-\d{2}(-\d{2}(T\d{2}(:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}(:?\d{2})?)?)?)?)?)?$"
    ),
    "termLabel [termID]": re.compile(r"^.+ \[[A-Za-z]+[_:]\d+\]$"),
    "PnYnMnWnDTnHnMnS": re.compile(r"^P(?!$)(\d+(\.\d+)?Y)?(\d+(\.\d+)?M)?(\d+(\.\d+)?W)?(\d+(\.\d+)?D)?(T(?=\d)(\d+(\.\d+)?H)?(\d+(\.\d+)?M)?(\d+(\.\d+)?S)?)?(/\S+)?$"),
    "<country or sea names>:<region>,<locality>": re.compile(r"^[^:]+(:.*)?$"),
}


GEO_NAMES = frozenset((*CURRENT, *HISTORICAL))


def check_item(term, item: str) -> str:
    """Category of one list item: ok, missing_term, other, case, bad."""
    if item in MISSING_VALUE_TERMS:
        return "missing_term"
    if item.lower() in MISSING_VALUE_TERMS:
        return "case"
    if re.match(r"^other: \S", item):
        return "other"  # decision D0: allowed for review in every field
    if term.name == "geo_loc_name":
        head = item.partition(":")[0].strip()
        return "ok" if head in GEO_NAMES else "bad"
    t = term.term_type
    if t == "controlled vocabulary" or t == "numeric or controlled vocabulary":
        if item in term.vocabulary:
            return "ok"
        if t.startswith("numeric") and is_number(item):
            return "ok"
        if re.match(r"^other: \S", item):
            return "other"
        if item.lower() in {v.lower() for v in term.vocabulary}:
            return "case"
        return "bad"
    if t == "numeric":
        return "ok" if is_number(item) else "bad"
    if t == "integer":
        return "ok" if is_number(item, integer=True) else "bad"
    if t == "Boolean":
        return "ok" if item in ("0", "1") else "bad"
    if t == "fixed format":
        pattern = FIXED_FORMAT_PATTERNS.get(term.fixed_format)
        if pattern is None:
            return "ok"
        return "ok" if pattern.match(item) else "bad"
    if t == "DOI":
        return "ok" if re.match(r"^(DOI: |https?://(dx\.)?doi\.org/)?10\.\d{4,9}/\S+$", item) else "bad"
    return "ok"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    terms = load_checklist()
    report = {}
    for table in TABLES:
        path = src / f"{table}.csv"
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            stats = {
                col: {"filled": 0, "items": 0, "multi": 0, "cats": collections.Counter(), "bad": collections.Counter(), "values": collections.Counter()}
                for col in header
            }
            n = 0
            for row in reader:
                n += 1
                for col, value in zip(header, row):
                    if not value:
                        continue
                    s = stats[col]
                    s["filled"] += 1
                    if len(s["values"]) < MAX_DISTINCT or value in s["values"]:
                        s["values"][value] += 1
                    term = terms.get(col)
                    if term is None:
                        continue
                    items = [p.strip() for p in value.split("|")] if term.term_type != "free text" else [value]
                    if len(items) > 1:
                        s["multi"] += 1
                    for item in items:
                        s["items"] += 1
                        cat = check_item(term, item)
                        s["cats"][cat] += 1
                        if cat in ("bad", "case") and (len(s["bad"]) < MAX_DISTINCT or item in s["bad"]):
                            s["bad"][item] += 1
        cols = {}
        for col in header:
            term = terms.get(col)
            s = stats[col]
            cols[col] = {
                "in_checklist": term is not None,
                "checklist_tables": list(term.tables) if term else [],
                "wrong_table": bool(term) and table not in term.tables,
                "requirement": term.requirement if term else "",
                "term_type": term.term_type if term else "",
                "unit": term.unit if term else "",
                "fixed_format": term.fixed_format if term else "",
                "filled": s["filled"],
                "fill_rate": round(s["filled"] / n, 4) if n else 0,
                "multi_valued_rows": s["multi"],
                "item_categories": dict(s["cats"]),
                "distinct_values": len(s["values"]),
                "top_values": s["values"].most_common(12),
                "top_nonconforming": s["bad"].most_common(40),
                "distinct_nonconforming": len(s["bad"]),
            }
        missing_terms = [
            t.name for t in terms.values() if table in t.tables and t.name not in header
        ]
        report[table] = {
            "rows": n,
            "columns": cols,
            "checklist_terms_absent_from_export": [
                {"term": name, "requirement": terms[name].requirement, "condition": terms[name].condition}
                for name in missing_terms
            ],
        }
        # full non-conforming distinct values, for writing rules later
        with (out / f"nonconforming_{table}.csv").open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["column", "value", "count"])
            for col in header:
                for value, count in stats[col]["bad"].most_common():
                    w.writerow([col, value, count])
        print(f"{table}: {n} rows profiled", flush=True)
    (out / "profile.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
