#!/usr/bin/env python3
"""Wall time per study from remap worker logs (2026-11-20).

Each "mapped N FAIRe standardized value(s) for study S" line's time minus
the previous one's is that study's full wall time (claim + mapping +
commit). Prints the time per bucket and the 20 slowest studies, and how
long the job had been on the study it was killed in.

Usage: python scripts/remap_log_times.py logs/remap_JOBID.out [...]
"""
from __future__ import annotations

import re
import sys
from datetime import datetime

LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) .*mapped (\d+) FAIRe standardized value\(s\) for study (\S+)")
START = re.compile(r"^Job started: (.+?)  db:")
BUCKETS = [(0, 10, "< 10 s"), (10, 60, "10-60 s"), (60, 600, "1-10 min"), (600, 3600, "10-60 min"),
           (3600, 10**9, "> 1 h")]


def main() -> None:
    rows = []
    for path in sys.argv[1:]:
        prev = None
        for line in open(path, errors="replace"):
            m = LINE.match(line)
            if m:
                t = datetime.fromisoformat(m.group(1))
                if prev is not None:
                    rows.append(((t - prev).total_seconds(), int(m.group(2)), m.group(3), path))
                prev = t
        if prev is not None:
            print(f"{path}: last study finished {prev}; check the .err for the kill time "
                  f"(time after that = the study it was killed in)")
    if not rows:
        print("no mapped lines found")
        return
    total = sum(r[0] for r in rows)
    print(f"\n{len(rows):,} studies timed, {total / 3600:.1f} h")
    print(f"{'wall time':<10} {'studies':>8} {'hours':>7} {'share':>6}")
    for lo, hi, name in BUCKETS:
        sel = [r for r in rows if lo <= r[0] < hi]
        h = sum(r[0] for r in sel) / 3600
        print(f"{name:<10} {len(sel):>8,} {h:>7.1f} {100 * h * 3600 / total:>5.0f}%")
    print(f"\n{'seconds':>8} {'values':>9}  study")
    for s, n, study, _ in sorted(rows, reverse=True)[:20]:
        print(f"{s:>8.0f} {n:>9,}  {study}")


if __name__ == "__main__":
    main()
