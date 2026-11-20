#!/usr/bin/env python3
"""Map the smallest studies first (2026-11-20).

The worker claims tasks by priority (lowest first), then created_at
(task_queue._claim_next_task_sqlite). Mapping time grows with a study's
sample count (~1 s per sample, scripts/profile_map_faire.py), so this sets
each pending MAP_FAIRE task's priority to its study's sample count: small
studies finish first, the giants wait for the speed fix. Only pending /
retry_pending MAP_FAIRE tasks change; other task types keep theirs.

Dry run by default (prints what it would do); --apply writes.

Usage: python scripts/order_map_queue.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import sqlite3

BUCKETS = [0, 100, 1_000, 5_000, 10**9]


def db_path() -> str:
    url = os.environ.get("FAIR_OCEAN_DATABASE_URL", "")
    return url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else "data/fair_ocean.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{db_path()}" + ("" if a.apply else "?mode=ro"), uri=True, timeout=300)
    tasks = conn.execute("select task_id, study_id from tasks where task_type='MAP_FAIRE' "
                         "and status in ('pending','retry_pending')").fetchall()
    samples = dict(conn.execute(
        "select es.study_id, count(*) from entity_studies es join entities e using (entity_id) "
        "where e.entity_level='sample' group by es.study_id").fetchall())
    new = [(samples.get(s, 0), t) for t, s in tasks]
    print(f"{len(new):,} pending MAP_FAIRE tasks")
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        n = [p for p, _ in new if lo <= p < hi]
        print(f"  {lo:>6,}-{(f'{hi:,}' if hi < 10**9 else ''):<6} samples: {len(n):>6,} studies, {sum(n):>9,} samples "
              f"(~{sum(n) / 3600:.0f} h at 1 s/sample)")
    if a.apply:
        conn.executemany("update tasks set priority=? where task_id=?", new)
        conn.commit()
        print("priorities set: smallest studies are claimed first")
    else:
        print("dry run; --apply to write")


if __name__ == "__main__":
    main()
