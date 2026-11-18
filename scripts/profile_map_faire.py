#!/usr/bin/env python3
"""Why is MAP_FAIRE slow? (2026-11-18: 280 studies mapped in 48 h, 9,218 left)

Read-only by default. Prints:
  1. the finished MAP_FAIRE tasks of the last 7 days: duration vs the
     study's raw facts and samples (slowest 15, and time per size bucket);
  2. the pending MAP_FAIRE studies by the same size buckets, with an
     estimate of the time left at the observed rate per bucket.

With --profile STUDY_ID it also maps that one study under cProfile and
ROLLS BACK (nothing is saved), printing the 30 most expensive calls. Run
it only while no worker job is writing to the database.

Usage: python scripts/profile_map_faire.py [--profile STUDY_ID]
"""
from __future__ import annotations

import argparse
import collections
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BUCKETS = [(0, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000), (1_000_000, 10**12)]


def db_path() -> str:
    url = os.environ.get("FAIR_OCEAN_DATABASE_URL", "")
    return url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else "data/fair_ocean.db"


def ts(value) -> datetime | None:
    if not value:
        return None
    d = datetime.fromisoformat(str(value))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def bucket(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n < hi:
            return f"{lo:>9,}-{hi:,}" if hi < 10**12 else f"{lo:>9,}+"
    return "?"


def sizes(conn, study_ids) -> dict[str, tuple[int, int]]:
    out = {}
    for s in study_ids:
        facts = conn.execute("select count(*) from raw_facts where study_id=?", (s,)).fetchone()[0]
        samples = conn.execute("select count(*) from entity_studies es join entities e using (entity_id) "
                               "where es.study_id=? and e.entity_level='sample'", (s,)).fetchone()[0]
        out[s] = (facts, samples)
    return out


def report() -> None:
    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    done = [(s, ts(a), ts(b)) for s, a, b in conn.execute(
        "select study_id, started_at, completed_at from tasks where task_type='MAP_FAIRE' "
        "and status='completed' and completed_at >= ?", (since,))]
    done = [(s, (b - a).total_seconds()) for s, a, b in done if a and b]
    pending = [s for (s,) in conn.execute(
        "select study_id from tasks where task_type='MAP_FAIRE' and status in ('pending','retry_pending','running')")]
    t0 = time.time()
    size = sizes(conn, {s for s, _ in done} | set(pending))
    print(f"(sizes counted in {time.time() - t0:.0f}s)\n")

    print(f"finished in the last 7 days: {len(done)} tasks, {sum(d for _, d in done) / 3600:.1f} h total")
    print(f"{'seconds':>9} {'raw_facts':>10} {'samples':>8}  study")
    for s, d in sorted(done, key=lambda x: -x[1])[:15]:
        print(f"{d:>9.0f} {size[s][0]:>10,} {size[s][1]:>8,}  {s}")

    rate: dict[str, list[float]] = collections.defaultdict(list)
    for s, d in done:
        rate[bucket(size[s][0])].append(d / max(size[s][0], 1))
    pend: dict[str, list[int]] = collections.defaultdict(list)
    for s in pending:
        pend[bucket(size[s][0])].append(size[s][0])

    print(f"\nby raw facts per study      done  median s  s/1k facts | pending  facts       est. hours")
    total_h = 0.0
    for lo, hi in BUCKETS:
        b = bucket(lo)
        r = sorted(rate.get(b, []))
        per = r[len(r) // 2] if r else None
        n_done = len(r)
        med_s = sorted(d for s, d in done if bucket(size[s][0]) == b)
        med = med_s[len(med_s) // 2] if med_s else 0
        facts = sum(pend.get(b, []))
        est = facts * per / 3600 if per is not None else None
        total_h += est or 0
        print(f"{b:<27} {n_done:>5} {med:>9.0f} {per * 1000 if per else 0:>11.2f} | {len(pend.get(b, [])):>7} "
              f"{facts:>12,} {('%.0f' % est) if est is not None else 'no data':>10}")
    print(f"estimated total for pending (one worker): {total_h:.0f} h = {total_h / 24:.1f} days")
    biggest = sorted(pending, key=lambda s: -size[s][0])[:10]
    print("\nbiggest pending: " + ", ".join(f"{s} ({size[s][0]:,} facts, {size[s][1]:,} samples)" for s in biggest))


def profile(study_id: str) -> None:
    import cProfile
    import pstats

    from fair_ocean_agent.database.session import get_session_factory
    from fair_ocean_agent.mapping.faire import map_study_to_faire

    session = get_session_factory()()
    prof = cProfile.Profile()
    t0 = time.time()
    try:
        prof.enable()
        created = map_study_to_faire(session, study_id)
        prof.disable()
    finally:
        session.rollback()
        session.close()
    print(f"\n{study_id}: {created:,} values in {time.time() - t0:.0f}s (rolled back, nothing saved)")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(30)
    pstats.Stats(prof).sort_stats("tottime").print_stats(15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", metavar="STUDY_ID")
    ap.add_argument("--skip-report", action="store_true")
    a = ap.parse_args()
    if not a.skip_report:
        report()
    if a.profile:
        profile(a.profile)


if __name__ == "__main__":
    main()
