#!/usr/bin/env python3
"""Reports the current state of the ENA seed-discovery database
(data/seed_discovery/mgnify_paper_seeds.sqlite by default) -- run counts,
study counts by resolution status, and crawl_state's own per-partition
progress. Read-only, DB-only, no network calls.

This is the most reliable way to check whether a long-running
cluster/run_ena_discovery.sbatch or cluster/run_ena_publication_resolution.sbatch
job is actually making progress, independent of its own log file: none of
those cluster/*.sbatch scripts used to set PYTHONUNBUFFERED, so a job's
own periodic "ENA discovery progress .../ENA aggregation progress .../
resolved ENA ..." log lines (seed_discovery/ena_discovery.py) could sit in
Python's stdout buffer for the job's ENTIRE runtime without ever reaching
the .out file -- looking exactly like a silently-stuck job even when it's
working fine. That's now fixed going forward (every cluster/*.sbatch
script exports it), but a currently-running job started before that fix
won't pick it up without being resubmitted. Until then, this script reads
the database directly instead of waiting on the log file.

Run this twice, a few minutes apart, while a job is running: if the
counts and crawl_state's own last_successful_request/updated_at
timestamps are advancing, discovery is actively working -- if everything
is frozen, something really is stuck.

Usage:
    python scripts/check_ena_discovery_progress.py
    python scripts/check_ena_discovery_progress.py --json
    python scripts/check_ena_discovery_progress.py --db /path/to/other.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data/seed_discovery/mgnify_paper_seeds.sqlite")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _counts_by_column(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    return {
        row[0]: row[1]
        for row in conn.execute(f"SELECT {column}, count(*) FROM {table} GROUP BY {column} ORDER BY 2 DESC")
    }


def build_report(db_path: Path) -> dict:
    if not db_path.exists():
        return {"db_path": str(db_path), "exists": False}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    report: dict = {"db_path": str(db_path), "exists": True, "size_mb": round(db_path.stat().st_size / 1024 / 1024, 1)}

    if _table_exists(conn, "ena_runs"):
        report["ena_runs_total"] = conn.execute("SELECT count(*) FROM ena_runs").fetchone()[0]
        report["ena_runs_by_accessibility_status"] = _counts_by_column(conn, "ena_runs", "sequence_accessibility_status")

    if _table_exists(conn, "ena_studies"):
        report["ena_studies_total"] = conn.execute("SELECT count(*) FROM ena_studies").fetchone()[0]
        report["ena_studies_by_bioproject_status"] = _counts_by_column(conn, "ena_studies", "bioproject_status")
        report["ena_studies_by_publication_resolution_status"] = _counts_by_column(
            conn, "ena_studies", "publication_resolution_status"
        )
        report["ena_studies_by_accessibility_status"] = _counts_by_column(conn, "ena_studies", "sequence_accessibility_status")

    if _table_exists(conn, "publication_candidates"):
        report["publication_candidates_total"] = conn.execute("SELECT count(*) FROM publication_candidates").fetchone()[0]

    if _table_exists(conn, "crawl_state"):
        report["crawl_state"] = [
            {
                "source": row["source"],
                "status": row["status"],
                "cursor": row["cursor"],
                "last_successful_request": row["last_successful_request"],
                "last_run_started": row["last_run_started"],
                "last_run_completed": row["last_run_completed"],
                "updated_at": row["updated_at"],
                "error": row["error"],
            }
            for row in conn.execute(
                "SELECT source, status, cursor, last_successful_request, last_run_started, "
                "last_run_completed, updated_at, error FROM crawl_state ORDER BY updated_at DESC"
            )
        ]

    conn.close()
    return report


def render_text(report: dict) -> str:
    if not report.get("exists"):
        return f"Database not found at {report['db_path']}."

    lines = [f"Database: {report['db_path']} ({report['size_mb']} MB)", ""]

    if "ena_runs_total" in report:
        lines.append(f"ENA runs discovered: {report['ena_runs_total']}")
        for status, count in report["ena_runs_by_accessibility_status"].items():
            lines.append(f"  {status}: {count}")
        lines.append("")

    if "ena_studies_total" in report:
        lines.append(f"ENA candidate studies (aggregated from runs): {report['ena_studies_total']}")
        lines.append("  by bioproject_status:")
        for status, count in report["ena_studies_by_bioproject_status"].items():
            lines.append(f"    {status}: {count}")
        lines.append("  by publication_resolution_status:")
        for status, count in report["ena_studies_by_publication_resolution_status"].items():
            lines.append(f"    {status}: {count}")
        lines.append("")

    if "publication_candidates_total" in report:
        lines.append(f"Publication candidates matched: {report['publication_candidates_total']}")
        lines.append("")

    if "crawl_state" in report:
        lines.append("Crawl state, most recently updated first (compare across two runs of this script to see if")
        lines.append("last_successful_request/updated_at are actually advancing):")
        for row in report["crawl_state"]:
            lines.append(f"  {row['source']}: status={row['status']} updated_at={row['updated_at']}")
            lines.append(f"    last_successful_request={row['last_successful_request']} cursor={row['cursor']}")
            if row["error"]:
                lines.append(f"    error={row['error']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"path to the seed-discovery SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a text report")
    args = parser.parse_args()

    report = build_report(args.db)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
