#!/usr/bin/env python3
"""Partitions the pending EXTRACT_TEXT_FACTS backlog across N isolated
local SQLite copies of the main database, so several parallel SLURM array
tasks can each process their own slice without ever touching the same
file concurrently.

Why this exists: `cluster/run_extraction_parallel.sbatch`'s job-array mode
(N array tasks all pointed at the SAME shared database) failed on every
array task with `sqlite3.OperationalError: locking protocol` on a plain
SELECT -- confirmed live that this cluster's Lustre-backed `/scratch`
mount doesn't reliably coordinate SQLite file locks across DIFFERENT
COMPUTE NODES. Each shard produced here is a full, independent snapshot of
the main database (via SQLite's own backup API, which is correct
regardless of journal mode -- a plain file copy of a live database is
not), filtered so it contains only ITS OWN slice of the pending backlog --
without that filtering, every shard would contain (and redundantly
reprocess) the entire backlog. Run `merge_extraction_shards.py` after
every array task finishes to stitch the results back into the main
database.

Usage:
    python scripts/shard_extraction_prep.py --shards 5
    python scripts/shard_extraction_prep.py --shards 5 --manifest data/shard_dbs/manifest.json
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from sqlalchemy import select

from _extraction_sharding import DEFAULT_MANIFEST_PATH, ShardManifest, ShardManifestEntry, sqlite_path_for_url

from fair_ocean_agent.config import REPO_ROOT, load_config
from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Task
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.workflow.handlers import enqueue_text_extraction_backfill

DEFAULT_SHARD_DIR = REPO_ROOT / "data" / "shard_dbs"


def _pending_extract_text_facts_tasks() -> list[tuple[str, str | None]]:
    """Returns (task_id, study_id) pairs, oldest first, for a deterministic
    partition regardless of how many times this is re-run."""
    with session_scope() as session:
        rows = session.execute(
            select(Task.task_id, Task.study_id)
            .where(Task.task_type == TaskType.EXTRACT_TEXT_FACTS.value, Task.status == TaskStatus.PENDING.value)
            .order_by(Task.created_at, Task.task_id)
        ).all()
        return [(task_id, study_id) for task_id, study_id in rows]


def _partition(tasks: list[tuple[str, str | None]], shard_count: int) -> list[list[tuple[str, str | None]]]:
    shards: list[list[tuple[str, str | None]]] = [[] for _ in range(shard_count)]
    for index, task in enumerate(tasks):
        shards[index % shard_count].append(task)
    return shards


def _snapshot_via_backup_api(main_path: Path, shard_path: Path) -> None:
    """A plain file copy (`cp`/`shutil.copy`) of a LIVE SQLite database can
    miss recent writes still sitting in a WAL/journal file, or copy a
    database mid-write. SQLite's own backup API (used here via the
    stdlib's sqlite3 binding) produces a correct, consistent snapshot
    regardless of journal mode -- the right way to copy a database that
    might still be open elsewhere."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if shard_path.exists():
        shard_path.unlink()
    source_conn = sqlite3.connect(str(main_path))
    dest_conn = sqlite3.connect(str(shard_path))
    try:
        source_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        source_conn.close()


def _filter_shard_to_its_own_tasks(shard_path: Path, keep_task_ids: set[str]) -> None:
    """Removes every OTHER shard's still-pending EXTRACT_TEXT_FACTS task
    from this shard's own copy -- the step that actually partitions the
    work. Without it, every shard would contain the entire backlog and
    every array task would redundantly reprocess everything."""
    conn = sqlite3.connect(str(shard_path))
    try:
        placeholders = ",".join("?" for _ in keep_task_ids)
        conn.execute(
            f"DELETE FROM tasks WHERE task_type = ? AND status = ? AND task_id NOT IN ({placeholders})",
            (TaskType.EXTRACT_TEXT_FACTS.value, TaskStatus.PENDING.value, *keep_task_ids),
        )
        conn.commit()
    finally:
        conn.close()


def build_shards(shard_count: int, shard_dir: Path) -> ShardManifest:
    with session_scope() as session:
        enqueue_text_extraction_backfill(session)

    tasks = _pending_extract_text_facts_tasks()
    if not tasks:
        raise SystemExit("No pending EXTRACT_TEXT_FACTS tasks found -- nothing to shard.")

    main_url = load_config().database.url
    main_path = sqlite_path_for_url(main_url)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    partitioned = _partition(tasks, shard_count)
    manifest = ShardManifest(main_db_path=str(main_path))
    for shard_index, shard_tasks in enumerate(partitioned, start=1):
        shard_path = shard_dir / f"shard_{shard_index}.db"
        _snapshot_via_backup_api(main_path, shard_path)
        task_ids = {task_id for task_id, _ in shard_tasks}
        _filter_shard_to_its_own_tasks(shard_path, task_ids)
        study_ids = sorted({study_id for _, study_id in shard_tasks if study_id is not None})
        manifest.shards.append(
            ShardManifestEntry(
                shard_index=shard_index,
                db_path=str(shard_path),
                task_ids=sorted(task_ids),
                study_ids=study_ids,
            )
        )
        print(f"shard {shard_index}: {len(shard_tasks)} task(s), {len(study_ids)} stud(y/ies) -> {shard_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, required=True, help="how many shard databases to create")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR, help="where to write shard_<N>.db files")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="where to write the manifest JSON")
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")

    manifest = build_shards(args.shards, args.shard_dir)
    manifest.write(args.manifest)
    total_tasks = sum(len(shard.task_ids) for shard in manifest.shards)
    print(f"\nWrote manifest for {len(manifest.shards)} shard(s), {total_tasks} task(s) total, to {args.manifest}")
    print("Next: submit the array job (see cluster/README.md), then run merge_extraction_shards.py.")


if __name__ == "__main__":
    main()
