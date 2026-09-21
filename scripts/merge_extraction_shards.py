#!/usr/bin/env python3
"""Merges every shard database produced by shard_extraction_prep.py back
into the main shared database, in a single process (no cross-node file
locking involved -- see that script's own docstring for why this two-step
shard/merge design exists at all).

Merge order per shard:
0. Release any stale claims left in the SHARD'S OWN isolated database --
   real gap found live: an array task killed mid-task (SIGTERM, preemption,
   node failure) leaves whatever it was processing stuck 'claimed'/
   'running' there, which nothing else would ever reclaim, and which step 4
   below would otherwise copy verbatim into the shared main database as a
   permanently-orphaned claim. See _release_stale_claims_in_shard's own
   comment for why 0 minutes is the correct staleness window here.

Then, all via one raw sqlite3 connection to the main database with the
shard ATTACHed:
1. `entities`: INSERT OR IGNORE. A shareable-level entity (SAMPLE/
   EXPERIMENT_RUN/SEQUENCING_RUN) this shard created independently for an
   accession another shard *also* independently created an entity for
   will silently lose this INSERT to `entities`' own partial unique index
   on (entity_level, external_identifier) -- expected, not an error.
2. Compute a per-shard entity_id_map (this shard's own entity_id -> the
   entity_id that actually won in main for the same identity) and use it
   to fix up THIS SHARD'S OWN copies of every table with an entity_id
   column, in place, before copying them -- so every subsequent copy step
   is a plain, safe `INSERT OR IGNORE ... SELECT *` with no dangling
   references to an entity_id that lost step 1 and doesn't exist in main.
3. Plain `INSERT OR IGNORE ... SELECT *` for sources, study_sources,
   entity_relationships, entity_studies, raw_facts, api_paper_corrections,
   and any brand-new tasks rows (e.g. DISCOVER_PRIMER_REFERENCE_STUDIES --
   its own idempotency_key UNIQUE constraint naturally dedupes two shards
   independently creating "the same" task).
4. Sync mutated columns via UPDATE ... FROM: raw_facts.review_status
   (quarantine flags this shard's own processing set on pre-existing
   facts), and this shard's OWN assigned EXTRACT_TEXT_FACTS tasks' status/
   attempt/claim lifecycle columns.

After every shard is merged: re-run map_study_to_faire for every touched
study_id against the now-fully-merged main database. This is deliberate,
not an afterthought -- map_study_to_faire runs exactly once per
EXTRACT_TEXT_FACTS task, *before* that task's own new facts are written,
so standardized_values/standardized_value_evidence are stale relative to
that task's own output even without sharding. Re-running it here, once,
with full cross-shard visibility, is simpler and more correct than trying
to merge those two (fully derived, delete-then-recreate-every-time) tables
directly -- it also self-heals the one real cross-study staleness this
design can introduce (resolve_primer_sequences_from_corpus's whole-corpus
primer lookup, and shared-entity-derived facts, both only ever see
whatever existed in a shard's isolated snapshot at partition time).

Real gap found live: at real scale (1702 touched studies on a real
cluster run) this remap pass legitimately takes hours, not minutes -- each
study runs several of its own DB round trips (entity/primer/sample-alias
resolution, a per-fact loop, a full delete-then-recreate of its
standardized_values), and Lustre's per-query latency adds up across
thousands of them. A single commit for the whole loop made that
indistinguishable from a genuine hang (nothing prints between studies) and
would have lost all 1702 studies' worth of work on any interruption. Each
study now commits (and prints) as soon as it finishes, so progress is
visible in real time and only the one study in flight at the moment of an
interruption is ever at risk.

Usage:
    python scripts/merge_extraction_shards.py
    python scripts/merge_extraction_shards.py --manifest data/shard_dbs/manifest.json
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _extraction_sharding import DEFAULT_MANIFEST_PATH, ShardManifest, with_lock_retry

from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.mapping.faire import map_study_to_faire
from fair_ocean_agent.workflow.task_queue import release_stale_claims

_SHAREABLE_ENTITY_LEVELS_SQL = ("'sample'", "'experiment_run'", "'sequencing_run'")

# Plain insert-or-ignore tables, in an order that keeps the entity_id_map
# fixups (below) applied before anything that depends on entity_id gets
# copied. new_id()'s own collision-free UUID scheme (database/ids.py)
# means every one of these is safe as a blind SELECT * once the shard's
# own entity_id columns have been corrected in place.
_BULK_COPY_TABLES = (
    "sources",
    "study_sources",
    "entity_relationships",
    "entity_studies",
    "raw_facts",
    "api_paper_corrections",
    "tasks",
)

# (table, column) pairs whose entity_id needs redirecting via the map
# before that table gets bulk-copied.
_ENTITY_ID_COLUMNS = (
    ("entity_relationships", "from_entity_id"),
    ("entity_relationships", "to_entity_id"),
    ("entity_studies", "entity_id"),
    ("raw_facts", "entity_id"),
    ("api_paper_corrections", "entity_id"),
)


def _merge_entities_and_build_id_map(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT OR IGNORE INTO main.entities SELECT * FROM shard.entities")
    conn.execute("DROP TABLE IF EXISTS temp.entity_id_map")
    conn.execute(
        f"""
        CREATE TEMP TABLE entity_id_map AS
        SELECT s.entity_id AS old_id,
               COALESCE(
                   (SELECT m.entity_id FROM main.entities m WHERE m.entity_id = s.entity_id),
                   (SELECT m.entity_id FROM main.entities m
                    WHERE m.entity_level = s.entity_level
                      AND m.external_identifier = s.external_identifier
                      AND s.entity_level IN ({",".join(_SHAREABLE_ENTITY_LEVELS_SQL)}))
               ) AS new_id
        FROM shard.entities s
        """
    )


def _redirect_shard_entity_ids(conn: sqlite3.Connection) -> None:
    """Fixes up the ATTACHED shard's own copies of dependent tables in
    place, before they get bulk-copied -- see module docstring step 2."""
    for table, column in _ENTITY_ID_COLUMNS:
        conn.execute(
            f"""
            UPDATE shard.{table}
            SET {column} = (SELECT new_id FROM entity_id_map WHERE old_id = shard.{table}.{column})
            WHERE {column} IN (
                SELECT old_id FROM entity_id_map WHERE new_id IS NOT NULL AND old_id != new_id
            )
            """
        )


def _bulk_copy_remaining_tables(conn: sqlite3.Connection) -> None:
    for table in _BULK_COPY_TABLES:
        conn.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM shard.{table}")


def _sync_raw_fact_review_status(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE main.raw_facts
        SET review_status = (SELECT s.review_status FROM shard.raw_facts s WHERE s.fact_id = main.raw_facts.fact_id)
        WHERE fact_id IN (
            SELECT s.fact_id FROM shard.raw_facts s
            WHERE s.fact_id = main.raw_facts.fact_id AND s.review_status != main.raw_facts.review_status
        )
        """
    )


_TASK_LIFECYCLE_COLUMNS = (
    "status", "attempt_count", "claimed_by", "claimed_at", "started_at", "completed_at", "last_error",
)


def _sync_task_lifecycle(conn: sqlite3.Connection, task_ids: list[str]) -> None:
    if not task_ids:
        return
    placeholders = ",".join("?" for _ in task_ids)
    set_clause = ", ".join(
        f"{column} = (SELECT s.{column} FROM shard.tasks s WHERE s.task_id = main.tasks.task_id)"
        for column in _TASK_LIFECYCLE_COLUMNS
    )
    conn.execute(
        f"UPDATE main.tasks SET {set_clause} WHERE task_id IN ({placeholders})",
        task_ids,
    )


def _release_stale_claims_in_shard(shard_db_path: str) -> int:
    """Real gap found live: a shard's own array task can be killed (e.g. by
    a SLURM SIGTERM, a preemption, a node failure) mid-task, leaving
    whatever it was actively processing stuck in 'claimed'/'running' inside
    THAT SHARD's own isolated database -- claim_next_task never reclaims
    those statuses on its own (see task_queue.py's own comment), so nothing
    would ever pick that task back up. Worse, _sync_task_lifecycle below
    copies a shard's task state verbatim into the shared main database, so
    merging as-is would silently bake a permanently-orphaned claim into it.
    By the time this runs, whatever array task owned this shard is long
    since finished (successfully or not) -- there is no live process left
    that could still be legitimately holding a claim in an ISOLATED shard
    file the way there might be in the shared main database, so 0 minutes
    is the correct staleness window here, not the 30-minute default used
    elsewhere for a database still being actively worked."""
    engine = create_engine(f"sqlite:///{shard_db_path}")
    try:
        session = sessionmaker(bind=engine)()
        try:
            released = release_stale_claims(session, stale_after_minutes=0)
            session.commit()
            return released
        finally:
            session.close()
    finally:
        engine.dispose()


def merge_shard(conn: sqlite3.Connection, shard_db_path: str, task_ids: list[str]) -> None:
    conn.execute("ATTACH DATABASE ? AS shard", (shard_db_path,))
    try:
        conn.execute("BEGIN")
        _merge_entities_and_build_id_map(conn)
        _redirect_shard_entity_ids(conn)
        _bulk_copy_remaining_tables(conn)
        _sync_raw_fact_review_status(conn)
        _sync_task_lifecycle(conn, task_ids)
        conn.execute("DROP TABLE IF EXISTS temp.entity_id_map")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("DETACH DATABASE shard")


def _remap_one_study(session, study_id: str) -> None:
    map_study_to_faire(session, study_id)
    # Real gap found live: with a single commit for the WHOLE loop, a real
    # 1702-study remap on the cluster ran silently for 1.5+ hours with zero
    # visible progress (nothing prints between studies) and would have lost
    # ALL of it on any interruption (SSH drop, SIGTERM, a transient locking
    # error) since nothing had actually been committed yet. Committing here,
    # per study, makes each one durable as soon as it finishes -- a later
    # interruption only ever loses the one study in flight, not the whole
    # run -- and is what makes the print below a real, truthful progress
    # signal rather than one big opaque transaction.
    session.commit()


def _remap_touched_studies(study_ids: list[str]) -> None:
    total = len(study_ids)
    started_all = time.monotonic()
    with session_scope() as session:
        for index, study_id in enumerate(study_ids, start=1):
            started = time.monotonic()
            with_lock_retry(_remap_one_study, session, study_id)
            elapsed = time.monotonic() - started
            total_elapsed = time.monotonic() - started_all
            print(
                f"  [{index}/{total}] re-mapped {study_id} ({elapsed:.1f}s, {total_elapsed / 60:.1f}m elapsed total)",
                flush=True,
            )


def merge_all_shards(manifest: ShardManifest) -> None:
    # Every real DB touch below goes through with_lock_retry -- see
    # _extraction_sharding.py's own comment: this cluster's Lustre-backed
    # scratch mount can raise a transient "locking protocol" error even
    # from a single, solitary process, almost certainly because some
    # OTHER process is concurrently touching the same shared database
    # file right now.
    conn = sqlite3.connect(manifest.main_db_path)
    try:
        for shard in manifest.shards:
            released = with_lock_retry(_release_stale_claims_in_shard, shard.db_path)
            if released:
                print(
                    f"shard {shard.shard_index}: released {released} orphaned claim(s) left by a "
                    "killed/crashed array task (e.g. a SIGTERM before it could finish) -- reset to "
                    "retry_pending/manual_review_required so they are not lost."
                )
            print(f"merging shard {shard.shard_index} ({shard.db_path}, {len(shard.task_ids)} task(s))...")
            with_lock_retry(merge_shard, conn, shard.db_path, shard.task_ids)
    finally:
        conn.close()

    study_ids = manifest.all_study_ids()
    print(f"re-mapping {len(study_ids)} touched stud(y/ies) with full cross-shard visibility...")
    with_lock_retry(_remap_touched_studies, study_ids)


def _archive_shards(manifest: ShardManifest, shard_dir: Path) -> None:
    archive_dir = shard_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for shard in manifest.shards:
        src = Path(shard.db_path)
        if src.exists():
            shutil.move(str(src), str(archive_dir / src.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--no-archive", action="store_true",
        help="leave shard_*.db files where they are instead of moving them to shard_dbs/archive/ after a successful merge",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"no manifest at {args.manifest} -- run shard_extraction_prep.py first")
    manifest = ShardManifest.read(args.manifest)

    merge_all_shards(manifest)

    if not args.no_archive:
        _archive_shards(manifest, args.manifest.parent)
        print(f"merged shard DBs moved to {args.manifest.parent / 'archive'} for inspection/recovery.")

    print("Merge complete.")


if __name__ == "__main__":
    main()
