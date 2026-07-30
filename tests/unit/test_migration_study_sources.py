"""Exercises the real `alembic upgrade`/`downgrade` for
87d37c7b8fe7_add_study_sources.py end to end (subprocess, matching how a
human/deploy script actually runs it) against a scratch SQLite database
seeded with a couple of pre-existing `sources` rows, confirming the
one-time backfill produces exactly one `study_sources` row per `sources`
row with the documented relationship_type/confidence defaults and
timestamps carried forward from each source row's own."""
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_alembic(*args: str, db_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env={"FAIR_OCEAN_DATABASE_URL": f"sqlite:///{db_path}", "PATH": "/usr/bin:/bin"},
        check=True,
        capture_output=True,
    )


@pytest.fixture
def scratch_db(tmp_path) -> Path:
    db_path = tmp_path / "scratch.db"
    _run_alembic("upgrade", "5b9e1d3c7a20", db_path=db_path)  # the revision just before this one
    return db_path


def _seed_source(db_path: Path, *, study_id: str, source_id: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO studies (study_id, canonical_status, marine_relevance_status, molecular_relevance_status, "
            "review_status, created_at, updated_at) VALUES (?, 'candidate', 'unknown', 'unknown', 'unreviewed', "
            "'2026-01-01 00:00:00', '2026-01-01 00:00:00')",
            (study_id,),
        )
        conn.execute(
            "INSERT INTO sources (source_id, study_id, source_type, source_name, access_status, "
            "inspection_status, inspection_level, is_mirror, fulltext_available, created_at, updated_at) "
            "VALUES (?, ?, 'repository_api', 'ena', 'unknown', 'not_inspected', 'none', 0, 0, "
            "'2026-02-03 04:05:06', '2026-02-03 04:05:06')",
            (source_id, study_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_migration_backfills_one_study_source_row_per_existing_source(scratch_db):
    study_id = f"STUDY-{uuid.uuid4().hex[:12]}"
    source_id = f"SRC-{uuid.uuid4().hex[:12]}"
    _seed_source(scratch_db, study_id=study_id, source_id=source_id)

    _run_alembic("upgrade", "87d37c7b8fe7", db_path=scratch_db)

    conn = sqlite3.connect(scratch_db)
    try:
        rows = conn.execute(
            "SELECT study_id, source_id, relationship_type, confidence, created_at, updated_at "
            "FROM study_sources WHERE source_id = ?",
            (source_id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row_study_id, row_source_id, relationship_type, confidence, created_at, updated_at = rows[0]
    assert row_study_id == study_id
    assert row_source_id == source_id
    assert relationship_type == "is_home_of"
    assert confidence == "structured_source"
    assert created_at.startswith("2026-02-03 04:05:06")
    assert updated_at.startswith("2026-02-03 04:05:06")


def test_migration_downgrade_drops_the_table(scratch_db):
    study_id = f"STUDY-{uuid.uuid4().hex[:12]}"
    source_id = f"SRC-{uuid.uuid4().hex[:12]}"
    _seed_source(scratch_db, study_id=study_id, source_id=source_id)

    _run_alembic("upgrade", "87d37c7b8fe7", db_path=scratch_db)
    _run_alembic("downgrade", "5b9e1d3c7a20", db_path=scratch_db)

    conn = sqlite3.connect(scratch_db)
    try:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='study_sources'"
        ).fetchall()
    finally:
        conn.close()
    assert result == []
