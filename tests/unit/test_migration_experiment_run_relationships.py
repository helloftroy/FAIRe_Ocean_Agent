"""Exercise the experiment-run relationship migration on scratch SQLite."""
import sqlite3
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_alembic(*args: str, db_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env={"FAIR_OCEAN_DATABASE_URL": f"sqlite:///{db_path}", "PATH": "/usr/bin:/bin"},
        check=True,
        capture_output=True,
    )


def test_experiment_run_relationship_migration_upgrades_and_downgrades(tmp_path):
    db_path = tmp_path / "experiment-runs.db"
    _run_alembic("upgrade", "e4b7c1d2a930", db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(entity_relationships)").fetchall()
        }
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(entity_relationships)").fetchall()
        }
    finally:
        conn.close()
    assert {
        "entity_relationship_id",
        "study_id",
        "from_entity_id",
        "to_entity_id",
        "relationship_type",
    }.issubset(columns)
    assert "ix_entity_relationships_study_type" in indexes

    _run_alembic("downgrade", "c31f0d8a62b4", db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='entity_relationships'"
        ).fetchall()
    finally:
        conn.close()
    assert tables == []
