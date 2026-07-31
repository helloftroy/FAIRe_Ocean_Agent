"""Exercise the prepared-source-text migration on a scratch SQLite DB."""
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


def test_prepared_source_text_migration_upgrades_and_downgrades(tmp_path):
    db_path = tmp_path / "prepared-text.db"
    _run_alembic("upgrade", "c31f0d8a62b4", db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(prepared_source_texts)").fetchall()
        }
    finally:
        conn.close()
    assert {
        "prepared_source_text_id",
        "text_content",
        "content_hash",
        "llm_model_name",
        "llm_prompt_version",
        "llm_extracted_at",
    }.issubset(columns)

    _run_alembic("downgrade", "87d37c7b8fe7", db_path=db_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='prepared_source_texts'"
        ).fetchall()
    finally:
        conn.close()
    assert tables == []
