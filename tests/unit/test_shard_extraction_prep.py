"""End-to-end test for scripts/shard_extraction_prep.py's build_shards --
partitions a real EXTRACT_TEXT_FACTS backlog into isolated on-disk SQLite
shard copies. See test_merge_extraction_shards.py for the merge side of
this same shard/merge design (cluster/README.md's "Speeding up
extraction")."""
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from shard_extraction_prep import build_shards  # noqa: E402

from fair_ocean_agent.config import reset_config_cache
from fair_ocean_agent.database.enums import IdentifierType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Base, ExternalIdentifier, Study, Task
from fair_ocean_agent.database.session import reset_engine_cache


@pytest.fixture
def main_db(tmp_path, monkeypatch):
    main_path = tmp_path / "main.db"
    engine = create_engine(f"sqlite:///{main_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    studies = [Study(title=f"Study {i}") for i in range(3)]
    session.add_all(studies)
    session.flush()
    for study in studies:
        session.add(
            ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.PMCID.value, identifier_value=f"PMC{study.study_id[-6:]}")
        )
    session.commit()
    session.close()
    engine.dispose()

    monkeypatch.setenv("FAIR_OCEAN_DATABASE_URL", f"sqlite:///{main_path}")
    reset_config_cache()
    reset_engine_cache()
    yield main_path
    reset_engine_cache()
    reset_config_cache()


def test_build_shards_partitions_the_backlog_across_isolated_copies(main_db, tmp_path):
    shard_dir = tmp_path / "shard_dbs"
    manifest = build_shards(shard_count=2, shard_dir=shard_dir)

    assert len(manifest.shards) == 2
    total_tasks = sum(len(shard.task_ids) for shard in manifest.shards)
    assert total_tasks == 3  # one EXTRACT_TEXT_FACTS task per study with a PMCID
    assert set(manifest.all_study_ids()) == {sid for shard in manifest.shards for sid in shard.study_ids}

    # Every shard's own copy contains ONLY its own slice of the backlog --
    # not the whole thing (that's the entire point of partitioning).
    for shard in manifest.shards:
        conn = sqlite3.connect(shard.db_path)
        try:
            rows = conn.execute(
                "SELECT task_id FROM tasks WHERE task_type = ? AND status = ?",
                (TaskType.EXTRACT_TEXT_FACTS.value, TaskStatus.PENDING.value),
            ).fetchall()
        finally:
            conn.close()
        assert sorted(r[0] for r in rows) == sorted(shard.task_ids)

    # Disjoint across shards -- no task double-counted.
    all_ids = [task_id for shard in manifest.shards for task_id in shard.task_ids]
    assert len(all_ids) == len(set(all_ids))


def test_build_shards_raises_when_nothing_is_pending(main_db, tmp_path):
    # First pass enqueues and shards the real backlog; mark every task
    # COMPLETED (not deleted -- enqueue_text_extraction_backfill's own
    # idempotency would otherwise just recreate a fresh pending row for
    # the same study on the next call) so a second pass finds nothing left.
    build_shards(shard_count=2, shard_dir=tmp_path / "shard_dbs_1")

    engine = create_engine(f"sqlite:///{main_db}", connect_args={"check_same_thread": False})
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    for task in session.query(Task).filter_by(task_type=TaskType.EXTRACT_TEXT_FACTS.value):
        task.status = TaskStatus.COMPLETED.value
    session.commit()
    session.close()
    engine.dispose()

    with pytest.raises(SystemExit):
        build_shards(shard_count=2, shard_dir=tmp_path / "shard_dbs_2")
