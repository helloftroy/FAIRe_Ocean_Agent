"""Real multi-process concurrency test for claim_next_task's SQLite path.

Runs actual separate OS processes (not threads -- this bug class only
manifests across genuinely separate connections/transactions, which a
single-process or in-memory-database test can't reproduce) against a real
on-disk SQLite file, mirroring exactly how several parallel SLURM jobs
sharing one data/fair_ocean.db would hit it.
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import TaskStatus, TaskType
from fair_ocean_agent.database.models import Base, Task
from fair_ocean_agent.workflow.task_queue import claim_next_task


def _engine_for(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    return engine


def _claim_until_empty(db_path_str: str, worker_id: str, result_path_str: str) -> None:
    """Module-level (not nested/lambda) so multiprocessing's spawn start
    method -- the default on macOS, and available everywhere -- can
    re-import it in the child process."""
    engine = _engine_for(Path(db_path_str))
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    claimed_ids: list[str] = []
    with session_factory() as session:
        while True:
            task = claim_next_task(session, worker_id=worker_id)
            session.commit()
            if task is None:
                break
            claimed_ids.append(task.task_id)
    engine.dispose()
    Path(result_path_str).write_text("\n".join(claimed_ids))


def test_claim_next_task_has_no_double_claims_under_real_multiprocess_contention(tmp_path):
    """Real gap found live: the old SQLite claim path (a plain SELECT,
    then a SEPARATE UPDATE via ORM attribute mutation + flush) produced
    416 duplicate claims out of 500 tasks under real 8-process contention
    -- a severe correctness bug (duplicated LLM extraction work, and risk
    of conflicting writes for the same study), not just a performance one,
    for the multi-job parallel-extraction workflow this pipeline needs
    when splitting a large backlog across several 48h SLURM jobs.

    Exercises the actual fix (task_queue.py's single atomic
    UPDATE ... RETURNING claim, combined with the WAL-mode + busy_timeout
    pragmas database/session.py registers) against real separate OS
    processes hitting a real on-disk file."""
    db_path = tmp_path / "concurrency.db"
    n_tasks = 200
    n_workers = 6

    engine = _engine_for(db_path)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        now = utcnow()
        for i in range(n_tasks):
            session.add(
                Task(
                    task_id=f"TASK-{i:05d}",
                    task_type=TaskType.DISCOVER_IDENTIFIERS.value,
                    study_id=f"STUDY-{i:05d}",
                    status=TaskStatus.PENDING.value,
                    priority=100,
                    max_attempts=3,
                    available_after=now,
                    idempotency_key=f"key-{i}",
                    created_at=now,
                    updated_at=now,
                )
            )
        session.commit()
    engine.dispose()

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    procs = []
    for i in range(n_workers):
        result_path = result_dir / f"worker_{i}.txt"
        p = multiprocessing.Process(
            target=_claim_until_empty,
            args=(str(db_path), f"worker-{i}", str(result_path)),
        )
        procs.append(p)
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert not p.is_alive(), "worker process hung past the 60s timeout"

    all_claimed: list[str] = []
    for result_path in result_dir.glob("*.txt"):
        text = result_path.read_text().strip()
        if text:
            all_claimed.extend(text.splitlines())

    assert len(all_claimed) == n_tasks, f"expected {n_tasks} total claims across all workers, got {len(all_claimed)}"
    assert len(set(all_claimed)) == n_tasks, (
        f"duplicate claim(s) detected across worker processes: "
        f"{len(all_claimed) - len(set(all_claimed))} duplicate(s)"
    )
