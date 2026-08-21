"""Engine/session management. One process-wide engine, created lazily so
importing this module never touches the filesystem or network."""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from fair_ocean_agent.config import REPO_ROOT, load_config

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = load_config().database.url
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            url = _resolve_sqlite_url(url)
        _engine = create_engine(url, connect_args=connect_args, future=True)
    return _engine


def _resolve_sqlite_url(url: str) -> str:
    """Rewrites a relative sqlite:///path.db URL to an absolute one anchored
    at REPO_ROOT, and ensures its parent directory exists.

    A relative path left in the URL is resolved by sqlite3 against the
    *process's current working directory* at connect time, not REPO_ROOT --
    those only coincide if the CLI happens to be invoked from inside the
    repo. This bit a real interactive session: a one-off analysis script
    run with a different cwd silently connected to (and would have
    created) a second, empty database, rather than to the one every
    `fair-ocean` command had been using. A cron job or systemd unit
    (Milestone 7) invoked from a different working directory would hit the
    exact same failure. Postgres URLs are host-qualified and don't have
    this problem, so this only applies to sqlite:// URLs.
    """
    # sqlite:///relative/path.db or sqlite:////absolute/path.db
    path_part = url.split("sqlite:///", 1)[-1]
    db_path = Path(path_part)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: commits on success, rolls back on
    exception, always closes."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Dispose of the cached engine/session factory; used by tests that need
    a fresh in-memory database per test."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def init_db() -> None:
    """Create all tables directly from the models (used for local/dev setup
    and by tests). In Postgres/production, prefer Alembic migrations."""
    from fair_ocean_agent.database.models import Base

    Base.metadata.create_all(get_engine())


def check_schema_drift() -> dict[str, list[str] | bool]:
    """Read-only diagnostic: a database that was ever bootstrapped via
    init_db()/create_all() -- rather than `alembic upgrade head` from the
    start -- can silently drift from the ORM models, because create_all()
    only creates tables that don't exist yet; it never adds a column that a
    later Alembic migration added to an already-existing table. Running
    `alembic upgrade head` directly against such a database fails outright
    (it has no alembic_version row, so Alembic tries to replay every
    migration from the very first one, including CREATE TABLE for tables
    that already exist -- confirmed live).

    Reports, without changing anything: whether the alembic_version tracking
    table exists, and for every table that already exists in the live
    database, which of its current ORM-model columns are missing from it.
    Does not attempt to fix anything or guess an Alembic revision to stamp
    -- what to do about any reported drift depends on exactly what's
    missing, which should be decided deliberately, not automatically,
    against a database holding real production data."""
    from sqlalchemy import inspect

    from fair_ocean_agent.database.models import Base

    engine = get_engine()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    missing_by_table: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        live_columns = {col["name"] for col in inspector.get_columns(table.name)}
        missing = [c.name for c in table.columns if c.name not in live_columns]
        if missing:
            missing_by_table[table.name] = missing

    return {
        "alembic_version_table_present": "alembic_version" in existing_tables,
        "missing_columns_by_table": missing_by_table,
    }


def reset_database(*, backup: bool = True) -> Path | None:
    """DESTRUCTIVE: drops every table -- including alembic_version -- and
    rebuilds an empty schema via `alembic upgrade head`, leaving zero
    studies/entities/facts/tasks. Intended only for active development/
    debugging, where re-running every seed paper from scratch against
    current code is more useful than debugging one study's stale state at
    a time (state accumulated incrementally across many discovery-logic
    changes is otherwise very hard to reason about in isolation).

    Never touches data/cache/ (the on-disk HTTP response cache) or any
    local/auto-fetched PDFs -- those hold real upstream API/publisher
    content, not pipeline-derived state, so there's nothing stale about
    them to reset; re-ingesting from a fresh database will still be fast
    because those responses are still cached.

    For a sqlite:// database, copies the on-disk file to a timestamped
    `<name>.bak.<UTC-timestamp>` sibling before dropping anything, unless
    backup=False, and returns that path. For any other backend, backup is
    the caller's own responsibility (e.g. a managed Postgres snapshot) and
    this always returns None -- there's no single on-disk file to copy."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect, text

    from fair_ocean_agent.database.models import Base

    engine = get_engine()
    backup_path: Path | None = None
    if backup and engine.url.get_backend_name() == "sqlite" and engine.url.database:
        db_path = Path(engine.url.database)
        if db_path.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = db_path.with_name(f"{db_path.name}.bak.{timestamp}")
            shutil.copy2(db_path, backup_path)

    inspector = inspect(engine)
    if "alembic_version" in inspector.get_table_names():
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE alembic_version"))
    Base.metadata.drop_all(engine)

    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

    return backup_path
