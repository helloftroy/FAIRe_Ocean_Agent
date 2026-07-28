"""Engine/session management. One process-wide engine, created lazily so
importing this module never touches the filesystem or network."""
from __future__ import annotations

from contextlib import contextmanager
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
