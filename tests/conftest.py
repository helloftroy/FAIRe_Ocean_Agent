import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fair_ocean_agent.config import RetrievalConfig
from fair_ocean_agent.database.models import Base


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    """cache_enabled=False so adapter tests never touch data/cache/ on disk."""
    return RetrievalConfig(
        request_timeout_seconds=5,
        cache_enabled=False,
        user_agent="fair-ocean-agent-tests/0.1 (contact: test@example.org)",
    )


@pytest.fixture
def db_session():
    """Fresh in-memory SQLite database per test -- fast, isolated, no shared
    state between tests (unlike the process-wide engine in
    database.session, which is for the CLI/worker, not tests)."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
