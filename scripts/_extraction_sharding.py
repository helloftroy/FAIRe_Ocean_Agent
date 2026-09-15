"""Shared manifest format + helpers for shard_extraction_prep.py and
merge_extraction_shards.py -- see cluster/README.md's "Speeding up
extraction" section for why these two scripts exist (a real Lustre/scratch
mount that can't coordinate SQLite file locks across compute nodes, found
live via a real --array=1-5 job that failed every task on a plain SELECT).

Kept intentionally tiny: just the manifest dataclasses + a path resolver,
no SQL of its own -- each script owns its own SQL so the actual database
operations stay easy to read top-to-bottom in one place.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from sqlalchemy.exc import OperationalError as _SQLAlchemyOperationalError

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.session import _resolve_sqlite_url

DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "shard_dbs" / "manifest.json"

_T = TypeVar("_T")

# Real gap found live: even a lone process on a login node (no concurrent
# array tasks at all) hit "sqlite3.OperationalError: locking protocol" on
# a plain SELECT, on the same Lustre-backed cluster scratch mount that
# made run_extraction_parallel.sbatch's shared-file array mode unusable
# for the same underlying reason (see database/session.py's WAL-fallback
# comment). Since nothing about THIS process was itself concurrent, this
# is almost certainly a still-running discovery/extraction job on some
# OTHER node concurrently touching the same shared database file --
# genuinely transient, timing-dependent contention, not a permanent
# failure. "locking protocol" is SQLite's own message for
# SQLITE_IOERR_LOCK, a lower-level fcntl() failure from the filesystem
# itself -- distinct from SQLITE_BUSY, which PRAGMA busy_timeout
# (database/session.py) already retries automatically on its own, so this
# needs its own explicit retry.
_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_DELAY_SECONDS = 3.0


def _is_transient_locking_protocol_error(exc: BaseException) -> bool:
    return "locking protocol" in str(exc)


def with_lock_retry(fn: Callable[..., _T], *args, attempts: int = _LOCK_RETRY_ATTEMPTS, **kwargs) -> _T:
    """Retries `fn(*args, **kwargs)` with a short linear backoff when it
    fails with SQLite's own "locking protocol" error -- see this module's
    own comment above for why. Any other exception (including a genuine
    SQLITE_BUSY that somehow outlasts busy_timeout) is never retried here,
    just re-raised immediately."""
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (sqlite3.OperationalError, _SQLAlchemyOperationalError) as exc:
            if not _is_transient_locking_protocol_error(exc):
                raise
            last_exc = exc
            if attempt == attempts:
                break
            delay = _LOCK_RETRY_BASE_DELAY_SECONDS * attempt
            print(
                f"transient SQLite locking-protocol error (attempt {attempt}/{attempts}), "
                f"retrying in {delay:.0f}s -- likely another process touching the same "
                f"database file right now: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


@dataclass
class ShardManifestEntry:
    shard_index: int
    db_path: str  # absolute filesystem path, NOT a sqlite:/// URL
    task_ids: list[str] = field(default_factory=list)
    study_ids: list[str] = field(default_factory=list)


@dataclass
class ShardManifest:
    main_db_path: str  # absolute filesystem path this manifest was cut from
    shards: list[ShardManifestEntry] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read(cls, path: Path) -> "ShardManifest":
        raw = json.loads(path.read_text())
        return cls(
            main_db_path=raw["main_db_path"],
            shards=[ShardManifestEntry(**entry) for entry in raw["shards"]],
        )

    def all_study_ids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for shard in self.shards:
            for study_id in shard.study_ids:
                if study_id not in seen:
                    seen.add(study_id)
                    ordered.append(study_id)
        return ordered


def sqlite_path_for_url(database_url: str) -> Path:
    """Returns the absolute filesystem path a `sqlite:///...` URL resolves
    to -- reuses database/session.py's own resolution (relative paths are
    anchored at REPO_ROOT, not the process cwd) so this always agrees with
    whatever `get_engine()` would actually open."""
    if not database_url.startswith("sqlite"):
        raise ValueError(f"only sqlite:// database URLs are supported by sharded extraction, got: {database_url}")
    resolved = _resolve_sqlite_url(database_url)
    return Path(resolved.split("sqlite:///", 1)[-1])
