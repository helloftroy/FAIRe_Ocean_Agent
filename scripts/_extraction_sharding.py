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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fair_ocean_agent.config import REPO_ROOT
from fair_ocean_agent.database.session import _resolve_sqlite_url

DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "shard_dbs" / "manifest.json"


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
