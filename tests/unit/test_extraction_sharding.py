import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _extraction_sharding import ShardManifest, ShardManifestEntry, sqlite_path_for_url  # noqa: E402


def test_manifest_round_trips_through_json(tmp_path):
    manifest = ShardManifest(
        main_db_path="/abs/path/main.db",
        shards=[
            ShardManifestEntry(shard_index=1, db_path="/abs/path/shard_1.db", task_ids=["TASK-1", "TASK-2"], study_ids=["STUDY-1"]),
            ShardManifestEntry(shard_index=2, db_path="/abs/path/shard_2.db", task_ids=["TASK-3"], study_ids=["STUDY-2", "STUDY-3"]),
        ],
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = ShardManifest.read(path)

    assert loaded.main_db_path == manifest.main_db_path
    assert [s.shard_index for s in loaded.shards] == [1, 2]
    assert loaded.shards[0].task_ids == ["TASK-1", "TASK-2"]


def test_all_study_ids_dedupes_preserving_first_occurrence_order():
    manifest = ShardManifest(
        main_db_path="/abs/path/main.db",
        shards=[
            ShardManifestEntry(shard_index=1, db_path="s1.db", study_ids=["STUDY-A", "STUDY-B"]),
            ShardManifestEntry(shard_index=2, db_path="s2.db", study_ids=["STUDY-B", "STUDY-C"]),
        ],
    )
    assert manifest.all_study_ids() == ["STUDY-A", "STUDY-B", "STUDY-C"]


def test_sqlite_path_for_url_resolves_relative_to_repo_root():
    from fair_ocean_agent.config import REPO_ROOT

    resolved = sqlite_path_for_url("sqlite:///data/fair_ocean.db")
    assert resolved == REPO_ROOT / "data" / "fair_ocean.db"


def test_sqlite_path_for_url_rejects_non_sqlite_urls():
    import pytest

    with pytest.raises(ValueError):
        sqlite_path_for_url("postgresql://localhost/fair_ocean")
