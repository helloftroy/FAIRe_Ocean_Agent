"""Tests for scripts/merge_extraction_shards.py against real, separate
on-disk SQLite files (not :memory:, since ATTACH DATABASE needs real
files) -- this is the correctness-critical part of the shard/merge design
documented in cluster/README.md's "Speeding up extraction" section, so
each real reconciliation case gets its own dedicated test rather than one
big fixture.
"""
import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from merge_extraction_shards import merge_all_shards, merge_shard  # noqa: E402
from shard_extraction_prep import _partition  # noqa: E402
from _extraction_sharding import ShardManifest, ShardManifestEntry  # noqa: E402

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType, TaskStatus, TaskType
from fair_ocean_agent.database.models import (
    ApiPaperCorrection,
    Base,
    Entity,
    EntityRelationship,
    EntityStudy,
    RawFact,
    Source,
    Study,
    Task,
)


def _file_db(path: Path):
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def _session_for(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)()


@pytest.fixture
def main_path(tmp_path) -> Path:
    return tmp_path / "main.db"


def _seed_main_with_one_study(main_path: Path) -> Study:
    engine = _file_db(main_path)
    session = _session_for(engine)
    study = Study(title="Shared test study")
    session.add(study)
    session.commit()
    session.close()
    engine.dispose()
    return study


def _copy_schema_and_rows_from(main_path: Path, shard_path: Path) -> None:
    """A shard db in real life is a full backup-API snapshot of main at
    partition time -- for these tests, a plain sqlite3 backup is
    equivalent and much simpler to set up."""
    src = sqlite3.connect(str(main_path))
    dst = sqlite3.connect(str(shard_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def test_plain_new_rows_are_copied_into_main(tmp_path, main_path):
    study = _seed_main_with_one_study(main_path)
    shard_path = tmp_path / "shard_1.db"
    _copy_schema_and_rows_from(main_path, shard_path)

    engine = _file_db(shard_path)
    session = _session_for(engine)
    source = Source(study_id=study.study_id, source_type="article_fulltext", source_name="europe_pmc_fulltext")
    session.add(source)
    session.flush()
    fact = RawFact(
        study_id=study.study_id, source_id=source.source_id, fact_type_candidate="target_gene",
        raw_field_name="target_gene", raw_value="16S rRNA", entity_level="study",
        support_type=SupportType.EXPLICIT.value,
    )
    session.add(fact)
    session.commit()
    session.close()
    engine.dispose()

    conn = sqlite3.connect(str(main_path))
    try:
        merge_shard(conn, str(shard_path), task_ids=[])
    finally:
        conn.close()

    verify_engine = _file_db(main_path)
    verify_session = _session_for(verify_engine)
    facts = verify_session.query(RawFact).all()
    assert len(facts) == 1
    assert facts[0].raw_value == "16S rRNA"
    verify_session.close()
    verify_engine.dispose()


def test_raw_fact_review_status_is_synced_from_shard(tmp_path, main_path):
    study = _seed_main_with_one_study(main_path)
    engine = _file_db(main_path)
    session = _session_for(engine)
    fact = RawFact(
        study_id=study.study_id, fact_type_candidate="elev", raw_field_name="elev", raw_value="10",
        entity_level="sample", support_type=SupportType.STRUCTURED_SOURCE.value,
        review_status=ReviewStatus.UNREVIEWED.value,
    )
    session.add(fact)
    session.commit()
    fact_id = fact.fact_id
    session.close()
    engine.dispose()

    shard_path = tmp_path / "shard_1.db"
    _copy_schema_and_rows_from(main_path, shard_path)
    shard_engine = _file_db(shard_path)
    shard_session = _session_for(shard_engine)
    shard_fact = shard_session.get(RawFact, fact_id)
    shard_fact.review_status = ReviewStatus.REJECTED.value
    shard_session.commit()
    shard_session.close()
    shard_engine.dispose()

    conn = sqlite3.connect(str(main_path))
    try:
        merge_shard(conn, str(shard_path), task_ids=[])
    finally:
        conn.close()

    verify_engine = _file_db(main_path)
    verify_session = _session_for(verify_engine)
    assert verify_session.get(RawFact, fact_id).review_status == ReviewStatus.REJECTED.value
    verify_session.close()
    verify_engine.dispose()


def test_task_lifecycle_columns_are_synced_for_this_shards_own_tasks(tmp_path, main_path):
    study = _seed_main_with_one_study(main_path)
    engine = _file_db(main_path)
    session = _session_for(engine)
    task = Task(
        task_type=TaskType.EXTRACT_TEXT_FACTS.value, study_id=study.study_id,
        status=TaskStatus.PENDING.value, idempotency_key=f"EXTRACT_TEXT_FACTS:{study.study_id}",
    )
    session.add(task)
    session.commit()
    task_id = task.task_id
    session.close()
    engine.dispose()

    shard_path = tmp_path / "shard_1.db"
    _copy_schema_and_rows_from(main_path, shard_path)
    shard_engine = _file_db(shard_path)
    shard_session = _session_for(shard_engine)
    shard_task = shard_session.get(Task, task_id)
    shard_task.status = TaskStatus.COMPLETED.value
    shard_task.attempt_count = 1
    shard_session.commit()
    shard_session.close()
    shard_engine.dispose()

    conn = sqlite3.connect(str(main_path))
    try:
        merge_shard(conn, str(shard_path), task_ids=[task_id])
    finally:
        conn.close()

    verify_engine = _file_db(main_path)
    verify_session = _session_for(verify_engine)
    merged_task = verify_session.get(Task, task_id)
    assert merged_task.status == TaskStatus.COMPLETED.value
    assert merged_task.attempt_count == 1
    verify_session.close()
    verify_engine.dispose()


def test_new_tasks_with_the_same_non_study_scoped_idempotency_key_dedupe(tmp_path, main_path):
    """Real gap found live: _resolve_and_seed_primer_references enqueues a
    DISCOVER_PRIMER_REFERENCE_STUDIES task with an explicit idempotency_key
    that is NOT study-scoped (f"DISCOVER_PRIMER_REFERENCE:{doi}") -- two
    shards each chasing a citation to the same DOI will each create a Task
    row with the identical idempotency_key but a different task_id. The
    table's own UniqueConstraint on idempotency_key must make INSERT OR
    IGNORE dedupe this automatically, with no bespoke logic."""
    _seed_main_with_one_study(main_path)

    shard_a = tmp_path / "shard_a.db"
    shard_b = tmp_path / "shard_b.db"
    _copy_schema_and_rows_from(main_path, shard_a)
    _copy_schema_and_rows_from(main_path, shard_b)

    shared_key = "DISCOVER_PRIMER_REFERENCE:10.1234/example"
    for shard_path in (shard_a, shard_b):
        engine = _file_db(shard_path)
        session = _session_for(engine)
        session.add(Task(task_type="DISCOVER_PRIMER_REFERENCE_STUDIES", status=TaskStatus.PENDING.value, idempotency_key=shared_key))
        session.commit()
        session.close()
        engine.dispose()

    conn = sqlite3.connect(str(main_path))
    try:
        merge_shard(conn, str(shard_a), task_ids=[])
        merge_shard(conn, str(shard_b), task_ids=[])
    finally:
        conn.close()

    verify_engine = _file_db(main_path)
    verify_session = _session_for(verify_engine)
    matches = verify_session.query(Task).filter_by(idempotency_key=shared_key).all()
    assert len(matches) == 1
    verify_session.close()
    verify_engine.dispose()


def test_duplicate_shareable_entities_across_shards_collapse_to_one_and_references_redirect(tmp_path, main_path):
    """Real gap found live: materialize_legacy_experiment_runs's SAMPLE/
    EXPERIMENT_RUN/SEQUENCING_RUN entity lookup is global (no study_id
    filter) in the real pipeline, but each shard's isolated snapshot can't
    see another shard's concurrent work -- two shards processing two
    different studies that reference the SAME real BioSample accession
    will each independently create their OWN new Entity row for it. The
    partial unique index on (entity_level, external_identifier) must
    reject the second one at merge time, and every row that referenced the
    losing entity_id (in either shard) must end up pointing at the
    survivor, not a dangling reference."""
    study_a = _seed_main_with_one_study(main_path)
    engine = _file_db(main_path)
    session = _session_for(engine)
    study_b = Study(title="Second study, same accession")
    session.add(study_b)
    session.commit()
    study_b_id = study_b.study_id
    session.close()
    engine.dispose()

    shard_a = tmp_path / "shard_a.db"
    shard_b = tmp_path / "shard_b.db"
    _copy_schema_and_rows_from(main_path, shard_a)
    _copy_schema_and_rows_from(main_path, shard_b)

    accession = "SAMN99999999"
    entity_ids: dict[str, str] = {}
    for shard_path, study_id in ((shard_a, study_a.study_id), (shard_b, study_b_id)):
        engine = _file_db(shard_path)
        session = _session_for(engine)
        entity = Entity(study_id=study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier=accession)
        session.add(entity)
        session.flush()
        session.add(EntityStudy(entity_id=entity.entity_id, study_id=study_id, relationship_type="is_home_of", confidence=SupportType.EXPLICIT.value))
        session.add(
            RawFact(
                study_id=study_id, entity_id=entity.entity_id, fact_type_candidate="organism",
                raw_field_name="organism", raw_value="seawater metagenome", entity_level="sample",
                support_type=SupportType.STRUCTURED_SOURCE.value,
            )
        )
        session.commit()
        entity_ids[shard_path.name] = entity.entity_id
        session.close()
        engine.dispose()

    assert entity_ids["shard_a.db"] != entity_ids["shard_b.db"]  # sanity: genuinely two different local ids

    conn = sqlite3.connect(str(main_path))
    try:
        merge_shard(conn, str(shard_a), task_ids=[])
        merge_shard(conn, str(shard_b), task_ids=[])
    finally:
        conn.close()

    verify_engine = _file_db(main_path)
    verify_session = _session_for(verify_engine)
    entities = verify_session.query(Entity).filter_by(entity_level=EntityLevel.SAMPLE.value, external_identifier=accession).all()
    assert len(entities) == 1, "the two shards' duplicate entities must collapse to exactly one"
    survivor_id = entities[0].entity_id

    facts = verify_session.query(RawFact).filter_by(fact_type_candidate="organism").all()
    assert len(facts) == 2, "both studies' organism facts must survive"
    assert all(f.entity_id == survivor_id for f in facts), "every fact must point at the surviving entity, not a dangling id"

    entity_studies = verify_session.query(EntityStudy).filter_by(entity_id=survivor_id).all()
    assert {es.study_id for es in entity_studies} == {study_a.study_id, study_b_id}, "both studies must still be linked to the surviving entity"

    verify_session.close()
    verify_engine.dispose()


def test_partition_splits_tasks_round_robin_and_is_deterministic():
    tasks = [(f"TASK-{i}", f"STUDY-{i}") for i in range(7)]
    shards = _partition(tasks, 3)
    assert [len(s) for s in shards] == [3, 2, 2]
    assert shards[0] == [tasks[0], tasks[3], tasks[6]]
    assert shards == _partition(tasks, 3)  # deterministic re-run


def test_merge_all_shards_regenerates_standardized_values_with_full_visibility(tmp_path, main_path, monkeypatch):
    """End-to-end via the manifest-driven entry point: after merging,
    map_study_to_faire must re-run for every touched study against the
    fully-merged database, not just the rows that shard happened to see in
    isolation."""
    from fair_ocean_agent.config import reset_config_cache
    from fair_ocean_agent.database.models import StandardizedValue
    from fair_ocean_agent.database.session import reset_engine_cache

    study = _seed_main_with_one_study(main_path)
    shard_path = tmp_path / "shard_1.db"
    _copy_schema_and_rows_from(main_path, shard_path)

    engine = _file_db(shard_path)
    session = _session_for(engine)
    session.add(
        RawFact(
            study_id=study.study_id, fact_type_candidate="target_gene", raw_field_name="target_gene",
            raw_value="16S rRNA", entity_level=EntityLevel.PROJECT.value, support_type=SupportType.EXPLICIT.value,
        )
    )
    session.commit()
    session.close()
    engine.dispose()

    manifest = ShardManifest(
        main_db_path=str(main_path),
        shards=[ShardManifestEntry(shard_index=1, db_path=str(shard_path), task_ids=[], study_ids=[study.study_id])],
    )

    monkeypatch.setenv("FAIR_OCEAN_DATABASE_URL", f"sqlite:///{main_path}")
    reset_config_cache()
    reset_engine_cache()
    try:
        merge_all_shards(manifest)
    finally:
        reset_engine_cache()
        reset_config_cache()

    verify_engine = _file_db(main_path)
    verify_session = _session_for(verify_engine)
    values = verify_session.query(StandardizedValue).filter_by(study_id=study.study_id, target_field="target_gene").all()
    assert len(values) == 1
    assert values[0].standardized_value == "16S rRNA"
    verify_session.close()
    verify_engine.dispose()
