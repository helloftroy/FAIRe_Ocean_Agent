from fair_ocean_agent.database.enums import EntityLevel, SupportType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, RawFact, StandardizedValue, Study, Task
from fair_ocean_agent.workflow.mapping_handlers import MAPPING_VERSION, enqueue_mapping_backfill, handle_map_faire
from fair_ocean_agent.workflow.task_queue import enqueue_task


def test_handle_map_faire_wraps_map_study_to_faire(db_session):
    study = Study(title="Handler test")
    db_session.add(study)
    db_session.flush()
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=sample.entity_id, raw_field_name="geo_loc_name",
            raw_value="USA: California", fact_type_candidate="geo_loc_name", entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    task = enqueue_task(db_session, TaskType.MAP_FAIRE, study_id=study.study_id)
    db_session.commit()

    handle_map_faire(db_session, task)
    db_session.commit()

    fields = {
        value.target_field
        for value in db_session.query(StandardizedValue).filter_by(study_id=study.study_id)
    }
    assert fields == {"geo_loc_name", "samp_name"}


def test_handle_map_faire_is_idempotent_on_retry(db_session):
    study = Study(title="Retry check")
    db_session.add(study)
    db_session.flush()
    sample = Entity(study_id=study.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    db_session.add(sample)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, entity_id=sample.entity_id, raw_field_name="geo_loc_name",
            raw_value="USA: California", fact_type_candidate="geo_loc_name", entity_level="sample",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()
    task = enqueue_task(db_session, TaskType.MAP_FAIRE, study_id=study.study_id)
    db_session.commit()

    handle_map_faire(db_session, task)
    db_session.commit()
    handle_map_faire(db_session, task)  # simulated retry
    db_session.commit()

    values = db_session.query(StandardizedValue).filter_by(study_id=study.study_id).all()
    assert len(values) == 2
    assert {value.target_field for value in values} == {"geo_loc_name", "samp_name"}


def test_enqueue_mapping_backfill_queues_one_per_study_with_facts(db_session):
    study = Study(title="Backfill check")
    db_session.add(study)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id, raw_field_name="x", raw_value="y", fact_type_candidate="x",
            entity_level="study", support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    count = enqueue_mapping_backfill(db_session)
    db_session.commit()

    assert count == 1
    task = db_session.query(Task).filter_by(task_type=TaskType.MAP_FAIRE.value).one()
    assert task.payload == {"mapping_version": MAPPING_VERSION}

    enqueue_mapping_backfill(db_session)  # idempotent
    db_session.commit()
    assert db_session.query(Task).filter_by(task_type=TaskType.MAP_FAIRE.value).count() == 1


def test_versioned_mapping_backfill_requeues_after_legacy_task_completed(db_session):
    study = Study(title="Legacy mapping task")
    db_session.add(study)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id,
            raw_field_name="x",
            raw_value="y",
            fact_type_candidate="x",
            entity_level="study",
            support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    legacy = enqueue_task(db_session, TaskType.MAP_FAIRE, study_id=study.study_id)
    legacy.status = TaskStatus.COMPLETED.value
    db_session.commit()

    enqueue_mapping_backfill(db_session)
    db_session.commit()

    tasks = db_session.query(Task).filter_by(task_type=TaskType.MAP_FAIRE.value).all()
    assert len(tasks) == 2
    assert {task.payload and task.payload.get("mapping_version") for task in tasks} == {
        None,
        MAPPING_VERSION,
    }
