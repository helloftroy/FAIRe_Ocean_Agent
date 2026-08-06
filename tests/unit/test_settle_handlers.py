"""Tests for workflow/settle_handlers.py -- the self-rescheduling
CHECK_COMPONENT_SETTLED poll task that gates root determination/MAP_FAIRE
until a connected component (identity/component.py) stops growing."""
from fair_ocean_agent.config import AppConfig, DiscoveryConfig
from fair_ocean_agent.database.enums import (
    ComponentStatus,
    EntityLevel,
    EntityRootStatus,
    RelationshipType,
    SupportType,
    TaskStatus,
    TaskType,
)
from fair_ocean_agent.database.models import Entity, EntityStudy, RawFact, Study, Task
from fair_ocean_agent.workflow import settle_handlers
from fair_ocean_agent.workflow.settle_handlers import (
    handle_check_component_settled,
    maybe_enqueue_settle_check,
    reopen_component_settle_check,
)
from fair_ocean_agent.workflow.task_queue import enqueue_task


def _study(session, **kwargs) -> Study:
    study = Study(**kwargs)
    session.add(study)
    session.flush()
    return study


def _shared_entity(session, home: Study, others: list[Study]) -> Entity:
    entity = Entity(study_id=home.study_id, entity_level=EntityLevel.SAMPLE.value, external_identifier="SAMN1")
    session.add(entity)
    session.flush()
    session.add(
        EntityStudy(
            entity_id=entity.entity_id, study_id=home.study_id,
            relationship_type=RelationshipType.IS_HOME_OF.value, confidence=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    for other in others:
        session.add(
            EntityStudy(
                entity_id=entity.entity_id, study_id=other.study_id,
                relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value,
                confidence=SupportType.STRUCTURED_SOURCE.value,
            )
        )
    session.flush()
    return entity


def _settle_task_for(session, study_id) -> Task:
    return session.query(Task).filter_by(
        task_type=TaskType.CHECK_COMPONENT_SETTLED.value, study_id=study_id
    ).order_by(Task.created_at.desc()).first()


def test_maybe_enqueue_settle_check_no_op_without_shareable_entity(db_session):
    study = _study(db_session, title="No repository data at all")
    db_session.commit()

    maybe_enqueue_settle_check(db_session, study.study_id)
    db_session.commit()

    assert db_session.query(Task).filter_by(task_type=TaskType.CHECK_COMPONENT_SETTLED.value).count() == 0


def test_maybe_enqueue_settle_check_marks_pending_and_enqueues(db_session):
    study = _study(db_session, title="Has a sample")
    _shared_entity(db_session, study, [])
    db_session.commit()

    maybe_enqueue_settle_check(db_session, study.study_id)
    db_session.commit()

    db_session.refresh(study)
    assert study.entity_component_status == ComponentStatus.PENDING.value
    task = _settle_task_for(db_session, study.study_id)
    assert task is not None
    assert task.payload == {"last_component_snapshot": [], "generation": 0}

    # Repeat call is a no-op (same idempotency key).
    maybe_enqueue_settle_check(db_session, study.study_id)
    db_session.commit()
    assert db_session.query(Task).filter_by(task_type=TaskType.CHECK_COMPONENT_SETTLED.value).count() == 1


def test_handler_reschedules_when_in_flight_discovery_task_exists(db_session):
    study_a = _study(db_session, title="Study A")
    study_b = _study(db_session, title="Study B")
    entity = _shared_entity(db_session, study_a, [study_b])
    db_session.add(
        Task(task_type=TaskType.DISCOVER_IDENTIFIERS.value, study_id=study_b.study_id, status=TaskStatus.PENDING.value, idempotency_key=f"test-in-flight-{study_b.study_id}")
    )
    db_session.commit()

    maybe_enqueue_settle_check(db_session, study_a.study_id)
    db_session.commit()
    task = _settle_task_for(db_session, study_a.study_id)

    handle_check_component_settled(db_session, task)
    db_session.commit()

    db_session.refresh(study_a)
    assert study_a.entity_component_status == ComponentStatus.PENDING.value
    assert db_session.query(RawFact).filter_by(fact_type_candidate="entity_root_ambiguous").count() == 0
    assert db_session.query(Task).filter_by(task_type=TaskType.MAP_FAIRE.value).count() == 0
    rescheduled = _settle_task_for(db_session, study_a.study_id)
    assert rescheduled.payload["generation"] == 1


def test_handler_settles_and_enqueues_map_faire_when_nothing_in_flight(db_session, monkeypatch):
    study_a = _study(db_session, title="Study A, 2020")
    study_b = _study(db_session, title="Study B, 2023")
    entity = _shared_entity(db_session, study_a, [study_b])
    db_session.add(
        RawFact(
            study_id=study_a.study_id, entity_id=entity.entity_id, fact_type_candidate="depth", raw_value="10",
            entity_level=EntityLevel.SAMPLE.value, support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.add(
        RawFact(
            study_id=study_b.study_id, entity_id=entity.entity_id, fact_type_candidate="depth", raw_value="10",
            entity_level=EntityLevel.SAMPLE.value, support_type=SupportType.STRUCTURED_SOURCE.value,
        )
    )
    db_session.commit()

    # First call: current component snapshot ({A, B}) differs from the
    # initial empty last_component_snapshot -- must reschedule, not settle.
    maybe_enqueue_settle_check(db_session, study_a.study_id)
    db_session.commit()
    first_task = _settle_task_for(db_session, study_a.study_id)
    handle_check_component_settled(db_session, first_task)
    db_session.commit()
    db_session.refresh(study_a)
    assert study_a.entity_component_status == ComponentStatus.PENDING.value

    second_task = _settle_task_for(db_session, study_a.study_id)
    assert second_task.payload["generation"] == 1
    handle_check_component_settled(db_session, second_task)
    db_session.commit()

    db_session.refresh(study_a)
    db_session.refresh(study_b)
    assert study_a.entity_component_status == ComponentStatus.SETTLED.value
    assert study_b.entity_component_status == ComponentStatus.SETTLED.value
    assert study_a.entity_component_settled_at is not None

    map_faire_study_ids = {
        t.study_id for t in db_session.query(Task).filter_by(task_type=TaskType.MAP_FAIRE.value).all()
    }
    assert map_faire_study_ids == {study_a.study_id, study_b.study_id}


def test_handler_marks_stalled_after_max_generations(db_session, monkeypatch):
    monkeypatch.setattr(
        settle_handlers, "load_config", lambda: AppConfig(discovery=DiscoveryConfig(max_settle_check_generations=1))
    )
    study_a = _study(db_session, title="Study A")
    study_b = _study(db_session, title="Study B")
    _shared_entity(db_session, study_a, [study_b])
    db_session.add(
        Task(task_type=TaskType.DISCOVER_IDENTIFIERS.value, study_id=study_b.study_id, status=TaskStatus.PENDING.value, idempotency_key=f"test-in-flight-{study_b.study_id}")
    )
    db_session.commit()

    task = enqueue_task(
        db_session, TaskType.CHECK_COMPONENT_SETTLED, study_id=study_a.study_id,
        idempotency_key=f"CHECK_COMPONENT_SETTLED:{study_a.study_id}:1",
        payload={"last_component_snapshot": [], "generation": 1},
    )
    db_session.commit()

    handle_check_component_settled(db_session, task)
    db_session.commit()

    db_session.refresh(study_a)
    db_session.refresh(study_b)
    assert study_a.entity_component_status == ComponentStatus.STALLED.value
    assert study_b.entity_component_status == ComponentStatus.STALLED.value
    assert db_session.query(RawFact).filter_by(fact_type_candidate="entity_component_stalled").count() == 1


def test_reopen_flips_settled_members_back_to_pending(db_session):
    study_a = _study(db_session, title="Study A", entity_component_status=ComponentStatus.SETTLED.value)
    study_b = _study(db_session, title="Study B", entity_component_status=ComponentStatus.SETTLED.value)
    _shared_entity(db_session, study_a, [study_b])
    db_session.commit()

    reopen_component_settle_check(db_session, study_a.study_id)
    db_session.commit()

    db_session.refresh(study_a)
    db_session.refresh(study_b)
    assert study_a.entity_component_status == ComponentStatus.PENDING.value
    assert study_b.entity_component_status == ComponentStatus.PENDING.value
    assert db_session.query(Task).filter_by(task_type=TaskType.CHECK_COMPONENT_SETTLED.value).count() == 1
