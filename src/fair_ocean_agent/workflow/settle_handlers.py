"""CHECK_COMPONENT_SETTLED: a self-rescheduling poll task that detects when
a connected component of Studies (identity/component.py) has stopped
growing, so root determination (identity/root_determination.py) and
MAP_FAIRE (workflow/mapping_handlers.py) can both wait for the FULL, final
component membership rather than acting on a partial one.

No built-in task-dependency/barrier mechanism exists in this task queue
(workflow/task_queue.py) -- this task IS that mechanism: it re-enqueues
itself with a fresh idempotency key (a monotonic `generation` counter
carried in its own payload, not a timestamp, so retries within one
generation never multiply tasks) and a delayed `available_after` until
nothing in the component is still discovering.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import (
    ComponentStatus,
    EntityLevel,
    ReviewStatus,
    SHAREABLE_ENTITY_LEVELS,
    SupportType,
    TaskStatus,
    TaskType,
)
from fair_ocean_agent.database.models import Entity, EntityStudy, RawFact, Study, Task
from fair_ocean_agent.config import load_config
from fair_ocean_agent.identity.component import compute_study_component
from fair_ocean_agent.identity.root_determination import determine_entity_root
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)

_IN_FLIGHT_DISCOVERY_STATUSES = (
    TaskStatus.PENDING.value, TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value, TaskStatus.RETRY_PENDING.value,
)
_DISCOVERY_TASK_TYPES = (TaskType.DISCOVER_IDENTIFIERS.value, TaskType.DISCOVER_CITING_STUDIES.value)


def _has_shareable_entity_link(session: Session, study_id: str) -> bool:
    return (
        session.query(EntityStudy.entity_study_id)
        .join(Entity, Entity.entity_id == EntityStudy.entity_id)
        .filter(
            EntityStudy.study_id == study_id,
            Entity.entity_level.in_([level.value for level in SHAREABLE_ENTITY_LEVELS]),
        )
        .first()
        is not None
    )


def maybe_enqueue_settle_check(session: Session, study_id: str) -> None:
    """First entry point into this machinery -- called after a study's own
    discovery resolves (workflow/handlers.py::handle_discover_identifiers)
    and from enqueue_mapping_backfill (workflow/mapping_handlers.py). A
    no-op for a study with no shareable-level entity links at all -- those
    never enter this machinery, matching "studies with no shareable
    entities proceed immediately, no waiting." Safe to call repeatedly for
    the same study: the generation-0 idempotency key makes a repeat call a
    no-op via enqueue_task's own de-duplication."""
    if not _has_shareable_entity_link(session, study_id):
        return
    study = session.get(Study, study_id)
    if study is not None and study.entity_component_status == ComponentStatus.NOT_APPLICABLE.value:
        study.entity_component_status = ComponentStatus.PENDING.value
    enqueue_task(
        session,
        TaskType.CHECK_COMPONENT_SETTLED,
        study_id=study_id,
        idempotency_key=f"CHECK_COMPONENT_SETTLED:{study_id}:0",
        payload={"last_component_snapshot": [], "generation": 0},
    )


def reopen_component_settle_check(session: Session, study_id: str) -> None:
    """Called when a previously-SETTLED component grows again (a new
    citing paper discovered months later, via
    scheduling/rediscovery.py::enqueue_citation_rediscovery_backfill or the
    eager hook in handle_discover_citing_studies) -- flips every current
    member back to PENDING and starts a fresh settle-check cycle. Uses a
    timestamp in the idempotency key (unlike the stable generation-0 key
    above) since this is deliberately a NEW cycle, not a retry of the
    original one."""
    component = compute_study_component(session, study_id)
    for member_id in component:
        member = session.get(Study, member_id)
        if member is not None and member.entity_component_status == ComponentStatus.SETTLED.value:
            member.entity_component_status = ComponentStatus.PENDING.value
    enqueue_task(
        session,
        TaskType.CHECK_COMPONENT_SETTLED,
        study_id=study_id,
        idempotency_key=f"CHECK_COMPONENT_SETTLED:reopen:{study_id}:{utcnow().isoformat()}",
        payload={"last_component_snapshot": [], "generation": 0},
    )


def _reschedule(session: Session, anchor_study_id: str, component: set[str], generation: int) -> None:
    poll_interval = load_config().discovery.component_settle_poll_interval_seconds
    enqueue_task(
        session,
        TaskType.CHECK_COMPONENT_SETTLED,
        study_id=anchor_study_id,
        available_after=utcnow() + timedelta(seconds=poll_interval),
        idempotency_key=f"CHECK_COMPONENT_SETTLED:{anchor_study_id}:{generation}",
        payload={"last_component_snapshot": sorted(component), "generation": generation},
    )


def _mark_stalled(session: Session, component: set[str]) -> None:
    for study_id in component:
        study = session.get(Study, study_id)
        if study is not None:
            study.entity_component_status = ComponentStatus.STALLED.value
    session.add(
        RawFact(
            study_id=next(iter(sorted(component))),
            entity_id=None,
            source_id=None,
            source_locator="workflow.settle_handlers",
            raw_field_name="entity_component_stalled",
            raw_value=f"component exceeded max_settle_check_generations: {sorted(component)}",
            fact_type_candidate="entity_component_stalled",
            entity_level=EntityLevel.STUDY.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method="workflow.settle_handlers",
            review_status=ReviewStatus.NEEDS_REVIEW.value,
            confidence_metadata={"component_study_ids": sorted(component)},
        )
    )
    logger.warning("component %s exceeded max_settle_check_generations; marked stalled", sorted(component))


def _mark_settled_and_determine_roots(session: Session, component: set[str]) -> None:
    now = utcnow()
    shared_entity_ids: set[str] = set()
    for study_id in component:
        study = session.get(Study, study_id)
        # Derived from the study's own entity links, not its current
        # entity_component_status -- a component member that hasn't
        # independently called maybe_enqueue_settle_check yet (e.g. its own
        # DISCOVER_IDENTIFIERS completed in a different order) must still
        # be marked SETTLED here rather than silently staying
        # NOT_APPLICABLE just because nothing touched it first.
        if study is not None and _has_shareable_entity_link(session, study_id):
            study.entity_component_status = ComponentStatus.SETTLED.value
            study.entity_component_settled_at = now
        shared_entity_ids |= set(
            session.scalars(select(EntityStudy.entity_id).where(EntityStudy.study_id == study_id)).all()
        )

    for entity_id in shared_entity_ids:
        entity = session.get(Entity, entity_id)
        if entity is not None and entity.root_status == "pending":
            determine_entity_root(session, entity)

    for study_id in component:
        has_raw_facts = session.query(RawFact.fact_id).filter(RawFact.study_id == study_id).first() is not None
        if has_raw_facts:
            enqueue_task(session, TaskType.MAP_FAIRE, study_id=study_id)


def handle_check_component_settled(session: Session, task: Task) -> None:
    anchor_study_id = task.study_id
    component = compute_study_component(session, anchor_study_id)

    in_flight = (
        session.query(Task.task_id)
        .filter(
            Task.study_id.in_(component),
            Task.task_type.in_(_DISCOVERY_TASK_TYPES),
            Task.status.in_(_IN_FLIGHT_DISCOVERY_STATUSES),
        )
        .first()
        is not None
    )

    payload = task.payload or {}
    last_snapshot = frozenset(payload.get("last_component_snapshot") or [])
    generation = payload.get("generation", 0)
    current_snapshot = frozenset(component)

    if in_flight or current_snapshot != last_snapshot:
        max_generations = load_config().discovery.max_settle_check_generations
        if generation + 1 > max_generations:
            _mark_stalled(session, component)
            return
        _reschedule(session, anchor_study_id, component, generation + 1)
        return

    _mark_settled_and_determine_roots(session, component)


TASK_HANDLERS[TaskType.CHECK_COMPONENT_SETTLED] = handle_check_component_settled
