"""Task handler for Milestone 6: MAP_FAIRE. Wraps mapping/faire.py's
already-idempotent map_study_to_faire so it can run through the same
task queue as every other handler (retryable, resumable, one task per
study). BeBOP mapping (MAP_BEBOP) is intentionally not wired here -- see
mapping/bebop.py -- since the authoritative miop schema hasn't been
vendored yet; a MAP_BEBOP task would fail with "no handler registered"
rather than run with fabricated field definitions.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import SHAREABLE_ENTITY_LEVELS, TaskType
from fair_ocean_agent.database.models import Entity, RawFact, Task
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.mapping.faire import map_study_to_faire
from fair_ocean_agent.workflow.settle_handlers import maybe_enqueue_settle_check
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import TASK_HANDLERS

logger = get_logger(__name__)

MAPPING_VERSION = "faire-mapping-v2-experiment-runs"


def handle_map_faire(session: Session, task: Task) -> None:
    created = map_study_to_faire(session, task.study_id)
    logger.info("mapped %d FAIRe standardized value(s) for study %s", created, task.study_id)


def enqueue_mapping_backfill(session: Session) -> int:
    """Queues one MAP_FAIRE task per study that has at least one raw_fact --
    except a study with >=1 shareable-level entity (SAMPLE/EXPERIMENT_RUN/
    SEQUENCING_RUN), which is routed through
    workflow/settle_handlers.py::maybe_enqueue_settle_check instead: MAP_FAIRE
    for it only gets enqueued once its whole connected component
    (identity/component.py) stops growing and root determination
    (identity/root_determination.py) has run, so exports/faire.py never has
    to guess which linked study's broadcast facts are authoritative for a
    shared entity. A study with no shareable entities has nothing to wait
    for -- proceeds exactly as before, immediately.

    Idempotent the same way enqueue_data_asset_and_validation_backfill is --
    enqueue_task itself de-duplicates pending/running tasks of the same
    type for the same study."""
    study_ids = list(session.scalars(select(RawFact.study_id).distinct()))
    queued_immediately = 0
    for study_id in study_ids:
        has_shareable_entity = (
            session.query(Entity.entity_id)
            .filter(
                Entity.study_id == study_id,
                Entity.entity_level.in_([level.value for level in SHAREABLE_ENTITY_LEVELS]),
            )
            .first()
            is not None
        )
        if has_shareable_entity:
            maybe_enqueue_settle_check(session, study_id)
            continue
        enqueue_task(
            session,
            TaskType.MAP_FAIRE,
            study_id=study_id,
            payload={"mapping_version": MAPPING_VERSION},
        )
        queued_immediately += 1
    return queued_immediately


TASK_HANDLERS[TaskType.MAP_FAIRE] = handle_map_faire
