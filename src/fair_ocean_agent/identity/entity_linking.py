"""Single choke point for creating an Entity row and its corresponding
entity_studies link, atomically -- exact mirror of identity/source_linking.py
for Entity/EntityStudy instead of Source/StudySource.

Every Entity-creation call site (currently just
workflow/handlers.py::_get_or_create_entity) routes through create_entity()
below, so entities.study_id (kept, unchanged -- see Entity's own docstring
in database/models.py) and the new entity_studies join row are never written
out of step with each other. entities.study_id stays the fast "home" read
path everywhere else in the codebase; this module is the only place a NEW
Entity row (or an additional, non-home study link for an already-existing
shareable-level Entity, via link_entity_to_study) gets created going
forward.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRootStatus,
    RelationshipType,
    SHAREABLE_ENTITY_LEVELS,
    SupportType,
)
from fair_ocean_agent.database.models import Entity, EntityStudy


def create_entity(
    session: Session, entity: Entity, *, confidence: SupportType = SupportType.STRUCTURED_SOURCE
) -> Entity:
    """Adds `entity` (already constructed by the caller, study_id already
    set on it) and its home entity_studies row in one flush. `confidence`
    defaults to STRUCTURED_SOURCE: a home link is definitionally certain --
    this study's own extraction pipeline is what caused the entity to exist
    at all -- regardless of whether the specific facts landing on it later
    turn out to be structured or LLM-derived (same reasoning as
    create_source's own default).

    A brand-new entity is unambiguously its own root (identity/
    root_determination.py) -- set eagerly here rather than left `pending`,
    since the common (non-shared) case needs no algorithm at all and
    shouldn't wait on a settle-check that may never even fire for it."""
    entity.root_study_id = entity.study_id
    entity.root_status = EntityRootStatus.DETERMINED.value
    entity.root_determined_at = utcnow()
    session.add(entity)
    session.flush()  # entity.entity_id must exist before the FK below
    session.add(
        EntityStudy(
            entity_id=entity.entity_id,
            study_id=entity.study_id,
            relationship_type=RelationshipType.IS_HOME_OF.value,
            confidence=confidence.value,
        )
    )
    session.flush()
    return entity


def link_entity_to_study(
    session: Session,
    entity: Entity,
    study_id: str,
    *,
    relationship_type: RelationshipType,
    confidence: SupportType,
) -> EntityStudy:
    """Adds a (non-home or additional) entity_studies row for an ALREADY-
    EXISTING Entity -- used by workflow/handlers.py's _get_or_create_entity
    when a different study resolves the same real-world sample/experiment/
    sequencing-run accession an earlier study already created (a second
    paper reusing the same deposited data). Idempotent by convention, not
    by DB constraint: callers check first via a query filtered on
    (study_id, entity_id) before calling, same idempotency-guard style used
    throughout workflow/handlers.py, since a second call for the same pair
    would hit entity_studies' own uq_entity_study constraint."""
    link = EntityStudy(
        entity_id=entity.entity_id,
        study_id=study_id,
        relationship_type=relationship_type.value,
        confidence=confidence.value,
    )
    session.add(link)
    session.flush()
    return link


def get_or_create_entity(
    session: Session,
    study_id: str,
    entity_level: EntityLevel,
    external_identifier: str,
    label: str | None = None,
) -> Entity:
    """Single choke point for entity get-or-create across the codebase --
    workflow/handlers.py's fact-materialization path AND
    extraction/experiment_runs.py's legacy sequencing-run-to-experiment-run
    materialization path both call this rather than keeping their own
    independent implementations. (They used to: two divergent
    _get_or_create_entity functions existed, and only one of them knew
    about EntityStudy -- exactly the kind of silent gap this single choke
    point exists to prevent, the same way create_source/create_entity above
    prevent it for Source/Entity creation itself.)

    For SHAREABLE_ENTITY_LEVELS (SAMPLE/EXPERIMENT_RUN/SEQUENCING_RUN --
    see database/enums.py), the lookup is global (no study_id filter): a
    second study resolving the same real BioSample/run accession an
    earlier study already created gets linked to that SAME Entity row via
    link_entity_to_study, never a duplicate. entities.study_id never moves
    off the entity's original home study when this happens -- only a new,
    non-home entity_studies row is added. The lookup is safe (at most one
    match) because of Entity's partial unique index on (entity_level,
    external_identifier) for these levels (database/models.py). Every
    other entity_level keeps the original study-scoped lookup unchanged:
    they're either handled by the separate Study-merge/sibling-split
    machinery (PROJECT) or typically unaccessioned, so there's no stable
    cross-study matching key regardless."""
    shareable = entity_level in SHAREABLE_ENTITY_LEVELS
    query = session.query(Entity).filter_by(entity_level=entity_level.value, external_identifier=external_identifier)
    if not shareable:
        query = query.filter_by(study_id=study_id)
    existing = query.one_or_none()

    if existing is not None:
        if label and not existing.label:
            existing.label = label
        if shareable and existing.study_id != study_id:
            already_linked = (
                session.query(EntityStudy)
                .filter_by(study_id=study_id, entity_id=existing.entity_id)
                .first()
            )
            if already_linked is None:
                link_entity_to_study(
                    session,
                    existing,
                    study_id,
                    relationship_type=RelationshipType.SHARES_ACCESSION_WITH,
                    confidence=SupportType.STRUCTURED_SOURCE,
                )
                # A genuinely NEW claimant invalidates whatever root answer
                # was previously settled (or eagerly set at creation) --
                # identity/root_determination.py only re-runs once this
                # entity's whole component has settled again. Not flipped
                # for an idempotent retry of a study that already links to
                # this entity (already_linked would be non-None then).
                existing.root_status = EntityRootStatus.PENDING.value
                existing.root_study_id = None
                existing.root_determined_at = None
        return existing

    return create_entity(
        session,
        Entity(
            study_id=study_id,
            entity_level=entity_level.value,
            external_identifier=external_identifier,
            label=label,
        ),
    )
