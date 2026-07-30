"""Single choke point for creating a Source row and its corresponding
study_sources link, atomically.

Every Source-creation call site in this codebase (workflow/handlers.py's
_persist_source_and_facts, workflow/supplement_handlers.py's
discover_supplements_for_study, workflow/refresh_handlers.py's
_persist_refreshed_source_and_facts) routes through create_source() below,
so sources.study_id (kept, unchanged -- see Source's own docstring) and the
new study_sources join row are never written out of step with each other.
Per the schema decision: sources.study_id stays the fast "home" read path
everywhere else in the codebase; this module is the only place a NEW
Source row (or an additional, non-home study link for an already-existing
Source, via link_source_to_study) gets created going forward.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import RelationshipType, SupportType
from fair_ocean_agent.database.models import Source, StudySource


def create_source(
    session: Session, source: Source, *, confidence: SupportType = SupportType.STRUCTURED_SOURCE
) -> Source:
    """Adds `source` (already constructed by the caller, study_id already
    set on it -- unchanged from today) and its home study_sources row in
    one flush. `confidence` defaults to STRUCTURED_SOURCE: every existing
    caller creates a Source from a directly-resolved adapter fetch_record()
    or a structurally-parsed supplement reference -- genuinely tier-1
    evidence that this Source belongs to this Study (same reasoning as the
    study_sources backfill migration's own default)."""
    session.add(source)
    session.flush()  # source.source_id must exist before the FK below
    session.add(
        StudySource(
            study_id=source.study_id,
            source_id=source.source_id,
            relationship_type=RelationshipType.IS_HOME_OF.value,
            confidence=confidence.value,
        )
    )
    session.flush()
    return source


def link_source_to_study(
    session: Session,
    source: Source,
    study_id: str,
    *,
    relationship_type: RelationshipType,
    confidence: SupportType,
) -> StudySource:
    """Adds a (non-home or additional) study_sources row for an ALREADY-
    EXISTING Source -- used by identity/resolution.py's
    resolve_or_create_study() to link a Source to a newly-created sibling
    Study (relationship_type=SHARES_ACCESSION_WITH) without touching
    sources.study_id there. Idempotent by convention, not by DB constraint:
    callers check first via a query filtered on (study_id, source_id)
    before calling, same idempotency-guard style used throughout
    workflow/handlers.py, since a second call for the same pair would hit
    study_sources' own uq_study_source constraint."""
    link = StudySource(
        study_id=study_id,
        source_id=source.source_id,
        relationship_type=relationship_type.value,
        confidence=confidence.value,
    )
    session.add(link)
    session.flush()
    return link
