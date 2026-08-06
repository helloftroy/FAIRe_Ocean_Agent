"""Decides which linked Study is the authoritative ("root") source for a
shared Entity's broadcast-style (study-wide LLM/text) facts --
exports/faire.py consults this to decide whose defaults (lat/lon/depth/
date/...) fill a shared sample's blanks, never an arbitrary or every
linked study's.

Only ever called from workflow/settle_handlers.py, once a connected
component (identity/component.py) has SETTLED -- an earlier answer could
be invalidated by a not-yet-processed sibling still in the discovery
queue, so this never runs mid-flight.

Algorithm, in priority order:
1. Earliest publication date among linked studies wins outright -- a paper
   cannot analyze data that doesn't exist yet. Year-granularity only
   (Source.publication_year via the existing, reused
   workflow/validation_handlers.py::_publication_year_for_study); Crossref's
   own day-level "published" RawFact is stored as a str()-stringified
   Python dict, not JSON, too fragile to parse reliably for finer-grained
   tie-breaking.
2. BioProject registration date (the "submitted" RawFact NcbiBioProjectAdapter
   already writes) is read as corroboration/plausibility context only --
   attached to the ambiguous-flag fact below when relevant, never used to
   override step 1's pick on its own.
3. Genuinely ambiguous (same year, or no publication-date signal at all for
   the tied studies) -> flag for human review, never guess. Matches this
   codebase's established flag-don't-guess convention
   (workflow/handlers.py::_record_citation_expansion_capped_fact,
   CandidateMatch-on-ambiguity in identity/resolution.py).

Submitter/institution matching (a weaker tiebreaker signal) is explicitly
out of scope: no adapter in this codebase parses structured author
affiliations today (Crossref only captures given/family name; OpenAlex's
raw `authorships` blob, which does carry institutions in the real API, is
stored unparsed here). Ties that would need this signal go straight to
ambiguous instead of a fragile guess.
"""
from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityRootStatus, ReviewStatus, SupportType
from fair_ocean_agent.database.models import Entity, EntityStudy, RawFact

_YEAR_RE = re.compile(r"^\s*(\d{4})")


def _registration_year_for_entity_home(session: Session, entity: Entity) -> int | None:
    """Best-effort corroboration only: the "submitted" RawFact
    (entity_level=PROJECT, no entity_id -- see NcbiBioProjectAdapter's own
    `add()` helper) is a property of the BioProject itself, not of
    whichever paper's Source row happened to fetch it, so any linked
    study's own copy should agree -- reading it from the entity's home
    study is a reasonable approximation given there's no explicit schema
    link from a shared SAMPLE/EXPERIMENT_RUN/SEQUENCING_RUN entity to the
    specific PROJECT entity it descended from."""
    raw_value = session.scalars(
        select(RawFact.raw_value).where(
            RawFact.study_id == entity.study_id,
            RawFact.entity_id.is_(None),
            RawFact.fact_type_candidate == "submitted",
        )
    ).first()
    if not raw_value:
        return None
    match = _YEAR_RE.match(raw_value)
    return int(match.group(1)) if match else None


def _flag_ambiguous(
    session: Session, entity: Entity, linked_study_ids: list[str], publication_years: dict[str, int | None]
) -> None:
    entity.root_status = EntityRootStatus.AMBIGUOUS.value
    entity.root_study_id = None
    entity.root_determined_at = None

    already_flagged = session.scalars(
        select(RawFact.fact_id).where(
            RawFact.entity_id == entity.entity_id,
            RawFact.fact_type_candidate == "entity_root_ambiguous",
            RawFact.review_status == ReviewStatus.NEEDS_REVIEW.value,
        )
    ).first()
    if already_flagged is not None:
        return  # already flagged for review from an earlier settle -- don't pile up duplicates

    session.add(
        RawFact(
            study_id=entity.study_id,
            entity_id=entity.entity_id,
            source_id=None,
            source_locator="identity.root_determination",
            raw_field_name="entity_root_ambiguous",
            raw_value=f"no unambiguous root among linked studies: {sorted(linked_study_ids)}",
            fact_type_candidate="entity_root_ambiguous",
            entity_level=entity.entity_level,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method="identity.root_determination",
            review_status=ReviewStatus.NEEDS_REVIEW.value,
            confidence_metadata={
                "candidate_study_ids": sorted(linked_study_ids),
                "publication_years": publication_years,
                "registration_year": _registration_year_for_entity_home(session, entity),
            },
        )
    )


def determine_entity_root(session: Session, entity: Entity) -> None:
    linked_study_ids = sorted(
        set(session.scalars(select(EntityStudy.study_id).where(EntityStudy.entity_id == entity.entity_id)).all())
    )
    if len(linked_study_ids) <= 1:
        # Not the normal path (create_entity already sets DETERMINED
        # eagerly for the non-shared case), but stays correct if this is
        # ever called directly against an entity that isn't actually shared.
        if linked_study_ids:
            entity.root_study_id = linked_study_ids[0]
            entity.root_status = EntityRootStatus.DETERMINED.value
            entity.root_determined_at = utcnow()
        return

    from fair_ocean_agent.workflow.validation_handlers import _publication_year_for_study

    publication_years = {study_id: _publication_year_for_study(session, study_id) for study_id in linked_study_ids}
    dated = {study_id: year for study_id, year in publication_years.items() if year is not None}

    winner: str | None = None
    if dated:
        earliest_year = min(dated.values())
        earliest_study_ids = [study_id for study_id, year in dated.items() if year == earliest_year]
        if len(earliest_study_ids) == 1:
            winner = earliest_study_ids[0]

    if winner is None:
        _flag_ambiguous(session, entity, linked_study_ids, publication_years)
        return

    entity.root_study_id = winner
    entity.root_status = EntityRootStatus.DETERMINED.value
    entity.root_determined_at = utcnow()
