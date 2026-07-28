"""Stage 1 (exact identifier) and Stage 2 (explicit relationship, via
merge_study_into) deduplication. Stage 3 (probabilistic candidate scoring,
see the candidate_matches table) needs title/author/etc. similarity scoring
and is not implemented yet -- that's Milestone 3+.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import CanonicalStatus, IdentifierType
from fair_ocean_agent.database.models import (
    CandidateMatch,
    DataAsset,
    Entity,
    ExternalIdentifier,
    RawFact,
    Source,
    StandardizedValue,
    Study,
    Task,
)
from fair_ocean_agent.identity.identifiers import normalize_identifier

_STUDY_FK_MODELS = (Entity, Source, RawFact, StandardizedValue, DataAsset, Task)


def find_existing_study_by_identifier(
    session: Session, identifier_type: IdentifierType, raw_value: str
) -> Study | None:
    """Exact-match lookup only. Returns the first matching canonical study,
    or None if no external_identifiers row has this normalized value."""
    normalized = normalize_identifier(identifier_type, raw_value)
    stmt = select(ExternalIdentifier).where(
        ExternalIdentifier.identifier_type == identifier_type.value,
        ExternalIdentifier.identifier_value == normalized,
    )
    existing = session.scalars(stmt).first()
    if existing is None:
        return None
    return session.get(Study, existing.study_id)


def find_existing_study_by_any_identifier(
    session: Session, identifiers: list[tuple[IdentifierType, str]]
) -> Study | None:
    """Try each (type, raw_value) pair in order; return the first exact
    match. Invalid identifiers (fail normalization) are silently skipped --
    the caller (seed loader) is responsible for surfacing normalization
    errors separately."""
    from fair_ocean_agent.identity.identifiers import IdentifierError

    for identifier_type, raw_value in identifiers:
        if not raw_value:
            continue
        try:
            match = find_existing_study_by_identifier(session, identifier_type, raw_value)
        except IdentifierError:
            continue
        if match is not None:
            return match
    return None


def merge_study_into(session: Session, absorb: Study, into: Study) -> Study:
    """Stage 2 (explicit relationship) merge: a source adapter resolved an
    identifier (e.g. a PMID from Europe PMC) that already belongs to a
    *different* canonical study than the one we started from -- both
    external_identifiers rows describe the same real study, so reassign
    every foreign key from `absorb` to `into` and mark `absorb` merged
    rather than leaving two canonical rows for one study. Returns `into` so
    callers can keep a single variable across a merge.

    Unlike Stage 3 candidate matches, this never needs human review: an
    explicit identifier match (not a similarity score) is the whole point
    of "exact identifiers before fuzzy matching" (section 8).
    """
    if absorb.study_id == into.study_id:
        return into

    into_identifier_keys = {
        (ei.identifier_type, ei.identifier_value) for ei in into.external_identifiers
    }
    for ei in list(
        session.scalars(select(ExternalIdentifier).where(ExternalIdentifier.study_id == absorb.study_id))
    ):
        if (ei.identifier_type, ei.identifier_value) in into_identifier_keys:
            session.delete(ei)  # already present on `into`; avoid a unique-constraint collision
        else:
            ei.study_id = into.study_id
    session.flush()

    for model in _STUDY_FK_MODELS:
        session.query(model).filter(model.study_id == absorb.study_id).update(
            {"study_id": into.study_id}, synchronize_session=False
        )

    session.query(CandidateMatch).filter(CandidateMatch.study_a_id == absorb.study_id).update(
        {"study_a_id": into.study_id}, synchronize_session=False
    )
    session.query(CandidateMatch).filter(CandidateMatch.study_b_id == absorb.study_id).update(
        {"study_b_id": into.study_id}, synchronize_session=False
    )

    absorb.canonical_status = CanonicalStatus.MERGED.value
    session.flush()
    session.expire(into)
    return into
