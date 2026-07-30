"""The single function every discovery path (DOI-seeded or accession-seeded)
calls to reconcile a newly-discovered identifier against pre-existing
studies -- resolve_or_create_study(). Neither path grows its own merge
logic; both delegate here via workflow/handlers.py's
_apply_related_identifiers, so DOI-led and accession-led discovery can
never silently diverge in how they decide to merge, split, or flag.

Evidence-tier gating (RelatedIdentifier.confidence, see sources/base.py)
decides how much scrutiny a match needs before it's trusted:

- STRUCTURED_SOURCE (tier 1, e.g. DataCite's own relatedIdentifiers,
  Europe PMC's PMID/PMCID): safe to auto-link, no consistency check.
- DETERMINISTICALLY_DERIVED (tier 2, a regex-matched accession already
  confirmed against its own source API) and INFERRED (tier 3, an LLM
  prose claim -- no producer exists yet, plumbing only): always
  consistency-checked (identity/consistency.py) before merging. Tier 3
  additionally never merges on its own even when consistent -- it always
  also lands a CandidateMatch for human review.

This module imports from identity/deduplication.py (Stage 1/2 dedup +
merge_study_into), not the reverse -- Stage 1/2 stays narrow and untouched;
this is the orchestration layer built on top of it.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import CandidateMatchMethod, CandidateMatchReviewStatus, CanonicalStatus, RelationshipType, SupportType
from fair_ocean_agent.database.models import CandidateMatch, Entity, ExternalIdentifier, RawFact, Source, Study
from fair_ocean_agent.identity.consistency import check_study_consistency
from fair_ocean_agent.identity.deduplication import find_all_existing_studies_by_identifier, merge_study_into
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier
from fair_ocean_agent.identity.source_linking import link_source_to_study
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import RelatedIdentifier

logger = get_logger(__name__)


def resolve_or_create_study(
    session: Session, study: Study, related: list[RelatedIdentifier], source: Source | None
) -> Study:
    """The shared merge-decision function. `study` is the Study the current
    task/adapter call is already working against (may or may not survive
    this call unchanged). `source` is the Source row this evidence came
    from -- used only for the sibling-split case, to know exactly which
    RawFact/Entity rows are "this new evidence"; None for call sites with
    no single triggering Source (fulltext-mined identifiers, supplement
    discovery), which degrades the sibling-split case to a CandidateMatch-
    only flag (see _create_sibling_and_flag). Returns the study to keep
    using afterward, same contract as the _apply_related_identifiers this
    replaces."""
    for rel in related:
        try:
            normalized_value = normalize_identifier(rel.identifier_type, rel.value)
        except IdentifierError:
            continue

        matches = find_all_existing_studies_by_identifier(session, rel.identifier_type, normalized_value)

        if not matches:
            # Case A: no match anywhere -- record the identifier, nothing to reconcile.
            session.add(
                ExternalIdentifier(
                    study_id=study.study_id,
                    identifier_type=rel.identifier_type.value,
                    identifier_value=normalized_value,
                    source=rel.source,
                    verified=(rel.confidence != SupportType.INFERRED),
                )
            )
            session.flush()
            continue

        if len(matches) == 1 and matches[0].study_id == study.study_id:
            continue  # already recorded on this same study, no-op

        candidates = [match for match in matches if match.study_id != study.study_id]
        if not candidates:
            continue

        if len(candidates) == 1:
            study = _resolve_against_one_candidate(session, study, candidates[0], rel, source)
        else:
            study = _resolve_against_many_candidates(session, study, candidates, rel, source)

    return study


def _resolve_against_one_candidate(
    session: Session, study: Study, existing_study: Study, rel: RelatedIdentifier, source: Source | None
) -> Study:
    if rel.confidence == SupportType.STRUCTURED_SOURCE:
        # Tier 1: safe to auto-link, no consistency check -- matches every
        # existing adapter's today-unconditional merge behavior exactly.
        return merge_study_into(session, absorb=existing_study, into=study)

    outcome = check_study_consistency(session, existing_study=existing_study, new_source=source)
    if outcome.status == "consistent" and rel.confidence != SupportType.INFERRED:
        return merge_study_into(session, absorb=existing_study, into=study)

    return _create_sibling_and_flag(
        session, study, [existing_study], rel, source,
        reason=outcome.message if outcome.status != "consistent" else "tier-3 evidence never merges alone",
    )


def _resolve_against_many_candidates(
    session: Session, study: Study, candidates: list[Study], rel: RelatedIdentifier, source: Source | None
) -> Study:
    best: Study | None = None
    if rel.confidence != SupportType.INFERRED:
        for candidate in candidates:
            outcome = check_study_consistency(session, existing_study=candidate, new_source=source)
            if outcome.status == "consistent":
                best = candidate
                break

    if best is not None:
        return merge_study_into(session, absorb=best, into=study)

    return _create_sibling_and_flag(session, study, candidates, rel, source, reason="multiple_existing_claims")


def _create_sibling_and_flag(
    session: Session,
    study: Study,
    conflicting_studies: list[Study],
    rel: RelatedIdentifier,
    source: Source | None,
    reason: str,
) -> Study:
    """The hard case: this new evidence looks like it shares an accession
    with `conflicting_studies` but fails the consistency check (or is
    tier-3, which never merges alone), so it must NOT be merged in. Splits
    off a new sibling Study carrying just this evidence, links the
    ambiguous identifier to it via ExternalIdentifier +
    RelationshipType.SHARES_ACCESSION_WITH, and drops a PENDING
    CandidateMatch for human review. `study` itself is returned unchanged
    apart from this one Source's ownership possibly moving off of it."""
    sibling = Study(canonical_status=CanonicalStatus.CANDIDATE.value)
    session.add(sibling)
    session.flush()

    session.add(
        ExternalIdentifier(
            study_id=sibling.study_id,
            identifier_type=rel.identifier_type.value,
            identifier_value=normalize_identifier(rel.identifier_type, rel.value),
            source=rel.source,
            relationship_type=RelationshipType.SHARES_ACCESSION_WITH.value,
            verified=False,
        )
    )

    if source is not None:
        source.study_id = sibling.study_id
        link_source_to_study(
            session, source, sibling.study_id,
            relationship_type=RelationshipType.IS_HOME_OF, confidence=rel.confidence,
        )

        session.query(RawFact).filter(RawFact.source_id == source.source_id).update(
            {"study_id": sibling.study_id}, synchronize_session=False
        )

        # Entity ownership rule: only move an Entity if 100% of its
        # RawFacts came from this one source_id -- Entity has no source_id
        # of its own, so partial ownership is unresolvable without
        # inventing a new column. A false negative (Entity stays behind)
        # is visible and recoverable later; a false positive (moving an
        # Entity that shouldn't move) would silently scatter one physical
        # sample's facts across two studies, which is strictly worse.
        candidate_entity_ids = {
            row.entity_id
            for row in session.query(RawFact.entity_id)
            .filter(RawFact.source_id == source.source_id, RawFact.entity_id.isnot(None))
            .distinct()
        }
        for entity_id in candidate_entity_ids:
            other_fact_exists = (
                session.query(RawFact.fact_id)
                .filter(RawFact.entity_id == entity_id, RawFact.source_id != source.source_id)
                .first()
                is not None
            )
            if not other_fact_exists:
                session.query(Entity).filter(Entity.entity_id == entity_id).update(
                    {"study_id": sibling.study_id}, synchronize_session=False
                )
    else:
        logger.warning(
            "identifier %s=%s could not be attributed to one Source for reassignment -- "
            "flagged for review, no rows moved",
            rel.identifier_type.value, rel.value,
        )

    session.add(
        CandidateMatch(
            study_a_id=sibling.study_id,
            study_b_id=conflicting_studies[0].study_id if len(conflicting_studies) == 1 else None,
            candidate_record_ref=(
                None if len(conflicting_studies) == 1
                else ",".join(candidate.study_id for candidate in conflicting_studies)
            ),
            match_method=CandidateMatchMethod.EXPLICIT_RELATIONSHIP.value,
            review_status=CandidateMatchReviewStatus.PENDING.value,
            supporting_features={
                "identifier_type": rel.identifier_type.value,
                "identifier_value": rel.value,
                "confidence": rel.confidence.value,
                "reason": reason,
            },
        )
    )
    session.flush()
    return study
