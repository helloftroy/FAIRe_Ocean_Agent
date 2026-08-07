"""Within-one-study SAMPLE entity reconciliation: a paper's own native
sample naming (e.g. a supplement table's own sample-ID column, materialized
by sources/supplement_parsing.py -- "GC04_1") vs. the real, structurally-
resolved NCBI BioSample accession for the same physical sample
("SAMN11268098"). These get created as two entirely separate Entity rows
today, since entity matching is strict equality on (entity_level,
external_identifier), and the two naming schemes never coincide as strings.

Distinct from identity/entity_linking.py's cross-study EntityStudy sharing
(the same real accession claimed by more than one Study) -- this reconciles
two DIFFERENT identifier schemes for what is provably the same physical
object WITHIN one study, using a real, already-present cross-reference: a
BioSample's own `source_material_id`-family attribute frequently embeds the
submitter's own native sample name (confirmed live: a real BioSample's
value was "GS14-GC08-1", and the same paper's own supplement table names
the identical physical sample "GC04_1" -- once elsewhere in this same
codebase's own comments; see sources/ncbi.py's `_derive_depth_from_source_
material_id`, a different, narrower use of this same loosely-structured
free-text field).

Deliberately NOT a destructive Entity-level merge (no RawFact.entity_id
reassignment): the matching evidence here is DETERMINISTICALLY_DERIVED
(a token derivation over free text), the same evidence tier
identity/resolution.py's own docstring says must never merge Study rows
unreviewed -- there's no un-merge mechanism in this schema for a false
positive at the Entity level either. Instead, a confident match is recorded
as a durable, queryable EntityRelationship
(EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS), and exports/faire.py folds
the two entities' values into one export row (pipe-joining genuine
conflicts) without touching the underlying rows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, ReviewStatus, SupportType
from fair_ocean_agent.database.models import Entity, EntityRelationship, EntityStudy, RawFact
from fair_ocean_agent.identity.entity_linking import get_or_create_entity_relationship

# NcbiBioSampleAdapter's own extraction_method for every structurally-
# resolved BioSample fact (workflow/handlers.py::_persist_source_and_facts,
# f"adapter:{adapter.name}") -- narrow and explicit on purpose. The ALIAS
# side is deliberately NOT keyed on a matching "known unaccessioned
# adapter" allow-list; it's defined by the ABSENCE of an accessioned fact,
# so a future PANGAEA/BCO-DMO-derived alias entity is picked up for free
# without this module needing to know about every adapter that could ever
# produce one (matches sources/supplement_parsing.py's own framing of
# itself as one of several possible producers of this kind of entity).
_ACCESSIONED_SAMPLE_EXTRACTION_METHODS = frozenset({"adapter:ncbi_biosample"})

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_SOURCE_MATERIAL_ID_NORMALIZED = "sourcematerialid"


@dataclass
class ReconciliationResult:
    matched: int
    ambiguous: int
    alias_candidates_considered: int


def _normalize_fact_type(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_SPLIT_RE.split(value) if token]


def _is_trailing_subsequence(alias_tokens: list[str], candidate_tokens: list[str]) -> bool:
    n = len(alias_tokens)
    if n == 0 or len(candidate_tokens) < n:
        return False
    return candidate_tokens[-n:] == alias_tokens


def _has_accessioned_fact(session: Session, entity_id: str) -> bool:
    return (
        session.query(RawFact.fact_id)
        .filter(RawFact.entity_id == entity_id, RawFact.extraction_method.in_(_ACCESSIONED_SAMPLE_EXTRACTION_METHODS))
        .first()
        is not None
    )


def _source_material_id_values(session: Session, entity_id: str) -> set[str]:
    """Distinct source_material_id-family raw values for one entity --
    deliberately a SET, not a list of rows: a shared entity carries one
    duplicate source-material-id RawFact per linked study's own
    independent resolution pass (confirmed live), and counting rows
    instead of distinct values would produce false "ambiguity" against a
    single, already-correct value repeated twice."""
    rows = session.execute(
        select(RawFact.fact_type_candidate, RawFact.raw_value).where(RawFact.entity_id == entity_id)
    ).all()
    return {
        raw_value
        for fact_type, raw_value in rows
        if fact_type and raw_value and _normalize_fact_type(fact_type) == _SOURCE_MATERIAL_ID_NORMALIZED
    }


def _alias_candidates(session: Session, study_id: str) -> list[Entity]:
    already_aliased_ids = set(
        session.scalars(
            select(EntityRelationship.from_entity_id).where(
                EntityRelationship.relationship_type == EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS.value
            )
        ).all()
    )
    candidates = []
    for entity in session.scalars(
        select(Entity).where(Entity.study_id == study_id, Entity.entity_level == EntityLevel.SAMPLE.value)
    ):
        if entity.entity_id in already_aliased_ids:
            continue
        if _has_accessioned_fact(session, entity.entity_id):
            continue
        candidates.append(entity)
    return candidates


def _canonical_candidates(session: Session, study_id: str) -> list[Entity]:
    linked_entity_ids = set(
        session.scalars(select(EntityStudy.entity_id).where(EntityStudy.study_id == study_id)).all()
    )
    candidates = []
    for entity_id in linked_entity_ids:
        entity = session.get(Entity, entity_id)
        if entity is None or entity.entity_level != EntityLevel.SAMPLE.value:
            continue
        if _has_accessioned_fact(session, entity.entity_id):
            candidates.append(entity)
    return candidates


def _record_match(session: Session, study_id: str, alias: Entity, canonical_entity_id: str) -> None:
    get_or_create_entity_relationship(
        session,
        study_id,
        from_entity_id=alias.entity_id,
        to_entity_id=canonical_entity_id,
        relationship_type=EntityRelationshipType.SAME_PHYSICAL_SAMPLE_AS,
    )
    already_recorded = session.scalars(
        select(RawFact.fact_id).where(
            RawFact.entity_id == alias.entity_id, RawFact.fact_type_candidate == "sample_alias_match"
        )
    ).first()
    if already_recorded is not None:
        return
    session.add(
        RawFact(
            study_id=study_id,
            entity_id=alias.entity_id,
            source_id=None,
            source_locator="identity.sample_alias_reconciliation",
            raw_field_name="sample_alias_match",
            raw_value=canonical_entity_id,
            fact_type_candidate="sample_alias_match",
            entity_level=EntityLevel.SAMPLE.value,
            support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
            extraction_method="identity.sample_alias_reconciliation",
            review_status=ReviewStatus.ACCEPTED.value,
            confidence_metadata={"canonical_entity_id": canonical_entity_id},
        )
    )


def _flag_ambiguous(session: Session, study_id: str, alias: Entity, candidate_entity_ids: set[str]) -> None:
    already_flagged = session.scalars(
        select(RawFact.fact_id).where(
            RawFact.entity_id == alias.entity_id,
            RawFact.fact_type_candidate == "sample_alias_ambiguous",
            RawFact.review_status == ReviewStatus.NEEDS_REVIEW.value,
        )
    ).first()
    if already_flagged is not None:
        return
    session.add(
        RawFact(
            study_id=study_id,
            entity_id=alias.entity_id,
            source_id=None,
            source_locator="identity.sample_alias_reconciliation",
            raw_field_name="sample_alias_ambiguous",
            raw_value=f"no unambiguous canonical match among: {sorted(candidate_entity_ids)}",
            fact_type_candidate="sample_alias_ambiguous",
            entity_level=EntityLevel.SAMPLE.value,
            support_type=SupportType.DETERMINISTICALLY_DERIVED.value,
            extraction_method="identity.sample_alias_reconciliation",
            review_status=ReviewStatus.NEEDS_REVIEW.value,
            confidence_metadata={"candidate_entity_ids": sorted(candidate_entity_ids)},
        )
    )


def reconcile_sample_aliases(session: Session, study_id: str) -> ReconciliationResult:
    """Idempotent, safe to call every time (e.g. from mapping/faire.py's
    map_study_to_faire) -- self-heals if the "losing" side of a pairing
    (typically the real BioSample resolution, or the supplement table)
    shows up on a later run."""
    alias_entities = _alias_candidates(session, study_id)
    canonical_entities = _canonical_candidates(session, study_id)
    canonical_token_sets = {
        entity.entity_id: [_tokenize(value) for value in _source_material_id_values(session, entity.entity_id)]
        for entity in canonical_entities
    }

    matched = 0
    ambiguous = 0
    for alias in alias_entities:
        alias_tokens = _tokenize(alias.external_identifier or "")
        if not alias_tokens:
            continue

        distinct_matches: set[str] = set()
        for entity_id, token_sets in canonical_token_sets.items():
            if any(_is_trailing_subsequence(alias_tokens, tokens) for tokens in token_sets):
                distinct_matches.add(entity_id)

        if len(distinct_matches) == 1:
            _record_match(session, study_id, alias, next(iter(distinct_matches)))
            matched += 1
        elif len(distinct_matches) >= 2:
            _flag_ambiguous(session, study_id, alias, distinct_matches)
            ambiguous += 1
        # zero matches: silent no-op -- the common/expected case.

    return ReconciliationResult(
        matched=matched, ambiguous=ambiguous, alias_candidates_considered=len(alias_entities)
    )
