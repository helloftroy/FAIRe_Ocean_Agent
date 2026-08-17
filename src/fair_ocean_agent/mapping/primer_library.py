"""Corpus-wide primer name -> sequence lookup: treats every study's own
raw_facts as a shared reference library, so a primer's sequence -- once
found in ANY paper's own text, whether stated directly or eventually
recovered via a chased citation (extraction/publication_metadata.py::
extract_primer_reference_citations + workflow/handlers.py::
handle_discover_primer_reference_study) -- becomes available to every
OTHER paper that only names the same primer without giving its own
sequence. Per an explicit user request: "so that it becomes easy to pull
out that sequence for all papers in the future." No new table: a plain
cross-study RawFact query, same idiom as identity/deduplication.py's
find_existing_study_by_identifier.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import RawFact

# forward/reverse primer name <-> sequence field pairs.
PRIMER_NAME_TO_SEQUENCE_FIELD = {
    "pcr_primer_name_forward": "pcr_primer_forward",
    "pcr_primer_name_reverse": "pcr_primer_reverse",
}
# fact_type_candidate a corpus-backfilled sequence is stored under --
# deliberately distinct from "pcr_primer_forward"/"_reverse" (this paper's
# own extraction) so mapping/rules.py can give it its own, always-
# review_required=True MappingRule rather than inheriting whichever
# review_required setting the paper's-own-extraction rule happens to have.
INHERITED_SEQUENCE_FIELD = {
    "pcr_primer_forward": "pcr_primer_forward_inherited",
    "pcr_primer_reverse": "pcr_primer_reverse_inherited",
}


def study_primer_name(session: Session, study_id: str, name_field: str) -> str | None:
    """The oldest surviving (non-rejected) value for a primer-name field
    on this study -- same "first fact wins" convention as mapping/
    faire.py's own map_study_to_faire."""
    return session.scalars(
        select(RawFact.raw_value)
        .where(
            RawFact.study_id == study_id,
            RawFact.fact_type_candidate == name_field,
            RawFact.review_status != ReviewStatus.REJECTED.value,
            RawFact.raw_value.is_not(None),
        )
        .order_by(RawFact.created_at)
    ).first()


def _study_has_own_sequence(session: Session, study_id: str, sequence_field: str) -> bool:
    return (
        session.scalars(
            select(RawFact.fact_id).where(
                RawFact.study_id == study_id,
                RawFact.fact_type_candidate.in_((sequence_field, INHERITED_SEQUENCE_FIELD[sequence_field])),
                RawFact.review_status != ReviewStatus.REJECTED.value,
            )
        ).first()
        is not None
    )


def corpus_primer_sequence(session: Session, primer_name: str, sequence_field: str) -> str | None:
    """Any study (including this one) that has this exact primer name
    (case-insensitive; no fuzzy matching -- "515F" and "515f" match,
    "515F" and "515F-Y" do not, see module docstring) together with its
    own real sequence for `sequence_field` is a usable source. Oldest
    match wins."""
    name_field = next((k for k, v in PRIMER_NAME_TO_SEQUENCE_FIELD.items() if v == sequence_field), None)
    if name_field is None or not primer_name.strip():
        return None
    target = primer_name.strip().casefold()
    name_fact = aliased(RawFact)
    sequence_fact = aliased(RawFact)
    rows = session.execute(
        select(name_fact.raw_value, sequence_fact.raw_value, sequence_fact.created_at)
        .join(name_fact, name_fact.study_id == sequence_fact.study_id)
        .where(
            sequence_fact.fact_type_candidate == sequence_field,
            sequence_fact.review_status != ReviewStatus.REJECTED.value,
            sequence_fact.raw_value.is_not(None),
            name_fact.fact_type_candidate == name_field,
            name_fact.review_status != ReviewStatus.REJECTED.value,
        )
        .order_by(sequence_fact.created_at)
    ).all()
    for name_value, sequence_value, _created_at in rows:
        if name_value and name_value.strip().casefold() == target:
            return sequence_value
    return None


def resolve_primer_sequences_from_corpus(session: Session, study_id: str) -> None:
    """Idempotent, safe to call every time (e.g. from mapping/faire.py's
    map_study_to_faire) -- self-heals if the corpus later learns a
    primer's sequence it didn't know on an earlier run (e.g. once a
    chased reference paper finishes processing)."""
    for name_field, sequence_field in PRIMER_NAME_TO_SEQUENCE_FIELD.items():
        if _study_has_own_sequence(session, study_id, sequence_field):
            continue
        primer_name = study_primer_name(session, study_id, name_field)
        if not primer_name:
            continue
        sequence_value = corpus_primer_sequence(session, primer_name, sequence_field)
        if not sequence_value:
            continue
        inherited_field = INHERITED_SEQUENCE_FIELD[sequence_field]
        session.add(
            RawFact(
                study_id=study_id,
                entity_id=None,
                source_id=None,
                source_locator="mapping.primer_library",
                raw_field_name=inherited_field,
                raw_value=sequence_value,
                fact_type_candidate=inherited_field,
                entity_level=EntityLevel.STUDY.value,
                support_type=SupportType.INFERRED.value,
                extraction_method="mapping.primer_library",
                review_status=ReviewStatus.NEEDS_REVIEW.value,
                confidence_metadata={"primer_name": primer_name, "detector": "primer_library_corpus_lookup"},
            )
        )
