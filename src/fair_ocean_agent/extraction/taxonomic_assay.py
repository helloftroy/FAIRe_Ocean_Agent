"""Structured (non-LLM) production of `assay_target_taxa` (FAIRe
`targetTaxonomicAssay`) facts, derived from real BioSample records rather
than free text.

A real NCBI BioSample's own `organism` attribute (`sources/ncbi.py`,
`fact_type_candidate="organism"`, one row per linked SAMPLE entity) is
submitter-supplied taxonomic identity for that physical sample, and for an
eDNA/metabarcoding BioSample this is very often the same taxon family the
paper's assay was designed to target. This is the "API" signal referenced
by `mapping/faire.py`'s own `targetTaxonomicAssay` conflict resolution --
distinct from (and preferred over, but never replacing) the LLM's own
free-text search over the paper itself, which can report a more specific
target than any one BioSample's `organism` field does.

An earlier version of this module instead scanned only title/abstract/
keyword publication metadata for target-taxon phrases -- retired in favor
of this BioSample-derived signal plus a proper phrase-search-then-LLM-
judged pass over the full paper text
(`extraction/search_flags.LLM_JUDGED_SEARCH_FIELDS`'s own
`assay_target_taxa`/`study_target_taxonomic_scope` entries), which covers
PCR-methods-section language the title/abstract/keyword scope never saw.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import EntityStudy, RawFact


def _distinct_organism_values(session: Session, study_id: str) -> list[str]:
    """Every distinct `organism` value across the study's linked SAMPLE (or
    any other) entities, in first-seen order -- deliberately entity-level
    agnostic and not filtered to SAMPLE only, since `organism` is only ever
    emitted by the BioSample adapter at SAMPLE level today, but a future
    adapter emitting it elsewhere should be picked up for free rather than
    silently ignored. Queried via `EntityStudy` (home-or-shared, matching
    every other study-wide aggregation in this codebase) rather than
    `Entity.study_id`, so a sample this study only *shares* (never homes)
    still contributes its own organism value."""
    entity_ids = select(EntityStudy.entity_id).where(EntityStudy.study_id == study_id)
    rows = session.execute(
        select(RawFact.raw_value).where(
            RawFact.entity_id.in_(entity_ids),
            RawFact.fact_type_candidate == "organism",
            RawFact.review_status != ReviewStatus.REJECTED.value,
        )
    ).all()
    values: list[str] = []
    seen: set[str] = set()
    for (raw_value,) in rows:
        if not raw_value:
            continue
        value = raw_value.strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def sync_assay_target_taxa_from_biosample_organisms(session: Session, study_id: str) -> None:
    """Idempotent, re-run-safe: called from `mapping/faire.py::
    map_study_to_faire` (mirroring `identity/sample_alias_reconciliation.py
    ::reconcile_sample_aliases`'s own pattern) so it always reflects every
    SAMPLE entity currently linked to the study regardless of discovery
    order, and self-heals if more BioSamples resolve on a later run.

    Writes (or updates in place) a single study-wide `assay_target_taxa`
    RawFact with `support_type=STRUCTURED_SOURCE`, distinguishing it from
    the LLM-judged-search path's `SupportType.EXPLICIT` facts of the same
    `fact_type_candidate` -- `mapping/faire.py`'s conflict resolution uses
    exactly that distinction to prefer this API-derived signal without
    discarding the LLM's own values."""
    values = _distinct_organism_values(session, study_id)
    existing = session.scalar(
        select(RawFact).where(
            RawFact.study_id == study_id,
            RawFact.entity_id.is_(None),
            RawFact.fact_type_candidate == "assay_target_taxa",
            RawFact.extraction_method == "derived:biosample_organism_aggregation",
        )
    )
    if not values:
        if existing is not None:
            existing.review_status = ReviewStatus.REJECTED.value
        return

    raw_value = " | ".join(values)
    if existing is not None:
        if existing.raw_value != raw_value:
            existing.raw_value = raw_value
            existing.confidence_metadata = {
                "detector": "biosample_organism_aggregation",
                "distinct_organism_values": values,
            }
        existing.review_status = ReviewStatus.ACCEPTED.value
        return

    session.add(
        RawFact(
            study_id=study_id,
            entity_id=None,
            source_id=None,
            source_locator="biosample_organism_aggregation",
            raw_field_name="organism",
            raw_value=raw_value,
            fact_type_candidate="assay_target_taxa",
            entity_level=EntityLevel.STUDY.value,
            support_type=SupportType.STRUCTURED_SOURCE.value,
            extraction_method="derived:biosample_organism_aggregation",
            review_status=ReviewStatus.ACCEPTED.value,
            confidence_metadata={
                "detector": "biosample_organism_aggregation",
                "distinct_organism_values": values,
            },
        )
    )
