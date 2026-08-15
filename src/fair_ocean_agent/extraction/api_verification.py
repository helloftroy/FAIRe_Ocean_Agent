"""Verifies specific structured API (BioSample/ENA/...) values against the
paper's own text, but only when there's a real, evidenced reason to
suspect a mislabeling -- never a blanket "fact-check everything" pass.
Per an explicit user request, scoped narrowly to one concrete, confirmed
failure mode: a BioSample submitter reporting a sediment/soil sample's
site water depth under "elevation" instead of FAIRe's own dedicated
`tot_depth_water_col` field (real audit: 10.1093/ismejo/wrae013,
STUDY-295abf4a8f43 -- BioSample says "elevation = 34 m", the paper says
"at a site with 34 m water depth"). `minimumDepthInMeters`/
`maximumDepthInMeters` are deliberately left untouched by this check --
for a sediment/soil sample those correctly describe depth WITHIN the
sediment/soil (a real, separately-reported BioSample "depth" attribute,
e.g. "0-1 cm sediment"), a different concept from the water column depth
above the site, which is exactly what `tot_depth_water_col` is for (see
its own FAIRe definition's own worked example: "if a sea surface water
sample was collected at a sampling site where the water depth was 15m,
enter 15 here and 0 under the terms minimumDepthInMeters and
maximumDepthInMeters").

A confirmed mismatch is corrected in place (elev quarantined,
tot_depth_water_col written) AND logged to the durable
`api_paper_corrections` table -- also an explicit user request: "this new
table isn't manual, this should be built into the code."
"""

from __future__ import annotations

import re

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import EntityLevel, ReviewStatus, SupportType
from fair_ocean_agent.database.models import ApiPaperCorrection, Entity, RawFact, StandardizedValue
from fair_ocean_agent.extraction.search_flags import confirm_value_described_as_depth
from fair_ocean_agent.llm.base import LLMBackend
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA

_ELEV_DEPTH_MISLABEL_DETECTOR = "elev_depth_mislabel_check"
_SEDIMENT_SOIL_ENV_MEDIUM_RE = re.compile(r"\b(?:sediment|soil|mud|sand)\b", re.IGNORECASE)
_NUMERIC_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)")
_CORRECTED_TARGET_FIELD = "tot_depth_water_col"


def detect_and_correct_elev_depth_mislabeling(
    backend: LLMBackend,
    session: Session,
    study_id: str,
    section_texts: list[tuple[str, str]],
    *,
    source_id: str | None,
    locator_prefix: str,
) -> None:
    """Call after `mapping.faire.map_study_to_faire` has already run for
    this study (so elev/tot_depth_water_col/env_medium StandardizedValues
    reflect the study's current facts). For each SAMPLE entity where elev
    is populated, tot_depth_water_col is not, and env_medium indicates
    soil/sediment, asks the LLM to check whether the paper's own text ties
    that elev number to a water/site depth concept instead -- confirmed
    matches are grouped by elev value first, so one paper-text lookup
    covers every sample sharing the same reported site-level elevation."""
    sample_entities = list(
        session.scalars(
            select(Entity).where(
                Entity.study_id == study_id, Entity.entity_level == EntityLevel.SAMPLE.value
            )
        )
    )
    if not sample_entities:
        return

    candidates_by_elev_value: dict[str, list[Entity]] = {}
    for entity in sample_entities:
        elev_value = session.scalar(
            select(StandardizedValue.standardized_value).where(
                StandardizedValue.study_id == study_id,
                StandardizedValue.entity_id == entity.entity_id,
                StandardizedValue.target_schema == TARGET_SCHEMA,
                StandardizedValue.target_field == "elev",
            )
        )
        if not elev_value:
            continue
        has_tot_depth_water_col = session.scalar(
            select(StandardizedValue.standardized_value_id).where(
                StandardizedValue.study_id == study_id,
                StandardizedValue.entity_id == entity.entity_id,
                StandardizedValue.target_schema == TARGET_SCHEMA,
                StandardizedValue.target_field == _CORRECTED_TARGET_FIELD,
            )
        )
        if has_tot_depth_water_col is not None:
            continue
        env_medium = session.scalar(
            select(StandardizedValue.standardized_value).where(
                StandardizedValue.study_id == study_id,
                StandardizedValue.entity_id == entity.entity_id,
                StandardizedValue.target_schema == TARGET_SCHEMA,
                StandardizedValue.target_field == "env_medium",
            )
        )
        if not env_medium or not _SEDIMENT_SOIL_ENV_MEDIUM_RE.search(env_medium):
            continue
        candidates_by_elev_value.setdefault(elev_value, []).append(entity)

    for elev_value, entities in candidates_by_elev_value.items():
        numeric_match = _NUMERIC_VALUE_RE.search(elev_value)
        if numeric_match is None:
            continue
        numeric_value = numeric_match.group(1)
        confirmed_quote = confirm_value_described_as_depth(backend, numeric_value, section_texts)
        if confirmed_quote is None:
            continue
        for entity in entities:
            _apply_elev_depth_correction(
                session,
                study_id=study_id,
                entity=entity,
                elev_value=elev_value,
                numeric_value=numeric_value,
                supporting_quote=confirmed_quote,
                source_id=source_id,
                locator_prefix=locator_prefix,
            )


def _apply_elev_depth_correction(
    session: Session,
    *,
    study_id: str,
    entity: Entity,
    elev_value: str,
    numeric_value: str,
    supporting_quote: str,
    source_id: str | None,
    locator_prefix: str,
) -> None:
    already_corrected = session.scalar(
        select(ApiPaperCorrection.correction_id).where(
            ApiPaperCorrection.study_id == study_id,
            ApiPaperCorrection.entity_id == entity.entity_id,
            ApiPaperCorrection.detector == _ELEV_DEPTH_MISLABEL_DETECTOR,
        )
    )
    if already_corrected is not None:
        return  # idempotent: already corrected on a prior run

    # Quarantine the mislabeled elev fact(s) rather than deleting them --
    # same "REJECTED, not destroyed" audit-trail discipline used
    # throughout this pipeline (see workflow/handlers.py's own re-
    # extraction quarantine).
    session.execute(
        update(RawFact)
        .where(
            RawFact.study_id == study_id,
            RawFact.entity_id == entity.entity_id,
            RawFact.fact_type_candidate == "elev",
            RawFact.review_status != ReviewStatus.REJECTED.value,
        )
        .values(review_status=ReviewStatus.REJECTED.value)
    )

    session.add(
        RawFact(
            study_id=study_id,
            entity_id=entity.entity_id,
            entity_level=EntityLevel.SAMPLE.value,
            fact_type_candidate=_CORRECTED_TARGET_FIELD,
            raw_field_name=_CORRECTED_TARGET_FIELD,
            raw_value=f"{numeric_value} m",
            source_id=source_id,
            source_locator=f"{locator_prefix}:elev_depth_mislabel_check:{entity.entity_id}",
            support_type=SupportType.EXPLICIT.value,
            evidence_quote=supporting_quote,
            extraction_method="llm_verified_elev_depth_correction",
            confidence_metadata={
                "detector": _ELEV_DEPTH_MISLABEL_DETECTOR,
                "description": (
                    f"BioSample's own elev={elev_value} was confirmed by the paper's own text to "
                    "actually be the site's total water column depth, not elevation above sea level."
                ),
            },
        )
    )

    session.add(
        ApiPaperCorrection(
            study_id=study_id,
            entity_id=entity.entity_id,
            api_faire_term="elev",
            api_value=elev_value,
            corrected_faire_term=_CORRECTED_TARGET_FIELD,
            corrected_value=numeric_value,
            supporting_quote=supporting_quote,
            detector=_ELEV_DEPTH_MISLABEL_DETECTOR,
        )
    )
