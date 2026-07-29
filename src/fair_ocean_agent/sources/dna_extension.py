"""Helpers for DNA-derived-data fields embedded in repository occurrence records.

GBIF and OBIS primarily expose dataset/occurrence metadata through their
public APIs. When publishers include Darwin Core `associatedSequences` or
GBIF DNA-derived-data extension terms, those terms may appear either as
top-level occurrence fields or nested inside an `extensions` object. This
module extracts only exact FAIRe/extension terms and a few explicit camelCase
aliases; fuzzy label matching belongs in review tooling, not adapters.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from fair_ocean_agent.database.enums import EntityLevel
from fair_ocean_agent.sources.base import RawFactCandidate


DNA_DERIVED_FACT_ALIASES: dict[str, tuple[str, ...]] = {
    "associatedSequences": (
        "associatedSequences",
        "associated_sequences",
        "dwc:associatedSequences",
    ),
    "target_gene": (
        "target_gene",
        "targetGene",
    ),
    "target_subfragment": (
        "target_subfragment",
        "targetSubfragment",
    ),
    "pcr_primer_forward": (
        "pcr_primer_forward",
        "pcrPrimerForward",
        "forward_primer_sequence",
        "forwardPrimerSequence",
    ),
    "pcr_primer_reverse": (
        "pcr_primer_reverse",
        "pcrPrimerReverse",
        "reverse_primer_sequence",
        "reversePrimerSequence",
    ),
    "pcr_primer_name_forward": (
        "pcr_primer_name_forward",
        "pcrPrimerNameForward",
        "forward_primer_name",
        "forwardPrimerName",
    ),
    "pcr_primer_name_reverse": (
        "pcr_primer_name_reverse",
        "pcrPrimerNameReverse",
        "reverse_primer_name",
        "reversePrimerName",
    ),
    "pcr_primer_reference_forward": (
        "pcr_primer_reference_forward",
        "pcrPrimerReferenceForward",
    ),
    "pcr_primer_reference_reverse": (
        "pcr_primer_reference_reverse",
        "pcrPrimerReferenceReverse",
    ),
    "annealingTemp": (
        "annealingTemp",
        "annealing_temp",
        "annealingTemperature",
    ),
    "ampliconSize": (
        "ampliconSize",
        "amplicon_size",
    ),
    "pcr_cond": (
        "pcr_cond",
        "pcrConditions",
        "pcr_amplification_conditions",
    ),
    "assay_type": (
        "assay_type",
        "assayType",
    ),
    "assay_name": (
        "assay_name",
        "assayName",
    ),
}

_ALIAS_TO_CANONICAL = {
    alias.lower(): canonical
    for canonical, aliases in DNA_DERIVED_FACT_ALIASES.items()
    for alias in aliases
}


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _occurrence_rows(payload: Any) -> Iterable[tuple[int, Any]]:
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            yield index, item
        return
    if not isinstance(payload, dict):
        return
    for key in ("results", "records", "data", "occurrences"):
        value = payload.get(key)
        if isinstance(value, list):
            for index, item in enumerate(value):
                yield index, item
            return


def _iter_fields(value: Any, path: str) -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield key, child_path, child
            yield from _iter_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield from _iter_fields(child, child_path)


def extract_dna_derived_facts(source_name: str, occurrence_payload: Any) -> list[RawFactCandidate]:
    facts: list[RawFactCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    for row_index, occurrence in _occurrence_rows(occurrence_payload):
        for raw_key, path, value in _iter_fields(occurrence, f"{source_name}.occurrence[{row_index}]"):
            if value in (None, "", [], {}):
                continue
            canonical = _ALIAS_TO_CANONICAL.get(str(raw_key).lower())
            if canonical is None:
                continue
            raw_value = _stringify(value)
            dedupe_key = (canonical, raw_value, path)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=canonical,
                    raw_field_name=str(raw_key),
                    raw_value=raw_value,
                    source_locator=path,
                )
            )

    return facts
