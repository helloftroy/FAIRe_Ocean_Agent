"""Small ENVO label expansion helpers for MIxS environmental triad fields.

NCBI BioSample often returns only an ontology ID (e.g. ``ENVO:00010483``)
for env_broad_scale/env_local_scale/env_medium. FAIRe output is much easier
to inspect if we retain the accession but prepend the readable label:
``environmental material | ENVO:00010483``.

This is deliberately deterministic and local. Unknown IDs are preserved as
IDs so we never invent labels; add high-frequency terms here as they appear.
"""
from __future__ import annotations

import re

ENVO_LABELS = {
    "ENVO:00000015": "ocean",
    "ENVO:00000016": "sea",
    "ENVO:00000020": "lake",
    "ENVO:00000022": "river",
    "ENVO:00000023": "stream",
    "ENVO:00000063": "water body",
    "ENVO:00000067": "marine benthic feature",
    "ENVO:00000134": "sea floor",
    "ENVO:00000170": "coral reef",
    "ENVO:00000230": "reef",
    "ENVO:00000231": "estuary",
    "ENVO:00000254": "marine water body",
    "ENVO:00000316": "intertidal zone",
    "ENVO:00000428": "biome",
    "ENVO:00000446": "terrestrial biome",
    "ENVO:00000447": "marine biome",
    "ENVO:00000486": "shoreline",
    "ENVO:00001998": "soil",
    "ENVO:00002005": "air",
    "ENVO:00002006": "water",
    "ENVO:00002007": "sediment",
    "ENVO:00002010": "saline water",
    "ENVO:00002011": "fresh water",
    "ENVO:00002012": "hypersaline water",
    "ENVO:00002149": "sea water",
    "ENVO:00002150": "coastal sea water",
    "ENVO:00002160": "estuarine mud",
    "ENVO:00002164": "marine sediment",
    "ENVO:00002200": "sea ice",
    "ENVO:00002297": "environmental feature",
    "ENVO:00010483": "environmental material",
    "ENVO:01000020": "estuarine biome",
    "ENVO:01000023": "marine pelagic biome",
    "ENVO:01000024": "marine benthic biome",
    "ENVO:01000181": "mangrove biome",
    "ENVO:01000298": "continental margin",
    "ENVO:01000301": "estuarine water",
    "ENVO:01000317": "marine hydrothermal vent biome",
    "ENVO:01000320": "marine hydrothermal vent",
    "ENVO:01000324": "hydrothermal vent fluid",
    "ENVO:01000686": "marine reef biome",
    "ENVO:01000687": "coral reef biome",
    "ENVO:01000688": "marine coral reef biome",
    "ENVO:01000925": "marine water",
    "ENVO:01001191": "coastal water",
}

_ENVO_RE = re.compile(r"(?:http://purl\.obolibrary\.org/obo/)?ENVO[:_](\d{7,8})", re.IGNORECASE)
_LABELLED_ENVO_RE = re.compile(
    r"(?P<label>[^\[\]|;]+?)\s*\[\s*(?P<id>(?:http://purl\.obolibrary\.org/obo/)?ENVO[:_]\d{7,8})\s*\]",
    re.IGNORECASE,
)


def normalize_envo_id(value: str) -> str:
    match = _ENVO_RE.search(value)
    if not match:
        return value.strip()
    return f"ENVO:{match.group(1)}"


def expand_envo_term(value: str) -> str:
    """Return ``label | ENVO:id`` when an ENVO ID is present.

    Existing ``label [ENVO:id]`` values keep the source label. Bare unknown
    ENVO IDs stay bare instead of being guessed.
    """
    raw = " ".join(str(value).split()).strip()
    if not raw:
        return raw
    labelled = _LABELLED_ENVO_RE.search(raw)
    if labelled:
        label = labelled.group("label").strip()
        envo_id = normalize_envo_id(labelled.group("id"))
        return f"{label} | {envo_id}"
    match = _ENVO_RE.search(raw)
    if not match:
        return raw
    envo_id = f"ENVO:{match.group(1)}"
    label = ENVO_LABELS.get(envo_id.upper())
    return f"{label} | {envo_id}" if label else envo_id


def expand_envo_terms(value: str | None) -> str | None:
    if value is None:
        return None
    pieces = re.split(r"\s*(?:\||;)\s*", value)
    expanded = []
    seen = set()
    for piece in pieces:
        result = expand_envo_term(piece)
        if not result:
            continue
        key = result.casefold()
        if key in seen:
            continue
        seen.add(key)
        expanded.append(result)
    return " | ".join(expanded) if expanded else None
