"""Deterministic identifier mining from publication text.

This is intentionally narrow: only identifiers with strong public accession
grammars are emitted automatically. Similar-looking labels or generic DOIs
remain out of scope because DOI references in full text are usually citations,
not dataset links.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from fair_ocean_agent.database.enums import IdentifierType, RelationshipType
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier
from fair_ocean_agent.sources.base import RelatedIdentifier

_BIOPROJECT_PATTERN = re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.IGNORECASE)
_SRA_STUDY_PATTERN = re.compile(r"\b(?:SRP|ERP|DRP)\d+\b", re.IGNORECASE)
_PANGAEA_DOI_PATTERN = re.compile(r"\b10\.1594/PANGAEA\.\d+\b", re.IGNORECASE)
_BCODMO_DOI_PATTERN = re.compile(
    r"\b10\.(?:1575|26008)/1912/(?:bco[-.]dmo|bcodmo)[^\s<>()\[\]{}\"']*",
    re.IGNORECASE,
)


def xml_to_text(xml: str) -> str:
    """Collapse JATS/XML text content into a searchable plain-text string.

    Invalid XML falls back to the raw text so cached/plain-text payloads can
    still be scanned.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return xml
    return " ".join((text or "").strip() for text in root.itertext() if (text or "").strip())


def extract_repository_identifiers_from_text(text: str, *, source_name: str) -> list[RelatedIdentifier]:
    """Return repository identifiers explicitly present in publication text."""
    candidates: list[tuple[IdentifierType, str]] = []
    candidates.extend((IdentifierType.BIOPROJECT_ACCESSION, match.group(0)) for match in _BIOPROJECT_PATTERN.finditer(text))
    candidates.extend((IdentifierType.SRA_STUDY_ACCESSION, match.group(0)) for match in _SRA_STUDY_PATTERN.finditer(text))
    candidates.extend((IdentifierType.DATASET_DOI, match.group(0)) for match in _PANGAEA_DOI_PATTERN.finditer(text))
    candidates.extend((IdentifierType.DATASET_DOI, match.group(0)) for match in _BCODMO_DOI_PATTERN.finditer(text))

    seen: set[tuple[IdentifierType, str]] = set()
    related: list[RelatedIdentifier] = []
    for identifier_type, raw_value in candidates:
        try:
            value = normalize_identifier(identifier_type, raw_value.rstrip(".,;:"))
        except IdentifierError:
            continue
        key = (identifier_type, value)
        if key in seen:
            continue
        seen.add(key)
        related.append(
            RelatedIdentifier(
                identifier_type=identifier_type,
                value=value,
                relationship_type=RelationshipType.IS_DATASET_FOR,
                source=source_name,
            )
        )
    return related
