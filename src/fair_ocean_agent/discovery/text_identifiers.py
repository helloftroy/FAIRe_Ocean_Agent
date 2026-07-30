"""Deterministic identifier mining from publication text.

This is intentionally narrow: only identifiers with strong public accession
grammars are emitted automatically. Similar-looking labels or generic DOIs
remain out of scope because DOI references in full text are usually citations,
not dataset links.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from fair_ocean_agent.database.enums import IdentifierType, RelationshipType, SupportType
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier
from fair_ocean_agent.sources.base import RelatedIdentifier, SourceAdapter, SourceRecordNotFoundError

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
                # Tier 2: a regex-matched accession pattern, not (yet)
                # confirmed to exist -- verify_deterministic_identifier
                # below must resolve it against its own source API before
                # identity/resolution.py's resolve_or_create_study() trusts
                # it enough to auto-link or even consistency-check it.
                confidence=SupportType.DETERMINISTICALLY_DERIVED,
            )
        )
    return related


# Which adapters can plausibly confirm each identifier type actually
# resolves to a real record -- tried in order, first success wins. Mirrors
# workflow/handlers.py's _resolve_dataset_sources' own adapter-selection
# logic for dataset DOIs (DataCite as the generic authority, native
# repository adapters when the DOI prefix matches).
_VERIFICATION_ADAPTER_NAMES: dict[IdentifierType, tuple[str, ...]] = {
    IdentifierType.BIOPROJECT_ACCESSION: ("ncbi_bioproject", "ena"),
    IdentifierType.SRA_STUDY_ACCESSION: ("ena",),
    IdentifierType.DATASET_DOI: ("pangaea", "bcodmo", "datacite"),
}


def verify_deterministic_identifier(adapters: dict[str, SourceAdapter], related: RelatedIdentifier) -> bool:
    """Tier-2 gate: a regex-matched accession is only trustworthy once it's
    confirmed to actually resolve against (at least one of) that identifier
    type's own source APIs. A 404 from every enabled candidate adapter
    means "not verifiable" -- treated as absence of confirmation, not an
    error (an unresolvable value drops the hit, it never raises, matching
    validation/logical.py's own "unparseable is NOT_ASSESSED, never a
    false ERROR" principle). Returns False if no candidate adapter is
    enabled or every one 404s; True as soon as one confirms a real record."""
    candidate_names = _VERIFICATION_ADAPTER_NAMES.get(related.identifier_type, ())
    lowered_value = related.value.lower()
    for name in candidate_names:
        adapter = adapters.get(name)
        if adapter is None:
            continue
        # A dataset DOI's own prefix says which repository it belongs to --
        # skip an adapter that plainly can't own this DOI (mirrors
        # _resolve_dataset_sources' own "pangaea"/"bco-dmo" substring
        # checks) so e.g. a PANGAEA DOI isn't wastefully tried against
        # bcodmo's fetch_record first.
        if related.identifier_type == IdentifierType.DATASET_DOI:
            if name == "pangaea" and "pangaea" not in lowered_value:
                continue
            if name == "bcodmo" and "bco-dmo" not in lowered_value and "bcodmo" not in lowered_value:
                continue
        try:
            adapter.fetch_record(related.value)
            return True
        except SourceRecordNotFoundError:
            continue
    return False
