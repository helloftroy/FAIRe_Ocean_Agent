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
from fair_ocean_agent.identity.identifiers import IdentifierError, guess_identifier_type, normalize_identifier
from fair_ocean_agent.sources.base import RelatedIdentifier, SourceAdapter, SourceRecordNotFoundError
from fair_ocean_agent.sources.ena import EnaAdapter

_BIOPROJECT_PATTERN = re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.IGNORECASE)
_SRA_STUDY_PATTERN = re.compile(r"\b(?:SRP|ERP|DRP)\d+\b", re.IGNORECASE)
_PANGAEA_DOI_PATTERN = re.compile(r"\b10\.1594/PANGAEA\.\d+\b", re.IGNORECASE)
_BCODMO_DOI_PATTERN = re.compile(
    r"\b10\.(?:1575|26008)/1912/(?:bco[-.]dmo|bcodmo)[^\s<>()\[\]{}\"']*",
    re.IGNORECASE,
)
# SRA/ENA/DDBJ *sample*-level accessions -- confirmed live
# (10.1073/pnas.2103275118) that a paper's Data Availability statement can
# cite these directly ("SRS7105074 - SRS7105095") and never state its own
# study accession at all, and as a RANGE (often with an en dash) rather
# than every individual sample spelled out. Neither shape was previously
# recognized: _SRA_STUDY_PATTERN above only matches the study-level
# SRP/ERP/DRP prefixes, not sample-level SRS/ERS/DRS. Range pattern is
# tried first and its matched span is blanked out of the text before the
# plain single-accession pattern runs, so a range's own boundary values
# aren't independently double-matched as standalone accessions too.
_SRA_SAMPLE_PATTERN = re.compile(r"\b(SRS|ERS|DRS)\d+\b", re.IGNORECASE)
_SRA_SAMPLE_RANGE_PATTERN = re.compile(
    r"\b(SRS|ERS|DRS)(\d+)\s*[-–—]\s*(?:SRS|ERS|DRS)?(\d+)\b", re.IGNORECASE
)
# A typo'd or malformed range (e.g. transposed digits producing a huge
# span) shouldn't silently trigger hundreds of speculative API lookups --
# mirrors sources/ncbi.py's MAX_SAMPLES_PER_PROJECT-style safety caps.
_MAX_SRA_SAMPLE_RANGE = 500


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


def _expand_sra_sample_range(prefix: str, start_digits: str, end_digits: str) -> list[str]:
    width = len(start_digits)
    start, end = int(start_digits), int(end_digits)
    if end < start or (end - start + 1) > _MAX_SRA_SAMPLE_RANGE:
        return []
    return [f"{prefix}{str(n).zfill(width)}" for n in range(start, end + 1)]


def _extract_sra_sample_accessions(text: str) -> list[str]:
    seen: set[str] = set()
    accessions: list[str] = []

    def add(value: str) -> None:
        upper = value.upper()
        if upper not in seen:
            seen.add(upper)
            accessions.append(upper)

    range_spans: list[tuple[int, int]] = []
    for match in _SRA_SAMPLE_RANGE_PATTERN.finditer(text):
        prefix = match.group(1).upper()
        for value in _expand_sra_sample_range(prefix, match.group(2), match.group(3)):
            add(value)
        range_spans.append((match.start(), match.end()))

    remaining = text
    for start, end in sorted(range_spans, reverse=True):
        remaining = remaining[:start] + " " * (end - start) + remaining[end:]
    for match in _SRA_SAMPLE_PATTERN.finditer(remaining):
        add(match.group(0))

    return accessions


def resolve_sra_sample_accessions_to_studies(
    adapters: dict[str, SourceAdapter], text: str, *, source_name: str
) -> list[RelatedIdentifier]:
    """SRA/ENA/DDBJ sample accessions (SRS/ERS/DRS) found in `text` resolve
    to their real parent study via a live lookup here, rather than becoming
    their own new identifier type with its own downstream handling -- this
    pipeline's existing study-level resolution already knows how to expand
    a BioProject/ENA study accession into the full sibling sample set (see
    EnaAdapter.fetch_record / workflow/handlers.py's _resolve_repository_
    sources), which is also strictly better than only capturing the
    specific sample accessions a paper's own text happens to enumerate.

    ENA's own study_accession field for an NCBI/SRA-native submission
    (confirmed live: 10.1073/pnas.2103275118's own SRS7105074-95 range)
    resolves to a PRJNA... BioProject accession, not an ENA-native one --
    guess_identifier_type routes each resolved accession to whichever real
    identifier type its own prefix actually is, rather than assuming it's
    always ENA_STUDY_ACCESSION."""
    ena = adapters.get("ena")
    if not isinstance(ena, EnaAdapter):
        return []
    sample_accessions = _extract_sra_sample_accessions(text)
    if not sample_accessions:
        return []

    seen: set[tuple[IdentifierType, str]] = set()
    related: list[RelatedIdentifier] = []
    for sample_accession in sample_accessions:
        try:
            study_accession = ena.resolve_sample_to_study_accession(sample_accession)
        except SourceRecordNotFoundError:
            continue
        if not study_accession:
            continue
        identifier_type = guess_identifier_type(study_accession)
        if identifier_type is None:
            continue
        try:
            value = normalize_identifier(identifier_type, study_accession)
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
                # Already confirmed by the resolution lookup itself (it
                # only returns a value for a real, existing ENA record) --
                # SupportType.DETERMINISTICALLY_DERIVED matches every other
                # identifier this module emits, all of which still pass
                # through verify_deterministic_identifier below regardless.
                confidence=SupportType.DETERMINISTICALLY_DERIVED,
            )
        )
    return related


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
    IdentifierType.ENA_STUDY_ACCESSION: ("ena",),
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
