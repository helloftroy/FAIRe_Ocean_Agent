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
# SRA/ENA/DDBJ *sample*- and *run*-level accessions -- confirmed live
# (10.1073/pnas.2103275118) that a paper's Data Availability statement can
# cite these directly ("SRS7105074 - SRS7105095") and never state its own
# study accession at all, and as a RANGE (often with an en dash) rather
# than every individual accession spelled out. Neither shape was previously
# recognized: _SRA_STUDY_PATTERN above only matches the study-level
# SRP/ERP/DRP prefixes, not sample-level SRS/ERS/DRS or run-level
# SRR/ERR/DRR. Each range pattern is tried first and its matched span is
# blanked out of the text before the plain single-accession pattern runs,
# so a range's own boundary values aren't independently double-matched as
# standalone accessions too.
_SRA_SAMPLE_PATTERN = re.compile(r"\b(SRS|ERS|DRS)\d+\b", re.IGNORECASE)
_SRA_SAMPLE_RANGE_PATTERN = re.compile(
    r"\b(SRS|ERS|DRS)(\d+)\s*[-–—]\s*(?:SRS|ERS|DRS)?(\d+)\b", re.IGNORECASE
)
_SRA_RUN_PATTERN = re.compile(r"\b(SRR|ERR|DRR)\d+\b", re.IGNORECASE)
_SRA_RUN_RANGE_PATTERN = re.compile(
    r"\b(SRR|ERR|DRR)(\d+)\s*[-–—]\s*(?:SRR|ERR|DRR)?(\d+)\b", re.IGNORECASE
)
# A typo'd or malformed range (e.g. transposed digits producing a huge
# span) shouldn't silently trigger hundreds of speculative API lookups --
# mirrors sources/ncbi.py's MAX_SAMPLES_PER_PROJECT-style safety caps.
_MAX_SRA_ACCESSION_RANGE = 500

# Pass 2: general-purpose dataset repositories, tried only once Pass 1
# (BioProject/SRA/ENA accessions above) has found nothing -- confirmed live
# as real, distinct gaps: 10.1038/s41598-024-60762-8's Data Availability
# names only a Zenodo DOI, 10.1002/edn3.184's only a Dryad dataset DOI.
# Tagged DATASET_DOI, the same generic bucket Pangaea/BCO-DMO already share
# above -- no new IdentifierType needed for DOI recognition itself.
_ZENODO_DOI_PATTERN = re.compile(r"\b10\.5281/zenodo\.\d+\b", re.IGNORECASE)
_DRYAD_DOI_PATTERN = re.compile(r"\b10\.5061/dryad\.[a-z0-9]+\b", re.IGNORECASE)
_FIGSHARE_DOI_PATTERN = re.compile(r"\b10\.6084/m9\.figshare\.\d+(?:\.v\d+)?\b", re.IGNORECASE)
_OSF_DOI_PATTERN = re.compile(r"\b10\.17605/OSF\.IO/[A-Z0-9]+\b", re.IGNORECASE)


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


def _expand_sra_accession_range(prefix: str, start_digits: str, end_digits: str) -> list[str]:
    width = len(start_digits)
    start, end = int(start_digits), int(end_digits)
    if end < start or (end - start + 1) > _MAX_SRA_ACCESSION_RANGE:
        return []
    return [f"{prefix}{str(n).zfill(width)}" for n in range(start, end + 1)]


def _extract_sra_accessions(text: str, single_pattern: re.Pattern, range_pattern: re.Pattern) -> list[str]:
    seen: set[str] = set()
    accessions: list[str] = []

    def add(value: str) -> None:
        upper = value.upper()
        if upper not in seen:
            seen.add(upper)
            accessions.append(upper)

    range_spans: list[tuple[int, int]] = []
    for match in range_pattern.finditer(text):
        prefix = match.group(1).upper()
        for value in _expand_sra_accession_range(prefix, match.group(2), match.group(3)):
            add(value)
        range_spans.append((match.start(), match.end()))

    remaining = text
    for start, end in sorted(range_spans, reverse=True):
        remaining = remaining[:start] + " " * (end - start) + remaining[end:]
    for match in single_pattern.finditer(remaining):
        add(match.group(0))

    return accessions


def _extract_sra_sample_accessions(text: str) -> list[str]:
    return _extract_sra_accessions(text, _SRA_SAMPLE_PATTERN, _SRA_SAMPLE_RANGE_PATTERN)


def _extract_sra_run_accessions(text: str) -> list[str]:
    return _extract_sra_accessions(text, _SRA_RUN_PATTERN, _SRA_RUN_RANGE_PATTERN)


def _resolve_sra_accessions_to_studies(
    adapters: dict[str, SourceAdapter], accessions: list[str], *, source_name: str
) -> list[RelatedIdentifier]:
    """Shared by the sample- and run-accession resolvers below: each
    accession resolves to its real parent study via a live lookup, rather
    than becoming its own new identifier type with its own downstream
    handling -- this pipeline's existing study-level resolution already
    knows how to expand a BioProject/ENA study accession into the full
    sibling sample set (see EnaAdapter.fetch_record / workflow/handlers.py's
    _resolve_repository_sources), which is also strictly better than only
    capturing the specific accessions a paper's own text happens to
    enumerate.

    ENA's own study_accession field for an NCBI/SRA-native submission
    (confirmed live: 10.1073/pnas.2103275118's own SRS7105074-95 range)
    resolves to a PRJNA... BioProject accession, not an ENA-native one --
    guess_identifier_type routes each resolved accession to whichever real
    identifier type its own prefix actually is, rather than assuming it's
    always ENA_STUDY_ACCESSION."""
    ena = adapters.get("ena")
    if not isinstance(ena, EnaAdapter) or not accessions:
        return []

    seen: set[tuple[IdentifierType, str]] = set()
    related: list[RelatedIdentifier] = []
    for accession in accessions:
        try:
            study_accession = ena.resolve_read_accession_to_study_accession(accession)
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


def resolve_sra_sample_accessions_to_studies(
    adapters: dict[str, SourceAdapter], text: str, *, source_name: str
) -> list[RelatedIdentifier]:
    """SRA/ENA/DDBJ sample accessions (SRS/ERS/DRS) found in `text`."""
    return _resolve_sra_accessions_to_studies(adapters, _extract_sra_sample_accessions(text), source_name=source_name)


def resolve_sra_run_accessions_to_studies(
    adapters: dict[str, SourceAdapter], text: str, *, source_name: str
) -> list[RelatedIdentifier]:
    """SRA/ENA/DDBJ run accessions (SRR/ERR/DRR) found in `text`."""
    return _resolve_sra_accessions_to_studies(adapters, _extract_sra_run_accessions(text), source_name=source_name)


def extract_dataset_repository_identifiers_from_text(text: str, *, source_name: str) -> list[RelatedIdentifier]:
    """Pass 2: Zenodo/Dryad/Figshare/OSF dataset DOIs, tried only once Pass 1
    (extract_repository_identifiers_from_text + the SRA sample/run resolvers
    above) has found nothing -- see workflow/handlers.py's
    _discover_identifiers_from_fulltext for the gating. Same shape as
    extract_repository_identifiers_from_text; kept separate so callers can
    gate the two independently rather than always paying for both."""
    candidates: list[tuple[IdentifierType, str]] = []
    candidates.extend((IdentifierType.DATASET_DOI, match.group(0)) for match in _ZENODO_DOI_PATTERN.finditer(text))
    candidates.extend((IdentifierType.DATASET_DOI, match.group(0)) for match in _DRYAD_DOI_PATTERN.finditer(text))
    candidates.extend((IdentifierType.DATASET_DOI, match.group(0)) for match in _FIGSHARE_DOI_PATTERN.finditer(text))
    candidates.extend((IdentifierType.DATASET_DOI, match.group(0)) for match in _OSF_DOI_PATTERN.finditer(text))

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
    IdentifierType.DATASET_DOI: ("pangaea", "bcodmo", "zenodo", "dryad", "figshare", "osf", "datacite"),
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
            if name == "zenodo" and "zenodo" not in lowered_value:
                continue
            if name == "dryad" and "dryad" not in lowered_value:
                continue
            if name == "figshare" and "figshare" not in lowered_value:
                continue
            if name == "osf" and "osf.io" not in lowered_value:
                continue
        try:
            adapter.fetch_record(related.value)
            return True
        except SourceRecordNotFoundError:
            continue
    return False
