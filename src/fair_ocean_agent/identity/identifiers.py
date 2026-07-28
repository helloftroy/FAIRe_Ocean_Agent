"""Normalization and validation for persistent identifiers (section 8 of the
design brief: "exact identifiers before fuzzy matching"). These are the only
basis for deduplication in Milestone 1/2 -- probabilistic matching (title,
author, etc.) is a later milestone and lives in identity/candidate_matches.py.

Every normalize_* function is pure and total: given a non-empty string it
either returns a normalized value or raises IdentifierError. Callers decide
what to do with an invalid identifier (e.g. keep the seed row but skip
creating an external_identifiers record).
"""
from __future__ import annotations

import re

from fair_ocean_agent.database.enums import IdentifierType

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[^\s]+$")
DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
    "DOI:",
)

PMID_PATTERN = re.compile(r"^\d+$")
PMCID_PATTERN = re.compile(r"^PMC\d+$", re.IGNORECASE)

# NCBI BioProject: PRJNA (NCBI), PRJEB (ENA), PRJDB (DDBJ)
BIOPROJECT_PATTERN = re.compile(r"^PRJ(NA|EB|DB)\d+$", re.IGNORECASE)
# NCBI BioSample: SAMN (NCBI), SAME (ENA), SAMD (DDBJ)
BIOSAMPLE_PATTERN = re.compile(r"^SAM(N|E|D)[A-Z]?\d+$", re.IGNORECASE)
# SRA/ENA/DDBJ study accessions
SRA_STUDY_PATTERN = re.compile(r"^(SRP|ERP|DRP)\d+$", re.IGNORECASE)
# ENA study accessions overlap with SRA study accessions (ERP...) and also
# appear as the BioProject-equivalent PRJEB... for ENA-native submissions.
ENA_STUDY_PATTERN = re.compile(r"^(ERP\d+|PRJEB\d+)$", re.IGNORECASE)

GBIF_DATASET_KEY_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class IdentifierError(ValueError):
    """Raised when a raw value cannot be normalized to a valid identifier of
    the claimed type."""


def normalize_doi(raw: str) -> str:
    value = raw.strip()
    for prefix in DOI_URL_PREFIXES:
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix):]
            break
    value = value.strip().strip("/").lower()
    if not DOI_PATTERN.match(value):
        raise IdentifierError(f"Not a valid DOI: {raw!r}")
    return value


def normalize_pmid(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"(?i)^pmid:?\s*", "", value)
    if not PMID_PATTERN.match(value):
        raise IdentifierError(f"Not a valid PMID: {raw!r}")
    return value


def normalize_pmcid(raw: str) -> str:
    value = raw.strip().upper()
    if not value.startswith("PMC"):
        value = f"PMC{value}"
    if not PMCID_PATTERN.match(value):
        raise IdentifierError(f"Not a valid PMCID: {raw!r}")
    return value


def normalize_bioproject_accession(raw: str) -> str:
    value = raw.strip().upper()
    if not BIOPROJECT_PATTERN.match(value):
        raise IdentifierError(f"Not a valid BioProject accession: {raw!r}")
    return value


def normalize_biosample_accession(raw: str) -> str:
    value = raw.strip().upper()
    if not BIOSAMPLE_PATTERN.match(value):
        raise IdentifierError(f"Not a valid BioSample accession: {raw!r}")
    return value


def normalize_sra_study_accession(raw: str) -> str:
    value = raw.strip().upper()
    if not SRA_STUDY_PATTERN.match(value):
        raise IdentifierError(f"Not a valid SRA study accession: {raw!r}")
    return value


def normalize_ena_study_accession(raw: str) -> str:
    value = raw.strip().upper()
    if not ENA_STUDY_PATTERN.match(value):
        raise IdentifierError(f"Not a valid ENA study accession: {raw!r}")
    return value


def normalize_gbif_dataset_key(raw: str) -> str:
    value = raw.strip().lower()
    if not GBIF_DATASET_KEY_PATTERN.match(value):
        raise IdentifierError(f"Not a valid GBIF dataset key (expected UUID): {raw!r}")
    return value


def normalize_obis_dataset_uuid(raw: str) -> str:
    value = raw.strip().lower()
    if not GBIF_DATASET_KEY_PATTERN.match(value):
        raise IdentifierError(f"Not a valid OBIS dataset UUID: {raw!r}")
    return value


_NORMALIZERS = {
    IdentifierType.DOI: normalize_doi,
    IdentifierType.DATASET_DOI: normalize_doi,
    IdentifierType.PMID: normalize_pmid,
    IdentifierType.PMCID: normalize_pmcid,
    IdentifierType.BIOPROJECT_ACCESSION: normalize_bioproject_accession,
    IdentifierType.BIOSAMPLE_ACCESSION: normalize_biosample_accession,
    IdentifierType.SRA_STUDY_ACCESSION: normalize_sra_study_accession,
    IdentifierType.ENA_STUDY_ACCESSION: normalize_ena_study_accession,
    IdentifierType.GBIF_DATASET_KEY: normalize_gbif_dataset_key,
    IdentifierType.OBIS_DATASET_UUID: normalize_obis_dataset_uuid,
}


def normalize_identifier(identifier_type: IdentifierType, raw: str) -> str:
    """Normalize `raw` according to `identifier_type`. Types without a
    dedicated normalizer (e.g. CRUISE_ID, BCODMO_DATASET_ID, PANGAEA_ID,
    NCEI_ACCESSION, URL, OTHER) fall back to a light whitespace-strip, since
    those repositories don't have a single well-defined public accession
    grammar to validate against."""
    normalizer = _NORMALIZERS.get(identifier_type)
    if normalizer is None:
        return raw.strip()
    return normalizer(raw)


def guess_identifier_type(raw: str) -> IdentifierType | None:
    """Best-effort type detection for a bare identifier string of unknown
    type, e.g. a seed CSV column that just says "identifier". Returns None
    rather than guessing wrong -- callers should treat that as
    IdentifierType.OTHER and flag for review."""
    value = raw.strip()
    checks: list[tuple[IdentifierType, re.Pattern]] = [
        (IdentifierType.BIOPROJECT_ACCESSION, BIOPROJECT_PATTERN),
        (IdentifierType.BIOSAMPLE_ACCESSION, BIOSAMPLE_PATTERN),
        (IdentifierType.ENA_STUDY_ACCESSION, ENA_STUDY_PATTERN),
        (IdentifierType.SRA_STUDY_ACCESSION, SRA_STUDY_PATTERN),
        (IdentifierType.PMCID, PMCID_PATTERN),
    ]
    for identifier_type, pattern in checks:
        if pattern.match(value):
            return identifier_type

    stripped = value
    for prefix in DOI_URL_PREFIXES:
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix):]
            break
    if DOI_PATTERN.match(stripped.strip("/")):
        return IdentifierType.DOI

    if value.lower().startswith(("http://", "https://")):
        return IdentifierType.URL

    if PMID_PATTERN.match(value):
        return IdentifierType.PMID

    return None
