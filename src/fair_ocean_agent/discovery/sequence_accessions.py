"""Central parser and normalized result shape for sequence accessions.

Publication data-availability text is not consistent about which INSDC
identifier it cites: some papers cite BioProjects, some cite SRA/ENA/DDBJ
submissions, and some cite individual samples/experiments/runs. Keep prefix
classification here so workflow code consumes one normalized structure
instead of growing prefix checks in several places.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_identifier


@dataclass(frozen=True)
class SequenceAccessionMatch:
    accession: str
    identifier_type: IdentifierType
    archive: str
    start: int
    end: int


@dataclass
class ResolvedSequenceRecord:
    cited_accession: str
    cited_accession_type: str
    archive: str
    bioproject_accessions: list[str] = field(default_factory=list)
    biosample_accessions: list[str] = field(default_factory=list)
    sra_submission_accessions: list[str] = field(default_factory=list)
    sra_study_accessions: list[str] = field(default_factory=list)
    sra_sample_accessions: list[str] = field(default_factory=list)
    sra_experiment_accessions: list[str] = field(default_factory=list)
    sra_run_accessions: list[str] = field(default_factory=list)
    sra_analysis_accessions: list[str] = field(default_factory=list)
    assembly_accessions: list[str] = field(default_factory=list)
    raw_sequence_data_resolved: bool = False
    resolution_status: str = "unresolved"
    resolution_method: str = "not_attempted"


_ACCESSION_PATTERNS: tuple[tuple[IdentifierType, str], ...] = (
    (IdentifierType.CNCB_PROJECT_ACCESSION, r"\bPRJCA\d+\b"),
    (IdentifierType.BIOPROJECT_ACCESSION, r"\bPRJ(?:NA|EB|DB|DA|EA)\d+\b"),
    (IdentifierType.CNCB_BIOSAMPLE_ACCESSION, r"\bSAMC\d+\b"),
    (IdentifierType.BIOSAMPLE_ACCESSION, r"\bSAM(?:N|D|EA)\d+\b"),
    (IdentifierType.SRA_SUBMISSION_ACCESSION, r"\b(?:SRA|ERA|DRA)\d+\b"),
    (IdentifierType.SRA_STUDY_ACCESSION, r"\b(?:SRP|ERP|DRP)\d+\b"),
    (IdentifierType.SRA_SAMPLE_ACCESSION, r"\b(?:SRS|ERS|DRS)\d+\b"),
    (IdentifierType.SRA_EXPERIMENT_ACCESSION, r"\b(?:SRX|ERX|DRX)\d+\b"),
    (IdentifierType.SRA_RUN_ACCESSION, r"\b(?:SRR|ERR|DRR)\d+\b"),
    (IdentifierType.SRA_ANALYSIS_ACCESSION, r"\b(?:SRZ|ERZ|DRZ)\d+\b"),
    (IdentifierType.ASSEMBLY_ACCESSION, r"\bGC[AF]_\d{9}\.\d+\b"),
    (IdentifierType.CNCB_STUDY_ACCESSION, r"\bCRA\d+\b"),
    (IdentifierType.CNCB_EXPERIMENT_ACCESSION, r"\bCRX\d+\b"),
    (IdentifierType.CNCB_RUN_ACCESSION, r"\bCRR\d+\b"),
)

_COMPILED_ACCESSION_PATTERNS = tuple(
    (identifier_type, re.compile(pattern, re.IGNORECASE)) for identifier_type, pattern in _ACCESSION_PATTERNS
)

_RANGE_PATTERNS: tuple[tuple[IdentifierType, re.Pattern], ...] = (
    (
        IdentifierType.SRA_SAMPLE_ACCESSION,
        re.compile(r"\b(SRS|ERS|DRS)(\d+)\s*[-–—]\s*(?:SRS|ERS|DRS)?(\d+)\b", re.IGNORECASE),
    ),
    (
        IdentifierType.SRA_RUN_ACCESSION,
        re.compile(r"\b(SRR|ERR|DRR)(\d+)\s*[-–—]\s*(?:SRR|ERR|DRR)?(\d+)\b", re.IGNORECASE),
    ),
)

_MAX_ACCESSION_RANGE = 500


def accession_archive(accession: str, identifier_type: IdentifierType) -> str:
    value = accession.upper()
    if identifier_type.name.startswith("CNCB_"):
        return "CNCB"
    if identifier_type == IdentifierType.ASSEMBLY_ACCESSION:
        return "NCBI" if value.startswith("GCF_") else "INSDC"
    if value.startswith(("PRJDB", "SAMD", "DRA", "DRP", "DRS", "DRX", "DRR", "DRZ")):
        return "DDBJ"
    if value.startswith(("PRJEB", "SAMEA", "ERA", "ERP", "ERS", "ERX", "ERR", "ERZ")):
        return "ENA"
    if value.startswith(("PRJNA", "SAMN", "SRA", "SRP", "SRS", "SRX", "SRR", "SRZ")):
        return "NCBI"
    return "unknown"


def _expand_accession_range(prefix: str, start_digits: str, end_digits: str) -> list[str]:
    width = len(start_digits)
    start, end = int(start_digits), int(end_digits)
    if end < start or (end - start + 1) > _MAX_ACCESSION_RANGE:
        return []
    return [f"{prefix.upper()}{str(n).zfill(width)}" for n in range(start, end + 1)]


def find_sequence_accessions(text: str) -> list[SequenceAccessionMatch]:
    seen: set[tuple[IdentifierType, str]] = set()
    matches: list[SequenceAccessionMatch] = []
    masked = text

    def add(identifier_type: IdentifierType, raw_value: str, start: int, end: int) -> None:
        try:
            value = normalize_identifier(identifier_type, raw_value.rstrip(".,;:"))
        except IdentifierError:
            return
        key = (identifier_type, value)
        if key in seen:
            return
        seen.add(key)
        matches.append(
            SequenceAccessionMatch(
                accession=value,
                identifier_type=identifier_type,
                archive=accession_archive(value, identifier_type),
                start=start,
                end=end,
            )
        )

    for identifier_type, pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(text):
            expanded = _expand_accession_range(match.group(1), match.group(2), match.group(3))
            if expanded:
                for value in expanded:
                    add(identifier_type, value, match.start(), match.end())
                masked = masked[:match.start()] + " " * (match.end() - match.start()) + masked[match.end():]

    for identifier_type, pattern in _COMPILED_ACCESSION_PATTERNS:
        for match in pattern.finditer(masked):
            add(identifier_type, match.group(0), match.start(), match.end())

    return sorted(matches, key=lambda item: (item.start, item.end, item.accession))

