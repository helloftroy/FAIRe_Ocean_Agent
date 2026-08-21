"""Classify a repository's own file listing for real sequence data, from
filenames alone -- no download, no content parsing (section 18's general
"don't waste time parsing content that isn't there" principle applies
directly to Zenodo/Dryad/Figshare/OSF's dataset records, which can be
gigabytes).

Three tiers, not two, per a real live case: 10.5281/zenodo.10381280's own
files are "Elas02.rar"/"MiFish.rar" -- real raw sequencing archives (its own
record metadata says "eDNA metabarcoding," "MiFish and Elas02 primer sets"),
named after the primer/marker set rather than containing "fastq" anywhere in
the filename. A plain keyword-only classifier would have called this
"absent" and wrongly dropped a paper with genuinely accessible data. A large
archive file with no matching keyword is downgraded to LIKELY rather than
CONFIRMED, but still counts as accessible for give-up purposes -- treating
it as ABSENT risks losing real data the same way this live case would have.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, SupportType
from fair_ocean_agent.sources.base import EntityLinkCandidate, RawFactCandidate

_SEQUENCE_EXTENSIONS = (".fastq.gz", ".fastq", ".fq.gz", ".fq", ".fasta.gz", ".fasta", ".fa.gz", ".fa", ".fna")
_ARCHIVE_EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".tar", ".rar", ".7z")
_SEQUENCE_KEYWORDS = ("fastq", "fq_", "_fq.", "fasta", "reads", "sequence")
# A tiny archive (a README bundle, a figure pack) isn't worth trusting as
# real sequence data just for having an archive extension -- raw amplicon/
# shotgun sequencing data is realistically always well above this.
_MIN_LIKELY_ARCHIVE_BYTES = 1_000_000


class SequenceDataStatus(str, Enum):
    CONFIRMED = "confirmed"  # a listed file's own extension is fastq/fasta, or an archive's filename says so
    LIKELY = "likely"  # a large archive is present with no confirming keyword -- still counts as accessible
    ABSENT = "absent"  # nothing that looks like it could hold sequence data


@dataclass(frozen=True)
class ListedFile:
    name: str
    size_bytes: int | None = None


def classify_file_listing(files: list[ListedFile]) -> SequenceDataStatus:
    has_large_archive = False
    for listed in files:
        lowered = listed.name.lower()
        if lowered.endswith(_SEQUENCE_EXTENSIONS):
            return SequenceDataStatus.CONFIRMED
        if lowered.endswith(_ARCHIVE_EXTENSIONS):
            if any(keyword in lowered for keyword in _SEQUENCE_KEYWORDS):
                return SequenceDataStatus.CONFIRMED
            if (listed.size_bytes or 0) >= _MIN_LIKELY_ARCHIVE_BYTES:
                has_large_archive = True
    return SequenceDataStatus.LIKELY if has_large_archive else SequenceDataStatus.ABSENT


def synthesize_placeholder_sample_and_run_facts(
    *, repo: str, doi: str, status: SequenceDataStatus
) -> list[RawFactCandidate]:
    """One SAMPLE + one EXPERIMENT_RUN row, not one per file inside the
    dataset and no unzipping -- per an explicit user request: "just one
    line in the sample and experiment metadata tables, so that we can
    populate with anything found in the paper later." Called whenever a
    Pass 2 adapter's own file-listing check (CONFIRMED/LIKELY) found real
    sequence data, since none of Zenodo/Dryad/Figshare/OSF's records carry
    a structured per-sample breakdown the way a real BioSample/ENA record
    does -- true of every one of them, not just the Dryad-zip case that
    prompted this. support_type=INFERRED marks every emitted fact as a
    synthesized placeholder, not something the paper's own text stated.

    The materialSampleID fact (fact_type "sample_accession") uses
    mapping/faire.py's own by-value redirect (_resolve_entity_id: for that
    one target field, the SAMPLE entity is looked up by matching raw_value
    against its external_identifier, not by this fact's own entity_id) --
    confirmed live end to end against a real map_study_to_faire +
    export-faire run that this produces exactly one sampleMetadata row
    (materialSampleID populated) and one experimentRunMetadata row, no
    third entity/row."""
    if status == SequenceDataStatus.ABSENT:
        return []

    sample_id = f"internal:{repo}:{doi}:sample"
    run_id = f"internal:{repo}:{doi}:run"
    resource_url = f"https://doi.org/{doi}"

    facts = [
        RawFactCandidate(
            entity_level=EntityLevel.SAMPLE,
            fact_type_candidate="samp_category",
            raw_field_name="samp_category",
            raw_value="sample",
            source_locator=f"{repo}.placeholder_sample",
            support_type=SupportType.INFERRED,
            entity_external_id=sample_id,
            entity_label=f"Unresolved sample(s) from {repo} dataset {doi}",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.SEQUENCING_RUN,
            fact_type_candidate="sample_accession",
            raw_field_name="sample_accession",
            raw_value=sample_id,
            source_locator=f"{repo}.placeholder_sample_accession",
            support_type=SupportType.INFERRED,
        ),
    ]
    run_links = [
        EntityLinkCandidate(
            entity_level=EntityLevel.SAMPLE,
            external_identifier=sample_id,
            relationship_type=EntityRelationshipType.DERIVED_FROM_SAMPLE,
            label=sample_id,
        )
    ]
    for fact_type, raw_field, value in (
        ("samp_name", "sample_id", sample_id),
        ("associatedSequences", "dataset_url", resource_url),
    ):
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.EXPERIMENT_RUN,
                fact_type_candidate=fact_type,
                raw_field_name=raw_field,
                raw_value=value,
                source_locator=f"{repo}.placeholder_run.{raw_field}",
                support_type=SupportType.INFERRED,
                entity_external_id=run_id,
                entity_label=f"Unresolved run(s) from {repo} dataset {doi}",
                entity_links=run_links,
            )
        )
    return facts
