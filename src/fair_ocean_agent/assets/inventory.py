"""Data-asset inventory (section 13): materialize DataAsset rows from
already-extracted structured raw_facts -- no new network calls, and never
the file itself, only its accession/size/checksum/location.

Currently covers ENA sequencing-run file listings (Milestone 3's
per-run fastq_ftp/fastq_bytes/fastq_md5 facts, one row per run). Other
asset types from section 13's list (ASV/OTU tables, sample metadata
tables, code repositories, protocol files, ...) need either a dedicated
adapter (OBIS/GBIF/BCO-DMO/PANGAEA) or LLM-based classification of
Data Availability Statement text, neither built yet -- see
docs/architecture.md.
"""
from __future__ import annotations

import posixpath

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import AccessStatus, AssetType, EntityLevel, RawOrProcessed
from fair_ocean_agent.database.models import DataAsset, Entity, RawFact, Source

_FORMAT_SUFFIXES = (".fastq.gz", ".fastq", ".fq.gz", ".fq")


def _guess_format(file_name: str | None) -> str | None:
    if not file_name:
        return None
    lowered = file_name.lower()
    for suffix in _FORMAT_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix.lstrip(".")
    return lowered.rsplit(".", 1)[-1] if "." in lowered else None


def _run_facts_by_entity(session: Session, study_id: str, source_id: str) -> dict[str, dict[str, str]]:
    facts = session.scalars(
        select(RawFact).where(
            RawFact.study_id == study_id,
            RawFact.source_id == source_id,
            RawFact.entity_id.is_not(None),
        )
    ).all()
    grouped: dict[str, dict[str, str]] = {}
    for fact in facts:
        grouped.setdefault(fact.entity_id, {})[fact.raw_field_name] = fact.raw_value
    return grouped


def inventory_ena_run_assets(session: Session, study_id: str) -> int:
    """Creates one RAW_SEQUENCE_FILES DataAsset per ENA sequencing_run
    entity that has a fastq_ftp fact. Idempotent: skips entities that
    already have a DataAsset row from this source, so retrying a task
    (or re-running the inventory pass later) never duplicates rows.
    Returns the number of new assets created."""
    ena_source = session.scalars(
        select(Source).where(Source.study_id == study_id, Source.source_name == "ena")
    ).first()
    if ena_source is None:
        return 0

    by_entity = _run_facts_by_entity(session, study_id, ena_source.source_id)

    existing_entity_ids = set(
        session.scalars(
            select(DataAsset.entity_id).where(
                DataAsset.study_id == study_id, DataAsset.source_id == ena_source.source_id
            )
        ).all()
    )

    created = 0
    for entity_id, fields in by_entity.items():
        if entity_id in existing_entity_ids:
            continue

        fastq_ftp = fields.get("fastq_ftp")
        if not fastq_ftp:
            continue  # run metadata without a listed file (e.g. no public fastq yet)

        entity = session.get(Entity, entity_id)
        if entity is None or entity.entity_level != EntityLevel.SEQUENCING_RUN.value:
            continue

        file_name = posixpath.basename(fastq_ftp)
        fastq_bytes = fields.get("fastq_bytes", "")
        size_bytes = int(fastq_bytes) if fastq_bytes.isdigit() else None

        description = ", ".join(
            fields[key] for key in ("library_strategy", "library_source", "instrument_platform") if fields.get(key)
        )

        session.add(
            DataAsset(
                study_id=study_id,
                entity_id=entity_id,
                asset_type=AssetType.RAW_SEQUENCE_FILES.value,
                repository="ENA",
                identifier=entity.external_identifier,
                file_name=file_name,
                format=_guess_format(file_name),
                size_bytes=size_bytes,
                access_status=AccessStatus.OPEN.value,  # ENA's read_run only lists files it can serve publicly
                raw_or_processed=RawOrProcessed.RAW.value,
                description=description or None,
                source_id=ena_source.source_id,
            )
        )
        created += 1

    return created
