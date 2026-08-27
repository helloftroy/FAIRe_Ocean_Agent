#!/usr/bin/env python3
"""Backfills physicochemical/environmental facts (and the sample entities
themselves) from MG-RAST's own per-sample scrape (mgrast_samples, populated
by mgrast_discovery.py's metadata phase) onto the main pipeline database --
same idea, same reasoning, and the same shape as
apply_cncb_physicochemical_enrichment.py: MG-RAST's own discovery pipeline
already scraped rich per-sample data into its own local database, so a
backfill from that local database is more efficient than a live adapter
re-fetching the same data from MG-RAST's API, and no MG-RAST source
adapter exists in workflow/handlers.py's _REPOSITORY_ADAPTER_CLASSES to do
that live fetch anyway.

Unlike CNCB (whose native study identifier -- CRA -- is one level removed
from the project accession the samples are actually keyed by, requiring a
cra_accessions_json indirection), an MG-RAST study's own
IdentifierType.MGRAST_PROJECT_ID value IS the exact key mgrast_samples.
mgrast_project_id is keyed by -- no indirection needed.

For each Study with an MGRAST_PROJECT_ID identifier, get-or-creates one
real SAMPLE Entity per mgrast_samples row for that project (via
identity/entity_linking.py's get_or_create_entity, the same established,
idempotent, cross-study-safe choke point used throughout this pipeline),
preferring the sample's own real BioSample accession as the external
identifier when MG-RAST's own metadata included a valid one (so a sample
independently discovered via NCBI/ENA later merges into the same Entity
rather than duplicating it) and falling back to a namespaced
"mgrast:<mgrast_sample_id>" identifier otherwise -- MG-RAST's own sample
ID is only unique per-project (mgrast_samples' own primary key is
(mgrast_project_id, mgrast_sample_id)), so it needs the namespace prefix
to be safe as a global external_identifier, unlike CNCB's SAMC accessions
which are already globally unique on their own.

Writes RawFact rows (support_type=structured_source) using the same
fact_type_candidate names apply_gold_physicochemical_enrichment.py and
apply_cncb_physicochemical_enrichment.py already write (temp/salinity/ph/
diss_oxygen/chlorophyll/depth/latitude/longitude/collection_date/
env_broad_scale/env_local_scale/env_medium/size_frac) -- no new mapping
rules needed.

Deliberately out of scope, matching the CNCB script's own precedent:
MG-RAST's own nitrate/nitrite/ammonium/phosphate columns have no existing
MappingRule and aren't backfilled here; nor is mgrast_datasets' own
sequencing/library metadata (platform, primers, target gene) -- this
script is physicochemical/sample-level only.

Idempotent and dry-run by default, same as the GOLD/CNCB scripts. Reports
which touched studies already completed MAP_FAIRE.

Usage:
    python scripts/apply_mgrast_physicochemical_enrichment.py
    python scripts/apply_mgrast_physicochemical_enrichment.py --mgrast-db data/seed_discovery/mgrast_paper_seeds.sqlite --apply
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, SupportType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Task
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.identity.entity_linking import get_or_create_entity
from fair_ocean_agent.seed_discovery.mgrast_discovery import BIOSAMPLE_RE

SOURCE_LOCATOR = "mgrast_enrichment"

# (mgrast_samples column, fact_type_candidate expected by an existing
# mapping/rules.py MappingRule at EntityLevel.SAMPLE) -- same fact_type
# names the GOLD/CNCB enrichment scripts already write.
MGRAST_FIELD_TO_FACT_TYPE = (
    ("latitude", "latitude"),
    ("longitude", "longitude"),
    ("collection_date", "collection_date"),
    ("depth", "depth"),
    ("env_broad_scale", "env_broad_scale"),
    ("env_local_scale", "env_local_scale"),
    ("env_medium", "env_medium"),
    ("size_fraction", "size_frac"),
    ("temperature", "temp"),
    ("salinity", "salinity"),
    ("ph", "ph"),
    ("dissolved_oxygen", "diss_oxygen"),
    ("oxygen", "diss_oxygen"),
    ("chlorophyll", "chlorophyll"),
)


def _mgrast_samples_by_project(mgrast_db_path: Path) -> dict[str, list[sqlite3.Row]]:
    conn = sqlite3.connect(mgrast_db_path)
    conn.row_factory = sqlite3.Row
    samples_by_project: dict[str, list[sqlite3.Row]] = {}
    for row in conn.execute("SELECT * FROM mgrast_samples WHERE mgrast_sample_id IS NOT NULL AND mgrast_sample_id != ''"):
        samples_by_project.setdefault(row["mgrast_project_id"], []).append(row)
    conn.close()
    return samples_by_project


def _sample_external_identifier(sample: sqlite3.Row) -> str:
    accession = (sample["biosample_accession"] or "").strip().upper()
    if BIOSAMPLE_RE.fullmatch(accession):
        return accession
    return f"mgrast:{sample['mgrast_sample_id']}"


def apply_enrichment(session, mgrast_db_path: Path, *, apply: bool) -> dict:
    samples_by_project = _mgrast_samples_by_project(mgrast_db_path)

    project_ids = list(
        session.scalars(
            select(ExternalIdentifier.identifier_value).where(
                ExternalIdentifier.identifier_type == IdentifierType.MGRAST_PROJECT_ID.value
            )
        ).all()
    )
    study_ids_by_project: dict[str, set[str]] = {}
    for study_id, project_id in session.execute(
        select(ExternalIdentifier.study_id, ExternalIdentifier.identifier_value).where(
            ExternalIdentifier.identifier_type == IdentifierType.MGRAST_PROJECT_ID.value
        )
    ).all():
        study_ids_by_project.setdefault(project_id, set()).add(study_id)

    counts = {
        "studies_with_mgrast_identifier": len({sid for sids in study_ids_by_project.values() for sid in sids}),
        "studies_matched_to_mgrast_samples": 0,
        "sample_rows_checked": 0,
        "sample_entities_new": 0,
        "sample_entities_already_existed": 0,
        "facts_created": 0,
        "facts_already_present": 0,
    }
    touched_study_ids: set[str] = set()

    existing_facts = {
        (entity_id, fact_type)
        for entity_id, fact_type in session.execute(
            select(RawFact.entity_id, RawFact.fact_type_candidate).where(RawFact.source_locator == SOURCE_LOCATOR)
        ).all()
    }

    for project_id in project_ids:
        samples = samples_by_project.get(project_id)
        study_ids = study_ids_by_project.get(project_id, set())
        if not samples or not study_ids:
            continue
        for study_id in study_ids:
            counts["studies_matched_to_mgrast_samples"] += 1
            for sample in samples:
                counts["sample_rows_checked"] += 1
                accession = _sample_external_identifier(sample)

                existing_entity = session.scalars(
                    select(Entity).where(Entity.entity_level == EntityLevel.SAMPLE.value, Entity.external_identifier == accession)
                ).first()
                if existing_entity is not None:
                    counts["sample_entities_already_existed"] += 1
                else:
                    counts["sample_entities_new"] += 1

                entity = None
                if apply:
                    entity = get_or_create_entity(session, study_id, EntityLevel.SAMPLE, accession, label=sample["sample_name"])

                for field, fact_type in MGRAST_FIELD_TO_FACT_TYPE:
                    value = sample[field]
                    if value in (None, ""):
                        continue
                    if existing_entity is not None and (existing_entity.entity_id, fact_type) in existing_facts:
                        counts["facts_already_present"] += 1
                        continue
                    counts["facts_created"] += 1
                    touched_study_ids.add(study_id)
                    if apply:
                        session.add(
                            RawFact(
                                study_id=entity.study_id,
                                entity_id=entity.entity_id,
                                entity_level=EntityLevel.SAMPLE.value,
                                fact_type_candidate=fact_type,
                                raw_field_name=field,
                                raw_value=str(value),
                                source_locator=SOURCE_LOCATOR,
                                support_type=SupportType.STRUCTURED_SOURCE.value,
                                extraction_method="mgrast_enrichment_join",
                            )
                        )
                        existing_facts.add((entity.entity_id, fact_type))

    if apply:
        session.flush()

    needing_remap = sorted(
        session.scalars(
            select(Task.study_id).where(
                Task.task_type == TaskType.MAP_FAIRE.value,
                Task.status == TaskStatus.COMPLETED.value,
                Task.study_id.in_(touched_study_ids),
            )
        ).all()
    ) if touched_study_ids else []

    counts["studies_touched"] = len(touched_study_ids)
    counts["studies_needing_remap"] = needing_remap
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mgrast-db", type=Path, default=Path("data/seed_discovery/mgrast_paper_seeds.sqlite"))
    parser.add_argument("--apply", action="store_true", help="Actually write the entities/facts; otherwise dry-run only.")
    args = parser.parse_args()

    with session_scope() as session:
        result = apply_enrichment(session, args.mgrast_db, apply=args.apply)

    print(f"Studies with an MG-RAST project identifier:                  {result['studies_with_mgrast_identifier']}")
    print(f"Studies matched to real MG-RAST sample rows:                 {result['studies_matched_to_mgrast_samples']}")
    print(f"Sample rows checked:                                         {result['sample_rows_checked']}")
    print(f"Sample entities {'created' if args.apply else 'that would be created'}:" + " " * 16 + f"{result['sample_entities_new']}")
    print(f"Sample entities already existing (merged via accession):     {result['sample_entities_already_existed']}")
    print(f"New facts {'written' if args.apply else 'that would be written'}:" + " " * 24 + f"{result['facts_created']}")
    print(f"Facts already present (skipped, safe re-run):                {result['facts_already_present']}")
    print(f"Studies touched:                                             {result['studies_touched']}")

    if result["studies_needing_remap"]:
        remap = result["studies_needing_remap"]
        print(f"\n{len(remap)} touched studies already completed MAP_FAIRE before this ran -- these won't")
        print("show these new facts in their FAIRe export until explicitly re-mapped:")
        for study_id in remap[:20]:
            print(f"  {study_id}")
        if len(remap) > 20:
            print(f"  ... and {len(remap) - 20} more")

    if not args.apply:
        print("\nDry run only -- pass --apply to write these sample entities/facts.")


if __name__ == "__main__":
    main()
