#!/usr/bin/env python3
"""Backfills physicochemical/environmental facts (and the sample entities
themselves) from CNCB/GSA's own per-BioSample scrape (cncb_samples,
populated by cncb_gsa_discovery.py's metadata phase) onto the main
pipeline database -- the same idea as apply_gold_physicochemical_enrichment.py,
but for a real structural reason it can't just enrich existing entities the
way that script does.

GOLD's script could assume a matching sample Entity already existed,
because a GOLD-sourced BioProject always resolves through the *existing*
NCBI BioProject -> BioSample pathway (a real, independent discovery route
that has nothing to do with GOLD's own bulk database) -- GOLD's script only
had to join onto samples the main pipeline had already created via a real
NCBI BioSample accession.

CNCB/GSA has no equivalent: a CNCB-native study (no INSDC BioProject
crosslink -- most of them) never gets its individual samples enumerated by
anything else in this pipeline, since no CNCB/GSA source adapter exists
(see cncb_gsa_discovery.py's own docstring precedent decision -- native
CNCB accessions are recognized and verified during discovery, but nothing
walks a resolved CNCB study and fetches its real samples the way ena.py or
qiita.py do). So this script does both jobs at once: for every Study that
has a CNCB_PROJECT_ACCESSION or CNCB_STUDY_ACCESSION identifier, get-or-
create one real SAMPLE Entity per real SAMC accession CNCB/GSA already
scraped for that project (via identity/entity_linking.py's
get_or_create_entity -- the same established, idempotent, cross-study-safe
choke point workflow/handlers.py and sources/qiita.py already use, so a
SAMC accession independently discovered some other way, e.g. free-text
mining, merges into the same Entity rather than duplicating it), then
writes the physicochemical facts onto it.

Writes RawFact rows (support_type=structured_source) using the same
fact_type_candidate names mapping/rules.py's existing MappingRules already
expect (temp/salinity/ph/diss_oxygen/chlorophyll/depth/latitude/longitude/
collection_date/env_broad_scale/env_local_scale/env_medium/size_frac/
samp_name) -- no new mapping rules needed, this rides the same rules GOLD's
enrichment already uses.

Deliberately out of scope (same discipline as the GOLD script not
inventing new fact types): CNCB's own nitrate/nitrite/ammonium/phosphate/
pressure columns have no existing MappingRule and aren't official FAIRe
fields under those exact names -- left as other_environmental_measurements_
json in cncb_samples for now, not backfilled here. Also out of scope:
cncb_experiments' sequencing/library metadata (platform, primers, target
gene) -- this script is physicochemical/sample-level only, matching what
was actually asked for; a study's own real sequencing metadata is a
separate, later concern.

Idempotent: skips a (entity, fact_type) pair that already has a fact from
this same source_locator, and get_or_create_entity is itself idempotent by
accession, so safe to re-run as CNCB discovery adds more projects/samples
over time. Dry-run by default -- in dry-run mode no entities are created
either (a fact can't be attached to an entity that doesn't exist yet), so
dry-run's facts_created count is an estimate against not-yet-created
entities, not a literal preview of what rows would exist.

A study that already completed MAP_FAIRE before this backfill ran won't
automatically pick up these new facts (same documented gap as the GOLD
script) -- reported here the same way, so you know which ones need an
explicit re-map.

Usage:
    python scripts/apply_cncb_physicochemical_enrichment.py
    python scripts/apply_cncb_physicochemical_enrichment.py --cncb-db data/seed_discovery/cncb_gsa_paper_seeds.sqlite --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, SupportType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Task
from fair_ocean_agent.database.session import session_scope
from fair_ocean_agent.identity.entity_linking import get_or_create_entity

SOURCE_LOCATOR = "cncb_gsa_enrichment"

# (cncb_samples column, fact_type_candidate expected by an existing
# mapping/rules.py MappingRule at EntityLevel.SAMPLE) -- same fact_type
# names apply_gold_physicochemical_enrichment.py already writes. Deliberately
# excludes sample_name/samp_name: mapping/faire.py's own MAP_FAIRE step
# already derives sampleMetadata's samp_name/materialSampleID directly from
# Entity.external_identifier (the real SAMC accession, set below via
# get_or_create_entity) for every SAMPLE entity regardless of source -- no
# MappingRule exists for it at this level, so a samp_name RawFact here would
# just sit unused. CNCB's own free-text sample_name is instead kept as the
# entity's `label` (passed to get_or_create_entity below), separate from the
# FAIRe export field.
CNCB_FIELD_TO_FACT_TYPE = (
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


def _cncb_bioproject_lookups(cncb_db_path: Path) -> tuple[dict[str, str], dict[str, list[sqlite3.Row]]]:
    conn = sqlite3.connect(cncb_db_path)
    conn.row_factory = sqlite3.Row
    cra_to_bioproject: dict[str, str] = {}
    for row in conn.execute("SELECT cncb_bioproject, cra_accessions_json FROM cncb_projects"):
        for cra in json.loads(row["cra_accessions_json"] or "[]"):
            cra_to_bioproject[cra.strip().upper()] = row["cncb_bioproject"]
    samples_by_bioproject: dict[str, list[sqlite3.Row]] = {}
    for row in conn.execute(
        "SELECT * FROM cncb_samples WHERE samc_accession IS NOT NULL AND samc_accession != ''"
    ):
        samples_by_bioproject.setdefault(row["cncb_bioproject"], []).append(row)
    conn.close()
    return cra_to_bioproject, samples_by_bioproject


def _resolve_study_bioprojects(session, cra_to_bioproject: dict[str, str]) -> dict[str, set[str]]:
    study_bioprojects: dict[str, set[str]] = {}
    identifiers = session.execute(
        select(ExternalIdentifier.study_id, ExternalIdentifier.identifier_type, ExternalIdentifier.identifier_value).where(
            ExternalIdentifier.identifier_type.in_(
                [IdentifierType.CNCB_PROJECT_ACCESSION.value, IdentifierType.CNCB_STUDY_ACCESSION.value]
            )
        )
    ).all()
    for study_id, identifier_type, identifier_value in identifiers:
        value = identifier_value.strip().upper()
        bioproject = value if identifier_type == IdentifierType.CNCB_PROJECT_ACCESSION.value else cra_to_bioproject.get(value)
        if bioproject is None:
            continue
        study_bioprojects.setdefault(study_id, set()).add(bioproject)
    return study_bioprojects


def apply_enrichment(session, cncb_db_path: Path, *, apply: bool) -> dict:
    cra_to_bioproject, samples_by_bioproject = _cncb_bioproject_lookups(cncb_db_path)
    study_bioprojects = _resolve_study_bioprojects(session, cra_to_bioproject)

    counts = {
        "studies_with_cncb_identifier": len(study_bioprojects),
        "studies_matched_to_cncb_samples": 0,
        "sample_rows_checked": 0,
        "sample_entities_new": 0,
        "sample_entities_already_existed": 0,
        "facts_created": 0,
        "facts_already_present": 0,
    }
    touched_study_ids: set[str] = set()

    # One bulk query for every fact this source has ever written -- same
    # rationale as the GOLD script: a per-(sample, field) existence check
    # would be one DB round trip per check.
    existing_facts = {
        (entity_id, fact_type)
        for entity_id, fact_type in session.execute(
            select(RawFact.entity_id, RawFact.fact_type_candidate).where(RawFact.source_locator == SOURCE_LOCATOR)
        ).all()
    }

    for study_id, bioprojects in study_bioprojects.items():
        study_matched = False
        for bioproject in bioprojects:
            samples = samples_by_bioproject.get(bioproject)
            if not samples:
                continue
            study_matched = True
            for sample in samples:
                counts["sample_rows_checked"] += 1
                accession = sample["samc_accession"].strip().upper()

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

                for field, fact_type in CNCB_FIELD_TO_FACT_TYPE:
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
                                extraction_method="cncb_gsa_enrichment_join",
                            )
                        )
                        existing_facts.add((entity.entity_id, fact_type))
        if study_matched:
            counts["studies_matched_to_cncb_samples"] += 1

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
    parser.add_argument("--cncb-db", type=Path, default=Path("data/seed_discovery/cncb_gsa_paper_seeds.sqlite"))
    parser.add_argument("--apply", action="store_true", help="Actually write the entities/facts; otherwise dry-run only.")
    args = parser.parse_args()

    with session_scope() as session:
        result = apply_enrichment(session, args.cncb_db, apply=args.apply)

    print(f"Studies with a CNCB project/study identifier:               {result['studies_with_cncb_identifier']}")
    print(f"Studies matched to real CNCB/GSA sample rows:                {result['studies_matched_to_cncb_samples']}")
    print(f"Sample rows checked:                                        {result['sample_rows_checked']}")
    print(f"Sample entities {'created' if args.apply else 'that would be created'}:" + " " * 16 + f"{result['sample_entities_new']}")
    print(f"Sample entities already existing (merged via accession):    {result['sample_entities_already_existed']}")
    print(f"New facts {'written' if args.apply else 'that would be written'}:" + " " * 24 + f"{result['facts_created']}")
    print(f"Facts already present (skipped, safe re-run):               {result['facts_already_present']}")
    print(f"Studies touched:                                            {result['studies_touched']}")

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
