#!/usr/bin/env python3
"""Backfills physicochemical/environmental facts onto sample Entities in
the main pipeline database from GOLD's own per-BioSample enrichment table
(gold_faire_enrichment) -- a real, structured data source (233,504 rows:
temperature/salinity/ph/oxygen/chlorophyll/depth/lat-lon/env_broad_scale/
env_local_scale/env_medium/size_fraction/collection date) that nothing in
the main pipeline consumes on its own, since a GOLD-sourced BioProject's
real samples get discovered the normal way (NCBI BioProject -> BioSample
API), completely independent of GOLD's own bulk-downloaded database.

The join key is Entity.external_identifier (a sample's real NCBI BioSample
accession, set when DISCOVER_IDENTIFIERS resolves it) against
gold_sequencing_projects.ncbi_biosample_accession -- NOT
gold_biosamples.ncbi_biosample_accession, which is almost entirely empty
(41 of 233,504). Confirmed live before trusting this: 233,638 of 261,544
gold_sequencing_projects rows (89%) have this field populated, joining
207,520 of 233,504 GOLD biosamples (89%) to a real, usable NCBI accession.

Writes RawFact rows (support_type=structured_source, matching how every
other repository-API-sourced fact is tagged) using the same
fact_type_candidate names mapping/rules.py's existing MappingRules
already expect (temp/salinity/ph/depth/latitude/longitude/
collection_date/env_broad_scale/env_local_scale/env_medium/size_frac),
so MAP_FAIRE picks these up with no further wiring -- plus chlorophyll/
diss_oxygen, which had no MappingRule at any level before this (added
alongside this script).

Idempotent: skips a (entity, fact_type) pair that already has a fact
from this same source_locator, safe to re-run as more samples get
discovered. Dry-run by default.

A study that already completed MAP_FAIRE before this backfill ran won't
automatically pick up these new facts -- nothing in the pipeline
re-triggers mapping on its own (see the pipeline's own docstrings on
this). This script reports which touched studies are in that state so
you know which ones need an explicit re-map once you're ready.

Usage:
    python scripts/apply_gold_physicochemical_enrichment.py
    python scripts/apply_gold_physicochemical_enrichment.py --gold-db data/jgi_gold/gold_sharded.sqlite --apply
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from sqlalchemy import select

from fair_ocean_agent.database.enums import EntityLevel, SupportType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, RawFact, Task
from fair_ocean_agent.database.session import session_scope

SOURCE_LOCATOR = "gold_faire_enrichment"

# (gold_faire_enrichment column, fact_type_candidate expected by an
# existing mapping/rules.py MappingRule at EntityLevel.SAMPLE)
GOLD_FIELD_TO_FACT_TYPE = (
    ("decimalLatitude", "latitude"),
    ("decimalLongitude", "longitude"),
    ("eventDate", "collection_date"),
    ("depth", "depth"),
    ("env_broad_scale", "env_broad_scale"),
    ("env_local_scale", "env_local_scale"),
    ("env_medium", "env_medium"),
    ("size_fraction", "size_frac"),
    ("temperature", "temp"),
    ("salinity", "salinity"),
    ("ph", "ph"),
    ("oxygen", "diss_oxygen"),
    ("chlorophyll", "chlorophyll"),
)


def _gold_enrichment_by_biosample_accession(gold_db_path: Path) -> dict[str, sqlite3.Row]:
    conn = sqlite3.connect(gold_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT sp.ncbi_biosample_accession AS accession, fe.*
        FROM gold_sequencing_projects sp
        JOIN gold_faire_enrichment fe ON fe.gold_biosample_id = sp.gold_biosample_id
        WHERE sp.ncbi_biosample_accession IS NOT NULL AND sp.ncbi_biosample_accession != ''
        ORDER BY sp.gold_biosample_id
        """
    ).fetchall()
    conn.close()
    by_accession: dict[str, sqlite3.Row] = {}
    for row in rows:
        # A BioSample accession reachable from more than one GOLD biosample
        # row is rare and unverifiable which is "correct" -- first one wins,
        # deterministic given the ORDER BY above.
        by_accession.setdefault(row["accession"], row)
    return by_accession


def apply_enrichment(session, gold_db_path: Path, *, apply: bool) -> dict:
    enrichment_by_accession = _gold_enrichment_by_biosample_accession(gold_db_path)

    counts = {"samples_checked": 0, "samples_matched": 0, "facts_created": 0, "facts_already_present": 0}
    touched_study_ids: set[str] = set()

    # One bulk query for every fact this source has ever written, checked
    # in-memory below -- a per-(sample, field) existence query here would
    # be one DB round trip per check, i.e. up to ~13x the sample count;
    # confirmed live this matters at real scale (hundreds of thousands of
    # GOLD-linked samples), not just theoretical.
    existing_facts = {
        (entity_id, fact_type)
        for entity_id, fact_type in session.execute(
            select(RawFact.entity_id, RawFact.fact_type_candidate).where(RawFact.source_locator == SOURCE_LOCATOR)
        ).all()
    }

    samples = session.scalars(
        select(Entity).where(Entity.entity_level == EntityLevel.SAMPLE.value, Entity.external_identifier.isnot(None))
    ).all()
    for sample in samples:
        counts["samples_checked"] += 1
        enrichment = enrichment_by_accession.get(sample.external_identifier)
        if enrichment is None:
            continue
        counts["samples_matched"] += 1
        for gold_field, fact_type in GOLD_FIELD_TO_FACT_TYPE:
            value = enrichment[gold_field]
            if value in (None, ""):
                continue
            if (sample.entity_id, fact_type) in existing_facts:
                counts["facts_already_present"] += 1
                continue
            counts["facts_created"] += 1
            touched_study_ids.add(sample.study_id)
            if apply:
                session.add(
                    RawFact(
                        study_id=sample.study_id,
                        entity_id=sample.entity_id,
                        entity_level=EntityLevel.SAMPLE.value,
                        fact_type_candidate=fact_type,
                        raw_field_name=gold_field,
                        raw_value=str(value),
                        source_locator=SOURCE_LOCATOR,
                        support_type=SupportType.STRUCTURED_SOURCE.value,
                        extraction_method="gold_faire_enrichment_join",
                    )
                )
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
    parser.add_argument("--gold-db", type=Path, default=Path("data/jgi_gold/gold_sharded.sqlite"))
    parser.add_argument("--apply", action="store_true", help="Actually write the facts; otherwise dry-run only.")
    args = parser.parse_args()

    with session_scope() as session:
        result = apply_enrichment(session, args.gold_db, apply=args.apply)

    print(f"Sample entities checked (have a real external_identifier): {result['samples_checked']}")
    print(f"Matched to a GOLD biosample:                                {result['samples_matched']}")
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
        print("\nDry run only -- pass --apply to write these facts.")


if __name__ == "__main__":
    main()
