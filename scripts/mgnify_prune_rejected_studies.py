#!/usr/bin/env python3
"""Re-evaluates every already-discovered MGnify study against the CURRENT
is_marine_study/rejected_biome_terms logic and removes any that no longer
pass -- a real gap in mgnify_discovery.py's own runner: _discover_studies
only ever calls db.upsert_study() on the ACCEPT branch, so a study already
sitting in the database as "accepted" is never retroactively re-evaluated
or removed just because a later code/config change (a parsing fix, a new
rejected_biome_terms entry) would now reject it -- a plain --refresh/
--no-resume re-run leaves it untouched.

Re-parses each study's own stored raw_json (not a fresh network call --
this is a pure re-classification pass over data already on disk) through
the current parse_study + is_marine_study, so it always reflects whatever
filters.py/config.py say right now, not whatever they said when a study
was first discovered.

Dry-run by default -- prints exactly what would be removed and why.
Pass --apply to actually delete (cascades to publication_candidates via
the schema's own ON DELETE CASCADE).

Usage:
    python scripts/mgnify_prune_rejected_studies.py
    python scripts/mgnify_prune_rejected_studies.py --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_ocean_agent.seed_discovery.clients.mgnify import parse_study
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.filters import is_marine_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/seed_discovery/mgnify_paper_seeds.sqlite"))
    parser.add_argument("--apply", action="store_true", help="Actually delete; otherwise dry-run only.")
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="MGnify accession to exclude from removal even if it fails the filter "
        "(e.g. a host-associated study on a marine organism, which the broad "
        "'animal' reject term can't distinguish from a terrestrial one). "
        "Repeatable.",
    )
    args = parser.parse_args()
    keep = set(args.keep)

    config = SeedDiscoveryConfig()
    db = SeedDiscoveryDB(args.db)
    rows = db.conn.execute("SELECT id, mgnify_accession, study_name, raw_json FROM mgnify_studies").fetchall()

    to_remove: list[tuple[int, str, str]] = []
    for row in rows:
        if not row["raw_json"]:
            continue
        study = parse_study(json.loads(row["raw_json"]))
        if not is_marine_study(study, config) and row["mgnify_accession"] not in keep:
            to_remove.append((row["id"], row["mgnify_accession"], row["study_name"] or ""))

    print(f"{len(rows)} studies checked, {len(to_remove)} no longer pass the current filter:")
    for _id, accession, name in to_remove:
        print(f"  {accession}  {name}")
    if keep:
        print(f"\nKept despite failing the filter (--keep): {sorted(keep)}")

    if not args.apply:
        print("\nDry run only -- pass --apply to actually remove these.")
        db.close()
        return

    for study_id, _accession, _name in to_remove:
        db.conn.execute("DELETE FROM mgnify_studies WHERE id = ?", (study_id,))
    db.conn.commit()
    print(f"\nRemoved {len(to_remove)} studies.")
    db.close()


if __name__ == "__main__":
    main()
