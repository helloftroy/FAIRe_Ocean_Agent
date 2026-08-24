#!/usr/bin/env python3
"""Picks one 'primary' DOI per GOLD study from the publication candidates
GOLD's own bulk export already lists (gold_study_publications), and writes
it to gold_studies.primary_doi -- the resolution step GOLD provides no
equivalent of on its own.

Deliberately does not touch the network or re-resolve any paper's
identity: GOLD already lists every paper that mentions a study/project,
the only open question is which one actually produced the sequencing
data versus one that's just reusing/citing it. The signal used is how
many distinct NCBI BioProjects a candidate DOI is linked to across the
whole GOLD corpus ("bioproject_fanout") -- a paper tied to a handful of
BioProjects generated that data; a paper tied to dozens or hundreds is a
consortium/overview/reanalysis paper. See
jgi_gold.resolve_gold_primary_publications's own docstring for the full
reasoning and the empirical check behind the default threshold.

Dry-run by default -- prints outcome counts only. Pass --apply to
actually write gold_studies.primary_doi/primary_doi_status/
primary_doi_selection_reason/primary_doi_bioproject_fanout.

Usage:
    python scripts/resolve_gold_primary_publications.py
    python scripts/resolve_gold_primary_publications.py --apply
    python scripts/resolve_gold_primary_publications.py --low-fanout-threshold 3 --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.jgi_gold import DEFAULT_LOW_FANOUT_THRESHOLD, resolve_gold_primary_publications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/jgi_gold/gold_sharded.sqlite"),
        help="GOLD sqlite snapshot to resolve against (default: the canonical sharded snapshot).",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write the resolution; otherwise dry-run only.")
    parser.add_argument(
        "--low-fanout-threshold",
        type=int,
        default=DEFAULT_LOW_FANOUT_THRESHOLD,
        help=f"A candidate DOI linked to more than this many distinct BioProjects is treated as a "
        f"reanalysis/consortium paper, not a source paper (default: {DEFAULT_LOW_FANOUT_THRESHOLD}).",
    )
    args = parser.parse_args()

    db = SeedDiscoveryDB(args.db)
    db.initialize()
    counts = resolve_gold_primary_publications(db, low_fanout_threshold=args.low_fanout_threshold, apply=args.apply)
    db.close()

    print(f"GOLD studies with at least one candidate DOI: {counts['gold_studies_with_candidates']}")
    print(f"  resolved (single lowest-fanout candidate):   {counts['resolved']}")
    print(f"  resolved_ambiguous (tied lowest fanout):      {counts['resolved_ambiguous']}")
    print(f"  likely_reanalysis_only (no low-fanout candidate): {counts['likely_reanalysis_only']}")
    if not args.apply:
        print("\nDry run only -- pass --apply to write these into gold_studies.")


if __name__ == "__main__":
    main()
