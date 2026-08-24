#!/usr/bin/env python3
"""Export seed-discovery MGnify + ENA paper seeds to ingest-seeds CSV.

This is the clearer name for scripts/mgnify_seeds_to_csv.py, which already
exports both MGnify and ENA rows from the seed-discovery paper_seeds view.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mgnify_seeds_to_csv import convert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/seed_discovery/mgnify_paper_seeds.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("cluster/seeds_seed_discovery.csv"))
    args = parser.parse_args()

    written, no_doi = convert(args.db, args.out)
    print(f"Wrote {written} seed rows to {args.out} ({no_doi} with no resolved DOI -- repository-only via BioProject)")


if __name__ == "__main__":
    main()
