from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.gold_bioproject_publications import (
    DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
    GoldBioprojectPublicationSearchRunner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Europe PMC + NCBI (deliberately no OpenAlex -- see the runner's own "
        "docstring) for the paper behind each real NCBI BioProject accession GOLD recorded, instead "
        "of trusting GOLD's own bulk publication field."
    )
    parser.add_argument("--db", default="data/jgi_gold/gold_sharded.sqlite", help="GOLD sqlite snapshot (default: the canonical sharded snapshot).")
    parser.add_argument("--limit", type=int, help="Only check this many not-yet-checked BioProject accessions this run.")
    parser.add_argument("--refresh", action="store_true", help="Re-check every accession, including ones already checked.")
    parser.add_argument(
        "--max-consecutive-429s",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
        help="Stop the whole run once this many accessions in a row fail with 429 Too Many Requests. 0 disables this.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = SeedDiscoveryConfig(db_path=Path(args.db))
    db = SeedDiscoveryDB(args.db)
    runner = GoldBioprojectPublicationSearchRunner(config, db)
    runner.install_signal_handlers()
    try:
        counts = runner.run(
            limit=args.limit,
            refresh=args.refresh,
            max_consecutive_rate_limit_failures=args.max_consecutive_429s if args.max_consecutive_429s > 0 else None,
        )
    finally:
        runner.close()
        db.close()

    for key in sorted(k for k in counts if k != "stopped_reason"):
        logging.info("%s: %s", key, counts[key])
    if counts.get("stopped_reason"):
        logging.warning("Stopped early: %s", counts["stopped_reason"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
