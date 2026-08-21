from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fair_ocean_agent.seed_discovery.config import RunLimits, SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.ena_discovery import EnaSeedDiscoveryRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover ENA marine sequence publication seeds.")
    parser.add_argument("--db", default="data/seed_discovery/mgnify_paper_seeds.sqlite", help="SQLite database path.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume ENA query partitions from saved cursors.")
    parser.add_argument("--no-resume", action="store_true", help="Start ENA query partitions at offset 0.")
    parser.add_argument("--refresh", action="store_true", help="Refresh API cache and revisit pending/error publication links.")
    parser.add_argument("--phase", choices=("all", "discovery", "publications"), default="all")
    parser.add_argument("--max-pages", type=int, help="Maximum ENA result pages to scan across all query partitions.")
    parser.add_argument("--max-studies", type=int, help="Maximum ENA studies to resolve publications for.")
    parser.add_argument("--include-secondary", action="store_true", help="Include low-confidence tags, taxonomy, and keyword searches.")
    parser.add_argument("--no-openalex", action="store_true", help="Skip OpenAlex; unresolved rows remain queued for later reprocessing.")
    parser.add_argument("--openalex-api-key", help="Optional OpenAlex API key. Email contact is sent by default.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = SeedDiscoveryConfig(
        db_path=Path(args.db),
        openalex_enabled=not args.no_openalex,
        openalex_api_key=args.openalex_api_key,
    )
    limits = RunLimits(
        max_pages=args.max_pages,
        max_studies=args.max_studies,
        refresh=args.refresh,
        resume=not args.no_resume,
        phase=args.phase,
        include_secondary=args.include_secondary,
    )
    runner = EnaSeedDiscoveryRunner(config)
    runner.install_signal_handlers()
    try:
        counts = runner.run(limits)
    finally:
        runner.close()
    for key in sorted(counts):
        logging.info("%s: %s", key, counts[key])
    logging.info("paper seed database: %s", config.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

