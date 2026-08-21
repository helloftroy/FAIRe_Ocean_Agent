from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fair_ocean_agent.seed_discovery.config import RunLimits, SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.mgnify_discovery import MgnifySeedDiscoveryRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover MGnify marine study publication seeds.")
    parser.add_argument("--db", default="data/seed_discovery/mgnify_paper_seeds.sqlite", help="SQLite database path.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from the last MGnify page cursor.")
    parser.add_argument("--no-resume", action="store_true", help="Start discovery at page 1.")
    parser.add_argument("--refresh", action="store_true", help="Revisit already discovered studies and API calls.")
    parser.add_argument("--resolve-only", action="store_true", help="Skip MGnify enumeration and only resolve saved studies.")
    parser.add_argument("--accession", help="Process one MGnify study accession only.")
    parser.add_argument("--max-pages", type=int, help="Maximum MGnify pages to scan.")
    parser.add_argument("--max-studies", type=int, help="Maximum accepted studies to discover/resolve.")
    parser.add_argument("--metadata-search", action="store_true", help="Enable low-confidence OpenAlex metadata fallback.")
    parser.add_argument("--no-openalex", action="store_true", help="Skip OpenAlex calls and mark unresolved studies for later OpenAlex reprocessing.")
    parser.add_argument("--openalex-api-key", help="Optional OpenAlex API key. Email contact is sent by default.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = SeedDiscoveryConfig(
        db_path=Path(args.db),
        metadata_search_enabled=args.metadata_search,
        openalex_enabled=not args.no_openalex,
        openalex_api_key=args.openalex_api_key,
    )
    limits = RunLimits(
        max_pages=args.max_pages,
        max_studies=args.max_studies,
        resolve_only=args.resolve_only,
        refresh=args.refresh,
        resume=not args.no_resume,
        accession=args.accession,
    )
    runner = MgnifySeedDiscoveryRunner(config)
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
