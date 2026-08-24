from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.jgi_gold import (
    download_gold_snapshot,
    inspect_gold_snapshot,
    print_gold_report,
    process_gold_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/process JGI GOLD bulk metadata for seed discovery and FAIRe enrichment.")
    parser.add_argument("--db", default="data/jgi_gold/gold.sqlite", help="SQLite database path.")
    parser.add_argument("--phase", choices=("all", "download", "inspect", "ingest"), default="all")
    parser.add_argument("--snapshot", help="Snapshot name/date. Defaults to today's date for download or latest local snapshot for processing.")
    parser.add_argument("--raw-dir", help="Existing raw snapshot directory for inspect/ingest.")
    parser.add_argument("--gold-data-dir", default="data/jgi_gold", help="Local GOLD data directory.")
    parser.add_argument("--refresh", action="store_true", help="Download a fresh snapshot before processing.")
    parser.add_argument("--workbook", help="Only ingest a workbook filename, for example public_studies_biosamples_sps_aps_organisms.xlsx.")
    parser.add_argument("--sheet", help="Only ingest one sheet name, for example Biosample or Sequencing Project.")
    parser.add_argument("--start-row", type=int, default=2, help="First Excel row number to ingest. Header is row 1.")
    parser.add_argument("--max-rows", type=int, help="Maximum non-header rows to scan in this run.")
    parser.add_argument("--reset-gold-db", action="store_true", help="Drop/recreate GOLD staging tables before this run.")
    parser.add_argument("--store-source-rows", action="store_true", help="Also store every inserted raw GOLD row in gold_source_rows. This makes the DB much larger.")
    parser.add_argument("--write-reports", action="store_true", help="Write final report CSVs after this run. Defaults off for shard runs.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = SeedDiscoveryConfig(db_path=Path(args.db), gold_data_dir=Path(args.gold_data_dir))

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    if args.phase in {"all", "download"} or args.refresh:
        raw_dir = download_gold_snapshot(config, snapshot=args.snapshot)
        if args.phase == "download":
            logging.info("GOLD raw snapshot: %s", raw_dir)
            return 0
    if raw_dir is None:
        raw_dir = latest_raw_snapshot(config.gold_data_dir, snapshot=args.snapshot)

    if args.phase == "inspect":
        inventory = inspect_gold_snapshot(raw_dir, config.gold_data_dir / "processed" / raw_dir.name)
        logging.info("GOLD schema inventory written for %s workbooks", len(inventory.get("workbooks", {})))
        return 0

    locations = process_gold_snapshot(
        config,
        raw_dir,
        snapshot=args.snapshot or raw_dir.name,
        workbook=args.workbook,
        sheet=args.sheet,
        start_row=args.start_row,
        max_rows=args.max_rows,
        reset_gold_db=args.reset_gold_db,
        store_source_rows=args.store_source_rows,
        write_reports_after_ingest=args.write_reports or not (args.workbook or args.sheet or args.max_rows),
    )
    print_gold_report(locations)
    return 0


def latest_raw_snapshot(gold_data_dir: Path, *, snapshot: str | None = None) -> Path:
    raw_root = gold_data_dir / "raw"
    if snapshot:
        path = raw_root / snapshot
        if not path.exists():
            raise SystemExit(f"raw snapshot does not exist: {path}")
        return path
    candidates = sorted(path for path in raw_root.glob("*") if path.is_dir())
    if not candidates:
        raise SystemExit("no GOLD raw snapshots found; run --phase download first")
    return candidates[-1]


if __name__ == "__main__":
    raise SystemExit(main())
