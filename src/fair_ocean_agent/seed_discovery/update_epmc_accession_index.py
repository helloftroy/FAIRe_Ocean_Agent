from __future__ import annotations

import argparse
import csv
import gzip
import logging
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import httpx

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.local_epmc import normalize_epmc_accession

logger = logging.getLogger(__name__)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


@dataclass(frozen=True)
class BulkFile:
    database_name: str
    filename: str
    url: str


def discover_accession_files(config: SeedDiscoveryConfig) -> list[BulkFile]:
    with httpx.Client(timeout=config.request_timeout_seconds, follow_redirects=True) as client:
        response = client.get(config.epmc_accession_bulk_base_url.rstrip("/") + "/")
        response.raise_for_status()
    parser = _HrefParser()
    parser.feed(response.text)
    files: list[BulkFile] = []
    for href in parser.hrefs:
        if not href.endswith(".csv"):
            continue
        filename = Path(href).name
        database = filename.removesuffix(".csv")
        files.append(BulkFile(database, filename, urljoin(config.epmc_accession_bulk_base_url.rstrip("/") + "/", filename)))
    return sorted(files, key=lambda item: item.filename)


def download_file(url: str, destination: Path, *, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded empty file: {url}")
    tmp.replace(destination)


def parse_accession_csv(path: Path, *, database_name: str, snapshot_date: str, batch_size: int = 10000):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{path} has no CSV header")
        accession_column = reader.fieldnames[0]
        normalized_headers = {name.casefold(): name for name in reader.fieldnames}
        pmcid_column = normalized_headers.get("pmcid")
        extid_column = normalized_headers.get("extid") or normalized_headers.get("ext_id")
        source_column = normalized_headers.get("source") or normalized_headers.get("src")
        if not (pmcid_column and extid_column and source_column):
            raise RuntimeError(f"{path} does not look like a Europe PMC accession CSV: {reader.fieldnames}")
        batch = []
        for row in reader:
            normalized = normalize_epmc_accession(row.get(accession_column))
            if not normalized:
                continue
            batch.append(
                {
                    "database_name": database_name,
                    "accession": row.get(accession_column) or normalized,
                    "normalized_accession": normalized,
                    "pmcid": _clean(row.get(pmcid_column)),
                    "article_source": _clean(row.get(source_column)),
                    "article_external_id": _clean(row.get(extid_column)),
                    "snapshot_date": snapshot_date,
                    "source_file": path.name,
                }
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def parse_id_mapping(path: Path, *, snapshot_date: str, batch_size: int = 10000):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "ORA-" in sample or "ERROR:" in sample:
            raise RuntimeError(f"{path} is not a valid mapping export; Europe PMC returned an error payload")
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{path} has no CSV header")
        headers = {name.casefold().replace("_", ""): name for name in reader.fieldnames}
        pmid_column = headers.get("pmid")
        pmcid_column = headers.get("pmcid")
        doi_column = headers.get("doi")
        if not pmid_column:
            raise RuntimeError(f"{path} does not include a PMID column: {reader.fieldnames}")
        batch = []
        for row in reader:
            pmid = _clean(row.get(pmid_column))
            if not pmid:
                continue
            batch.append(
                {
                    "pmid": pmid,
                    "pmcid": _clean(row.get(pmcid_column)) if pmcid_column else None,
                    "doi": _clean(row.get(doi_column)) if doi_column else None,
                    "snapshot_date": snapshot_date,
                    "source_file": path.name,
                }
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().strip('"')
    return value or None


def run_update(config: SeedDiscoveryConfig, *, selected_databases: tuple[str, ...], download: bool = True, id_mapping: bool = True) -> dict[str, int]:
    db = SeedDiscoveryDB(config.db_path)
    db.initialize()
    snapshot_date = utcnow().date().isoformat()
    try:
        available = discover_accession_files(config)
        logger.info("Europe PMC accession files discovered: %s", len(available))
        selected = [item for item in available if item.database_name in set(selected_databases)]
        logger.info("Relevant biological database files selected: %s", len(selected))
        for item in selected:
            logger.info("selected Europe PMC accession file: %s", item.filename)

        counts: dict[str, int] = {}
        accession_dir = config.epmc_bulk_dir / "accessions"
        for item in selected:
            path = accession_dir / item.filename
            if download:
                download_file(item.url, path, timeout=config.request_timeout_seconds)
            imported = 0
            for batch in parse_accession_csv(path, database_name=item.database_name, snapshot_date=snapshot_date):
                imported += db.upsert_epmc_accession_links(batch)
            counts[f"accession_links_{item.database_name}"] = imported
            logger.info("imported %s accession/article links from %s", imported, item.filename)

        if id_mapping:
            mapping_path = config.epmc_bulk_dir / "ids" / Path(config.epmc_id_mapping_url).name
            if download:
                download_file(config.epmc_id_mapping_url, mapping_path, timeout=config.request_timeout_seconds)
            imported = 0
            try:
                for batch in parse_id_mapping(mapping_path, snapshot_date=snapshot_date):
                    imported += db.upsert_epmc_article_ids(batch)
            except RuntimeError as exc:
                counts["article_id_mappings"] = 0
                counts["article_id_mapping_skipped"] = 1
                logger.warning("skipping PMID/PMCID/DOI mapping import: %s", exc)
            else:
                counts["article_id_mappings"] = imported
                logger.info("imported %s PMID/PMCID/DOI mapping rows", imported)
        return counts
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update the local Europe PMC accession-to-publication index.")
    parser.add_argument("--db", default="data/seed_discovery/mgnify_paper_seeds.sqlite", help="SQLite database path.")
    parser.add_argument("--bootstrap", action="store_true", help="Download and ingest the current Europe PMC bulk snapshot.")
    parser.add_argument("--refresh", action="store_true", help="Alias for --bootstrap; current snapshots replace routine API searching.")
    parser.add_argument("--no-download", action="store_true", help="Ingest already downloaded files from the local bulk directory.")
    parser.add_argument("--skip-id-mapping", action="store_true", help="Skip PMID/PMCID/DOI mapping import.")
    parser.add_argument("--databases", help="Comma-separated Europe PMC accession database names to import.")
    parser.add_argument("--bulk-dir", default=None, help="Local staging directory for downloaded Europe PMC files.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    default_config = SeedDiscoveryConfig()
    databases = tuple(
        item.strip()
        for item in (args.databases.split(",") if args.databases else default_config.epmc_selected_accession_databases)
        if item.strip()
    )
    config = SeedDiscoveryConfig(db_path=Path(args.db))
    if args.bulk_dir:
        config = SeedDiscoveryConfig(db_path=Path(args.db), epmc_bulk_dir=Path(args.bulk_dir))
    if not (args.bootstrap or args.refresh or args.no_download):
        logging.error("Pass --bootstrap/--refresh to download, or --no-download to ingest existing staged files.")
        return 2
    counts = run_update(config, selected_databases=databases, download=not args.no_download, id_mapping=not args.skip_id_mapping)
    for key in sorted(counts):
        logging.info("%s: %s", key, counts[key])
    logging.info("paper seed database: %s", config.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
