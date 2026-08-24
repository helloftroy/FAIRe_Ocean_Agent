from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import httpx

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB, utc_iso
from fair_ocean_agent.seed_discovery.models import ResolutionStatus

logger = logging.getLogger(__name__)

DOWNLOAD_MODES = {
    "public_studies_biosamples_sps_aps_organisms": "site_excel",
    "public_sra_biome_img_annotations": "sra_biome_img_excel",
    "gold_cvs": "cv_excel",
    "ecosystem_paths": "ecosystempaths",
}

MARINE_TERMS = (
    "marine",
    "ocean",
    "sea water",
    "seawater",
    "marine sediment",
    "coastal",
    "estuary",
    "estuarine",
    "brackish",
    "coral reef",
    "sea ice",
    "hydrothermal vent",
    "pelagic",
    "benthic",
    "continental shelf",
    "deep ocean",
    "envo_00000447",
    "envo_00001999",
    "envo_00002149",
)

ENVIRONMENTAL_STRATEGY_TERMS = (
    "metagenome",
    "metagenomic",
    "metatranscriptome",
    "metatranscriptomic",
    "amplicon",
    "targeted",
    "environmental",
)

FIELD_CANDIDATES = {
    "gold_study_id": ("gold study id", "study gold id", "study id", "gold id"),
    "gold_biosample_id": ("gold biosample id", "biosample gold id", "biosample id"),
    "gold_project_id": ("project gold id", "gold sequencing project id", "sequencing project id", "gold project id", "project id", "ap project gold ids"),
    "gold_analysis_project_id": ("ap gold id", "gold analysis project id", "analysis project id"),
    "gold_study_id_ref": ("study gold id", "gold study id", "ap study gold id", "study id"),
    "gold_biosample_id_ref": ("biosample gold id", "gold biosample id", "biosample id"),
    "study_name": ("study name", "name"),
    "study_description": ("study description", "description"),
    "biosample_name": ("biosample name", "sample name", "name"),
    "collection_date": ("biosample sample collection date", "sample collection date", "collection date", "sampling date"),
    "latitude": ("biosample latitude", "latitude", "lat"),
    "longitude": ("biosample longitude", "longitude", "lon", "longitude"),
    "depth": ("depth", "water depth", "sample depth"),
    "ecosystem": ("biosample ecosystem", "ecosystem"),
    "ecosystem_category": ("biosample ecosystem category", "ecosystem category"),
    "ecosystem_type": ("biosample ecosystem type", "ecosystem type"),
    "ecosystem_subtype": ("biosample ecosystem subtype", "ecosystem subtype", "ecosystem sub type"),
    "specific_ecosystem": ("biosample specific ecosystem", "specific ecosystem"),
    "ecosystem_path_id": ("biosample ecosystem path id", "ecosystem path id"),
    "env_broad_scale": ("broad-scale environmental context", "broad scale environmental context", "env broad scale"),
    "env_local_scale": ("local environmental context", "local-scale environmental context", "env local scale"),
    "env_medium": ("environmental medium", "env medium"),
    "habitat": ("habitat",),
    "geo_loc_name": ("biosample geographic location", "biosample sample collection site", "geographic location", "geographic location name", "sample isolation country/ocean", "country/ocean"),
    "ncbi_bioproject": ("ncbi bioproject accession", "bioproject accession", "ncbi bioproject"),
    "ncbi_biosample": ("ncbi biosample accession", "biosample accession", "ncbi biosample"),
    "sequencing_strategy": ("sequencing strategy", "seq method", "sequencing method", "project type"),
    "project_status": ("project status", "sequencing status", "status"),
    "jgi_project_id": ("its sequencing project id", "jgi project id", "jgi portal id", "portal id"),
    "publication_title": ("publication title", "pub title", "paper title", "article title"),
    "pmid": ("project genome publication pubmed id", "project other publication pubmed id", "pmid", "pubmed id", "pubmed"),
    "pmcid": ("pmcid", "pmc id"),
    "analysis_project_type": ("ap type", "analysis project type", "analysis type"),
    "img_identifier": ("ap img taxon id", "img taxon id", "img genome id", "img id", "img identifier"),
    "sample_collection_method": ("sample collection method", "collection method"),
    "size_fraction": ("size fraction",),
    "temperature": ("temperature", "sample collection temperature"),
    "salinity": ("salinity", "salinity concentration"),
    "ph": ("ph", "pH"),
    "oxygen": ("oxygen", "oxygen concentration", "dissolved oxygen"),
    "chlorophyll": ("chlorophyll", "chlorophyll concentration"),
}

INGEST_ENTITY_TYPES = {"study", "biosample", "sequencing_project", "analysis_project"}
INGEST_COMMIT_INTERVAL = 5_000
INGEST_PROGRESS_INTERVAL = 50_000

FAIRE_MAPPING_ROWS = (
    ("Latitude", "decimalLatitude", "high", "Direct coordinate candidate; preserve original units/format."),
    ("Longitude", "decimalLongitude", "high", "Direct coordinate candidate; preserve original units/format."),
    ("Sample Collection Date", "eventDate", "high", "Candidate event date."),
    ("Depth", "depth", "medium", "May need water/sediment/sample-depth context."),
    ("Geographic Location", "geo_loc_name", "medium", "Candidate locality/country-ocean string."),
    ("Broad-scale Environmental Context", "env_broad_scale", "high", "MIxS/ENVO-like broad context."),
    ("Local Environmental Context", "env_local_scale", "high", "MIxS/ENVO-like local context."),
    ("Environmental Medium", "env_medium", "high", "MIxS/ENVO-like material/medium."),
    ("Sample Collection Method", "samp_collect_method", "medium", "Candidate sampling method; review context."),
    ("Size Fraction", "size_frac", "medium", "Candidate size fraction/filter information."),
    ("Temperature", "temperature", "medium", "Environmental measurement staging."),
    ("Salinity", "salinity", "medium", "Environmental measurement staging."),
    ("pH", "ph", "medium", "Environmental measurement staging."),
    ("Oxygen", "oxygen", "medium", "Environmental measurement staging."),
    ("Chlorophyll", "chlorophyll", "medium", "Environmental measurement staging."),
)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text_parts.append(data.strip())


@dataclass(frozen=True)
class GoldDownload:
    key: str
    mode: str
    url: str
    filename: str
    source_last_generated_date: str | None = None


def discover_gold_downloads(config: SeedDiscoveryConfig) -> list[GoldDownload]:
    with httpx.Client(timeout=config.request_timeout_seconds, follow_redirects=True) as client:
        response = client.get(config.gold_downloads_url)
        response.raise_for_status()
    parser = _HrefParser()
    parser.feed(response.text)
    page_text = " ".join(parser.text_parts)
    downloads: list[GoldDownload] = []
    for key, mode in DOWNLOAD_MODES.items():
        href = next((href for href in parser.hrefs if f"mode={mode}" in href), f"/download?mode={mode}")
        generated = _generated_date_for_mode(page_text, key)
        downloads.append(
            GoldDownload(
                key=key,
                mode=mode,
                url=urljoin(config.gold_base_url.rstrip("/") + "/", href),
                filename=f"{key}.xlsx",
                source_last_generated_date=generated,
            )
        )
    return downloads


def download_gold_snapshot(config: SeedDiscoveryConfig, *, snapshot: str | None = None) -> Path:
    snapshot_name = snapshot or date.today().isoformat()
    raw_dir = config.gold_data_dir / "raw" / snapshot_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloads = discover_gold_downloads(config)
    manifest = {
        "download_timestamp": utcnow().isoformat(),
        "source": config.gold_downloads_url,
        "snapshot": snapshot_name,
        "files": [],
    }
    with httpx.Client(timeout=config.request_timeout_seconds, follow_redirects=True) as client:
        for item in downloads:
            path = raw_dir / item.filename
            tmp = path.with_suffix(path.suffix + ".tmp")
            logger.info("downloading GOLD %s from %s", item.key, item.url)
            with client.stream("GET", item.url) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"downloaded empty GOLD file: {item.url}")
            tmp.replace(path)
            manifest["files"].append(
                {
                    "key": item.key,
                    "source_url": item.url,
                    "source_last_generated_date": item.source_last_generated_date,
                    "filename": path.name,
                    "file_size": path.stat().st_size,
                    "checksum_sha256": sha256_file(path),
                }
            )
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return raw_dir


def inspect_gold_snapshot(raw_dir: Path, processed_dir: Path) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    inventory = {"raw_dir": str(raw_dir), "workbooks": {}}
    for workbook_path in sorted(raw_dir.glob("*.xlsx")):
        sheets = {}
        for sheet_name, rows in iter_xlsx_sheets(workbook_path):
            header = next(rows, None)
            columns = [str(value).strip() for value in header if value is not None] if header else []
            sampled_rows = 0
            for row in rows:
                if any(value not in (None, "") for value in row):
                    sampled_rows += 1
                if sampled_rows >= 25:
                    break
            sheets[sheet_name] = {
                "rows": None,
                "row_count_note": "not counted during quick inspection because GOLD exports can contain multi-GB worksheet XML",
                "sampled_nonempty_rows": sampled_rows,
                "columns": columns,
                "inferred_entity_type": infer_entity_type(workbook_path.name, sheet_name, columns),
            }
            logger.info("GOLD workbook=%s sheet=%s sampled_rows=%s columns=%s", workbook_path.name, sheet_name, sampled_rows, len(columns))
        inventory["workbooks"][workbook_path.name] = {"sheets": sheets}
    path = processed_dir / "schema_inventory.json"
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    return inventory


def process_gold_snapshot(
    config: SeedDiscoveryConfig,
    raw_dir: Path,
    *,
    snapshot: str | None = None,
    workbook: str | None = None,
    sheet: str | None = None,
    start_row: int = 2,
    max_rows: int | None = None,
    reset_gold_db: bool = True,
    store_source_rows: bool = True,
    write_reports_after_ingest: bool = True,
) -> dict[str, Path | str | dict]:
    db = SeedDiscoveryDB(config.db_path)
    db.initialize()
    ensure_gold_runtime_indexes(db)
    snapshot_name = snapshot or raw_dir.name
    processed_dir = config.gold_data_dir / "processed" / snapshot_name
    reports_dir = config.gold_data_dir / "reports" / snapshot_name
    file_manifest_dir = config.gold_data_dir / "file_manifests" / snapshot_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    file_manifest_dir.mkdir(parents=True, exist_ok=True)
    try:
        inventory = inspect_gold_snapshot(raw_dir, processed_dir)
        if reset_gold_db:
            clear_gold_snapshot(db, snapshot_name)
        progress_path = reports_dir / "ingest_progress.json"
        counts = ingest_gold_workbooks(
            db,
            raw_dir,
            snapshot_name,
            progress_path=progress_path,
            workbook_filter=workbook,
            sheet_filter=sheet,
            start_row=start_row,
            max_rows=max_rows,
            store_source_rows=store_source_rows,
        )
        write_faire_mapping_candidates(processed_dir / "faire_mapping_candidates.csv")
        write_jgi_file_manifest(db, file_manifest_dir / "jgi_file_manifest.csv")
        if write_reports_after_ingest:
            write_reports(db, reports_dir)
        output_locations = {
            "snapshot": snapshot_name,
            "shard": {
                "workbook": workbook,
                "sheet": sheet,
                "start_row": start_row,
                "max_rows": max_rows,
                "reset_gold_db": reset_gold_db,
                "store_source_rows": store_source_rows,
                "write_reports_after_ingest": write_reports_after_ingest,
            },
            "raw_download_dir": str(raw_dir.resolve()),
            "manifest": str((raw_dir / "manifest.json").resolve()),
            "schema_inventory": str((processed_dir / "schema_inventory.json").resolve()),
            "database": str(config.db_path),
            "tables": {
                "source_rows": "gold_source_rows",
                "studies": "gold_studies",
                "biosamples": "gold_biosamples",
                "sequencing_projects": "gold_sequencing_projects",
                "analysis_projects": "gold_analysis_projects",
                "faire_enrichment": "gold_faire_enrichment",
            },
            "jgi_file_manifest_dir": str(file_manifest_dir.resolve()),
            "jgi_file_manifest": str((file_manifest_dir / "jgi_file_manifest.csv").resolve()),
            "reports": {
                "new_studies": str((reports_dir / "new_studies.csv").resolve()),
                "enrichment_existing": str((reports_dir / "enrichment_existing_studies.csv").resolve()),
                "metadata_completeness": str((reports_dir / "metadata_completeness.csv").resolve()),
                "faire_mapping_candidates": str((processed_dir / "faire_mapping_candidates.csv").resolve()),
                "ingest_progress": str(progress_path.resolve()),
            },
            "counts": counts,
        }
        output_path = config.gold_data_dir / "OUTPUT_LOCATIONS.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_locations, indent=2, sort_keys=True), encoding="utf-8")
        return output_locations
    finally:
        db.close()


def clear_gold_snapshot(db: SeedDiscoveryDB, snapshot: str) -> None:
    logger.info("resetting GOLD staging tables before ingesting snapshot=%s", snapshot)
    db.conn.executescript(
        """
        DROP TABLE IF EXISTS gold_source_rows;
        DROP TABLE IF EXISTS gold_studies;
        DROP TABLE IF EXISTS gold_biosamples;
        DROP TABLE IF EXISTS gold_sequencing_projects;
        DROP TABLE IF EXISTS gold_analysis_projects;
        DROP TABLE IF EXISTS gold_study_publications;
        DROP TABLE IF EXISTS gold_project_jgi_files;
        DROP TABLE IF EXISTS gold_faire_enrichment;
        """
    )
    db.conn.commit()
    db.initialize()
    ensure_gold_runtime_indexes(db)


def ensure_gold_runtime_indexes(db: SeedDiscoveryDB) -> None:
    db.conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_gold_publications_identity
          ON gold_study_publications(
            COALESCE(gold_study_id, ''),
            COALESCE(gold_project_id, ''),
            COALESCE(doi, ''),
            COALESCE(pmid, ''),
            COALESCE(pmcid, ''),
            COALESCE(title, ''),
            COALESCE(source_snapshot_date, '')
          );
        CREATE INDEX IF NOT EXISTS idx_gold_publications_doi
          ON gold_study_publications(doi);
        CREATE INDEX IF NOT EXISTS idx_gold_publications_project
          ON gold_study_publications(gold_project_id);
        CREATE INDEX IF NOT EXISTS idx_gold_jgi_files_placeholder
          ON gold_project_jgi_files(
            COALESCE(gold_project_id, ''),
            COALESCE(jgi_project_id, ''),
            COALESCE(source_snapshot_date, ''),
            availability_status
          );
        CREATE INDEX IF NOT EXISTS idx_gold_jgi_files_project
          ON gold_project_jgi_files(gold_project_id, jgi_project_id);
        """
    )
    db.conn.commit()


DEFAULT_LOW_FANOUT_THRESHOLD = 5


def resolve_gold_primary_publications(
    db: SeedDiscoveryDB,
    *,
    low_fanout_threshold: int = DEFAULT_LOW_FANOUT_THRESHOLD,
    apply: bool = False,
) -> dict[str, int]:
    """Picks one 'primary' DOI per GOLD study and writes it to
    gold_studies.primary_doi -- the resolution step GOLD's own bulk
    publication field never provides. Unlike MGnify/ENA, where
    publication_resolver.py searches OpenAlex/Europe PMC to *find* a
    study's paper, GOLD already lists every paper that mentions a
    study/project directly in its own export (see upsert_gold_publication,
    match_method='gold_bulk_publication_field') -- the open question here
    isn't which papers exist, it's which one actually produced the
    sequencing data versus one that's just reusing/citing it. Per an
    explicit user request, this deliberately never touches the network or
    re-resolves a paper's identity to answer that.

    The one signal available without polling anything: how many distinct
    NCBI BioProjects a candidate DOI is linked to across the *whole* GOLD
    corpus (its "bioproject_fanout"), not just this one study's own
    candidates. A paper that only ever appears against one or a handful
    of BioProjects generated that data; a paper linked to dozens or
    hundreds is a consortium/overview/reanalysis paper that cites
    already-published data, not its source. Confirmed empirically against
    data/jgi_gold/gold_sharded.sqlite before writing this: of 635
    candidate DOIs, only 35 touch exactly one BioProject while 344 touch
    21 or more -- a bimodal split, not a continuum, so a single threshold
    cleanly separates "wrote this data" from "reused this data" papers.
    default=5 keeps room for a genuine multi-site study depositing a
    handful of related BioProjects from one paper, per the user's own
    "some papers might have more than one" caveat.

    A study-level publication row (gold_project_id IS NULL) is treated as
    covering every BioProject under that study's own sequencing projects
    -- GOLD records it once per study rather than once per project, but
    it's making the same claim about all of them either way.

    Ties for the lowest fanout are broken by DOI string (deterministic,
    reproducible) and recorded as RESOLVED_AMBIGUOUS rather than silently
    picked as if certain. A study whose only candidates all exceed the
    threshold is left with no primary_doi at all
    (LIKELY_REANALYSIS_ONLY) -- guessing among only-reanalysis candidates
    would be worse than leaving it for a real citation-chase/human pass to
    find the actual source paper later. A candidate DOI with no
    computable fanout at all (none of its linked projects have a
    populated ncbi_bioproject_accession -- true for ~10% of GOLD
    sequencing projects) is treated the same as an over-threshold
    candidate: unverifiable is not the same as verified-safe.

    apply=False (the default) computes and returns the outcome counts
    without writing anything -- callers doing a dry run first should pass
    apply=False, inspect the counts, then call again with apply=True."""
    db.conn.executescript(
        """
        DROP TABLE IF EXISTS _gold_doi_bioproject_fanout;
        CREATE TEMP TABLE _gold_doi_bioproject_fanout AS
        WITH doi_bioprojects AS (
            SELECT gsp.doi AS doi, sp.ncbi_bioproject_accession AS bioproject
            FROM gold_study_publications gsp
            JOIN gold_sequencing_projects sp ON sp.gold_study_id = gsp.gold_study_id
            WHERE gsp.doi IS NOT NULL AND gsp.doi != '' AND gsp.gold_project_id IS NULL
              AND sp.ncbi_bioproject_accession IS NOT NULL AND sp.ncbi_bioproject_accession != ''
            UNION
            SELECT gsp.doi AS doi, sp.ncbi_bioproject_accession AS bioproject
            FROM gold_study_publications gsp
            JOIN gold_sequencing_projects sp ON sp.gold_project_id = gsp.gold_project_id
            WHERE gsp.doi IS NOT NULL AND gsp.doi != '' AND gsp.gold_project_id IS NOT NULL
              AND sp.ncbi_bioproject_accession IS NOT NULL AND sp.ncbi_bioproject_accession != ''
        )
        SELECT doi, COUNT(DISTINCT bioproject) AS fanout
        FROM doi_bioprojects
        GROUP BY doi;

        DROP TABLE IF EXISTS _gold_study_doi_candidates;
        CREATE TEMP TABLE _gold_study_doi_candidates AS
        SELECT DISTINCT gsp.gold_study_id AS gold_study_id, gsp.doi AS doi
        FROM gold_study_publications gsp
        WHERE gsp.doi IS NOT NULL AND gsp.doi != '' AND gsp.gold_study_id IS NOT NULL
        UNION
        SELECT DISTINCT sp.gold_study_id AS gold_study_id, gsp.doi AS doi
        FROM gold_study_publications gsp
        JOIN gold_sequencing_projects sp ON sp.gold_project_id = gsp.gold_project_id
        WHERE gsp.doi IS NOT NULL AND gsp.doi != '' AND sp.gold_study_id IS NOT NULL;
        """
    )

    rows = db.conn.execute(
        """
        SELECT c.gold_study_id AS gold_study_id, c.doi AS doi, f.fanout AS fanout
        FROM _gold_study_doi_candidates c
        LEFT JOIN _gold_doi_bioproject_fanout f ON f.doi = c.doi
        ORDER BY c.gold_study_id
        """
    ).fetchall()

    candidates_by_study: dict[str, list[tuple[str, int | None]]] = {}
    for row in rows:
        candidates_by_study.setdefault(row["gold_study_id"], []).append((row["doi"], row["fanout"]))

    counts = {
        "gold_studies_with_candidates": len(candidates_by_study),
        "resolved": 0,
        "resolved_ambiguous": 0,
        "likely_reanalysis_only": 0,
    }
    updates: list[tuple[str | None, str, str, int | None, str]] = []
    for gold_study_id, candidates in candidates_by_study.items():
        in_range = sorted(
            ((doi, fanout) for doi, fanout in candidates if fanout is not None and fanout <= low_fanout_threshold),
            key=lambda pair: (pair[1], pair[0]),
        )
        if not in_range:
            counts["likely_reanalysis_only"] += 1
            updates.append(
                (
                    None,
                    ResolutionStatus.LIKELY_REANALYSIS_ONLY.value,
                    f"{len(candidates)} candidate DOI(s), none with a bioproject_fanout <= {low_fanout_threshold}",
                    None,
                    gold_study_id,
                )
            )
            continue
        best_fanout = in_range[0][1]
        tied = [doi for doi, fanout in in_range if fanout == best_fanout]
        ambiguous = len(tied) > 1
        primary_doi = tied[0]
        if ambiguous:
            counts["resolved_ambiguous"] += 1
            status = ResolutionStatus.RESOLVED_AMBIGUOUS
            reason = f"{len(tied)} candidates tied at bioproject_fanout={best_fanout} of {len(candidates)} total; picked lexicographically first DOI"
        else:
            counts["resolved"] += 1
            status = ResolutionStatus.RESOLVED
            reason = f"lowest bioproject_fanout ({best_fanout}) among {len(candidates)} candidate DOI(s)"
        updates.append((primary_doi, status.value, reason, best_fanout, gold_study_id))

    if apply:
        db.conn.executemany(
            """
            UPDATE gold_studies
            SET primary_doi = ?, primary_doi_status = ?, primary_doi_selection_reason = ?,
                primary_doi_bioproject_fanout = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%S', 'now')
            WHERE gold_study_id = ?
            """,
            updates,
        )
        db.conn.commit()

    return counts


def ingest_gold_workbooks(
    db: SeedDiscoveryDB,
    raw_dir: Path,
    snapshot: str,
    *,
    progress_path: Path | None = None,
    workbook_filter: str | None = None,
    sheet_filter: str | None = None,
    start_row: int = 2,
    max_rows: int | None = None,
    store_source_rows: bool = True,
) -> dict[str, int]:
    counts: dict[str, int] = {
        "gold_source_rows": 0,
        "gold_studies": 0,
        "gold_biosamples": 0,
        "gold_sequencing_projects": 0,
        "gold_analysis_projects": 0,
        "gold_faire_enrichment": 0,
        "gold_study_publications": 0,
        "gold_project_jgi_files": 0,
        "gold_rows_scanned": 0,
        "gold_rows_skipped_not_priority": 0,
    }
    now = utc_iso()
    pending_writes = 0
    for workbook_path in sorted(raw_dir.glob("*.xlsx")):
        if workbook_filter and workbook_path.name != workbook_filter:
            continue
        for sheet_name, rows in iter_xlsx_sheets(workbook_path):
            if sheet_filter and sheet_name != sheet_filter:
                continue
            header = next(rows, None)
            if not header:
                continue
            columns = [str(value).strip() if value is not None else "" for value in header]
            entity_type = infer_entity_type(workbook_path.name, sheet_name, columns)
            if entity_type not in INGEST_ENTITY_TYPES:
                continue
            logger.info("GOLD ingest starting workbook=%s sheet=%s entity=%s", workbook_path.name, sheet_name, entity_type)
            write_ingest_progress(progress_path, counts, workbook_path.name, sheet_name, entity_type, "started")
            sheet_inserted = 0
            last_row_number = None
            scanned_in_shard = 0
            progress_interval = shard_progress_interval(max_rows)
            for row_number, values in enumerate(rows, start=2):
                if row_number < start_row:
                    continue
                if max_rows is not None and scanned_in_shard >= max_rows:
                    logger.info(
                        "GOLD ingest shard limit reached workbook=%s sheet=%s start_row=%s max_rows=%s",
                        workbook_path.name,
                        sheet_name,
                        start_row,
                        max_rows,
                    )
                    break
                scanned_in_shard += 1
                last_row_number = row_number
                counts["gold_rows_scanned"] += 1
                if scanned_in_shard % progress_interval == 0:
                    logger.info(
                        "GOLD ingest progress workbook=%s sheet=%s shard_scanned=%s total_scanned=%s inserted_rows=%s skipped_not_priority=%s row_number=%s",
                        workbook_path.name,
                        sheet_name,
                        scanned_in_shard,
                        counts["gold_rows_scanned"],
                        sheet_inserted,
                        counts["gold_rows_skipped_not_priority"],
                        row_number,
                    )
                    write_ingest_progress(progress_path, counts, workbook_path.name, sheet_name, entity_type, "running", row_number=row_number)
                row = _row_dict(columns, values)
                if not any(value not in (None, "") for value in row.values()):
                    continue
                if not should_ingest_gold_row(row, entity_type, workbook_path.name):
                    counts["gold_rows_skipped_not_priority"] += 1
                    continue
                entity_id = entity_id_for_row(row, entity_type)
                source_json = json.dumps(row, sort_keys=True, default=str)
                if store_source_rows:
                    db.conn.execute(
                        """
                        INSERT INTO gold_source_rows(
                          snapshot_date, workbook_name, sheet_name, row_number, entity_type, source_entity_id,
                          source_metadata_json, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(snapshot_date, workbook_name, sheet_name, row_number) DO UPDATE SET
                          entity_type=excluded.entity_type,
                          source_entity_id=excluded.source_entity_id,
                          source_metadata_json=excluded.source_metadata_json,
                          updated_at=excluded.updated_at
                        """,
                        (snapshot, workbook_path.name, sheet_name, row_number, entity_type, entity_id, source_json, now, now),
                    )
                    counts["gold_source_rows"] += 1
                pending_writes += 1
                sheet_inserted += 1
                if entity_type == "study":
                    counts["gold_studies"] += upsert_gold_study(db, row, source_json, snapshot, now)
                    counts["gold_study_publications"] += upsert_gold_publication(
                        db,
                        row,
                        source_json,
                        snapshot,
                        now,
                        gold_study_id=pick(row, "gold_study_id") or entity_id_for_row(row, "study"),
                        gold_project_id=None,
                    )
                elif entity_type == "biosample":
                    counts["gold_biosamples"] += upsert_gold_biosample(db, row, source_json, snapshot, now)
                    counts["gold_faire_enrichment"] += upsert_gold_faire_enrichment(db, row, source_json, snapshot, now)
                elif entity_type == "sequencing_project":
                    counts["gold_sequencing_projects"] += upsert_gold_project(db, row, source_json, snapshot, now)
                    project_id = pick(row, "gold_project_id") or entity_id_for_row(row, "sequencing_project")
                    counts["gold_study_publications"] += upsert_gold_publication(
                        db,
                        row,
                        source_json,
                        snapshot,
                        now,
                        gold_study_id=pick(row, "gold_study_id_ref"),
                        gold_project_id=project_id,
                    )
                    counts["gold_project_jgi_files"] += upsert_gold_jgi_file_placeholder(
                        db,
                        row,
                        source_json,
                        snapshot,
                        now,
                        gold_project_id=project_id,
                    )
                elif entity_type == "analysis_project":
                    counts["gold_analysis_projects"] += upsert_gold_analysis_project(db, row, source_json, snapshot, now)
                if pending_writes >= INGEST_COMMIT_INTERVAL:
                    db.conn.commit()
                    pending_writes = 0
            db.conn.commit()
            pending_writes = 0
            logger.info(
                "GOLD ingest finished workbook=%s sheet=%s entity=%s inserted_source_rows=%s",
                workbook_path.name,
                sheet_name,
                entity_type,
                sheet_inserted,
            )
            write_ingest_progress(progress_path, counts, workbook_path.name, sheet_name, entity_type, "finished", row_number=last_row_number)
    db.conn.commit()
    write_ingest_progress(progress_path, counts, None, None, None, "complete")
    return counts


def shard_progress_interval(max_rows: int | None) -> int:
    if max_rows is None:
        return INGEST_PROGRESS_INTERVAL
    if max_rows <= 10_000:
        return 1_000
    if max_rows <= 50_000:
        return 5_000
    return INGEST_PROGRESS_INTERVAL


def write_ingest_progress(
    path: Path | None,
    counts: dict[str, int],
    workbook_name: str | None,
    sheet_name: str | None,
    entity_type: str | None,
    status: str,
    *,
    row_number: int | None = None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": utcnow().isoformat(),
        "status": status,
        "workbook_name": workbook_name,
        "sheet_name": sheet_name,
        "entity_type": entity_type,
        "row_number": row_number,
        "counts": counts,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def should_ingest_gold_row(row: dict, entity_type: str, workbook_name: str) -> bool:
    if entity_type == "study":
        return True
    marine_confidence, _methods = classify_marine(row)
    if marine_confidence != "low":
        return True
    if row_is_environmental_priority(row):
        return True
    if entity_type == "biosample":
        return False
    if workbook_name == "public_sra_biome_img_annotations.xlsx":
        return True
    strategy = pick(row, "sequencing_strategy") or pick(row, "analysis_project_type") or ""
    return any(term in strategy.casefold() for term in ENVIRONMENTAL_STRATEGY_TERMS)


def iter_xlsx_sheets(workbook_path: Path):
    with ZipFile(workbook_path) as archive:
        shared_strings = load_shared_strings(archive)
        for sheet_name, sheet_path in xlsx_sheet_paths(archive):
            yield sheet_name, iter_xlsx_rows(archive, sheet_path, shared_strings)


def xlsx_sheet_paths(archive: ZipFile) -> list[tuple[str, str]]:
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    office_rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    package_rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_root.findall(f"{package_rel_ns}Relationship")
        if "Id" in rel.attrib and "Target" in rel.attrib
    }
    paths: list[tuple[str, str]] = []
    for sheet in workbook_root.findall(f"{main_ns}sheets/{main_ns}sheet"):
        rel_id = sheet.attrib.get(f"{office_rel_ns}id")
        if not rel_id or rel_id not in rel_targets:
            continue
        target = rel_targets[rel_id]
        path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        paths.append((sheet.attrib.get("name", rel_id), path))
    return paths


def load_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    strings: list[str] = []
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with archive.open("xl/sharedStrings.xml") as handle:
        for _event, element in ET.iterparse(handle, events=("end",)):
            if element.tag == f"{main_ns}si":
                strings.append("".join(text.text or "" for text in element.iter(f"{main_ns}t")))
                element.clear()
    return strings


def iter_xlsx_rows(archive: ZipFile, sheet_path: str, shared_strings: list[str]):
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with archive.open(sheet_path) as handle:
        for _event, element in ET.iterparse(handle, events=("end",)):
            if element.tag != f"{main_ns}row":
                continue
            values: dict[int, str | None] = {}
            max_index = -1
            for cell in element.findall(f"{main_ns}c"):
                index = cell_index(cell.attrib.get("r", ""))
                max_index = max(max_index, index)
                values[index] = cell_value(cell, shared_strings)
            yield [values.get(index) for index in range(max_index + 1)] if max_index >= 0 else []
            element.clear()


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(f"{main_ns}t")) or None
    value = cell.find(f"{main_ns}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        try:
            return shared_strings[int(value.text)]
        except (IndexError, ValueError):
            return value.text
    return value.text


def cell_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def upsert_gold_study(db: SeedDiscoveryDB, row: dict, source_json: str, snapshot: str, now: str) -> int:
    gold_id = pick(row, "gold_study_id") or entity_id_for_row(row, "study")
    if not gold_id:
        return 0
    marine_confidence, methods = classify_marine(row)
    db.conn.execute(
        """
        INSERT INTO gold_studies(
          gold_study_id, study_name, study_description, marine_confidence, marine_match_methods,
          primary_bioproject, primary_doi, source_snapshot_date, source_metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(gold_study_id) DO UPDATE SET
          study_name=COALESCE(excluded.study_name, gold_studies.study_name),
          study_description=COALESCE(excluded.study_description, gold_studies.study_description),
          marine_confidence=excluded.marine_confidence,
          marine_match_methods=excluded.marine_match_methods,
          primary_bioproject=COALESCE(excluded.primary_bioproject, gold_studies.primary_bioproject),
          primary_doi=COALESCE(excluded.primary_doi, gold_studies.primary_doi),
          source_snapshot_date=excluded.source_snapshot_date,
          source_metadata_json=excluded.source_metadata_json,
          updated_at=excluded.updated_at
        """,
        (
            gold_id,
            pick(row, "study_name"),
            pick(row, "study_description"),
            marine_confidence,
            methods,
            pick(row, "ncbi_bioproject"),
            find_doi(row),
            snapshot,
            source_json,
            now,
            now,
        ),
    )
    return 1


def upsert_gold_biosample(db: SeedDiscoveryDB, row: dict, source_json: str, snapshot: str, now: str) -> int:
    gold_id = pick(row, "gold_biosample_id") or entity_id_for_row(row, "biosample")
    if not gold_id:
        return 0
    marine_confidence, methods = classify_marine(row)
    db.conn.execute(
        """
        INSERT INTO gold_biosamples(
          gold_biosample_id, gold_study_id, ncbi_biosample_accession, biosample_name, collection_date,
          latitude, longitude, depth, ecosystem, ecosystem_category, ecosystem_type, ecosystem_subtype,
          specific_ecosystem, env_broad_scale, env_local_scale, env_medium, marine_confidence,
          marine_match_methods, source_snapshot_date, source_metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(gold_biosample_id) DO UPDATE SET
          gold_study_id=COALESCE(excluded.gold_study_id, gold_biosamples.gold_study_id),
          ncbi_biosample_accession=COALESCE(excluded.ncbi_biosample_accession, gold_biosamples.ncbi_biosample_accession),
          biosample_name=COALESCE(excluded.biosample_name, gold_biosamples.biosample_name),
          collection_date=COALESCE(excluded.collection_date, gold_biosamples.collection_date),
          latitude=COALESCE(excluded.latitude, gold_biosamples.latitude),
          longitude=COALESCE(excluded.longitude, gold_biosamples.longitude),
          depth=COALESCE(excluded.depth, gold_biosamples.depth),
          marine_confidence=excluded.marine_confidence,
          marine_match_methods=excluded.marine_match_methods,
          source_snapshot_date=excluded.source_snapshot_date,
          source_metadata_json=excluded.source_metadata_json,
          updated_at=excluded.updated_at
        """,
        (
            gold_id,
            pick(row, "gold_study_id_ref"),
            pick(row, "ncbi_biosample"),
            pick(row, "biosample_name"),
            pick(row, "collection_date"),
            pick(row, "latitude"),
            pick(row, "longitude"),
            pick(row, "depth"),
            pick(row, "ecosystem"),
            pick(row, "ecosystem_category"),
            pick(row, "ecosystem_type"),
            pick(row, "ecosystem_subtype"),
            pick(row, "specific_ecosystem"),
            pick(row, "env_broad_scale"),
            pick(row, "env_local_scale"),
            pick(row, "env_medium"),
            marine_confidence,
            methods,
            snapshot,
            source_json,
            now,
            now,
        ),
    )
    return 1


def upsert_gold_project(db: SeedDiscoveryDB, row: dict, source_json: str, snapshot: str, now: str) -> int:
    gold_id = pick(row, "gold_project_id") or entity_id_for_row(row, "sequencing_project")
    if not gold_id:
        return 0
    db.conn.execute(
        """
        INSERT INTO gold_sequencing_projects(
          gold_project_id, gold_study_id, gold_biosample_id, ncbi_bioproject_accession,
          ncbi_biosample_accession, sequencing_strategy, project_status, jgi_project_id,
          source_snapshot_date, source_metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(gold_project_id) DO UPDATE SET
          gold_study_id=COALESCE(excluded.gold_study_id, gold_sequencing_projects.gold_study_id),
          gold_biosample_id=COALESCE(excluded.gold_biosample_id, gold_sequencing_projects.gold_biosample_id),
          ncbi_bioproject_accession=COALESCE(excluded.ncbi_bioproject_accession, gold_sequencing_projects.ncbi_bioproject_accession),
          ncbi_biosample_accession=COALESCE(excluded.ncbi_biosample_accession, gold_sequencing_projects.ncbi_biosample_accession),
          sequencing_strategy=COALESCE(excluded.sequencing_strategy, gold_sequencing_projects.sequencing_strategy),
          project_status=COALESCE(excluded.project_status, gold_sequencing_projects.project_status),
          jgi_project_id=COALESCE(excluded.jgi_project_id, gold_sequencing_projects.jgi_project_id),
          source_snapshot_date=excluded.source_snapshot_date,
          source_metadata_json=excluded.source_metadata_json,
          updated_at=excluded.updated_at
        """,
        (
            gold_id,
            pick(row, "gold_study_id_ref"),
            pick(row, "gold_biosample_id_ref"),
            pick(row, "ncbi_bioproject"),
            pick(row, "ncbi_biosample"),
            pick(row, "sequencing_strategy"),
            pick(row, "project_status"),
            pick(row, "jgi_project_id"),
            snapshot,
            source_json,
            now,
            now,
        ),
    )
    return 1


def upsert_gold_analysis_project(db: SeedDiscoveryDB, row: dict, source_json: str, snapshot: str, now: str) -> int:
    gold_id = pick(row, "gold_analysis_project_id") or entity_id_for_row(row, "analysis_project")
    if not gold_id:
        return 0
    db.conn.execute(
        """
        INSERT INTO gold_analysis_projects(
          gold_analysis_project_id, gold_project_id, gold_biosample_id, gold_study_id,
          analysis_project_type, img_identifier, source_snapshot_date, source_metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(gold_analysis_project_id) DO UPDATE SET
          gold_project_id=COALESCE(excluded.gold_project_id, gold_analysis_projects.gold_project_id),
          gold_biosample_id=COALESCE(excluded.gold_biosample_id, gold_analysis_projects.gold_biosample_id),
          gold_study_id=COALESCE(excluded.gold_study_id, gold_analysis_projects.gold_study_id),
          analysis_project_type=COALESCE(excluded.analysis_project_type, gold_analysis_projects.analysis_project_type),
          img_identifier=COALESCE(excluded.img_identifier, gold_analysis_projects.img_identifier),
          source_snapshot_date=excluded.source_snapshot_date,
          source_metadata_json=excluded.source_metadata_json,
          updated_at=excluded.updated_at
        """,
        (
            gold_id,
            pick(row, "gold_project_id"),
            pick(row, "gold_biosample_id_ref"),
            pick(row, "gold_study_id_ref"),
            pick(row, "analysis_project_type"),
            pick(row, "img_identifier"),
            snapshot,
            source_json,
            now,
            now,
        ),
    )
    return 1


def upsert_gold_faire_enrichment(db: SeedDiscoveryDB, row: dict, source_json: str, snapshot: str, now: str) -> int:
    gold_id = pick(row, "gold_biosample_id") or entity_id_for_row(row, "biosample")
    if not gold_id:
        return 0
    provenance = {
        "source": "GOLD",
        "gold_biosample_id": gold_id,
        "snapshot_date": snapshot,
        "source_metadata_json": json.loads(source_json),
    }
    db.conn.execute(
        """
        INSERT INTO gold_faire_enrichment(
          canonical_biosample, gold_biosample_id, gold_study_id, ncbi_bioproject,
          decimalLatitude, decimalLongitude, eventDate, depth, geo_loc_name,
          env_broad_scale, env_local_scale, env_medium, sample_collection_method, size_fraction,
          temperature, salinity, ph, oxygen, chlorophyll, source, source_snapshot_date,
          provenance_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'GOLD', ?, ?, ?, ?)
        """,
        (
            pick(row, "ncbi_biosample") or gold_id,
            gold_id,
            pick(row, "gold_study_id_ref"),
            pick(row, "ncbi_bioproject"),
            pick(row, "latitude"),
            pick(row, "longitude"),
            pick(row, "collection_date"),
            pick(row, "depth"),
            pick(row, "geo_loc_name"),
            pick(row, "env_broad_scale"),
            pick(row, "env_local_scale"),
            pick(row, "env_medium"),
            pick(row, "sample_collection_method"),
            pick(row, "size_fraction"),
            pick(row, "temperature"),
            pick(row, "salinity"),
            pick(row, "ph"),
            pick(row, "oxygen"),
            pick(row, "chlorophyll"),
            snapshot,
            json.dumps(provenance, sort_keys=True),
            now,
            now,
        ),
    )
    return 1


def upsert_gold_publication(
    db: SeedDiscoveryDB,
    row: dict,
    source_json: str,
    snapshot: str,
    now: str,
    *,
    gold_study_id: str | None,
    gold_project_id: str | None,
) -> int:
    doi = find_doi(row)
    pmid = pick(row, "pmid") or find_identifier(row, r"\bPMID[:\s]*([0-9]{5,})\b")
    pmcid = pick(row, "pmcid") or find_identifier(row, r"\b(PMC[0-9]{5,})\b")
    title = pick(row, "publication_title")
    matched_identifier = doi or pmid or pmcid or title
    if not matched_identifier:
        return 0
    existing = db.conn.execute(
        """
        SELECT id FROM gold_study_publications
        WHERE COALESCE(gold_study_id, '') = ?
          AND COALESCE(gold_project_id, '') = ?
          AND COALESCE(doi, '') = ?
          AND COALESCE(pmid, '') = ?
          AND COALESCE(pmcid, '') = ?
          AND COALESCE(title, '') = ?
          AND COALESCE(source_snapshot_date, '') = ?
        """,
        (gold_study_id or "", gold_project_id or "", doi or "", pmid or "", pmcid or "", title or "", snapshot),
    ).fetchone()
    if existing:
        db.conn.execute("UPDATE gold_study_publications SET raw_json = ?, updated_at = ? WHERE id = ?", (source_json, now, existing["id"]))
        return 0
    db.conn.execute(
        """
        INSERT INTO gold_study_publications(
          gold_study_id, gold_project_id, doi, pmid, pmcid, title, match_method,
          matched_identifier, match_confidence, match_score, is_primary, source_snapshot_date,
          raw_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            gold_study_id,
            gold_project_id,
            doi,
            pmid,
            pmcid,
            title,
            "gold_bulk_publication_field",
            matched_identifier,
            "medium" if title and not (doi or pmid or pmcid) else "high",
            1.0 if doi or pmid or pmcid else 0.5,
            1 if doi or pmid or pmcid else 0,
            snapshot,
            source_json,
            now,
            now,
        ),
    )
    return 1


def upsert_gold_jgi_file_placeholder(
    db: SeedDiscoveryDB,
    row: dict,
    source_json: str,
    snapshot: str,
    now: str,
    *,
    gold_project_id: str | None,
) -> int:
    jgi_project_id = pick(row, "jgi_project_id")
    if not gold_project_id and not jgi_project_id:
        return 0
    existing = db.conn.execute(
        """
        SELECT id FROM gold_project_jgi_files
        WHERE COALESCE(gold_project_id, '') = ?
          AND COALESCE(jgi_project_id, '') = ?
          AND COALESCE(source_snapshot_date, '') = ?
          AND availability_status = 'metadata_only_auth_required_for_file_listing'
        """,
        (gold_project_id or "", jgi_project_id or "", snapshot),
    ).fetchone()
    if existing:
        db.conn.execute("UPDATE gold_project_jgi_files SET raw_json = ?, updated_at = ? WHERE id = ?", (source_json, now, existing["id"]))
        return 0
    db.conn.execute(
        """
        INSERT INTO gold_project_jgi_files(
          gold_project_id, jgi_project_id, availability_status, source_snapshot_date,
          raw_json, created_at, updated_at
        )
        VALUES (?, ?, 'metadata_only_auth_required_for_file_listing', ?, ?, ?, ?)
        """,
        (gold_project_id, jgi_project_id, snapshot, source_json, now, now),
    )
    return 1


def infer_entity_type(workbook_name: str, sheet_name: str, columns: Iterable[str]) -> str:
    sheet_haystack = f"{sheet_name} {' '.join(columns)}".casefold()
    workbook_haystack = workbook_name.casefold()
    normalized_columns = {normalize_header(column) for column in columns}
    if "ap gold id" in normalized_columns:
        return "analysis_project"
    if "project gold id" in normalized_columns:
        return "sequencing_project"
    if "biosample gold id" in normalized_columns:
        return "biosample"
    if "study gold id" in normalized_columns and len(normalized_columns) <= 8:
        return "study"
    if "analysis" in sheet_haystack and "project" in sheet_haystack:
        return "analysis_project"
    if "sequencing" in sheet_haystack and "project" in sheet_haystack:
        return "sequencing_project"
    if "biosample" in sheet_haystack or "bio sample" in sheet_haystack:
        return "biosample"
    if re.search(r"\bstudy\b|\bstudies\b", sheet_haystack):
        return "study"
    if "organism" in sheet_haystack:
        return "organism"
    if "ecosystem" in sheet_haystack:
        return "ecosystem_path"
    if "cv" in workbook_haystack:
        return "controlled_vocabulary"
    if "sra" in workbook_haystack and "biome" in workbook_haystack:
        return "analysis_project"
    return "unknown"


def classify_marine(row: dict) -> tuple[str, str | None]:
    structured_keys = (
        "ecosystem",
        "ecosystem_category",
        "ecosystem_type",
        "ecosystem_subtype",
        "specific_ecosystem",
        "env_broad_scale",
        "env_local_scale",
        "env_medium",
        "habitat",
        "geo_loc_name",
    )
    matches = []
    for key in structured_keys:
        value = pick(row, key)
        if value and any(term in value.casefold() for term in MARINE_TERMS):
            matches.append(f"{key}:{value}")
    if matches:
        return "high", "|".join(matches[:10])
    text = " ".join(str(value) for value in row.values() if value is not None).casefold()
    if any(term in text for term in MARINE_TERMS):
        return "medium", "free_text_marine_term"
    return "low", None


def row_is_environmental_priority(row: dict) -> bool:
    text = " ".join(str(value) for value in row.values() if value is not None).casefold()
    return any(term in text for term in ENVIRONMENTAL_STRATEGY_TERMS)


def pick(row: dict, logical_field: str) -> str | None:
    headers = {normalize_header(key): key for key in row}
    for candidate in FIELD_CANDIDATES.get(logical_field, (logical_field,)):
        key = headers.get(normalize_header(candidate))
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return str(value).strip()
    if logical_field == "ncbi_bioproject":
        return find_accession(row, r"\bPRJ(?:NA|EB|DB)\d+\b")
    if logical_field == "ncbi_biosample":
        return find_accession(row, r"\b(?:SAMN|SAMEA|SAMD)\d+\b")
    return None


def find_accession(row: dict, pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for value in row.values():
        if value is None:
            continue
        match = regex.search(str(value))
        if match:
            return match.group(0).upper()
    return None


def find_doi(row: dict) -> str | None:
    regex = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
    for key, value in row.items():
        if value is None:
            continue
        if "doi" in key.casefold() or "publication" in key.casefold() or "paper" in key.casefold():
            match = regex.search(str(value))
            if match:
                return match.group(0).rstrip(".,;")
    return None


def find_identifier(row: dict, pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for value in row.values():
        if value is None:
            continue
        match = regex.search(str(value))
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return None


def entity_id_for_row(row: dict, entity_type: str) -> str | None:
    prefixes = {
        "study": ("Gs", "GOLD Study"),
        "biosample": ("Gb", "GOLD Biosample"),
        "sequencing_project": ("Gp", "GOLD Project"),
        "analysis_project": ("Ga", "GOLD Analysis"),
    }
    direct = {
        "study": pick(row, "gold_study_id"),
        "biosample": pick(row, "gold_biosample_id"),
        "sequencing_project": pick(row, "gold_project_id"),
        "analysis_project": pick(row, "gold_analysis_project_id"),
    }.get(entity_type)
    if direct:
        return direct
    regex = re.compile(r"\bG[absmp]\d+\b", re.IGNORECASE)
    for value in row.values():
        if value is None:
            continue
        match = regex.search(str(value))
        if match:
            return match.group(0)
    return None


def _row_dict(columns: list[str], values: tuple) -> dict:
    out = {}
    for idx, column in enumerate(columns):
        if not column:
            continue
        value = values[idx] if idx < len(values) else None
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        out[column] = value
    return out


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def write_faire_mapping_candidates(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_field", "possible_faire_field", "mapping_confidence", "notes"])
        writer.writerows(FAIRE_MAPPING_ROWS)


def write_jgi_file_manifest(db: SeedDiscoveryDB, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gold_project_id",
                "jgi_project_id",
                "file_id",
                "filename",
                "file_type",
                "size_bytes",
                "checksum",
                "availability_status",
                "download_locator",
            ]
        )
        for row in db.conn.execute(
            """
            SELECT gold_project_id, jgi_project_id, file_id, filename, file_type, size_bytes,
                   checksum, availability_status, download_locator
            FROM gold_project_jgi_files
            ORDER BY gold_project_id, jgi_project_id, filename
            """
        ):
            writer.writerow(
                [
                    row["gold_project_id"],
                    row["jgi_project_id"],
                    row["file_id"],
                    row["filename"],
                    row["file_type"],
                    row["size_bytes"],
                    row["checksum"],
                    row["availability_status"],
                    row["download_locator"],
                ]
            )


def write_reports(db: SeedDiscoveryDB, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    known_bioproject_sql = """
        SELECT DISTINCT bioproject_accession AS bioproject FROM ena_studies WHERE bioproject_accession IS NOT NULL
        UNION
        SELECT DISTINCT bioproject_accession FROM mgnify_studies WHERE bioproject_accession IS NOT NULL
    """
    with (reports_dir / "new_studies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_project_id", "ncbi_bioproject_accession", "gold_study_id", "sequencing_strategy"])
        for row in db.conn.execute(
            f"""
            SELECT gold_project_id, ncbi_bioproject_accession, gold_study_id, sequencing_strategy
            FROM gold_sequencing_projects
            WHERE ncbi_bioproject_accession IS NOT NULL
              AND ncbi_bioproject_accession NOT IN ({known_bioproject_sql})
            ORDER BY gold_project_id
            """
        ):
            writer.writerow([row["gold_project_id"], row["ncbi_bioproject_accession"], row["gold_study_id"], row["sequencing_strategy"]])
    with (reports_dir / "enrichment_existing_studies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ncbi_bioproject_accession", "gold_project_id", "gold_study_id", "gold_biosample_id", "gold_fields_present"])
        for row in db.conn.execute(
            f"""
            SELECT p.ncbi_bioproject_accession, p.gold_project_id, p.gold_study_id, b.gold_biosample_id,
                   b.latitude, b.longitude, b.depth, b.env_broad_scale, b.env_local_scale, b.env_medium
            FROM gold_sequencing_projects p
            LEFT JOIN gold_biosamples b ON b.gold_biosample_id = p.gold_biosample_id
            WHERE p.ncbi_bioproject_accession IN ({known_bioproject_sql})
            ORDER BY p.ncbi_bioproject_accession
            """
        ):
            fields = [key for key in ("latitude", "longitude", "depth", "env_broad_scale", "env_local_scale", "env_medium") if row[key]]
            writer.writerow([row["ncbi_bioproject_accession"], row["gold_project_id"], row["gold_study_id"], row["gold_biosample_id"], "|".join(fields)])
    write_metadata_completeness(db, reports_dir / "metadata_completeness.csv")


def write_metadata_completeness(db: SeedDiscoveryDB, path: Path) -> None:
    metrics = {
        "biosamples": "count(*)",
        "with_ncbi_biosample": "sum(ncbi_biosample_accession IS NOT NULL)",
        "with_collection_date": "sum(collection_date IS NOT NULL)",
        "with_lat_lon": "sum(latitude IS NOT NULL AND longitude IS NOT NULL)",
        "with_depth": "sum(depth IS NOT NULL)",
        "with_env_broad_scale": "sum(env_broad_scale IS NOT NULL)",
        "with_env_local_scale": "sum(env_local_scale IS NOT NULL)",
        "with_env_medium": "sum(env_medium IS NOT NULL)",
    }
    row = db.conn.execute(f"SELECT {', '.join(f'{expr} AS {name}' for name, expr in metrics.items())} FROM gold_biosamples").fetchone()
    total = int(row["biosamples"] or 0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "count", "percent"])
        for metric in metrics:
            count = int(row[metric] or 0)
            pct = round((count / total) * 100, 2) if total else 0.0
            writer.writerow([metric, count, pct])


def print_gold_report(locations: dict) -> None:
    counts = locations.get("counts", {})
    print("=" * 60)
    print("JGI GOLD INGEST COMPLETE")
    print("=" * 60)
    print("\nRAW GOLD DOWNLOADS")
    print(locations["raw_download_dir"])
    print("\nRAW DOWNLOAD MANIFEST")
    print(locations["manifest"])
    print("\nGOLD SCHEMA INVENTORY")
    print(locations["schema_inventory"])
    print("\nNORMALIZED GOLD DATABASE")
    print(f"SQLite database: {locations['database']}")
    print("Tables:")
    for table in locations["tables"].values():
        print(f"  {table}")
    print("\nFAIRe FIELD MAPPING CANDIDATES")
    print(locations["reports"]["faire_mapping_candidates"])
    print("\nJGI FILE MANIFESTS")
    print(locations["jgi_file_manifest_dir"])
    print("\nREPORTS")
    for path in locations["reports"].values():
        print(path)
    print("\nCOUNTS")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    print("=" * 60)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_date_for_mode(page_text: str, key: str) -> str | None:
    label = {
        "public_studies_biosamples_sps_aps_organisms": "Public Studies",
        "public_sra_biome_img_annotations": "Public SRA",
        "gold_cvs": "GOLD CVs",
        "ecosystem_paths": "Ecosystem",
    }.get(key, "")
    if not label:
        return None
    match = re.search(re.escape(label) + r".{0,200}?Last generated:\s*([0-9]{1,2}\s+\w+,?\s+[0-9]{4})", page_text, re.IGNORECASE)
    return match.group(1) if match else None
