from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi
from fair_ocean_agent.seed_discovery.models import EnaRun, EnaStudy, MatchConfidence, MgnifyStudy, PublicationCandidate, ResolutionStatus


def utc_iso() -> str:
    return utcnow().isoformat()


def normalize_doi_for_seed(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip().strip(".,;")
    try:
        return normalize_doi(value)
    except IdentifierError:
        return None


class SeedDiscoveryDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 60000")
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS mgnify_studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mgnify_accession TEXT NOT NULL UNIQUE,
                bioproject_accession TEXT,
                secondary_study_accession TEXT,
                study_name TEXT,
                study_abstract TEXT,
                centre_name TEXT,
                public_release_date TEXT,
                sample_count INTEGER,
                biome TEXT,
                experiment_types TEXT,
                mgnify_last_updated TEXT,
                raw_json TEXT,
                discovery_status TEXT NOT NULL DEFAULT 'accepted',
                publication_resolution_status TEXT NOT NULL DEFAULT 'not_yet_processed',
                first_seen_at TEXT NOT NULL,
                last_checked_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS publication_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mgnify_study_id INTEGER REFERENCES mgnify_studies(id) ON DELETE CASCADE,
                ena_study_id INTEGER REFERENCES ena_studies(id) ON DELETE CASCADE,
                doi TEXT,
                normalized_doi TEXT,
                pmid TEXT,
                pmcid TEXT,
                openalex_id TEXT,
                title TEXT,
                title_key TEXT,
                publication_date TEXT,
                publication_year INTEGER,
                publication_type TEXT,
                match_method TEXT NOT NULL,
                matched_identifier TEXT,
                match_confidence TEXT NOT NULL,
                match_score REAL NOT NULL DEFAULT 0,
                raw_json TEXT,
                is_primary INTEGER NOT NULL DEFAULT 0,
                primary_selection_reason TEXT,
                first_seen_at TEXT NOT NULL,
                last_verified_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(mgnify_study_id, normalized_doi),
                UNIQUE(mgnify_study_id, pmid),
                UNIQUE(mgnify_study_id, pmcid),
                UNIQUE(mgnify_study_id, openalex_id),
                UNIQUE(mgnify_study_id, title_key, publication_year),
                UNIQUE(ena_study_id, normalized_doi),
                UNIQUE(ena_study_id, pmid),
                UNIQUE(ena_study_id, pmcid),
                UNIQUE(ena_study_id, openalex_id),
                UNIQUE(ena_study_id, title_key, publication_year)
            );

            CREATE TABLE IF NOT EXISTS ena_studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_dataset_id TEXT NOT NULL UNIQUE,
                ena_study_accession TEXT,
                secondary_study_accession TEXT,
                bioproject_accession TEXT,
                bioproject_resolution_method TEXT,
                bioproject_status TEXT NOT NULL DEFAULT 'unresolved',
                ncbi_bioproject_verified INTEGER NOT NULL DEFAULT 0,
                study_title TEXT,
                project_name TEXT,
                centre_name TEXT,
                first_public TEXT,
                marine_confidence TEXT,
                marine_match_methods TEXT,
                marine_tags TEXT,
                sample_count INTEGER NOT NULL DEFAULT 0,
                run_count INTEGER NOT NULL DEFAULT 0,
                downloadable_run_count INTEGER NOT NULL DEFAULT 0,
                fastq_run_count INTEGER NOT NULL DEFAULT 0,
                fastq_bytes_total INTEGER NOT NULL DEFAULT 0,
                sequence_accessibility_status TEXT NOT NULL DEFAULT 'no_downloadable_reads',
                metadata_completeness_json TEXT,
                metadata_usefulness_score INTEGER NOT NULL DEFAULT 0,
                publication_resolution_status TEXT NOT NULL DEFAULT 'not_yet_processed',
                raw_json TEXT,
                first_seen_at TEXT NOT NULL,
                last_checked_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ena_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_accession TEXT NOT NULL UNIQUE,
                experiment_accession TEXT,
                sample_accession TEXT,
                secondary_sample_accession TEXT,
                study_accession TEXT,
                secondary_study_accession TEXT,
                bioproject_accession TEXT,
                submission_accession TEXT,
                study_title TEXT,
                project_name TEXT,
                centre_name TEXT,
                first_public TEXT,
                fastq_ftp TEXT,
                fastq_md5 TEXT,
                fastq_bytes TEXT,
                submitted_ftp TEXT,
                submitted_md5 TEXT,
                submitted_bytes TEXT,
                submitted_format TEXT,
                sra_ftp TEXT,
                sra_md5 TEXT,
                sra_bytes TEXT,
                library_strategy TEXT,
                library_source TEXT,
                library_selection TEXT,
                library_layout TEXT,
                instrument_platform TEXT,
                instrument_model TEXT,
                target_gene TEXT,
                collection_date TEXT,
                lat TEXT,
                lon TEXT,
                depth TEXT,
                country TEXT,
                marine_region TEXT,
                environment_biome TEXT,
                environment_feature TEXT,
                environment_material TEXT,
                sample_collection TEXT,
                extraction_protocol TEXT,
                library_construction_protocol TEXT,
                marine_tag TEXT,
                marine_confidence TEXT,
                marine_match_methods TEXT,
                sequence_accessibility_status TEXT NOT NULL DEFAULT 'no_downloadable_reads',
                raw_json TEXT,
                first_seen_at TEXT NOT NULL,
                last_checked_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_cache (
                request_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                response_json TEXT NOT NULL,
                status_code INTEGER,
                fetched_at TEXT NOT NULL,
                expires_at TEXT
            );

            CREATE TABLE IF NOT EXISTS crawl_state (
                source TEXT PRIMARY KEY,
                cursor TEXT,
                last_successful_request TEXT,
                last_run_started TEXT,
                last_run_completed TEXT,
                status TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS epmc_accession_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                database_name TEXT NOT NULL,
                accession TEXT NOT NULL,
                normalized_accession TEXT NOT NULL,
                pmcid TEXT,
                article_source TEXT,
                article_external_id TEXT,
                snapshot_date TEXT NOT NULL,
                source_file TEXT NOT NULL,
                active_in_latest_snapshot INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS uq_epmc_accession_links_identity
              ON epmc_accession_links(
                database_name,
                normalized_accession,
                COALESCE(pmcid, ''),
                COALESCE(article_source, ''),
                COALESCE(article_external_id, '')
              );
            CREATE INDEX IF NOT EXISTS idx_epmc_accession_links_norm
              ON epmc_accession_links(normalized_accession);
            CREATE INDEX IF NOT EXISTS idx_epmc_accession_links_db_norm
              ON epmc_accession_links(database_name, normalized_accession);
            CREATE INDEX IF NOT EXISTS idx_epmc_accession_links_pmcid
              ON epmc_accession_links(pmcid);
            CREATE INDEX IF NOT EXISTS idx_epmc_accession_links_article
              ON epmc_accession_links(article_source, article_external_id);

            CREATE TABLE IF NOT EXISTS epmc_article_ids (
                pmid TEXT PRIMARY KEY,
                pmcid TEXT,
                doi TEXT,
                normalized_doi TEXT,
                snapshot_date TEXT NOT NULL,
                source_file TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_epmc_article_ids_pmcid
              ON epmc_article_ids(pmcid);
            CREATE INDEX IF NOT EXISTS idx_epmc_article_ids_norm_doi
              ON epmc_article_ids(normalized_doi);

            CREATE TABLE IF NOT EXISTS gold_source_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                workbook_name TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                source_entity_id TEXT,
                source_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(snapshot_date, workbook_name, sheet_name, row_number)
            );

            CREATE INDEX IF NOT EXISTS idx_gold_source_rows_entity
              ON gold_source_rows(entity_type, source_entity_id);

            CREATE TABLE IF NOT EXISTS gold_studies (
                gold_study_id TEXT PRIMARY KEY,
                study_name TEXT,
                study_description TEXT,
                marine_confidence TEXT,
                marine_match_methods TEXT,
                primary_bioproject TEXT,
                primary_doi TEXT,
                primary_doi_status TEXT,
                primary_doi_selection_reason TEXT,
                primary_doi_bioproject_fanout INTEGER,
                source_snapshot_date TEXT,
                source_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gold_biosamples (
                gold_biosample_id TEXT PRIMARY KEY,
                gold_study_id TEXT,
                ncbi_biosample_accession TEXT,
                biosample_name TEXT,
                collection_date TEXT,
                latitude TEXT,
                longitude TEXT,
                depth TEXT,
                ecosystem TEXT,
                ecosystem_category TEXT,
                ecosystem_type TEXT,
                ecosystem_subtype TEXT,
                specific_ecosystem TEXT,
                env_broad_scale TEXT,
                env_local_scale TEXT,
                env_medium TEXT,
                marine_confidence TEXT,
                marine_match_methods TEXT,
                source_snapshot_date TEXT,
                source_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gold_biosamples_ncbi
              ON gold_biosamples(ncbi_biosample_accession);
            CREATE INDEX IF NOT EXISTS idx_gold_biosamples_study
              ON gold_biosamples(gold_study_id);

            CREATE TABLE IF NOT EXISTS gold_sequencing_projects (
                gold_project_id TEXT PRIMARY KEY,
                gold_study_id TEXT,
                gold_biosample_id TEXT,
                ncbi_bioproject_accession TEXT,
                ncbi_biosample_accession TEXT,
                sequencing_strategy TEXT,
                project_status TEXT,
                jgi_project_id TEXT,
                source_snapshot_date TEXT,
                source_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gold_projects_bioproject
              ON gold_sequencing_projects(ncbi_bioproject_accession);
            CREATE INDEX IF NOT EXISTS idx_gold_projects_biosample
              ON gold_sequencing_projects(ncbi_biosample_accession);

            CREATE TABLE IF NOT EXISTS gold_analysis_projects (
                gold_analysis_project_id TEXT PRIMARY KEY,
                gold_project_id TEXT,
                gold_biosample_id TEXT,
                gold_study_id TEXT,
                analysis_project_type TEXT,
                img_identifier TEXT,
                source_snapshot_date TEXT,
                source_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gold_study_publications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gold_study_id TEXT,
                gold_project_id TEXT,
                doi TEXT,
                pmid TEXT,
                pmcid TEXT,
                title TEXT,
                match_method TEXT NOT NULL,
                matched_identifier TEXT,
                match_confidence TEXT,
                match_score REAL DEFAULT 0,
                is_primary INTEGER NOT NULL DEFAULT 0,
                source_snapshot_date TEXT,
                raw_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

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

            CREATE TABLE IF NOT EXISTS gold_bioproject_publication_search (
                bioproject_accession TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                candidates_found INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                checked_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gold_project_jgi_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gold_project_id TEXT,
                jgi_project_id TEXT,
                file_id TEXT,
                filename TEXT,
                file_type TEXT,
                size_bytes INTEGER,
                checksum TEXT,
                availability_status TEXT,
                download_locator TEXT,
                source_snapshot_date TEXT,
                raw_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gold_jgi_files_placeholder
              ON gold_project_jgi_files(
                COALESCE(gold_project_id, ''),
                COALESCE(jgi_project_id, ''),
                COALESCE(source_snapshot_date, ''),
                availability_status
              );
            CREATE INDEX IF NOT EXISTS idx_gold_jgi_files_project
              ON gold_project_jgi_files(gold_project_id, jgi_project_id);

            CREATE TABLE IF NOT EXISTS gold_faire_enrichment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_biosample TEXT,
                gold_biosample_id TEXT,
                gold_study_id TEXT,
                ncbi_bioproject TEXT,
                decimalLatitude TEXT,
                decimalLongitude TEXT,
                eventDate TEXT,
                depth TEXT,
                geo_loc_name TEXT,
                env_broad_scale TEXT,
                env_local_scale TEXT,
                env_medium TEXT,
                sample_collection_method TEXT,
                size_fraction TEXT,
                temperature TEXT,
                salinity TEXT,
                ph TEXT,
                oxygen TEXT,
                chlorophyll TEXT,
                source TEXT NOT NULL DEFAULT 'GOLD',
                source_snapshot_date TEXT,
                provenance_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_gold_faire_biosample
              ON gold_faire_enrichment(canonical_biosample);

            """
        )
        self._ensure_publication_candidates_schema()
        self._ensure_gold_studies_primary_doi_columns()
        self._create_paper_seeds_view()
        self.conn.commit()

    def _ensure_publication_candidates_schema(self) -> None:
        columns = {str(row["name"]): row for row in self.conn.execute("PRAGMA table_info(publication_candidates)").fetchall()}
        if "ena_study_id" in columns and int(columns["mgnify_study_id"]["notnull"]) == 0:
            return
        self.conn.executescript(
            """
            DROP VIEW IF EXISTS paper_seeds;
            ALTER TABLE publication_candidates RENAME TO publication_candidates_old;
            CREATE TABLE publication_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mgnify_study_id INTEGER REFERENCES mgnify_studies(id) ON DELETE CASCADE,
                ena_study_id INTEGER REFERENCES ena_studies(id) ON DELETE CASCADE,
                doi TEXT,
                normalized_doi TEXT,
                pmid TEXT,
                pmcid TEXT,
                openalex_id TEXT,
                title TEXT,
                title_key TEXT,
                publication_date TEXT,
                publication_year INTEGER,
                publication_type TEXT,
                match_method TEXT NOT NULL,
                matched_identifier TEXT,
                match_confidence TEXT NOT NULL,
                match_score REAL NOT NULL DEFAULT 0,
                raw_json TEXT,
                is_primary INTEGER NOT NULL DEFAULT 0,
                primary_selection_reason TEXT,
                first_seen_at TEXT NOT NULL,
                last_verified_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(mgnify_study_id, normalized_doi),
                UNIQUE(mgnify_study_id, pmid),
                UNIQUE(mgnify_study_id, pmcid),
                UNIQUE(mgnify_study_id, openalex_id),
                UNIQUE(mgnify_study_id, title_key, publication_year),
                UNIQUE(ena_study_id, normalized_doi),
                UNIQUE(ena_study_id, pmid),
                UNIQUE(ena_study_id, pmcid),
                UNIQUE(ena_study_id, openalex_id),
                UNIQUE(ena_study_id, title_key, publication_year)
            );
            INSERT INTO publication_candidates(
                id, mgnify_study_id, ena_study_id, doi, normalized_doi, pmid, pmcid, openalex_id,
                title, title_key, publication_date, publication_year, publication_type,
                match_method, matched_identifier, match_confidence, match_score, raw_json,
                is_primary, primary_selection_reason, first_seen_at, last_verified_at, created_at, updated_at
            )
            SELECT
                id, mgnify_study_id, NULL, doi, normalized_doi, pmid, pmcid, openalex_id,
                title, title_key, publication_date, publication_year, publication_type,
                match_method, matched_identifier, match_confidence, match_score, raw_json,
                is_primary, primary_selection_reason, first_seen_at, last_verified_at, created_at, updated_at
            FROM publication_candidates_old;
            DROP TABLE publication_candidates_old;
            """
        )

    def _ensure_gold_studies_primary_doi_columns(self) -> None:
        """CREATE TABLE IF NOT EXISTS gold_studies above only creates these
        columns on a brand-new database -- an existing gold_studies table
        (every GOLD sqlite snapshot on disk as of this writing) predates
        them and needs a plain ADD COLUMN. Unlike
        _ensure_publication_candidates_schema, these are new nullable
        columns with no UNIQUE constraint involved, so a rename-and-rebuild
        isn't needed -- ALTER TABLE ADD COLUMN is safe and instant even on
        a multi-GB table."""
        columns = {str(row["name"]) for row in self.conn.execute("PRAGMA table_info(gold_studies)").fetchall()}
        for column, ddl_type in (
            ("primary_doi_status", "TEXT"),
            ("primary_doi_selection_reason", "TEXT"),
            ("primary_doi_bioproject_fanout", "INTEGER"),
        ):
            if column not in columns:
                self.conn.execute(f"ALTER TABLE gold_studies ADD COLUMN {column} {ddl_type}")

    def _create_paper_seeds_view(self) -> None:
        self.conn.executescript(
            """
            DROP VIEW IF EXISTS paper_seeds;
            CREATE VIEW paper_seeds AS
            SELECT
                'mgnify' AS seed_source,
                COALESCE(s.bioproject_accession, s.secondary_study_accession, s.mgnify_accession) AS canonical_dataset_id,
                s.mgnify_accession,
                s.bioproject_accession,
                CASE WHEN s.bioproject_accession IS NOT NULL THEN 'resolved' ELSE 'unresolved' END AS bioproject_status,
                NULL AS ena_study_accession,
                s.secondary_study_accession,
                s.study_name AS study_title,
                s.biome,
                s.experiment_types,
                s.sample_count,
                NULL AS run_count,
                NULL AS downloadable_run_count,
                NULL AS sequence_accessibility_status,
                NULL AS marine_confidence,
                NULL AS metadata_completeness_json,
                p.normalized_doi AS primary_doi,
                p.pmid AS primary_pmid,
                p.pmcid AS primary_pmcid,
                p.openalex_id AS primary_openalex_id,
                p.title AS primary_paper_title,
                p.publication_date AS primary_publication_date,
                p.match_method AS publication_match_method,
                p.match_confidence AS publication_match_confidence,
                (
                    SELECT group_concat(pc.normalized_doi, '|')
                    FROM publication_candidates pc
                    WHERE pc.mgnify_study_id = s.id
                      AND pc.normalized_doi IS NOT NULL
                      AND pc.is_primary = 0
                ) AS alternate_dois,
                (
                    SELECT count(*)
                    FROM publication_candidates pc
                    WHERE pc.mgnify_study_id = s.id
                ) AS publication_candidate_count,
                s.publication_resolution_status,
                CASE
                    WHEN p.normalized_doi IS NOT NULL AND p.match_confidence IN ('very_high', 'high') THEN 'complete'
                    WHEN s.publication_resolution_status = 'openalex_no_resolve_reprocess' THEN 'sequence_data_found_no_paper'
                    ELSE s.publication_resolution_status
                END AS seed_status,
                s.first_seen_at,
                s.last_checked_at
            FROM mgnify_studies s
            LEFT JOIN publication_candidates p
              ON p.mgnify_study_id = s.id AND p.is_primary = 1
            UNION ALL
            SELECT
                'ena' AS seed_source,
                s.canonical_dataset_id,
                NULL AS mgnify_accession,
                s.bioproject_accession,
                s.bioproject_status,
                s.ena_study_accession,
                s.secondary_study_accession,
                s.study_title,
                NULL AS biome,
                NULL AS experiment_types,
                s.sample_count,
                s.run_count,
                s.downloadable_run_count,
                s.sequence_accessibility_status,
                s.marine_confidence,
                s.metadata_completeness_json,
                p.normalized_doi AS primary_doi,
                p.pmid AS primary_pmid,
                p.pmcid AS primary_pmcid,
                p.openalex_id AS primary_openalex_id,
                p.title AS primary_paper_title,
                p.publication_date AS primary_publication_date,
                p.match_method AS publication_match_method,
                p.match_confidence AS publication_match_confidence,
                (
                    SELECT group_concat(pc.normalized_doi, '|')
                    FROM publication_candidates pc
                    WHERE pc.ena_study_id = s.id
                      AND pc.normalized_doi IS NOT NULL
                      AND pc.is_primary = 0
                ) AS alternate_dois,
                (
                    SELECT count(*)
                    FROM publication_candidates pc
                    WHERE pc.ena_study_id = s.id
                ) AS publication_candidate_count,
                s.publication_resolution_status,
                CASE
                    WHEN s.downloadable_run_count > 0
                      AND p.normalized_doi IS NOT NULL
                      AND p.match_confidence IN ('very_high', 'high')
                    THEN 'complete'
                    WHEN s.publication_resolution_status = 'publication_candidates_low_confidence' THEN 'publication_candidates_low_confidence'
                    ELSE 'sequence_data_found_no_paper'
                END AS seed_status,
                s.first_seen_at,
                s.last_checked_at
            FROM ena_studies s
            LEFT JOIN publication_candidates p
              ON p.ena_study_id = s.id AND p.is_primary = 1;
            """
        )

    def mark_run_started(self, source: str, cursor: str | None = None) -> None:
        now = utc_iso()
        self.conn.execute(
            """
            INSERT INTO crawl_state(source, cursor, last_run_started, status, updated_at)
            VALUES (?, ?, ?, 'running', ?)
            ON CONFLICT(source) DO UPDATE SET
              cursor=COALESCE(excluded.cursor, crawl_state.cursor),
              last_run_started=excluded.last_run_started,
              status='running',
              error=NULL,
              updated_at=excluded.updated_at
            """,
            (source, cursor, now, now),
        )
        self.conn.commit()

    def update_crawl_state(self, source: str, *, cursor: str | None = None, request: str | None = None,
                           status: str | None = None, error: str | None = None, completed: bool = False) -> None:
        now = utc_iso()
        row = self.conn.execute("SELECT source FROM crawl_state WHERE source = ?", (source,)).fetchone()
        if row is None:
            self.mark_run_started(source, cursor)
        self.conn.execute(
            """
            UPDATE crawl_state
            SET cursor=COALESCE(?, cursor),
                last_successful_request=COALESCE(?, last_successful_request),
                last_run_completed=CASE WHEN ? THEN ? ELSE last_run_completed END,
                status=COALESCE(?, status),
                error=?,
                updated_at=?
            WHERE source=?
            """,
            (cursor, request, 1 if completed else 0, now, status, error, now, source),
        )
        self.conn.commit()

    def crawl_cursor(self, source: str) -> str | None:
        row = self.conn.execute("SELECT cursor FROM crawl_state WHERE source = ?", (source,)).fetchone()
        return str(row["cursor"]) if row and row["cursor"] is not None else None

    def crawl_status(self, source: str) -> str | None:
        row = self.conn.execute("SELECT status FROM crawl_state WHERE source = ?", (source,)).fetchone()
        return str(row["status"]) if row and row["status"] is not None else None

    def crawl_updated_at(self, source: str) -> str | None:
        row = self.conn.execute("SELECT updated_at FROM crawl_state WHERE source = ?", (source,)).fetchone()
        return str(row["updated_at"]) if row and row["updated_at"] is not None else None

    def get_cache(self, request_key: str) -> dict | list | None:
        row = self.conn.execute("SELECT response_json FROM api_cache WHERE request_key = ?", (request_key,)).fetchone()
        if row is None:
            return None
        return json.loads(row["response_json"])

    def set_cache(self, request_key: str, source: str, payload: dict | list, status_code: int | None = None) -> None:
        self.conn.execute(
            """
            INSERT INTO api_cache(request_key, source, response_json, status_code, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(request_key) DO UPDATE SET
              response_json=excluded.response_json,
              status_code=excluded.status_code,
              fetched_at=excluded.fetched_at
            """,
            (request_key, source, json.dumps(payload, sort_keys=True), status_code, utc_iso()),
        )
        self.conn.commit()

    def upsert_epmc_accession_links(self, rows: Iterable[dict]) -> int:
        now = utc_iso()
        values = [
            (
                row["database_name"],
                row["accession"],
                row["normalized_accession"],
                row.get("pmcid"),
                row.get("article_source"),
                row.get("article_external_id"),
                row["snapshot_date"],
                row["source_file"],
                now,
                now,
                now,
                now,
            )
            for row in rows
        ]
        if not values:
            return 0
        self.conn.executemany(
            """
            INSERT INTO epmc_accession_links(
                database_name, accession, normalized_accession, pmcid, article_source, article_external_id,
                snapshot_date, source_file, active_in_latest_snapshot, first_seen_at, last_seen_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(database_name, normalized_accession, COALESCE(pmcid, ''), COALESCE(article_source, ''), COALESCE(article_external_id, ''))
            DO UPDATE SET
              accession=excluded.accession,
              pmcid=COALESCE(excluded.pmcid, epmc_accession_links.pmcid),
              article_source=COALESCE(excluded.article_source, epmc_accession_links.article_source),
              article_external_id=COALESCE(excluded.article_external_id, epmc_accession_links.article_external_id),
              snapshot_date=excluded.snapshot_date,
              source_file=excluded.source_file,
              active_in_latest_snapshot=1,
              last_seen_at=excluded.last_seen_at,
              updated_at=excluded.updated_at
            """,
            values,
        )
        self.conn.commit()
        return len(values)

    def upsert_epmc_article_ids(self, rows: Iterable[dict]) -> int:
        now = utc_iso()
        values = [
            (
                row["pmid"],
                row.get("pmcid"),
                row.get("doi"),
                normalize_doi_for_seed(row.get("doi")),
                row["snapshot_date"],
                row.get("source_file"),
                now,
                now,
            )
            for row in rows
            if row.get("pmid")
        ]
        if not values:
            return 0
        self.conn.executemany(
            """
            INSERT INTO epmc_article_ids(pmid, pmcid, doi, normalized_doi, snapshot_date, source_file, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
              pmcid=COALESCE(excluded.pmcid, epmc_article_ids.pmcid),
              doi=COALESCE(excluded.doi, epmc_article_ids.doi),
              normalized_doi=COALESCE(excluded.normalized_doi, epmc_article_ids.normalized_doi),
              snapshot_date=excluded.snapshot_date,
              source_file=excluded.source_file,
              updated_at=excluded.updated_at
            """,
            values,
        )
        self.conn.commit()
        return len(values)

    def epmc_links_for_accessions(self, normalized_accessions: Iterable[str]) -> list[sqlite3.Row]:
        values = sorted({value for value in normalized_accessions if value})
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        return list(
            self.conn.execute(
                f"""
                SELECT
                  l.*,
                  ids.pmid AS mapped_pmid,
                  ids.pmcid AS mapped_pmcid,
                  ids.doi AS mapped_doi,
                  ids.normalized_doi AS mapped_normalized_doi
                FROM epmc_accession_links l
                LEFT JOIN epmc_article_ids ids
                  ON (l.article_source = 'MED' AND ids.pmid = l.article_external_id)
                  OR (l.pmcid IS NOT NULL AND ids.pmcid = l.pmcid)
                WHERE l.normalized_accession IN ({placeholders})
                  AND l.active_in_latest_snapshot = 1
                """,
                values,
            )
        )

    def clear_api_cache(self) -> None:
        self.conn.execute("DELETE FROM api_cache")
        self.conn.commit()

    def upsert_study(self, study: MgnifyStudy) -> int:
        now = utc_iso()
        self.conn.execute(
            """
            INSERT INTO mgnify_studies(
                mgnify_accession, bioproject_accession, secondary_study_accession, study_name, study_abstract,
                centre_name, public_release_date, sample_count, biome, experiment_types, mgnify_last_updated,
                raw_json, discovery_status, publication_resolution_status, first_seen_at, last_checked_at,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', 'not_yet_processed', ?, ?, ?, ?)
            ON CONFLICT(mgnify_accession) DO UPDATE SET
              bioproject_accession=excluded.bioproject_accession,
              secondary_study_accession=excluded.secondary_study_accession,
              study_name=excluded.study_name,
              study_abstract=excluded.study_abstract,
              centre_name=excluded.centre_name,
              public_release_date=excluded.public_release_date,
              sample_count=excluded.sample_count,
              biome=excluded.biome,
              experiment_types=excluded.experiment_types,
              mgnify_last_updated=excluded.mgnify_last_updated,
              raw_json=excluded.raw_json,
              last_checked_at=excluded.last_checked_at,
              updated_at=excluded.updated_at
            """,
            (
                study.mgnify_accession,
                study.bioproject_accession,
                study.secondary_study_accession,
                study.study_name,
                study.study_abstract,
                study.centre_name,
                study.public_release_date,
                study.sample_count,
                study.biome,
                study.experiment_types,
                study.mgnify_last_updated,
                study.raw_json,
                now,
                now,
                now,
                now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM mgnify_studies WHERE mgnify_accession = ?", (study.mgnify_accession,)).fetchone()
        return int(row["id"])

    def studies_for_resolution(self, *, refresh: bool = False, limit: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM mgnify_studies"
        params: list[object] = []
        if not refresh:
            sql += " WHERE publication_resolution_status IN (?, ?, ?)"
            params.extend([
                ResolutionStatus.NOT_YET_PROCESSED.value,
                ResolutionStatus.API_ERROR.value,
                ResolutionStatus.OPENALEX_REPROCESS.value,
            ])
        sql += """
        ORDER BY
          CASE publication_resolution_status
            WHEN 'not_yet_processed' THEN 0
            WHEN 'api_error' THEN 1
            WHEN 'openalex_no_resolve_reprocess' THEN 2
            WHEN 'no_publication_link_found' THEN 3
            WHEN 'publication_candidates_low_confidence' THEN 4
            ELSE 4
          END,
          COALESCE(last_checked_at, ''),
          id
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return list(self.conn.execute(sql, params))

    def upsert_ena_run(self, run: EnaRun) -> None:
        now = utc_iso()
        values = (
            run.run_accession,
            run.experiment_accession,
            run.sample_accession,
            run.secondary_sample_accession,
            run.study_accession,
            run.secondary_study_accession,
            run.bioproject_accession,
            run.submission_accession,
            run.study_title,
            run.project_name,
            run.centre_name,
            run.first_public,
            run.fastq_ftp,
            run.fastq_md5,
            run.fastq_bytes,
            run.submitted_ftp,
            run.submitted_md5,
            run.submitted_bytes,
            run.submitted_format,
            run.sra_ftp,
            run.sra_md5,
            run.sra_bytes,
            run.library_strategy,
            run.library_source,
            run.library_selection,
            run.library_layout,
            run.instrument_platform,
            run.instrument_model,
            run.target_gene,
            run.collection_date,
            run.lat,
            run.lon,
            run.depth,
            run.country,
            run.marine_region,
            run.environment_biome,
            run.environment_feature,
            run.environment_material,
            run.sample_collection,
            run.extraction_protocol,
            run.library_construction_protocol,
            run.marine_tag,
            run.marine_confidence,
            run.marine_match_methods,
            run.sequence_accessibility_status,
            run.raw_json,
            now,
            now,
            now,
            now,
        )
        self.conn.execute(
            """
            INSERT INTO ena_runs(
                run_accession, experiment_accession, sample_accession, secondary_sample_accession,
                study_accession, secondary_study_accession, bioproject_accession, submission_accession,
                study_title, project_name, centre_name, first_public,
                fastq_ftp, fastq_md5, fastq_bytes, submitted_ftp, submitted_md5, submitted_bytes,
                submitted_format, sra_ftp, sra_md5, sra_bytes,
                library_strategy, library_source, library_selection, library_layout,
                instrument_platform, instrument_model, target_gene,
                collection_date, lat, lon, depth, country, marine_region,
                environment_biome, environment_feature, environment_material, sample_collection,
                extraction_protocol, library_construction_protocol,
                marine_tag, marine_confidence, marine_match_methods, sequence_accessibility_status,
                raw_json, first_seen_at, last_checked_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_accession) DO UPDATE SET
              experiment_accession=excluded.experiment_accession,
              sample_accession=excluded.sample_accession,
              secondary_sample_accession=excluded.secondary_sample_accession,
              study_accession=excluded.study_accession,
              secondary_study_accession=excluded.secondary_study_accession,
              bioproject_accession=excluded.bioproject_accession,
              submission_accession=excluded.submission_accession,
              study_title=excluded.study_title,
              project_name=excluded.project_name,
              centre_name=excluded.centre_name,
              first_public=excluded.first_public,
              fastq_ftp=excluded.fastq_ftp,
              fastq_md5=excluded.fastq_md5,
              fastq_bytes=excluded.fastq_bytes,
              submitted_ftp=excluded.submitted_ftp,
              submitted_md5=excluded.submitted_md5,
              submitted_bytes=excluded.submitted_bytes,
              submitted_format=excluded.submitted_format,
              sra_ftp=excluded.sra_ftp,
              sra_md5=excluded.sra_md5,
              sra_bytes=excluded.sra_bytes,
              library_strategy=excluded.library_strategy,
              library_source=excluded.library_source,
              library_selection=excluded.library_selection,
              library_layout=excluded.library_layout,
              instrument_platform=excluded.instrument_platform,
              instrument_model=excluded.instrument_model,
              target_gene=excluded.target_gene,
              collection_date=excluded.collection_date,
              lat=excluded.lat,
              lon=excluded.lon,
              depth=excluded.depth,
              country=excluded.country,
              marine_region=excluded.marine_region,
              environment_biome=excluded.environment_biome,
              environment_feature=excluded.environment_feature,
              environment_material=excluded.environment_material,
              sample_collection=excluded.sample_collection,
              extraction_protocol=excluded.extraction_protocol,
              library_construction_protocol=excluded.library_construction_protocol,
              marine_tag=excluded.marine_tag,
              marine_confidence=excluded.marine_confidence,
              marine_match_methods=excluded.marine_match_methods,
              sequence_accessibility_status=excluded.sequence_accessibility_status,
              raw_json=excluded.raw_json,
              last_checked_at=excluded.last_checked_at,
              updated_at=excluded.updated_at
            """,
            values,
        )
        self.conn.commit()

    def ena_run_groups(self) -> dict[str, list[sqlite3.Row]]:
        rows = self.conn.execute(
            """
            SELECT * FROM ena_runs
            WHERE COALESCE(study_accession, secondary_study_accession, bioproject_accession) IS NOT NULL
            ORDER BY COALESCE(bioproject_accession, secondary_study_accession, study_accession), run_accession
            """
        ).fetchall()
        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            key = row["bioproject_accession"] or row["secondary_study_accession"] or row["study_accession"]
            groups.setdefault(str(key), []).append(row)
        return groups

    def ena_accession_family_for_study(self, study_row: sqlite3.Row, *, sample_limit: int = 500, run_limit: int = 1000) -> dict[str, list[str]]:
        study_values = {
            str(value)
            for value in (
                study_row["bioproject_accession"],
                study_row["secondary_study_accession"],
                study_row["ena_study_accession"],
                study_row["canonical_dataset_id"],
            )
            if value
        }
        rows = self.conn.execute(
            """
            SELECT sample_accession, secondary_sample_accession, experiment_accession, run_accession
            FROM ena_runs
            WHERE bioproject_accession = ?
               OR secondary_study_accession = ?
               OR study_accession = ?
               OR study_accession = ?
            ORDER BY run_accession
            """,
            (
                study_row["bioproject_accession"],
                study_row["secondary_study_accession"],
                study_row["ena_study_accession"],
                study_row["canonical_dataset_id"],
            ),
        ).fetchall()
        biosamples: list[str] = []
        experiments: list[str] = []
        runs: list[str] = []
        seen_biosamples: set[str] = set()
        seen_experiments: set[str] = set()
        seen_runs: set[str] = set()
        for row in rows:
            for value in (row["sample_accession"], row["secondary_sample_accession"]):
                if value and value not in seen_biosamples and len(biosamples) < sample_limit:
                    seen_biosamples.add(str(value))
                    biosamples.append(str(value))
            value = row["experiment_accession"]
            if value and value not in seen_experiments and len(experiments) < run_limit:
                seen_experiments.add(str(value))
                experiments.append(str(value))
            value = row["run_accession"]
            if value and value not in seen_runs and len(runs) < run_limit:
                seen_runs.add(str(value))
                runs.append(str(value))
        return {
            "study_accessions": sorted(study_values),
            "biosamples": biosamples,
            "experiments": experiments,
            "runs": runs,
        }

    def upsert_ena_study(self, study: EnaStudy) -> int:
        now = utc_iso()
        self.conn.execute(
            """
            INSERT INTO ena_studies(
                canonical_dataset_id, ena_study_accession, secondary_study_accession,
                bioproject_accession, bioproject_resolution_method, bioproject_status,
                ncbi_bioproject_verified, study_title, project_name, centre_name, first_public,
                marine_confidence, marine_match_methods, marine_tags,
                sample_count, run_count, downloadable_run_count, fastq_run_count, fastq_bytes_total,
                sequence_accessibility_status, metadata_completeness_json, metadata_usefulness_score,
                publication_resolution_status, raw_json, first_seen_at, last_checked_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_dataset_id) DO UPDATE SET
              ena_study_accession=excluded.ena_study_accession,
              secondary_study_accession=excluded.secondary_study_accession,
              bioproject_accession=excluded.bioproject_accession,
              bioproject_resolution_method=excluded.bioproject_resolution_method,
              bioproject_status=excluded.bioproject_status,
              ncbi_bioproject_verified=excluded.ncbi_bioproject_verified,
              study_title=excluded.study_title,
              project_name=excluded.project_name,
              centre_name=excluded.centre_name,
              first_public=excluded.first_public,
              marine_confidence=excluded.marine_confidence,
              marine_match_methods=excluded.marine_match_methods,
              marine_tags=excluded.marine_tags,
              sample_count=excluded.sample_count,
              run_count=excluded.run_count,
              downloadable_run_count=excluded.downloadable_run_count,
              fastq_run_count=excluded.fastq_run_count,
              fastq_bytes_total=excluded.fastq_bytes_total,
              sequence_accessibility_status=excluded.sequence_accessibility_status,
              metadata_completeness_json=excluded.metadata_completeness_json,
              metadata_usefulness_score=excluded.metadata_usefulness_score,
              raw_json=excluded.raw_json,
              last_checked_at=excluded.last_checked_at,
              updated_at=excluded.updated_at
            """,
            (
                study.canonical_dataset_id,
                study.ena_study_accession,
                study.secondary_study_accession,
                study.bioproject_accession,
                study.bioproject_resolution_method,
                study.bioproject_status,
                1 if study.ncbi_bioproject_verified else 0,
                study.study_title,
                study.project_name,
                study.centre_name,
                study.first_public,
                study.marine_confidence,
                study.marine_match_methods,
                study.marine_tags,
                study.sample_count,
                study.run_count,
                study.downloadable_run_count,
                study.fastq_run_count,
                study.fastq_bytes_total,
                study.sequence_accessibility_status,
                study.metadata_completeness_json,
                study.metadata_usefulness_score,
                study.publication_resolution_status if isinstance(study.publication_resolution_status, str) else study.publication_resolution_status.value,
                study.raw_json,
                now,
                now,
                now,
                now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM ena_studies WHERE canonical_dataset_id = ?", (study.canonical_dataset_id,)).fetchone()
        return int(row["id"])

    def ena_studies_for_resolution(self, *, refresh: bool = False, limit: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM ena_studies"
        params: list[object] = []
        if not refresh:
            sql += " WHERE publication_resolution_status IN (?, ?, ?, ?)"
            params.extend([
                ResolutionStatus.NOT_YET_PROCESSED.value,
                ResolutionStatus.API_ERROR.value,
                ResolutionStatus.OPENALEX_REPROCESS.value,
                ResolutionStatus.LOW_CONFIDENCE.value,
            ])
        sql += """
        ORDER BY
          CASE publication_resolution_status
            WHEN 'not_yet_processed' THEN 0
            WHEN 'api_error' THEN 1
            WHEN 'openalex_no_resolve_reprocess' THEN 2
            WHEN 'publication_candidates_low_confidence' THEN 3
            ELSE 4
          END,
          COALESCE(last_checked_at, ''),
          id
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return list(self.conn.execute(sql, params))

    def upsert_publication_candidate(self, study_id: int, candidate: PublicationCandidate) -> None:
        self._upsert_publication_candidate(mgnify_study_id=study_id, ena_study_id=None, candidate=candidate)

    def upsert_ena_publication_candidate(self, ena_study_id: int, candidate: PublicationCandidate) -> None:
        self._upsert_publication_candidate(mgnify_study_id=None, ena_study_id=ena_study_id, candidate=candidate)

    def _upsert_publication_candidate(
        self,
        *,
        mgnify_study_id: int | None,
        ena_study_id: int | None,
        candidate: PublicationCandidate,
    ) -> None:
        now = utc_iso()
        normalized_doi = normalize_doi_for_seed(candidate.doi)
        title_key = title_dedupe_key(candidate.title)
        values = (
            mgnify_study_id,
            ena_study_id,
            candidate.doi,
            normalized_doi,
            candidate.pmid,
            candidate.pmcid,
            candidate.openalex_id,
            candidate.title,
            title_key,
            candidate.publication_date,
            candidate.publication_year,
            candidate.publication_type,
            candidate.match_method,
            candidate.matched_identifier,
            candidate.match_confidence.value if isinstance(candidate.match_confidence, MatchConfidence) else str(candidate.match_confidence),
            candidate.match_score,
            candidate.raw_json,
            now,
            now,
            now,
            now,
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO publication_candidates(
                mgnify_study_id, ena_study_id, doi, normalized_doi, pmid, pmcid, openalex_id, title, title_key,
                publication_date, publication_year, publication_type, match_method, matched_identifier,
                match_confidence, match_score, raw_json, first_seen_at, last_verified_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        self.conn.execute(
            """
            UPDATE publication_candidates
            SET doi=COALESCE(?, doi),
                normalized_doi=COALESCE(?, normalized_doi),
                pmid=COALESCE(?, pmid),
                pmcid=COALESCE(?, pmcid),
                openalex_id=COALESCE(?, openalex_id),
                title=COALESCE(?, title),
                publication_date=COALESCE(?, publication_date),
                publication_year=COALESCE(?, publication_year),
                publication_type=COALESCE(?, publication_type),
                match_score=MAX(match_score, ?),
                raw_json=COALESCE(?, raw_json),
                last_verified_at=?,
                updated_at=?
            WHERE COALESCE(mgnify_study_id, -1)=COALESCE(?, -1)
              AND COALESCE(ena_study_id, -1)=COALESCE(?, -1)
              AND (
                (normalized_doi IS NOT NULL AND normalized_doi = ?)
                OR (pmid IS NOT NULL AND pmid = ?)
                OR (pmcid IS NOT NULL AND pmcid = ?)
                OR (openalex_id IS NOT NULL AND openalex_id = ?)
                OR (title_key IS NOT NULL AND title_key = ? AND COALESCE(publication_year, -1) = COALESCE(?, -1))
              )
            """,
            (
                candidate.doi,
                normalized_doi,
                candidate.pmid,
                candidate.pmcid,
                candidate.openalex_id,
                candidate.title,
                candidate.publication_date,
                candidate.publication_year,
                candidate.publication_type,
                candidate.match_score,
                candidate.raw_json,
                now,
                now,
                mgnify_study_id,
                ena_study_id,
                normalized_doi,
                candidate.pmid,
                candidate.pmcid,
                candidate.openalex_id,
                title_key,
                candidate.publication_year,
            ),
        )
        self.conn.commit()

    def candidates_for_study(self, study_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM publication_candidates WHERE mgnify_study_id = ?", (study_id,)))

    def candidates_for_ena_study(self, ena_study_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM publication_candidates WHERE ena_study_id = ?", (ena_study_id,)))

    def set_primary(self, study_id: int, candidate_id: int | None, status: ResolutionStatus, reason: str | None = None) -> None:
        now = utc_iso()
        self.conn.execute("UPDATE publication_candidates SET is_primary = 0, primary_selection_reason = NULL WHERE mgnify_study_id = ?", (study_id,))
        if candidate_id is not None:
            self.conn.execute(
                "UPDATE publication_candidates SET is_primary = 1, primary_selection_reason = ? WHERE id = ?",
                (reason, candidate_id),
            )
        self.conn.execute(
            "UPDATE mgnify_studies SET publication_resolution_status = ?, last_checked_at = ?, updated_at = ? WHERE id = ?",
            (status.value, now, now, study_id),
        )
        self.conn.commit()

    def set_ena_primary(self, ena_study_id: int, candidate_id: int | None, status: ResolutionStatus, reason: str | None = None) -> None:
        now = utc_iso()
        self.conn.execute("UPDATE publication_candidates SET is_primary = 0, primary_selection_reason = NULL WHERE ena_study_id = ?", (ena_study_id,))
        if candidate_id is not None:
            self.conn.execute(
                "UPDATE publication_candidates SET is_primary = 1, primary_selection_reason = ? WHERE id = ?",
                (reason, candidate_id),
            )
        self.conn.execute(
            "UPDATE ena_studies SET publication_resolution_status = ?, last_checked_at = ?, updated_at = ? WHERE id = ?",
            (status.value, now, now, ena_study_id),
        )
        self.conn.commit()

    def count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
        return int(row["n"])


def title_dedupe_key(title: str | None) -> str | None:
    if not title:
        return None
    import re

    key = re.sub(r"\W+", " ", title.casefold()).strip()
    return key or None


_CONFIDENCE_RANK = {
    MatchConfidence.VERY_HIGH.value: 4,
    MatchConfidence.HIGH.value: 3,
    MatchConfidence.MEDIUM.value: 2,
    MatchConfidence.LOW.value: 1,
}


def choose_primary_candidate(candidates: Iterable[sqlite3.Row]) -> tuple[int | None, ResolutionStatus, str | None]:
    rows = list(candidates)
    if not rows:
        return None, ResolutionStatus.NO_PUBLICATION, None
    identifier_rows = [
        row for row in rows
        if row["normalized_doi"] or row["pmid"] or row["pmcid"] or row["openalex_id"]
    ]
    if not identifier_rows:
        return None, ResolutionStatus.LOW_CONFIDENCE, "publication candidates found, but none had DOI/PMID/PMCID/OpenAlex identifiers"
    high_rows = [row for row in identifier_rows if _CONFIDENCE_RANK.get(row["match_confidence"], 0) >= 3]
    if not high_rows:
        return None, ResolutionStatus.LOW_CONFIDENCE, "only low/medium confidence candidates found"

    def sort_key(row: sqlite3.Row) -> tuple[str, int, int]:
        date = row["publication_date"] or (f"{row['publication_year']:04d}" if row["publication_year"] else "9999")
        confidence_rank = -_CONFIDENCE_RANK.get(row["match_confidence"], 0)
        return (date, confidence_rank, int(row["id"]))

    sorted_rows = sorted(high_rows, key=sort_key)
    primary = sorted_rows[0]
    ambiguous = False
    if len(sorted_rows) > 1:
        first_key = (sort_key(sorted_rows[0])[0], sort_key(sorted_rows[0])[1])
        second_key = (sort_key(sorted_rows[1])[0], sort_key(sorted_rows[1])[1])
        ambiguous = first_key == second_key
    status = ResolutionStatus.RESOLVED_AMBIGUOUS if ambiguous else (
        ResolutionStatus.RESOLVED_MULTIPLE if len(rows) > 1 else ResolutionStatus.RESOLVED
    )
    return int(primary["id"]), status, "earliest high-confidence publication"
