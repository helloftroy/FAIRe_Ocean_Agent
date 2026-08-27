from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi

logger = logging.getLogger(__name__)

MGRAST_BASE_URL = "https://api.mg-rast.org"
MGRAST_LINK_BASE_URL = "https://mg-rast.org/linkin.cgi"
DEFAULT_MGRAST_DB = Path("data/seed_discovery/mgrast_paper_seeds.sqlite")
DEFAULT_MGRAST_DATA_DIR = Path("data/mgrast")
DEFAULT_MGNIFY_DB = Path("data/seed_discovery/mgnify_paper_seeds.sqlite")
DEFAULT_QIITA_DB = Path("data/seed_discovery/qiita_paper_seeds.sqlite")
DEFAULT_GOLD_DB = Path("data/jgi_gold/gold_sharded.sqlite")
DEFAULT_CNCB_DB = Path("data/seed_discovery/cncb_gsa_paper_seeds.sqlite")

MGP_RE = re.compile(r"\bmgp\d+\b", re.IGNORECASE)
MGS_RE = re.compile(r"\bmgs\d+\b", re.IGNORECASE)
MGM_RE = re.compile(r"\b(?:mgm)?\d{6,}\.\d+\b", re.IGNORECASE)
MGL_RE = re.compile(r"\bmgl\d+\b", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
PMID_RE = re.compile(r"\b(?:PMID|PubMed(?:\s+ID)?)[:\s]*([0-9]{5,})\b", re.IGNORECASE)
PMCID_RE = re.compile(r"\bPMC\d+\b", re.IGNORECASE)
BIOPROJECT_RE = re.compile(r"\bPRJ(?:NA|EB|DB|DA|EA)\d+\b", re.IGNORECASE)
SRA_STUDY_RE = re.compile(r"\b(?:SRP|ERP|DRP)\d+\b", re.IGNORECASE)
SRA_EXPERIMENT_RE = re.compile(r"\b(?:SRX|ERX|DRX)\d+\b", re.IGNORECASE)
SRA_RUN_RE = re.compile(r"\b(?:SRR|ERR|DRR)\d+\b", re.IGNORECASE)
BIOSAMPLE_RE = re.compile(r"\bSAM(?:N|E|D)[A-Z]?\d+\b", re.IGNORECASE)

MARINE_TERMS = (
    "marine",
    "ocean",
    "seawater",
    "sea water",
    "coastal",
    "estuary",
    "estuarine",
    "brackish",
    "coral",
    "reef",
    "mangrove",
    "salt marsh",
    "marine sediment",
    "seafloor",
    "pelagic",
    "benthic",
    "intertidal",
    "subtidal",
    "deep sea",
    "hydrothermal",
    "sea ice",
    "continental shelf",
)
ENVIRONMENT_TERMS = (
    "environmental",
    "metagenome",
    "metagenomic",
    "metatranscriptome",
    "metatranscriptomic",
    "microbiome",
    "microbial community",
    "amplicon",
    "mimarks",
    "16s",
    "18s",
    "its",
    "sediment",
    "soil",
    "water",
    "freshwater",
)
REJECT_TERMS = (
    "human",
    "clinical",
    "patient",
    "tumor",
    "cancer",
    "blood",
    "host-associated medical",
    "mouse",
    "mice",
    "rat ",
    "swine",
    "cattle",
    "isolate genome",
    "pure culture",
)

SAMPLE_FIELD_ALIASES = {
    "sample_name": ("sample_name", "sample name", "name"),
    "sample_description": ("sample_description", "description", "sample desc"),
    "collection_date": ("collection_date", "collection date", "samp_collect_date", "date"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "longitude"),
    "depth": ("depth", "samp_collect_depth", "water_depth"),
    "altitude": ("altitude", "elevation"),
    "geo_loc_name": ("geo_loc_name", "geographic location", "location"),
    "country": ("country",),
    "ocean_region": ("ocean_region", "ocean region", "marine region"),
    "env_broad_scale": ("env_broad_scale", "biome", "environment biome"),
    "env_local_scale": ("env_local_scale", "feature", "environment feature"),
    "env_medium": ("env_medium", "material", "environment material", "isolation source"),
    "biome": ("biome", "environment biome"),
    "feature": ("feature", "environment feature"),
    "material": ("material", "environment material", "sample material"),
    "sample_type": ("sample_type", "sample type"),
    "sampling_method": ("sampling_method", "sampling method"),
    "sample_collection_method": ("sample_collection_method", "sample collection method", "collection method"),
    "filter_type": ("filter_type", "filter type", "filter name"),
    "filter_pore_size": ("filter_pore_size", "pore size", "filter size"),
    "size_fraction": ("size_fraction", "size fraction"),
    "storage_method": ("storage_method", "samp_store_temp", "storage condition"),
    "preservation_method": ("preservation_method", "preservation"),
    "temperature": ("temperature", "temp", "water temperature"),
    "salinity": ("salinity",),
    "ph": ("ph", "pH"),
    "oxygen": ("oxygen",),
    "dissolved_oxygen": ("dissolved_oxygen", "dissolved oxygen"),
    "chlorophyll": ("chlorophyll", "chlorophyll a"),
    "nitrate": ("nitrate",),
    "nitrite": ("nitrite",),
    "ammonium": ("ammonium", "ammonia"),
    "phosphate": ("phosphate",),
    "pressure": ("pressure",),
}

DATASET_FIELD_ALIASES = {
    "sequence_type": ("sequence_type", "investigation_type", "type"),
    "data_type": ("data_type", "sequence_type_guess"),
    "library_strategy": ("investigation_type", "library_strategy"),
    "library_source": ("library_source",),
    "library_selection": ("library_selection",),
    "platform": ("platform", "sequencing_method"),
    "instrument_model": ("instrument_model", "instrument", "sequencing_method"),
    "target_gene": ("target_gene", "target gene", "gene"),
    "target_subfragment": ("target_subfragment", "target_subfragment", "region"),
    "primer_forward": ("primer_forward", "forward_primer", "pcr_primer_forward"),
    "primer_reverse": ("primer_reverse", "reverse_primer", "pcr_primer_reverse"),
    "extraction_method": ("extraction_method", "nucl_acid_ext"),
    "pcr_method": ("pcr_method", "pcr conditions", "pcr_cond"),
    "library_method": ("library_method", "library construction", "library protocol"),
    "read_length": ("read_length", "read length"),
}

ENV_MEASUREMENT_HINTS = {
    "temperature",
    "salinity",
    "ph",
    "oxygen",
    "dissolved oxygen",
    "chlorophyll",
    "nitrate",
    "nitrite",
    "ammonium",
    "ammonia",
    "phosphate",
    "sulfate",
    "sulfide",
    "conductivity",
    "turbidity",
    "pressure",
    "organic carbon",
    "dissolved organic carbon",
    "particulate organic carbon",
    "sample volume",
}


def utc_iso() -> str:
    return utcnow().isoformat()


def json_dumps(value) -> str:  # noqa: ANN001
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def normalize_doi_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return normalize_doi(value.strip().rstrip(".,;)</]"))
    except IdentifierError:
        return None


def unique(values) -> list[str]:  # noqa: ANN001
    out = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def flatten_metadata_value(value) -> str | None:  # noqa: ANN001
    if isinstance(value, dict):
        raw = value.get("value")
        unit = value.get("unit")
        if raw in (None, ""):
            return None
        text = str(raw).strip()
        if unit and str(unit).strip() and str(unit).strip() not in text:
            text = f"{text} {str(unit).strip()}"
        return text
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return None


def flatten_metadata_record(record: dict | None) -> dict[str, str]:
    flat: dict[str, str] = {}
    if not isinstance(record, dict):
        return flat
    for key, value in record.items():
        if key in {"data", "envPackage"} and isinstance(value, dict):
            flat.update(flatten_metadata_record(value.get("data") if "data" in value else value))
            continue
        text = flatten_metadata_value(value)
        if text:
            flat[key] = text
    return flat


def first_value(record: dict, aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize_key(key): key for key in record}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key and record.get(key):
            return str(record[key]).strip()
    return None


def text_blob(*values) -> str:  # noqa: ANN001
    return " ".join(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "") for value in values)


def extract_dois(*values) -> list[str]:
    found = []
    for raw in values:
        for match in DOI_RE.findall(str(raw or "")):
            doi = normalize_doi_or_none(match)
            if doi:
                found.append(doi)
    return unique(found)


def extract_pmids(*values) -> list[str]:
    found = []
    for raw in values:
        found.extend(PMID_RE.findall(str(raw or "")))
    return unique(found)


def extract_pmcids(*values) -> list[str]:
    return unique(match.upper() for raw in values for match in PMCID_RE.findall(str(raw or "")))


def extract_accessions(pattern: re.Pattern, *values) -> list[str]:
    return unique(match.upper() for raw in values for match in pattern.findall(str(raw or "")))


def classify_marine(*values) -> tuple[str, list[dict]]:
    blob = text_blob(*values).casefold()
    evidence = []
    for term in MARINE_TERMS:
        if term in blob:
            evidence.append({"term": term, "class": "marine"})
    for term in ENVIRONMENT_TERMS:
        if term in blob:
            evidence.append({"term": term, "class": "environment"})
    reject = [term for term in REJECT_TERMS if term in blob]
    if any(item["class"] == "marine" for item in evidence):
        return "high", evidence
    if evidence and not reject:
        return "medium", evidence
    if evidence:
        return "low", evidence + [{"term": term, "class": "reject"} for term in reject]
    return "not_marine", [{"term": term, "class": "reject"} for term in reject]


def infer_target(text: str) -> tuple[str | None, str | None]:
    lower = text.casefold()
    target = None
    subfragment = None
    if "16s" in lower:
        target = "16S rRNA"
    elif "18s" in lower:
        target = "18S rRNA"
    elif re.search(r"\bits\b", lower):
        target = "ITS"
    elif "coi" in lower or "cytochrome c" in lower:
        target = "COI"
    region = re.search(r"\b(V[1-9](?:[-\u2013]V[1-9])?)\b", text, re.IGNORECASE)
    if region:
        subfragment = region.group(1).upper().replace("\u2013", "-")
    return target, subfragment


def infer_layout(download_files: list[dict], metadata_blob: str) -> str | None:
    names = " ".join(str(item.get("file_name") or item.get("name") or "") for item in download_files).casefold()
    blob = f"{names} {metadata_blob.casefold()}"
    if re.search(r"(_r?1|\.1)\.(?:f(?:ast)?q|fa)", blob) and re.search(r"(_r?2|\.2)\.(?:f(?:ast)?q|fa)", blob):
        return "paired end"
    if "paired" in blob or "2 x " in blob or "2x" in blob:
        return "paired end"
    if "single" in blob:
        return "single end"
    return None


def is_sequence_file(row: dict) -> bool:
    name = str(row.get("file_name") or row.get("name") or "").casefold()
    data_type = str(row.get("data_type") or "").casefold()
    fmt = str(row.get("file_format") or "").casefold()
    return (
        data_type == "sequence"
        or fmt in {"fastq", "fq", "fasta", "fa", "fna"}
        or name.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz", ".fasta", ".fasta.gz", ".fa", ".fa.gz", ".fna", ".fna.gz"))
    )


def clean_publication_url(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if "doi.org/" in value or "pubmed" in value.casefold() or value.startswith(("http://", "https://")):
        return value
    return None


def publication_title_candidates(project: dict, export: dict) -> list[str]:
    candidates = []
    for record in (project.get("metadata") or {}, export.get("data") or {}, project):
        for key, value in flatten_metadata_record(record).items():
            key_norm = normalize_key(key)
            text = str(value).strip()
            if "publication" in key_norm or "citation" in key_norm or "article" in key_norm:
                if DOI_RE.search(text) or PMID_RE.search(text):
                    continue
                if len(text.split()) >= 4:
                    candidates.append(text)
    return unique(candidates)


@dataclass
class MgrastConfig:
    db_path: Path = DEFAULT_MGRAST_DB
    data_dir: Path = DEFAULT_MGRAST_DATA_DIR
    mgnify_db_path: Path = DEFAULT_MGNIFY_DB
    qiita_db_path: Path = DEFAULT_QIITA_DB
    gold_db_path: Path = DEFAULT_GOLD_DB
    cncb_db_path: Path = DEFAULT_CNCB_DB
    base_url: str = MGRAST_BASE_URL
    page_size: int = 100
    min_request_interval_seconds: float = 2.0
    request_timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_base_seconds: float = 2.0


class MgrastDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS mgrast_projects (
              mgrast_project_id TEXT PRIMARY KEY,
              title TEXT,
              description TEXT,
              abstract TEXT,
              project_type TEXT,
              project_public INTEGER,
              project_created_date TEXT,
              project_release_date TEXT,
              PI_name TEXT,
              contacts_json TEXT NOT NULL DEFAULT '[]',
              primary_doi TEXT,
              primary_pmid TEXT,
              primary_paper_title TEXT,
              primary_publication_date TEXT,
              publication_dois_json TEXT NOT NULL DEFAULT '[]',
              pmids_json TEXT NOT NULL DEFAULT '[]',
              paper_titles_json TEXT NOT NULL DEFAULT '[]',
              publication_urls_json TEXT NOT NULL DEFAULT '[]',
              publication_resolution_status TEXT NOT NULL DEFAULT 'not_yet_processed',
              publication_resolution_method TEXT,
              matched_mgrast_identifier TEXT,
              bioprojects_json TEXT NOT NULL DEFAULT '[]',
              insdc_study_accessions_json TEXT NOT NULL DEFAULT '[]',
              sequence_accessibility_status TEXT NOT NULL DEFAULT 'not_yet_processed',
              marine_confidence TEXT NOT NULL DEFAULT 'not_yet_processed',
              marine_match_methods_json TEXT NOT NULL DEFAULT '[]',
              sample_count INTEGER NOT NULL DEFAULT 0,
              dataset_count INTEGER NOT NULL DEFAULT 0,
              overlap_status TEXT,
              overlap_sources_json TEXT NOT NULL DEFAULT '{}',
              source_metadata_json TEXT NOT NULL DEFAULT '{}',
              first_seen_at TEXT NOT NULL,
              last_checked_at TEXT,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mgrast_samples (
              mgrast_project_id TEXT NOT NULL,
              mgrast_sample_id TEXT NOT NULL,
              sample_name TEXT,
              sample_description TEXT,
              biosample_accession TEXT,
              collection_date TEXT,
              latitude TEXT,
              longitude TEXT,
              depth TEXT,
              altitude TEXT,
              geo_loc_name TEXT,
              country TEXT,
              ocean_region TEXT,
              env_broad_scale TEXT,
              env_local_scale TEXT,
              env_medium TEXT,
              biome TEXT,
              feature TEXT,
              material TEXT,
              sample_type TEXT,
              sampling_method TEXT,
              sample_collection_method TEXT,
              filter_type TEXT,
              filter_pore_size TEXT,
              size_fraction TEXT,
              storage_method TEXT,
              preservation_method TEXT,
              temperature TEXT,
              salinity TEXT,
              ph TEXT,
              oxygen TEXT,
              dissolved_oxygen TEXT,
              chlorophyll TEXT,
              nitrate TEXT,
              nitrite TEXT,
              ammonium TEXT,
              phosphate TEXT,
              pressure TEXT,
              other_environmental_measurements_json TEXT NOT NULL DEFAULT '{}',
              source_metadata_json TEXT NOT NULL DEFAULT '{}',
              PRIMARY KEY (mgrast_project_id, mgrast_sample_id)
            );

            CREATE TABLE IF NOT EXISTS mgrast_datasets (
              mgrast_project_id TEXT NOT NULL,
              mgrast_sample_id TEXT,
              mgrast_dataset_id TEXT NOT NULL,
              dataset_name TEXT,
              sequence_type TEXT,
              data_type TEXT,
              library_strategy TEXT,
              library_source TEXT,
              library_selection TEXT,
              platform TEXT,
              instrument_model TEXT,
              target_gene TEXT,
              target_subfragment TEXT,
              primer_forward TEXT,
              primer_reverse TEXT,
              extraction_method TEXT,
              pcr_method TEXT,
              library_method TEXT,
              paired_end TEXT,
              read_length TEXT,
              raw_sequence_available INTEGER NOT NULL DEFAULT 0,
              input_filename TEXT,
              file_size TEXT,
              checksum TEXT,
              download_locator TEXT,
              bioproject_accession TEXT,
              biosample_accession TEXT,
              insdc_study_accessions_json TEXT NOT NULL DEFAULT '[]',
              experiment_accessions_json TEXT NOT NULL DEFAULT '[]',
              run_accessions_json TEXT NOT NULL DEFAULT '[]',
              source_metadata_json TEXT NOT NULL DEFAULT '{}',
              PRIMARY KEY (mgrast_project_id, mgrast_dataset_id)
            );

            CREATE VIEW IF NOT EXISTS paper_seeds AS
              SELECT
                'mgrast' AS seed_source,
                mgrast_project_id AS source_study_id,
                mgrast_project_id AS source_project_id,
                CASE
                  WHEN json_array_length(bioprojects_json) > 0 THEN json_extract(bioprojects_json, '$[0]')
                  ELSE NULL
                END AS bioproject_accession,
                mgrast_project_id AS native_project_accession,
                title AS study_title,
                primary_doi,
                primary_pmid,
                primary_paper_title,
                publication_dois_json,
                sequence_accessibility_status,
                marine_confidence,
                overlap_status,
                publication_resolution_status,
                first_seen_at,
                last_checked_at
              FROM mgrast_projects
              WHERE marine_confidence IN ('high', 'medium')
                AND sequence_accessibility_status IN (
                  'mgrast_raw_reads_confirmed',
                  'mgrast_public_sequence_confirmed',
                  'mgrast_and_insdc_raw_reads',
                  'files_listed_access_unverified'
                );

            CREATE VIEW IF NOT EXISTS mgrast_faire_sample_enrichment AS
              SELECT mgrast_project_id, mgrast_sample_id, biosample_accession,
                     collection_date, latitude, longitude, depth, geo_loc_name,
                     env_broad_scale, env_local_scale, env_medium, biome, feature,
                     material, sampling_method, sample_collection_method,
                     temperature, salinity, ph, oxygen, dissolved_oxygen,
                     chlorophyll, nitrate, nitrite, ammonium, phosphate
              FROM mgrast_samples;

            CREATE VIEW IF NOT EXISTS mgrast_faire_experiment_enrichment AS
              SELECT mgrast_project_id, mgrast_sample_id, mgrast_dataset_id,
                     platform, instrument_model, sequence_type, library_strategy,
                     target_gene, target_subfragment, primer_forward, primer_reverse,
                     extraction_method, pcr_method, library_method, paired_end,
                     input_filename, checksum, download_locator, insdc_study_accessions_json,
                     experiment_accessions_json, run_accessions_json
              FROM mgrast_datasets;
            """
        )
        self.conn.commit()

    def upsert_project(self, row: dict) -> None:
        row = dict(row)
        now = utc_iso()
        row.setdefault("first_seen_at", now)
        row.setdefault("updated_at", now)
        columns = (
            "mgrast_project_id", "title", "description", "abstract", "project_type", "project_public",
            "project_created_date", "project_release_date", "PI_name", "contacts_json", "primary_doi",
            "primary_pmid", "primary_paper_title", "primary_publication_date", "publication_dois_json",
            "pmids_json", "paper_titles_json", "publication_urls_json", "publication_resolution_status",
            "publication_resolution_method", "matched_mgrast_identifier", "bioprojects_json",
            "insdc_study_accessions_json", "sequence_accessibility_status", "marine_confidence",
            "marine_match_methods_json", "sample_count", "dataset_count", "overlap_status",
            "overlap_sources_json", "source_metadata_json", "first_seen_at", "last_checked_at", "updated_at",
        )
        payload = {column: row.get(column) for column in columns}
        for column, default in (
            ("contacts_json", "[]"),
            ("publication_dois_json", "[]"),
            ("pmids_json", "[]"),
            ("paper_titles_json", "[]"),
            ("publication_urls_json", "[]"),
            ("bioprojects_json", "[]"),
            ("insdc_study_accessions_json", "[]"),
            ("marine_match_methods_json", "[]"),
            ("overlap_sources_json", "{}"),
            ("source_metadata_json", "{}"),
        ):
            payload[column] = payload[column] or default
        payload["project_public"] = 1 if payload.get("project_public") else 0
        payload["publication_resolution_status"] = payload["publication_resolution_status"] or "not_yet_processed"
        payload["sequence_accessibility_status"] = payload["sequence_accessibility_status"] or "not_yet_processed"
        payload["marine_confidence"] = payload["marine_confidence"] or "not_yet_processed"
        payload["sample_count"] = payload["sample_count"] or 0
        payload["dataset_count"] = payload["dataset_count"] or 0
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"mgrast_project_id", "first_seen_at"})
        self.conn.execute(
            f"INSERT INTO mgrast_projects({', '.join(columns)}) VALUES({placeholders}) "
            f"ON CONFLICT(mgrast_project_id) DO UPDATE SET {updates}",
            [payload[column] for column in columns],
        )
        self.conn.commit()

    def upsert_sample(self, row: dict) -> None:
        columns = (
            "mgrast_project_id", "mgrast_sample_id", "sample_name", "sample_description", "biosample_accession",
            "collection_date", "latitude", "longitude", "depth", "altitude", "geo_loc_name", "country",
            "ocean_region", "env_broad_scale", "env_local_scale", "env_medium", "biome", "feature",
            "material", "sample_type", "sampling_method", "sample_collection_method", "filter_type",
            "filter_pore_size", "size_fraction", "storage_method", "preservation_method", "temperature",
            "salinity", "ph", "oxygen", "dissolved_oxygen", "chlorophyll", "nitrate", "nitrite",
            "ammonium", "phosphate", "pressure", "other_environmental_measurements_json", "source_metadata_json",
        )
        payload = {column: row.get(column) for column in columns}
        payload["other_environmental_measurements_json"] = payload["other_environmental_measurements_json"] or "{}"
        payload["source_metadata_json"] = payload["source_metadata_json"] or "{}"
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"mgrast_project_id", "mgrast_sample_id"})
        self.conn.execute(
            f"INSERT INTO mgrast_samples({', '.join(columns)}) VALUES({placeholders}) "
            f"ON CONFLICT(mgrast_project_id, mgrast_sample_id) DO UPDATE SET {updates}",
            [payload[column] for column in columns],
        )
        self.conn.commit()

    def upsert_dataset(self, row: dict) -> None:
        columns = (
            "mgrast_project_id", "mgrast_sample_id", "mgrast_dataset_id", "dataset_name", "sequence_type",
            "data_type", "library_strategy", "library_source", "library_selection", "platform",
            "instrument_model", "target_gene", "target_subfragment", "primer_forward", "primer_reverse",
            "extraction_method", "pcr_method", "library_method", "paired_end", "read_length",
            "raw_sequence_available", "input_filename", "file_size", "checksum", "download_locator",
            "bioproject_accession", "biosample_accession", "insdc_study_accessions_json",
            "experiment_accessions_json", "run_accessions_json", "source_metadata_json",
        )
        payload = {column: row.get(column) for column in columns}
        payload["raw_sequence_available"] = 1 if payload.get("raw_sequence_available") else 0
        for column in ("insdc_study_accessions_json", "experiment_accessions_json", "run_accessions_json"):
            payload[column] = payload[column] or "[]"
        payload["source_metadata_json"] = payload["source_metadata_json"] or "{}"
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"mgrast_project_id", "mgrast_dataset_id"})
        self.conn.execute(
            f"INSERT INTO mgrast_datasets({', '.join(columns)}) VALUES({placeholders}) "
            f"ON CONFLICT(mgrast_project_id, mgrast_dataset_id) DO UPDATE SET {updates}",
            [payload[column] for column in columns],
        )
        self.conn.commit()

    def project_ids_for_metadata(self, max_projects: int | None, *, refresh: bool) -> list[str]:
        where = "" if refresh else "WHERE last_checked_at IS NULL OR marine_confidence = 'not_yet_processed'"
        sql = f"SELECT mgrast_project_id FROM mgrast_projects {where} ORDER BY mgrast_project_id"
        if max_projects:
            sql += f" LIMIT {int(max_projects)}"
        return [row[0] for row in self.conn.execute(sql)]

    def all_projects(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM mgrast_projects ORDER BY mgrast_project_id").fetchall()


class MgrastClient:
    def __init__(self, config: MgrastConfig, *, refresh: bool = False, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.refresh = refresh
        self._last_request = 0.0
        self._client = httpx.Client(
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": "fair-ocean-agent-mgrast-discovery/0.1", "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait = self.config.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get_json(self, path: str, *, params: dict | None = None, cache_path: Path | None = None) -> dict:
        if cache_path and cache_path.exists() and not self.refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        # Bounded retry with exponential backoff for transient failures --
        # this client previously had none at all (unlike CncbClient/
        # QiitaClient, which both needed this fix live after a single
        # transient network blip crashed an entire discovery job outright).
        payload = None
        for attempt in range(1, self.config.max_retries + 1):
            self._sleep()
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
                break
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                non_retryable_client_error = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500
                if non_retryable_client_error or attempt >= self.config.max_retries:
                    raise
                sleep_for = self.config.retry_base_seconds * attempt
                logger.warning("retrying MG-RAST request %s after %ss (%s)", url, sleep_for, exc)
                time.sleep(sleep_for)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json_dumps(payload), encoding="utf-8")
        return payload

    def project_page(self, *, limit: int, offset: int) -> dict:
        return self.get_json("project", params={"limit": limit, "offset": offset})

    def project(self, project_id: str, raw_dir: Path) -> dict:
        return self.get_json(f"project/{project_id}", params={"verbosity": "full"}, cache_path=raw_dir / "project.json")

    def metadata_export(self, project_id: str, raw_dir: Path) -> dict:
        return self.get_json(f"metadata/export/{project_id}", cache_path=raw_dir / "metadata.json")

    def download_manifest(self, dataset_id: str, raw_dir: Path) -> dict:
        return self.get_json(f"download/{dataset_id}", cache_path=raw_dir / "downloads" / f"{dataset_id}.json")


def project_seed_from_item(item: dict) -> dict | None:
    project_id = str(item.get("id") or "").strip()
    if not MGP_RE.fullmatch(project_id):
        return None
    now = utc_iso()
    return {
        "mgrast_project_id": project_id,
        "title": item.get("name"),
        "project_public": str(item.get("status") or "").casefold() == "public",
        "project_created_date": item.get("created"),
        "PI_name": item.get("pi"),
        "sequence_accessibility_status": "not_yet_processed",
        "marine_confidence": "not_yet_processed",
        "publication_resolution_status": "not_yet_processed",
        "source_metadata_json": json_dumps({"project_list_item": item}),
        "first_seen_at": now,
        "updated_at": now,
    }


def metadata_sample_rows(project_id: str, project: dict, export: dict) -> list[dict]:
    sample_records: dict[str, dict] = {}
    for metagenome in project.get("metagenomes") or []:
        sample_id = str(metagenome.get("sample") or "").strip()
        if sample_id:
            sample_records.setdefault(sample_id, {}).update(metagenome)
    for sample in export.get("samples") or []:
        sample_id = str(sample.get("id") or sample.get("name") or "").strip()
        flat = flatten_metadata_record(sample)
        if sample_id:
            sample_records.setdefault(sample_id, {}).update(flat)
            sample_records[sample_id]["_source_export_record"] = sample

    rows = []
    for sample_id, record in sample_records.items():
        row = {"mgrast_project_id": project_id, "mgrast_sample_id": sample_id}
        for column, aliases in SAMPLE_FIELD_ALIASES.items():
            row[column] = first_value(record, aliases)
        coords = str(record.get("coordinates") or "")
        if coords and (not row.get("latitude") or not row.get("longitude")):
            parts = [part.strip() for part in coords.split(",")]
            if len(parts) >= 2:
                row["latitude"] = row.get("latitude") or parts[0]
                row["longitude"] = row.get("longitude") or parts[1]
        row["sample_name"] = row.get("sample_name") or sample_id
        row["biosample_accession"] = next(iter(extract_accessions(BIOSAMPLE_RE, record)), None)
        other_env = {}
        for key, value in record.items():
            norm = normalize_key(key)
            if any(hint in norm for hint in ENV_MEASUREMENT_HINTS) and key not in row:
                other_env[key] = value
        row["other_environmental_measurements_json"] = json_dumps(other_env)
        row["source_metadata_json"] = json_dumps(record)
        rows.append(row)
    return rows


def libraries_by_id(export: dict) -> dict[str, dict]:
    libraries = {}
    for sample in export.get("samples") or []:
        for library in sample.get("libraries") or []:
            library_id = str(library.get("id") or library.get("name") or "").strip()
            if library_id:
                libraries[library_id] = flatten_metadata_record(library)
                libraries[library_id]["_source_export_record"] = library
    return libraries


def dataset_rows(project_id: str, project: dict, export: dict, downloads_by_dataset: dict[str, dict]) -> list[dict]:
    library_records = libraries_by_id(export)
    rows = []
    for metagenome in project.get("metagenomes") or []:
        dataset_id = str(metagenome.get("metagenome_id") or "").strip()
        if not dataset_id:
            continue
        library_id = str(metagenome.get("library") or "").strip()
        merged = {}
        merged.update(metagenome)
        merged.update(metagenome.get("attributes") or {})
        merged.update(library_records.get(library_id, {}))
        download = downloads_by_dataset.get(dataset_id) or {}
        files = download.get("data") if isinstance(download, dict) else []
        if not isinstance(files, list):
            files = []
        sequence_files = [item for item in files if isinstance(item, dict) and is_sequence_file(item)]
        first_file = sequence_files[0] if sequence_files else (files[0] if files else {})
        blob = text_blob(merged, files)
        target, subfragment = infer_target(blob)
        row = {
            "mgrast_project_id": project_id,
            "mgrast_sample_id": metagenome.get("sample"),
            "mgrast_dataset_id": dataset_id,
            "dataset_name": metagenome.get("name"),
            "paired_end": infer_layout(files, blob),
            "raw_sequence_available": bool(sequence_files or files),
            "input_filename": first_file.get("file_name"),
            "file_size": first_file.get("file_size"),
            "checksum": first_file.get("file_md5") or first_file.get("checksum"),
            "download_locator": first_file.get("url"),
            "bioproject_accession": next(iter(extract_accessions(BIOPROJECT_RE, merged, files)), None),
            "biosample_accession": next(iter(extract_accessions(BIOSAMPLE_RE, merged, files)), None),
            "insdc_study_accessions_json": json_dumps(extract_accessions(SRA_STUDY_RE, merged, files)),
            "experiment_accessions_json": json_dumps(extract_accessions(SRA_EXPERIMENT_RE, merged, files)),
            "run_accessions_json": json_dumps(extract_accessions(SRA_RUN_RE, merged, files)),
            "source_metadata_json": json_dumps({"metagenome": metagenome, "library": library_records.get(library_id), "download": download}),
        }
        for column, aliases in DATASET_FIELD_ALIASES.items():
            row[column] = first_value(merged, aliases)
        row["target_gene"] = row.get("target_gene") or target
        row["target_subfragment"] = row.get("target_subfragment") or subfragment
        rows.append(row)
    return rows


def sequence_accessibility_status(rows: list[dict]) -> str:
    has_mgrast = any(row.get("raw_sequence_available") for row in rows)
    has_insdc = any(json.loads(row.get("run_accessions_json") or "[]") or json.loads(row.get("insdc_study_accessions_json") or "[]") for row in rows)
    if has_mgrast and has_insdc:
        return "mgrast_and_insdc_raw_reads"
    if has_mgrast:
        return "mgrast_raw_reads_confirmed"
    if has_insdc:
        return "insdc_raw_reads_confirmed"
    return "no_public_sequence_data"


def project_row_from_metadata(project_id: str, project: dict, export: dict, sample_rows: list[dict], dataset_rows_: list[dict]) -> dict:
    flat_project = flatten_metadata_record(project.get("metadata") or {})
    flat_export = flatten_metadata_record(export.get("data") or {})
    source_blob = text_blob(project, export)
    dois = extract_dois(source_blob)
    pmids = extract_pmids(source_blob)
    pmcids = extract_pmcids(source_blob)
    publication_urls = unique(
        url
        for url in re.findall(r"https?://[^\s\"'<>)]+", source_blob)
        if clean_publication_url(url)
    )
    titles = publication_title_candidates(project, export)
    primary_title = titles[0] if titles else None
    marine_confidence, marine_evidence = classify_marine(project, export)
    bioprojects = extract_accessions(BIOPROJECT_RE, source_blob)
    insdc_studies = extract_accessions(SRA_STUDY_RE, source_blob)
    methods = "missing"
    if dois:
        methods = "mgrast_explicit_doi"
    elif pmids:
        methods = "mgrast_explicit_pmid"
    elif primary_title:
        methods = "mgrast_explicit_title_or_citation"
    status = "resolved" if dois and (primary_title or project.get("name")) else ("title_known_doi_missing" if primary_title else ("pmid_only" if pmids else "missing"))
    now = utc_iso()
    description = project.get("description") or flat_project.get("project_description") or flat_export.get("project_description")
    return {
        "mgrast_project_id": project_id,
        "title": project.get("name") or export.get("name"),
        "description": description,
        "abstract": flat_project.get("project_abstract") or flat_export.get("project_abstract"),
        "project_type": flat_project.get("investigation_type") or flat_export.get("investigation_type"),
        "project_public": str(project.get("status") or "").casefold() == "public",
        "project_created_date": project.get("created") or project.get("created_on"),
        "project_release_date": project.get("public_date") or project.get("release_date"),
        "PI_name": project.get("pi") or flat_project.get("lastname"),
        "contacts_json": json_dumps(unique([project.get("pi"), flat_project.get("firstname"), flat_project.get("lastname")])),
        "primary_doi": dois[0] if dois else None,
        "primary_pmid": pmids[0] if pmids else None,
        "primary_paper_title": primary_title,
        "publication_dois_json": json_dumps(dois),
        "pmids_json": json_dumps(pmids),
        "paper_titles_json": json_dumps(titles),
        "publication_urls_json": json_dumps(publication_urls),
        "publication_resolution_status": status,
        "publication_resolution_method": methods,
        "matched_mgrast_identifier": project_id if status != "missing" else None,
        "bioprojects_json": json_dumps(bioprojects),
        "insdc_study_accessions_json": json_dumps(insdc_studies),
        "sequence_accessibility_status": sequence_accessibility_status(dataset_rows_),
        "marine_confidence": marine_confidence,
        "marine_match_methods_json": json_dumps(marine_evidence),
        "sample_count": len(sample_rows),
        "dataset_count": len(dataset_rows_),
        "source_metadata_json": json_dumps({"project": project, "metadata_export": export, "pmcids": pmcids}),
        "last_checked_at": now,
        "updated_at": now,
    }


def update_overlap(db: MgrastDB, config: MgrastConfig) -> None:
    sources_by_doi = {
        "mgnify": _doi_set(config.mgnify_db_path, "SELECT primary_doi FROM mgnify_studies WHERE primary_doi IS NOT NULL"),
        # ena_studies has no primary_doi column at all (confirmed live
        # 2026-08-26 against seed_discovery/db.py's own schema) -- an ENA
        # study's DOI lives in the shared publication_candidates table,
        # joined via ena_study_id, same shape cncb_gsa_discovery.py's own
        # load_mgnify_overlap already uses for mgnify_study_id. The
        # original flat query here would have silently returned an empty
        # set on every run (caught by _doi_set's own try/except), quietly
        # under-reporting every ENA DOI overlap rather than erroring.
        "ena": _doi_set(
            config.mgnify_db_path,
            "SELECT pc.normalized_doi FROM publication_candidates pc "
            "JOIN ena_studies es ON es.id = pc.ena_study_id "
            "WHERE pc.normalized_doi IS NOT NULL",
        ),
        "qiita": _doi_set(config.qiita_db_path, "SELECT primary_doi FROM qiita_studies WHERE primary_doi IS NOT NULL"),
        "gold": _doi_set(config.gold_db_path, "SELECT primary_doi FROM gold_studies WHERE primary_doi IS NOT NULL"),
        "cncb": _doi_set(config.cncb_db_path, "SELECT primary_doi FROM cncb_projects WHERE primary_doi IS NOT NULL"),
    }
    sources_by_bioproject = {
        "mgnify": _value_set(config.mgnify_db_path, "SELECT bioproject_accession FROM mgnify_studies WHERE bioproject_accession IS NOT NULL"),
        "ena": _value_set(config.mgnify_db_path, "SELECT bioproject_accession FROM ena_studies WHERE bioproject_accession IS NOT NULL"),
        "gold": _value_set(config.gold_db_path, "SELECT bioproject_accession FROM gold_studies WHERE bioproject_accession IS NOT NULL"),
    }
    for row in db.all_projects():
        overlaps = {key: [] for key in ("ena", "mgnify", "qiita", "jgi", "cncb")}
        doi = row["primary_doi"]
        for source, dois in sources_by_doi.items():
            if doi and doi in dois:
                overlaps["jgi" if source == "gold" else source].append(f"doi:{doi}")
        for bp in json.loads(row["bioprojects_json"] or "[]"):
            for source, values in sources_by_bioproject.items():
                if bp in values:
                    overlaps["jgi" if source == "gold" else source].append(f"bioproject:{bp}")
        nonempty = {key: unique(value) for key, value in overlaps.items() if value}
        if nonempty and row["sequence_accessibility_status"] != "no_public_sequence_data":
            status = "known_project_with_new_metadata"
        elif row["primary_doi"] and row["sequence_accessibility_status"] != "no_public_sequence_data":
            status = "net_new_project"
        else:
            status = "mgrast_only" if not json.loads(row["bioprojects_json"] or "[]") else "unknown"
        db.conn.execute(
            "UPDATE mgrast_projects SET overlap_status = ?, overlap_sources_json = ?, updated_at = ? WHERE mgrast_project_id = ?",
            (status, json_dumps(nonempty), utc_iso(), row["mgrast_project_id"]),
        )
    db.conn.commit()


def _value_set(path: Path, sql: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        rows = {str(row[0]).strip() for row in conn.execute(sql) if row[0]}
        conn.close()
        return rows
    except sqlite3.Error:
        return set()


def _doi_set(path: Path, sql: str) -> set[str]:
    return {doi for doi in (normalize_doi_or_none(value) for value in _value_set(path, sql)) if doi}


def write_manifest(data_dir: Path, project_id: str, rows: list[dict]) -> None:
    manifest_dir = data_dir / "file_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        if row.get("download_locator"):
            manifest.append(
                {
                    "mgrast_project_id": project_id,
                    "mgrast_dataset_id": row.get("mgrast_dataset_id"),
                    "mgrast_sample_id": row.get("mgrast_sample_id"),
                    "filename": row.get("input_filename"),
                    "file_size": row.get("file_size"),
                    "checksum": row.get("checksum"),
                    "download_locator": row.get("download_locator"),
                }
            )
    (manifest_dir / f"{project_id}.json").write_text(json_dumps(manifest), encoding="utf-8")


def write_reports(db: MgrastDB, data_dir: Path) -> None:
    reports = data_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    queries = {
        "new_projects.csv": "SELECT * FROM mgrast_projects WHERE overlap_status IN ('net_new_project', 'mgrast_only') ORDER BY mgrast_project_id",
        "existing_projects_enriched.csv": "SELECT * FROM mgrast_projects WHERE overlap_status = 'known_project_with_new_metadata' ORDER BY mgrast_project_id",
        "source_overlap.csv": "SELECT mgrast_project_id, title, primary_doi, bioprojects_json, overlap_status, overlap_sources_json FROM mgrast_projects ORDER BY mgrast_project_id",
        "unresolved_publications.csv": "SELECT * FROM mgrast_projects WHERE publication_resolution_status IN ('missing', 'not_yet_processed', 'ambiguous') ORDER BY mgrast_project_id",
        "paper_title_resolved_doi_missing.csv": "SELECT * FROM mgrast_projects WHERE primary_paper_title IS NOT NULL AND primary_doi IS NULL ORDER BY mgrast_project_id",
        "unresolved_sequence_access.csv": "SELECT * FROM mgrast_projects WHERE sequence_accessibility_status IN ('not_yet_processed', 'no_public_sequence_data') ORDER BY mgrast_project_id",
    }
    for filename, query in queries.items():
        rows = db.conn.execute(query).fetchall()
        with (reports / filename).open("w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(dict(row) for row in rows)
    write_metadata_completeness(db, reports / "metadata_completeness.csv")


def write_metadata_completeness(db: MgrastDB, path: Path) -> None:
    rows = []
    for table, columns in {
        "mgrast_samples": (
            "collection_date", "latitude", "longitude", "depth", "env_broad_scale", "env_local_scale",
            "env_medium", "sampling_method", "temperature", "salinity", "ph", "oxygen", "chlorophyll",
            "nitrate", "nitrite", "ammonium", "phosphate",
        ),
        "mgrast_datasets": (
            "sequence_type", "platform", "instrument_model", "target_gene", "target_subfragment",
            "primer_forward", "primer_reverse", "extraction_method", "pcr_method", "library_method",
        ),
    }.items():
        total = db.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for column in columns:
            populated = db.conn.execute(f"SELECT count(*) FROM {table} WHERE {column} IS NOT NULL AND {column} != ''").fetchone()[0]
            rows.append({"table": table, "field": column, "populated": populated, "total": total})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("table", "field", "populated", "total"))
        writer.writeheader()
        writer.writerows(rows)


def write_output_locations(config: MgrastConfig, seed_csv: Path = Path("cluster/seeds_mgrast.csv")) -> None:
    payload = {
        "database": str(config.db_path),
        "tables": ["mgrast_projects", "mgrast_samples", "mgrast_datasets"],
        "paper_seeds_view": "paper_seeds",
        "raw_data_dir": str(config.data_dir / "raw"),
        "file_manifest_dir": str(config.data_dir / "file_manifests"),
        "reports_dir": str(config.data_dir / "reports"),
        "seed_csv": str(seed_csv),
        "sbatch_script": "cluster/run_mgrast_discovery.sbatch",
    }
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "OUTPUT_LOCATIONS.json").write_text(json_dumps(payload), encoding="utf-8")


class MgrastDiscoveryRunner:
    def __init__(self, config: MgrastConfig, *, refresh: bool = False, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.refresh = refresh
        self.db = MgrastDB(config.db_path)
        self.client = MgrastClient(config, refresh=refresh, transport=transport)

    def close(self) -> None:
        self.client.close()
        self.db.close()

    def run(self, phase: str, *, max_projects: int | None = None) -> Counter:
        self.db.initialize()
        phases = ["discovery", "metadata", "overlap", "reports"] if phase == "all" else [phase]
        counts = Counter()
        for current in phases:
            if current == "discovery":
                counts.update(self._discovery(max_projects=max_projects))
            elif current in {"metadata", "files", "accessions", "publications"}:
                counts.update(self._metadata(max_projects=max_projects))
            elif current == "overlap":
                update_overlap(self.db, self.config)
                counts["overlap_complete"] += 1
            elif current == "reports":
                write_reports(self.db, self.config.data_dir)
                write_output_locations(self.config)
                counts["reports_written"] += 1
            else:
                raise ValueError(f"Unknown phase: {current}")
        print_mgrast_report(self.db, self.config)
        return counts

    def _discovery(self, *, max_projects: int | None) -> Counter:
        counts = Counter()
        offset = 0
        while True:
            try:
                page = self.client.project_page(limit=self.config.page_size, offset=offset)
            except httpx.HTTPError as exc:
                # Client-level retries (MgrastClient.get_json) are already
                # exhausted by this point -- stopping here (rather than
                # crashing run() outright, the same failure shape that hit
                # Qiita and CNCB live) still preserves every project found
                # so far (MgrastDB.upsert_project commits per-row), even
                # though a re-run currently restarts this walk from offset
                # 0 rather than resuming from here.
                counts["discovery_errors"] += 1
                logger.warning("MG-RAST project listing failed at offset=%s, stopping discovery: %s", offset, exc)
                break
            items = page.get("data") or []
            if not items:
                break
            for item in items:
                if max_projects and counts["projects_seen"] >= max_projects:
                    return counts
                row = project_seed_from_item(item)
                if row:
                    self.db.upsert_project(row)
                    counts["projects_seen"] += 1
            offset += self.config.page_size
            if not page.get("next"):
                break
            if counts["projects_seen"] % 100 == 0:
                logger.info("MG-RAST discovery scanned %s projects", counts["projects_seen"])
        return counts

    def _metadata(self, *, max_projects: int | None) -> Counter:
        counts = Counter()
        project_ids = self.db.project_ids_for_metadata(max_projects, refresh=self.refresh)
        for index, project_id in enumerate(project_ids, start=1):
            raw_dir = self.config.data_dir / "raw" / project_id
            try:
                project = self.client.project(project_id, raw_dir)
                export = self.client.metadata_export(project_id, raw_dir)
                downloads = {}
                for metagenome in project.get("metagenomes") or []:
                    dataset_id = str(metagenome.get("metagenome_id") or "").strip()
                    if dataset_id:
                        downloads[dataset_id] = self.client.download_manifest(dataset_id, raw_dir)
                samples = metadata_sample_rows(project_id, project, export)
                datasets = dataset_rows(project_id, project, export, downloads)
                for sample in samples:
                    self.db.upsert_sample(sample)
                for dataset in datasets:
                    self.db.upsert_dataset(dataset)
                self.db.upsert_project(project_row_from_metadata(project_id, project, export, samples, datasets))
                write_manifest(self.config.data_dir, project_id, datasets)
                counts["projects_metadata_processed"] += 1
                counts["samples"] += len(samples)
                counts["datasets"] += len(datasets)
            except Exception as exc:  # noqa: BLE001
                logger.exception("MG-RAST metadata failed for %s", project_id)
                self.db.conn.execute(
                    "UPDATE mgrast_projects SET last_checked_at = ?, sequence_accessibility_status = ?, publication_resolution_status = ?, updated_at = ? WHERE mgrast_project_id = ?",
                    (utc_iso(), "api_error", "api_error", utc_iso(), project_id),
                )
                self.db.conn.commit()
                counts["api_errors"] += 1
            if index % 100 == 0:
                logger.info("MG-RAST metadata processed %s/%s projects", index, len(project_ids))
        return counts


def count_where(db: MgrastDB, table: str, where: str = "1=1") -> int:
    return int(db.conn.execute(f"SELECT count(*) FROM {table} WHERE {where}").fetchone()[0])


def print_mgrast_report(db: MgrastDB, config: MgrastConfig) -> None:
    rows = db.conn.execute("SELECT * FROM mgrast_projects").fetchall()
    overlap_counter = Counter()
    for row in rows:
        for source, values in json.loads(row["overlap_sources_json"] or "{}").items():
            if values:
                overlap_counter[source] += 1
    marine_where = "marine_confidence IN ('high', 'medium')"
    public_sequence_where = (
        "sequence_accessibility_status != 'no_public_sequence_data' "
        "AND sequence_accessibility_status NOT IN ('not_yet_processed', 'api_error')"
    )
    title_where = "primary_paper_title IS NOT NULL AND primary_paper_title != ''"
    doi_where = "primary_doi IS NOT NULL AND primary_doi != ''"
    pmid_where = "primary_pmid IS NOT NULL AND primary_pmid != ''"
    with_bioproject_where = "json_array_length(bioprojects_json) > 0"
    without_bioproject_where = "json_array_length(bioprojects_json) = 0"
    net_new_where = "overlap_status = 'net_new_project'"
    title_no_doi_where = "primary_paper_title IS NOT NULL AND primary_doi IS NULL"
    unresolved_where = "publication_resolution_status IN ('missing', 'not_yet_processed')"
    print("\n============================================================")
    print("MG-RAST DISCOVERY COMPLETE")
    print("============================================================\n")
    print("DATABASE")
    print(config.db_path)
    print("\nTABLES")
    print("  mgrast_projects")
    print("  mgrast_samples")
    print("  mgrast_datasets")
    print("\nPAPER SEEDS VIEW")
    print("  paper_seeds")
    print("\nRAW SOURCE DATA")
    print(config.data_dir / "raw")
    print("\nFILE MANIFESTS")
    print(config.data_dir / "file_manifests")
    print("\nREPORTS")
    print(config.data_dir / "reports")
    print("\nSEED CSV")
    print("cluster/seeds_mgrast.csv")
    print("\nSBATCH SCRIPT")
    print("cluster/run_mgrast_discovery.sbatch")
    print("\n============================================================\n")
    print(f"PUBLIC PROJECTS SCANNED: {len(rows)}")
    print(f"MARINE/ENVIRONMENT PROJECTS: {count_where(db, 'mgrast_projects', marine_where)}")
    print(f"PROJECTS WITH PUBLIC SEQUENCE DATA: {count_where(db, 'mgrast_projects', public_sequence_where)}")
    print(f"PROJECTS WITH PAPER TITLE: {count_where(db, 'mgrast_projects', title_where)}")
    print(f"PROJECTS WITH DOI: {count_where(db, 'mgrast_projects', doi_where)}")
    print(f"PROJECTS WITH PMID: {count_where(db, 'mgrast_projects', pmid_where)}")
    print(f"PROJECTS WITH BIOPROJECT: {count_where(db, 'mgrast_projects', with_bioproject_where)}")
    print(f"PROJECTS WITHOUT BIOPROJECT: {count_where(db, 'mgrast_projects', without_bioproject_where)}")
    print(f"MG-RAST-ONLY PROJECTS: {count_where(db, 'mgrast_projects', without_bioproject_where)}")
    print(f"OVERLAP ENA: {overlap_counter['ena']}")
    print(f"OVERLAP MGNIFY: {overlap_counter['mgnify']}")
    print(f"OVERLAP QIITA: {overlap_counter['qiita']}")
    print(f"OVERLAP JGI: {overlap_counter['jgi']}")
    print(f"OVERLAP CNCB: {overlap_counter['cncb']}")
    print(f"NET-NEW PROJECTS: {count_where(db, 'mgrast_projects', net_new_where)}")
    print(f"PAPER TITLE KNOWN BUT DOI UNRESOLVED: {count_where(db, 'mgrast_projects', title_no_doi_where)}")
    print(f"PUBLICATION COMPLETELY UNRESOLVED: {count_where(db, 'mgrast_projects', unresolved_where)}")
    print("\n============================================================")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover MG-RAST marine/environmental project seeds.")
    parser.add_argument("--db", type=Path, default=DEFAULT_MGRAST_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_MGRAST_DATA_DIR)
    parser.add_argument("--mgnify-db", type=Path, default=DEFAULT_MGNIFY_DB)
    parser.add_argument("--qiita-db", type=Path, default=DEFAULT_QIITA_DB)
    parser.add_argument("--gold-db", type=Path, default=DEFAULT_GOLD_DB)
    parser.add_argument("--cncb-db", type=Path, default=DEFAULT_CNCB_DB)
    parser.add_argument("--phase", choices=("discovery", "metadata", "files", "accessions", "publications", "overlap", "reports", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Compatibility flag; resume is the default unless --refresh is used.")
    parser.add_argument("--refresh", action="store_true", help="Re-download cached MG-RAST project metadata.")
    parser.add_argument("--max-projects", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--mgrast-min-request-interval-seconds", type=float, default=2.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    # stream=sys.stdout, not the logging module's own stderr default -- the
    # sbatch wrapper redirects stdout/stderr to separate .out/.err files,
    # and every report section in this script already prints to stdout
    # (same real gap found and fixed in cncb_gsa_discovery.py: progress/
    # warning lines were landing unseen in the .err file).
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    config = MgrastConfig(
        db_path=args.db,
        data_dir=args.data_dir,
        mgnify_db_path=args.mgnify_db,
        qiita_db_path=args.qiita_db,
        gold_db_path=args.gold_db,
        cncb_db_path=args.cncb_db,
        page_size=args.page_size,
        min_request_interval_seconds=args.mgrast_min_request_interval_seconds,
    )
    runner = MgrastDiscoveryRunner(config, refresh=args.refresh)
    try:
        counts = runner.run(args.phase, max_projects=args.max_projects)
    finally:
        runner.close()

    print("\n=== MG-RAST discovery run counts ===")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
