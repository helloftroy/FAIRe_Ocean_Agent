from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import httpx

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi

logger = logging.getLogger(__name__)

CNCB_BASE_URL = "https://ngdc.cncb.ac.cn"
DEFAULT_CNCB_DB = Path("data/seed_discovery/cncb_gsa_paper_seeds.sqlite")
DEFAULT_CNCB_DATA_DIR = Path("data/cncb_gsa")
DEFAULT_MGNIFY_DB = Path("data/seed_discovery/mgnify_paper_seeds.sqlite")
DEFAULT_QIITA_DB = Path("data/seed_discovery/qiita_paper_seeds.sqlite")
DEFAULT_GOLD_DB = Path("data/jgi_gold/gold_sharded.sqlite")

CRA_RE = re.compile(r"\bCRA\d+\b", re.IGNORECASE)
PRJCA_RE = re.compile(r"\bPRJCA\d+\b", re.IGNORECASE)
SAMC_RE = re.compile(r"\bSAMC\d+\b", re.IGNORECASE)
CRX_RE = re.compile(r"\bCRX\d+\b", re.IGNORECASE)
CRR_RE = re.compile(r"\bCRR\d+\b", re.IGNORECASE)
INSDC_BIOPROJECT_RE = re.compile(r"\bPRJ(?:NA|EB|DB|DA|EA)\d+\b", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
PMID_RE = re.compile(r"\b(?:PMID|PubMed(?:\s+ID)?)[:\s]*([0-9]{5,})\b", re.IGNORECASE)

DISCOVERY_QUERIES = (
    "marine",
    "ocean",
    "seawater",
    '"sea water"',
    "coastal",
    "estuary",
    "estuarine",
    "brackish",
    '"marine sediment"',
    "sediment",
    "coral",
    "reef",
    "mangrove",
    '"salt marsh"',
    "pelagic",
    "benthic",
    "seafloor",
    "hydrothermal",
    '"deep sea"',
    '"sea ice"',
    '"continental shelf"',
    "intertidal",
    "subtidal",
    "metagenome",
    "metagenomic",
    "metatranscriptomic",
    "amplicon",
    '"environmental DNA"',
    '"environmental RNA"',
)

MARINE_TERMS = (
    "marine",
    "ocean",
    "seawater",
    "sea water",
    "coastal",
    "estuary",
    "estuarine",
    "brackish",
    "marine sediment",
    "coral",
    "reef",
    "mangrove",
    "salt marsh",
    "pelagic",
    "benthic",
    "seafloor",
    "hydrothermal",
    "deep sea",
    "sea ice",
    "continental shelf",
    "intertidal",
    "subtidal",
)
ENVIRONMENT_TERMS = (
    "environmental",
    "metagenome",
    "metagenomic",
    "metatranscriptomic",
    "microbiome",
    "microbial community",
    "amplicon",
    "16s",
    "18s",
    "its",
    "sediment",
    "soil",
    "water",
    "freshwater",
    "lake",
    "river",
)
REJECT_TERMS = (
    "human",
    "clinical",
    "patient",
    "tumor",
    "cancer",
    "blood",
    "cell line",
    "mouse",
    "mice",
    "rat ",
    "pig",
    "swine",
    "cattle",
    "bovine",
    "plant genome",
    "cultivar",
    "isolate genome",
    "whole genome resequencing",
)

SAMPLE_FIELD_ALIASES = {
    "sample_name": ("sample name", "sample_name", "name", "alias"),
    "sample_description": ("description", "sample description", "sample_description", "title"),
    "collection_date": ("collection date", "collection_date", "sampling date", "sample collection date", "date"),
    "latitude": ("latitude", "lat", "geo_loc_latitude"),
    "longitude": ("longitude", "lon", "lng", "geo_loc_longitude"),
    "depth": ("depth", "sample depth", "water depth"),
    "altitude": ("altitude", "elevation"),
    "geo_loc_name": ("geographic location", "geo_loc_name", "geo loc name", "location"),
    "country": ("country",),
    "ocean_region": ("ocean region", "marine region", "sea area"),
    "env_broad_scale": ("env_broad_scale", "environment biome", "biome"),
    "env_local_scale": ("env_local_scale", "environment feature", "feature"),
    "env_medium": ("env_medium", "environment material", "material", "isolation source"),
    "sample_type": ("sample type", "sample_type"),
    "habitat": ("habitat",),
    "sampling_method": ("sampling method", "sampling_method"),
    "sample_collection_method": ("sample collection method", "collection method"),
    "filter_type": ("filter type", "filter_type", "filter name"),
    "filter_pore_size": ("filter pore size", "pore size", "filter_size", "size fraction"),
    "size_fraction": ("size fraction", "size_fraction"),
    "storage_method": ("storage method", "storage condition"),
    "preservation_method": ("preservation method", "preservation"),
    "temperature": ("temperature", "water temperature", "temp"),
    "salinity": ("salinity",),
    "ph": ("ph", "pH"),
    "oxygen": ("oxygen",),
    "dissolved_oxygen": ("dissolved oxygen",),
    "chlorophyll": ("chlorophyll", "chlorophyll a"),
    "nitrate": ("nitrate",),
    "nitrite": ("nitrite",),
    "ammonium": ("ammonium", "ammonia"),
    "phosphate": ("phosphate",),
    "pressure": ("pressure",),
}

ENVIRONMENTAL_MEASUREMENT_KEYS = {
    "temperature",
    "salinity",
    "ph",
    "oxygen",
    "dissolved oxygen",
    "depth",
    "pressure",
    "chlorophyll",
    "nitrate",
    "nitrite",
    "ammonium",
    "ammonia",
    "phosphate",
    "conductivity",
    "organic carbon",
    "dissolved organic carbon",
    "particulate organic carbon",
    "turbidity",
    "sample volume",
}


def utc_iso() -> str:
    return utcnow().isoformat()


def json_dumps(value) -> str:  # noqa: ANN001
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize_doi_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return normalize_doi(value.strip().rstrip(".,;)</]"))
    except IdentifierError:
        return None


def html_to_text(raw_html: str) -> str:
    cleaned = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw_html)
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    return re.sub(r"[ \t\r\f\v]+", " ", html.unescape(cleaned)).strip()


def strip_tags(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_to_text(fragment)).strip()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def first_value(row: dict, aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize_key(key): key for key in row}
    for alias in aliases:
        key = normalized.get(normalize_key(alias))
        if key and row.get(key) not in (None, ""):
            return str(row[key]).strip()
    return None


def unique(values) -> list[str]:  # noqa: ANN001
    out = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class CncbConfig:
    db_path: Path = DEFAULT_CNCB_DB
    data_dir: Path = DEFAULT_CNCB_DATA_DIR
    mgnify_db_path: Path = DEFAULT_MGNIFY_DB
    qiita_db_path: Path = DEFAULT_QIITA_DB
    gold_db_path: Path = DEFAULT_GOLD_DB
    base_url: str = CNCB_BASE_URL
    min_request_interval_seconds: float = 2.0
    request_timeout_seconds: float = 120.0
    page_size: int = 50
    # db=bioproject's per-query totals run in the thousands (confirmed live
    # across all 28 DISCOVERY_QUERIES: max observed was "metagenome" at
    # ~65K, most terms far smaller), not the millions db=gsa's own mixed
    # INSDC-mirror index returns for the same terms -- a full scan to
    # exhaustion is tractable at this default, unlike the old 50-page cap
    # (2500 records) which found real native hits for the tested terms 0%
    # of the time against db=gsa's noise.
    max_pages_per_query: int = 300
    max_retries: int = 3
    retry_base_seconds: float = 2.0


class CncbDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 60000")

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cncb_projects (
                cncb_bioproject TEXT PRIMARY KEY,
                cra_accessions_json TEXT NOT NULL DEFAULT '[]',
                title TEXT,
                description TEXT,
                project_type TEXT,
                sequencing_strategy_json TEXT NOT NULL DEFAULT '[]',
                primary_doi TEXT,
                publication_dois_json TEXT NOT NULL DEFAULT '[]',
                pmids_json TEXT NOT NULL DEFAULT '[]',
                publication_resolution_status TEXT NOT NULL DEFAULT 'not_yet_processed',
                sample_count INTEGER NOT NULL DEFAULT 0,
                experiment_count INTEGER NOT NULL DEFAULT 0,
                run_count INTEGER NOT NULL DEFAULT 0,
                sequence_accessibility_status TEXT,
                marine_confidence TEXT,
                marine_match_methods_json TEXT NOT NULL DEFAULT '[]',
                insdc_bioprojects_json TEXT NOT NULL DEFAULT '[]',
                overlap_status TEXT NOT NULL DEFAULT 'not_checked',
                overlap_sources_json TEXT NOT NULL DEFAULT '{}',
                source_metadata_json TEXT NOT NULL DEFAULT '{}',
                first_seen_at TEXT NOT NULL,
                last_checked_at TEXT
            );

            CREATE TABLE IF NOT EXISTS cncb_samples (
                cncb_bioproject TEXT NOT NULL,
                cra_accession TEXT,
                samc_accession TEXT NOT NULL,
                sample_name TEXT,
                sample_description TEXT,
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
                sample_type TEXT,
                habitat TEXT,
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
                PRIMARY KEY (cncb_bioproject, samc_accession)
            );

            CREATE TABLE IF NOT EXISTS cncb_experiments (
                cncb_bioproject TEXT NOT NULL,
                cra_accession TEXT NOT NULL,
                crx_accession TEXT NOT NULL,
                crr_accessions_json TEXT NOT NULL DEFAULT '[]',
                samc_accession TEXT,
                experiment_title TEXT,
                species TEXT,
                library_strategy TEXT,
                library_source TEXT,
                library_selection TEXT,
                platform TEXT,
                instrument_model TEXT,
                layout TEXT,
                read_length TEXT,
                target_gene TEXT,
                target_subfragment TEXT,
                primer_forward TEXT,
                primer_reverse TEXT,
                library_protocol TEXT,
                extraction_protocol TEXT,
                pcr_protocol TEXT,
                file_names_json TEXT NOT NULL DEFAULT '[]',
                file_sizes_json TEXT NOT NULL DEFAULT '[]',
                checksums_json TEXT NOT NULL DEFAULT '[]',
                download_urls_json TEXT NOT NULL DEFAULT '[]',
                source_metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (cncb_bioproject, crx_accession)
            );

            CREATE VIEW IF NOT EXISTS paper_seeds AS
            SELECT
                'cncb_gsa' AS seed_source,
                COALESCE(json_extract(cra_accessions_json, '$[0]'), cncb_bioproject) AS source_study_id,
                COALESCE(json_extract(cra_accessions_json, '$[0]'), cncb_bioproject) AS source_project_id,
                CASE
                  WHEN json_array_length(insdc_bioprojects_json) > 0 THEN json_extract(insdc_bioprojects_json, '$[0]')
                  ELSE NULL
                END AS bioproject_accession,
                cncb_bioproject AS native_project_accession,
                title AS study_title,
                primary_doi,
                CASE WHEN json_array_length(pmids_json) > 0 THEN json_extract(pmids_json, '$[0]') ELSE NULL END AS primary_pmid,
                publication_dois_json,
                sequence_accessibility_status,
                marine_confidence,
                overlap_status,
                publication_resolution_status,
                first_seen_at,
                last_checked_at
            FROM cncb_projects
            WHERE marine_confidence IN ('high', 'medium', 'low');

            CREATE VIEW IF NOT EXISTS cncb_faire_sample_enrichment AS
            SELECT
                cncb_bioproject,
                cra_accession,
                samc_accession AS materialSampleID,
                sample_name AS samp_name,
                collection_date AS eventDate,
                latitude AS decimalLatitude,
                longitude AS decimalLongitude,
                depth,
                geo_loc_name,
                env_broad_scale,
                env_local_scale,
                env_medium,
                sample_collection_method,
                sampling_method,
                filter_type,
                filter_pore_size,
                size_fraction,
                temperature,
                salinity,
                ph,
                oxygen,
                dissolved_oxygen,
                chlorophyll,
                nitrate,
                nitrite,
                ammonium,
                phosphate
            FROM cncb_samples;

            CREATE VIEW IF NOT EXISTS cncb_faire_experiment_enrichment AS
            SELECT
                cncb_bioproject,
                cra_accession,
                crx_accession,
                crr_accessions_json,
                samc_accession,
                platform,
                instrument_model,
                library_strategy,
                target_gene,
                target_subfragment,
                primer_forward,
                primer_reverse,
                extraction_protocol,
                pcr_protocol,
                library_protocol,
                file_names_json,
                checksums_json,
                download_urls_json
            FROM cncb_experiments;
            """
        )
        self.conn.commit()

    def upsert_project(self, row: dict) -> None:
        now = utc_iso()
        row = dict(row)
        row.setdefault("first_seen_at", now)
        row.setdefault("last_checked_at", now)
        columns = [
            "cncb_bioproject", "cra_accessions_json", "title", "description", "project_type",
            "sequencing_strategy_json", "primary_doi", "publication_dois_json", "pmids_json",
            "publication_resolution_status", "sample_count", "experiment_count", "run_count",
            "sequence_accessibility_status", "marine_confidence", "marine_match_methods_json",
            "insdc_bioprojects_json", "overlap_status", "overlap_sources_json", "source_metadata_json",
            "first_seen_at", "last_checked_at",
        ]
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"cncb_bioproject", "first_seen_at"})
        self.conn.execute(
            f"""
            INSERT INTO cncb_projects({', '.join(columns)})
            VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(cncb_bioproject) DO UPDATE SET {assignments}
            """,
            [row.get(column) for column in columns],
        )

    def upsert_sample(self, row: dict) -> None:
        columns = [
            "cncb_bioproject", "cra_accession", "samc_accession", "sample_name", "sample_description",
            "collection_date", "latitude", "longitude", "depth", "altitude", "geo_loc_name", "country",
            "ocean_region", "env_broad_scale", "env_local_scale", "env_medium", "sample_type", "habitat",
            "sampling_method", "sample_collection_method", "filter_type", "filter_pore_size", "size_fraction",
            "storage_method", "preservation_method", "temperature", "salinity", "ph", "oxygen",
            "dissolved_oxygen", "chlorophyll", "nitrate", "nitrite", "ammonium", "phosphate", "pressure",
            "other_environmental_measurements_json", "source_metadata_json",
        ]
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"cncb_bioproject", "samc_accession"})
        self.conn.execute(
            f"""
            INSERT INTO cncb_samples({', '.join(columns)})
            VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(cncb_bioproject, samc_accession) DO UPDATE SET {assignments}
            """,
            [row.get(column) for column in columns],
        )

    def upsert_experiment(self, row: dict) -> None:
        columns = [
            "cncb_bioproject", "cra_accession", "crx_accession", "crr_accessions_json", "samc_accession",
            "experiment_title", "species", "library_strategy", "library_source", "library_selection",
            "platform", "instrument_model", "layout", "read_length", "target_gene", "target_subfragment",
            "primer_forward", "primer_reverse", "library_protocol", "extraction_protocol", "pcr_protocol",
            "file_names_json", "file_sizes_json", "checksums_json", "download_urls_json", "source_metadata_json",
        ]
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"cncb_bioproject", "crx_accession"})
        self.conn.execute(
            f"""
            INSERT INTO cncb_experiments({', '.join(columns)})
            VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(cncb_bioproject, crx_accession) DO UPDATE SET {assignments}
            """,
            [row.get(column) for column in columns],
        )

    def commit(self) -> None:
        self.conn.commit()


class CncbClient:
    def __init__(self, config: CncbConfig, *, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.last_request_at = 0.0
        self.client = httpx.Client(
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={
                "User-Agent": "fair-ocean-agent-cncb-gsa-seed-discovery/0.1 (mailto:parkmhelen@gmail.com)",
                "Accept": "application/json,text/html,*/*",
            },
        )

    def close(self) -> None:
        self.client.close()

    def get(self, url: str, *, params: dict | None = None) -> httpx.Response:
        for attempt in range(1, self.config.max_retries + 1):
            elapsed = time.monotonic() - self.last_request_at
            wait = self.config.min_request_interval_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.client.get(url, params=params)
                self.last_request_at = time.monotonic()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(int(retry_after))
                    raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                if attempt >= self.config.max_retries:
                    raise
                sleep_for = self.config.retry_base_seconds * attempt
                logger.warning("retrying CNCB request %s after %ss (%s)", url, sleep_for, exc)
                time.sleep(sleep_for)
        raise RuntimeError("unreachable retry loop")

    def search_bioprojects(self, query: str, *, start: int, size: int) -> dict:
        # Confirmed live (2026-08-26): db=gsa's own index is >99% INSDC-
        # mirrored Run/Experiment records -- a generic environmental term
        # like "amplicon" returns 18M total hits there, and a real
        # type=="GSA" (native project) record never surfaces within any
        # practical page budget (tested 3000 records deep for
        # amplicon/seawater/coral: zero found). db=bioproject is CNCB's own
        # dedicated BioProject index -- orders of magnitude smaller per
        # query (thousands, not millions) and each native record's own
        # attrs already carries Center=="GSA" plus its CrasAcc/SamplesAcc
        # cross-references directly, no HTML scrape needed just to find
        # them. See item_to_project_seed for the corresponding filter.
        response = self.get(
            urljoin(self.config.base_url, "/search/api/specific"),
            params={"q": query, "db": "bioproject", "size": size, "start": start},
        )
        return response.json()

    def gsa_html(self, cra_accession: str) -> str:
        return self.get(urljoin(self.config.base_url, f"/gsa/browse/{cra_accession}")).text

    def biosample_html(self, samc_accession: str) -> str:
        return self.get(urljoin(self.config.base_url, f"/biosample/browse/{samc_accession}")).text


def extract_search_items(payload: dict) -> tuple[int, list[dict]]:
    if str(payload.get("code")) != "200" or not isinstance(payload.get("result"), dict):
        return 0, []
    data = payload.get("result", {}).get("data", {})
    return int(data.get("recordsFiltered") or data.get("recordsTotal") or 0), list(data.get("data") or [])


def item_to_project_seed(item: dict) -> dict | None:
    attrs = item.get("attrs") or {}
    bioproject = (item.get("id") or attrs.get("Accession") or "").strip().upper()
    # attrs.Center distinguishes a native CNCB/GSA submission ("GSA") from
    # an INSDC-mirrored one (typically "SRA") within db=bioproject's mixed
    # index -- confirmed live against real records of both kinds. The
    # PRJCA_RE check is a belt-and-suspenders match on the accession's own
    # prefix, same defensive-double-check style as the old GSA-typed path.
    if item.get("type") != "BioProject" or attrs.get("Center") != "GSA" or not PRJCA_RE.match(bioproject):
        return None
    return {
        "cncb_bioproject": bioproject,
        "cra_accessions": unique(attrs.get("CrasAcc") or []),
        "title": item.get("title") or attrs.get("Title") or item.get("description"),
        "description": item.get("description") or attrs.get("Description"),
        "source": item,
    }


def classify_marine(project: dict, samples: list[dict] | None = None, experiments: list[dict] | None = None) -> tuple[str, list[str]]:
    chunks = [project.get("title"), project.get("description"), json_dumps(project.get("source") or project)]
    if samples:
        chunks.extend(json_dumps(sample) for sample in samples[:100])
    if experiments:
        chunks.extend(json_dumps(experiment) for experiment in experiments[:100])
    text = " ".join(chunk for chunk in chunks if chunk).casefold()
    methods = [f"marine_term:{term}" for term in MARINE_TERMS if term in text]
    if methods:
        return "high", methods[:30]
    env_hits = [term for term in ENVIRONMENT_TERMS if term in text]
    reject_hits = [term for term in REJECT_TERMS if term in text]
    if env_hits and not reject_hits:
        return "medium", [f"environment_term:{term}" for term in env_hits[:30]]
    if env_hits:
        return "not_marine", [f"environment_term:{term}" for term in env_hits[:10]] + [f"reject_context:{term}" for term in reject_hits[:20]]
    return "not_marine", [f"reject_context:{term}" for term in reject_hits[:20]]


def parse_label_text(raw_html: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for match in re.finditer(r"<b>\s*([^:<]+):\s*</b>\s*(.*?)(?=</span>|</div>|<br|</p>)", raw_html, re.IGNORECASE | re.DOTALL):
        key = strip_tags(match.group(1))
        value = strip_tags(match.group(2))
        if key and value:
            pairs[key] = value
    return pairs


def parse_gsa_html(cra_accession: str, raw_html: str) -> dict:
    labels = parse_label_text(raw_html)
    text = html_to_text(raw_html)
    prjcas = unique(PRJCA_RE.findall(raw_html))
    insdc_bioprojects = unique(INSDC_BIOPROJECT_RE.findall(raw_html))
    dois = sorted({doi for doi in (normalize_doi_or_none(m.group(0)) for m in DOI_RE.finditer(text)) if doi})
    pmids = sorted(set(PMID_RE.findall(text)))
    https_roots = sorted(set(re.findall(r"https://download\.cncb\.ac\.cn/[^\s\"'<>]+", raw_html)))
    ftp_roots = sorted(set(re.findall(r"ftp://download\.big\.ac\.cn/[^\s\"'<>]+", raw_html)))
    experiments = parse_experiments_from_gsa_html(cra_accession, raw_html, https_roots + ftp_roots)
    sample_accessions = unique([exp.get("samc_accession") for exp in experiments] + SAMC_RE.findall(raw_html))
    strategies = sorted({value for exp in experiments for value in (exp.get("library_strategy"), exp.get("target_gene")) if value})
    return {
        "cra_accession": cra_accession,
        "cncb_bioproject": prjcas[0] if prjcas else None,
        "cra_accessions": [cra_accession],
        "title": labels.get("标题") or labels.get("Title"),
        "description": labels.get("描述") or labels.get("Description"),
        "release_date": labels.get("发布日期") or labels.get("Release date"),
        "file_count": labels.get("文件个数"),
        "file_size": labels.get("文件大小"),
        "download_roots": https_roots + ftp_roots,
        "insdc_bioprojects": insdc_bioprojects,
        "publication_dois": dois,
        "pmids": pmids,
        "sample_accessions": sample_accessions,
        "experiments": experiments,
        "sequencing_strategy": strategies,
        "source": {"labels": labels, "html_text": text[:50000], "html_length": len(raw_html)},
    }


def parse_experiments_from_gsa_html(cra_accession: str, raw_html: str, download_roots: list[str]) -> list[dict]:
    parts = re.split(r'(?=<tr class="experiment">)', raw_html)
    experiments = []
    for part in parts:
        if '<tr class="experiment">' not in part:
            continue
        exp_block = part.split('<tr class="experiment">', 1)[1]
        exp_row = exp_block.split("</tr>", 1)[0]
        cells = re.findall(r"<td[^>]*>(.*?)</td>", exp_row, re.IGNORECASE | re.DOTALL)
        if len(cells) < 5:
            continue
        crx = next(iter(unique(CRX_RE.findall(cells[0]))), None)
        if not crx:
            continue
        title = strip_tags(cells[1])
        species = strip_tags(cells[2])
        platform = strip_tags(cells[3])
        samc = next(iter(unique(SAMC_RE.findall(cells[4]))), None)
        next_experiment = exp_block.split('<tr class="experiment">', 1)[0]
        run_rows = re.findall(r'<tr class="runTr">(.*?)</tr>', next_experiment, re.IGNORECASE | re.DOTALL)
        crrs = []
        file_names = []
        for run_row in run_rows:
            crrs.extend(CRR_RE.findall(run_row))
            file_names.extend(re.findall(r"<strong>\s*File:\s*</strong>\s*([^<\s]+)", run_row, re.IGNORECASE))
        crrs = unique(crrs)
        file_names = sorted(set(file_names))
        download_urls = []
        for root in download_roots:
            root = root.rstrip("/")
            download_urls.extend(f"{root}/{name}" for name in file_names)
        target_gene, target_subfragment = infer_marker_from_text(f"{title} {species}")
        layout = infer_layout(file_names)
        return_row = {
            "cncb_bioproject": None,
            "cra_accession": cra_accession,
            "crx_accession": crx,
            "crr_accessions_json": json_dumps(crrs),
            "samc_accession": samc,
            "experiment_title": title,
            "species": species,
            "platform": platform,
            "instrument_model": platform,
            "layout": layout,
            "target_gene": target_gene,
            "target_subfragment": target_subfragment,
            "file_names_json": json_dumps(file_names),
            "file_sizes_json": "[]",
            "checksums_json": "[]",
            "download_urls_json": json_dumps(download_urls),
            "source_metadata_json": json_dumps({"experiment_row": strip_tags(exp_row), "run_rows": [strip_tags(row) for row in run_rows]}),
        }
        experiments.append(return_row)
    return experiments


def infer_marker_from_text(text: str) -> tuple[str | None, str | None]:
    lowered = text.casefold()
    target_gene = None
    if "16s" in lowered or any(primer in lowered for primer in ("338f", "341f", "515f", "806r", "805r", "785r")):
        target_gene = "16S rRNA"
    elif "18s" in lowered:
        target_gene = "18S rRNA"
    elif "its" in lowered:
        target_gene = "ITS"
    elif "coi" in lowered:
        target_gene = "COI"
    region = None
    for pattern in (r"\bV[0-9](?:-V?[0-9])?\b", r"\bITS[12]\b"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            region = match.group(0)
            break
    return target_gene, region


def infer_layout(file_names: list[str]) -> str | None:
    joined = " ".join(file_names).casefold()
    if re.search(r"(_r?1|_1)\.f", joined) and re.search(r"(_r?2|_2)\.f", joined):
        return "paired end"
    if file_names:
        return "single end"
    return None


def parse_biosample_html(samc_accession: str, raw_html: str) -> dict:
    labels = parse_label_text(raw_html)
    text = html_to_text(raw_html)
    attrs = dict(labels)
    # BioSample pages also have many table-style key/value rows. This broad
    # pass preserves them without requiring a fragile page-specific DOM parser.
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", raw_html, re.IGNORECASE | re.DOTALL):
        cells = [strip_tags(cell) for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)]
        if len(cells) >= 2 and cells[0] and cells[1]:
            attrs.setdefault(cells[0].rstrip(":"), cells[1])
    source = {"labels": labels, "attributes": attrs, "html_text": text[:30000], "html_length": len(raw_html)}
    out = {"samc_accession": samc_accession, "source": source}
    for field, aliases in SAMPLE_FIELD_ALIASES.items():
        out[field] = first_value(attrs, aliases)
    if out.get("sample_name") is None:
        out["sample_name"] = samc_accession
    if out.get("sample_description") is None:
        out["sample_description"] = labels.get("Description") or labels.get("描述")
    other = {}
    for key, value in attrs.items():
        norm = normalize_key(key)
        if any(token in norm for token in ENVIRONMENTAL_MEASUREMENT_KEYS) and value:
            other[key] = value
    out["other_environmental_measurements_json"] = json_dumps(other)
    out["source_metadata_json"] = json_dumps(source)
    return out


def project_row_from_seed(seed: dict) -> dict:
    confidence, methods = classify_marine(seed)
    cra_accessions = seed.get("cra_accessions") or []
    return {
        "cncb_bioproject": seed["cncb_bioproject"],
        "cra_accessions_json": json_dumps(cra_accessions),
        "title": seed.get("title"),
        "description": seed.get("description"),
        "project_type": None,
        "sequencing_strategy_json": "[]",
        "primary_doi": None,
        "publication_dois_json": "[]",
        "pmids_json": "[]",
        "publication_resolution_status": "not_yet_processed",
        "sample_count": 0,
        "experiment_count": 0,
        "run_count": 0,
        # A native BioProject with no CrasAcc at all (e.g. a metabolome-only
        # submission, confirmed live) has nothing for the metadata phase to
        # ever scrape -- known upfront rather than left "not_yet_checked"
        # forever.
        "sequence_accessibility_status": "not_yet_checked" if cra_accessions else "no_downloadable_raw_data",
        "marine_confidence": confidence,
        "marine_match_methods_json": json_dumps(methods),
        "insdc_bioprojects_json": "[]",
        "overlap_status": "not_checked",
        "overlap_sources_json": "{}",
        "source_metadata_json": json_dumps(seed),
    }


def project_row_from_gsa(parsed: dict, existing_seed: dict | None = None) -> dict:
    seed = existing_seed or {}
    cncb_bioproject = parsed.get("cncb_bioproject") or seed.get("cncb_bioproject")
    experiments = parsed.get("experiments") or []
    sample_accessions = parsed.get("sample_accessions") or []
    crr_count = sum(len(json.loads(exp.get("crr_accessions_json") or "[]")) for exp in experiments)
    dois = parsed.get("publication_dois") or []
    pmids = parsed.get("pmids") or []
    confidence, methods = classify_marine({**seed, **parsed}, experiments=experiments)
    status = "gsa_files_listed" if crr_count or parsed.get("download_roots") else "no_downloadable_raw_data"
    return {
        "cncb_bioproject": cncb_bioproject,
        "cra_accessions_json": json_dumps(unique([*(json.loads(seed.get("cra_accessions_json", "[]")) if seed.get("cra_accessions_json") else []), parsed["cra_accession"]])),
        "title": parsed.get("title") or seed.get("title"),
        "description": parsed.get("description") or seed.get("description"),
        "project_type": None,
        "sequencing_strategy_json": json_dumps(parsed.get("sequencing_strategy") or []),
        "primary_doi": dois[0] if dois else None,
        "publication_dois_json": json_dumps(dois),
        "pmids_json": json_dumps(pmids),
        "publication_resolution_status": "resolved" if dois else ("pmid_only" if pmids else "missing"),
        "sample_count": len(sample_accessions),
        "experiment_count": len(experiments),
        "run_count": crr_count,
        "sequence_accessibility_status": status,
        "marine_confidence": confidence,
        "marine_match_methods_json": json_dumps(methods),
        "insdc_bioprojects_json": json_dumps(parsed.get("insdc_bioprojects") or []),
        "overlap_status": "not_checked",
        "overlap_sources_json": "{}",
        "source_metadata_json": json_dumps({"seed": seed, "gsa": parsed}),
    }


def sample_row_to_db(cncb_bioproject: str, cra_accession: str | None, sample: dict) -> dict:
    row = {"cncb_bioproject": cncb_bioproject, "cra_accession": cra_accession, "samc_accession": sample["samc_accession"]}
    for field in [
        "sample_name", "sample_description", "collection_date", "latitude", "longitude", "depth", "altitude",
        "geo_loc_name", "country", "ocean_region", "env_broad_scale", "env_local_scale", "env_medium",
        "sample_type", "habitat", "sampling_method", "sample_collection_method", "filter_type", "filter_pore_size",
        "size_fraction", "storage_method", "preservation_method", "temperature", "salinity", "ph", "oxygen",
        "dissolved_oxygen", "chlorophyll", "nitrate", "nitrite", "ammonium", "phosphate", "pressure",
        "other_environmental_measurements_json", "source_metadata_json",
    ]:
        row[field] = sample.get(field)
    row.setdefault("other_environmental_measurements_json", "{}")
    row.setdefault("source_metadata_json", json_dumps(sample))
    return row


def update_overlap(db: CncbDB, config: CncbConfig) -> Counter:
    counts: Counter = Counter()
    mgnify_by_bioproject, mgnify_by_doi = load_mgnify_overlap(config.mgnify_db_path)
    qiita_by_bioproject, qiita_by_doi = load_qiita_overlap(config.qiita_db_path)
    gold_by_bioproject, gold_by_doi = load_gold_overlap(config.gold_db_path)
    for row in db.conn.execute("SELECT * FROM cncb_projects").fetchall():
        overlaps: dict[str, list[str]] = {"ena": [], "mgnify": [], "qiita": [], "jgi": []}
        insdc = json.loads(row["insdc_bioprojects_json"] or "[]")
        dois = json.loads(row["publication_dois_json"] or "[]")
        for accession in insdc:
            if accession in mgnify_by_bioproject:
                overlaps["mgnify"].append(mgnify_by_bioproject[accession])
                overlaps["ena"].append(accession)
            if accession in qiita_by_bioproject:
                overlaps["qiita"].append(qiita_by_bioproject[accession])
            if accession in gold_by_bioproject:
                overlaps["jgi"].append(gold_by_bioproject[accession])
        for doi in dois:
            if doi in mgnify_by_doi:
                overlaps["mgnify"].append(mgnify_by_doi[doi])
            if doi in qiita_by_doi:
                overlaps["qiita"].append(qiita_by_doi[doi])
            if doi in gold_by_doi:
                overlaps["jgi"].append(gold_by_doi[doi])
        overlaps = {key: sorted(set(values)) for key, values in overlaps.items()}
        status = "net_new_project" if not any(overlaps.values()) else "existing_project_with_new_metadata"
        for key, values in overlaps.items():
            if values:
                counts[f"overlap_{key}"] += 1
        db.conn.execute(
            "UPDATE cncb_projects SET overlap_status = ?, overlap_sources_json = ?, last_checked_at = ? WHERE cncb_bioproject = ?",
            (status, json_dumps(overlaps), utc_iso(), row["cncb_bioproject"]),
        )
    db.commit()
    return counts


def load_mgnify_overlap(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        by_bioproject = {
            row["bioproject_accession"]: row["mgnify_accession"]
            for row in conn.execute("SELECT mgnify_accession, bioproject_accession FROM mgnify_studies WHERE bioproject_accession IS NOT NULL")
        }
        by_doi = {
            (row["normalized_doi"] or row["doi"]): row["mgnify_accession"]
            for row in conn.execute(
                """
                SELECT pc.normalized_doi, pc.doi, ms.mgnify_accession
                FROM publication_candidates pc
                JOIN mgnify_studies ms ON ms.id = pc.mgnify_study_id
                WHERE pc.doi IS NOT NULL
                """
            )
        }
    finally:
        conn.close()
    return by_bioproject, by_doi


def load_qiita_overlap(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        by_bioproject = {
            row["primary_bioproject"]: row["qiita_study_id"]
            for row in conn.execute("SELECT qiita_study_id, primary_bioproject FROM qiita_studies WHERE primary_bioproject IS NOT NULL")
        }
        by_doi = {}
        for row in conn.execute("SELECT qiita_study_id, publication_dois_json FROM qiita_studies"):
            for doi in json.loads(row["publication_dois_json"] or "[]"):
                by_doi[doi] = row["qiita_study_id"]
    except sqlite3.Error:
        return {}, {}
    finally:
        conn.close()
    return by_bioproject, by_doi


def load_gold_overlap(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        by_bioproject = {
            row["ncbi_bioproject_accession"]: row["gold_study_id"]
            for row in conn.execute("SELECT gold_study_id, ncbi_bioproject_accession FROM gold_sequencing_projects WHERE ncbi_bioproject_accession IS NOT NULL")
        }
        by_doi = {
            row["primary_doi"]: row["gold_study_id"]
            for row in conn.execute("SELECT gold_study_id, primary_doi FROM gold_studies WHERE primary_doi IS NOT NULL")
        }
    except sqlite3.Error:
        return {}, {}
    finally:
        conn.close()
    return by_bioproject, by_doi


class CncbGsaDiscoveryRunner:
    def __init__(self, config: CncbConfig, *, transport: httpx.BaseTransport | None = None):
        self.config = config
        self.db = CncbDB(config.db_path)
        self.client = CncbClient(config, transport=transport)

    def close(self) -> None:
        self.client.close()
        self.db.close()

    def run(self, *, phase: str = "all", max_projects: int | None = None, refresh: bool = False) -> dict:
        self.db.initialize()
        raw_dir = self.config.data_dir / "raw"
        reports_dir = self.config.data_dir / "reports"
        manifests_dir = self.config.data_dir / "file_manifests"
        cache_dir = self.config.data_dir / "cache"
        for path in (raw_dir, reports_dir, manifests_dir, cache_dir):
            path.mkdir(parents=True, exist_ok=True)
        counts: Counter = Counter()
        if phase in {"all", "discovery"}:
            self._discover(raw_dir, counts, max_projects=max_projects, refresh=refresh)
        if phase in {"all", "metadata", "accessions", "files", "publications"}:
            self._metadata(raw_dir, manifests_dir, counts, max_projects=max_projects, refresh=refresh)
        if phase in {"all", "overlap"}:
            counts.update(update_overlap(self.db, self.config))
        if phase in {"all", "reports"}:
            write_reports(self.db, reports_dir)
        write_output_locations(self.config, reports_dir, manifests_dir)
        print_cncb_report(self.config)
        return {"counts": dict(counts), "database": str(self.config.db_path)}

    def _discover(self, raw_dir: Path, counts: Counter, *, max_projects: int | None, refresh: bool) -> None:
        seen_projects: set[str] = {row["cncb_bioproject"] for row in self.db.conn.execute("SELECT cncb_bioproject FROM cncb_projects")}
        for query in DISCOVERY_QUERIES:
            start = 0
            pages = 0
            while True:
                if max_projects is not None and len(seen_projects) >= max_projects:
                    self.db.commit()
                    return
                # "bioproject" subdir (not the old flat "_search/") is
                # deliberate: real live bug found 2026-08-26 -- a cache key
                # of query+start alone survived the db=gsa -> db=bioproject
                # fix below unchanged, so a re-run against an existing raw_
                # dir kept silently replaying the OLD db=gsa responses
                # cached under the old path instead of ever re-querying the
                # live API. Namespacing the cache path by which index is
                # being searched means any future change to what's being
                # asked for automatically busts stale cache instead of
                # silently masking itself as "no new results found."
                cache_path = raw_dir / "_search" / "bioproject" / f"{re.sub(r'[^a-zA-Z0-9]+', '_', query).strip('_')}_{start}.json"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                if cache_path.exists() and not refresh:
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                else:
                    try:
                        payload = self.client.search_bioprojects(query, start=start, size=self.config.page_size)
                    except httpx.HTTPError as exc:
                        counts["discover_query_errors"] += 1
                        logger.warning("CNCB search failed for query=%s start=%s, moving to next query: %s", query, start, exc)
                        break
                    cache_path.write_text(json_dumps(payload), encoding="utf-8")
                total, items = extract_search_items(payload)
                counts["search_pages"] += 1
                for item in items:
                    seed = item_to_project_seed(item)
                    if seed is None:
                        continue
                    if max_projects is not None and len(seen_projects) >= max_projects and seed["cncb_bioproject"] not in seen_projects:
                        break
                    self.db.upsert_project(project_row_from_seed(seed))
                    seen_projects.add(seed["cncb_bioproject"])
                    counts["native_projects_seen"] += 1
                self.db.commit()
                pages += 1
                if len(items) < self.config.page_size or start + self.config.page_size >= total or pages >= self.config.max_pages_per_query:
                    break
                start += self.config.page_size
            logger.info("CNCB discovery query=%s projects=%s", query, len(seen_projects))

    def _metadata(self, raw_dir: Path, manifests_dir: Path, counts: Counter, *, max_projects: int | None, refresh: bool) -> None:
        rows = self.db.conn.execute("SELECT * FROM cncb_projects ORDER BY cncb_bioproject").fetchall()
        if max_projects is not None:
            rows = rows[:max_projects]
        for index, row in enumerate(rows, start=1):
            cra_accessions = json.loads(row["cra_accessions_json"] or "[]")
            all_experiments = []
            all_samples = []
            for cra in cra_accessions:
                project_dir = raw_dir / row["cncb_bioproject"]
                project_dir.mkdir(parents=True, exist_ok=True)
                html_path = project_dir / f"{cra}.html"
                if html_path.exists() and not refresh:
                    gsa_html = html_path.read_text(encoding="utf-8")
                else:
                    try:
                        gsa_html = self.client.gsa_html(cra)
                    except httpx.HTTPError as exc:
                        counts["metadata_cra_errors"] += 1
                        logger.warning("CNCB GSA page fetch failed for %s, skipping this run: %s", cra, exc)
                        continue
                    html_path.write_text(gsa_html, encoding="utf-8")
                parsed = parse_gsa_html(cra, gsa_html)
                merged_row = project_row_from_gsa(parsed, dict(row))
                self.db.upsert_project(merged_row)
                experiments = parsed.get("experiments") or []
                manifest_files = []
                for experiment in experiments:
                    experiment["cncb_bioproject"] = merged_row["cncb_bioproject"]
                    self.db.upsert_experiment(experiment)
                    for file_name, url in zip(json.loads(experiment["file_names_json"]), json.loads(experiment["download_urls_json"])):
                        manifest_files.append(
                            {
                                "cncb_bioproject": merged_row["cncb_bioproject"],
                                "cra_accession": cra,
                                "crx_accession": experiment["crx_accession"],
                                "crr_accessions": json.loads(experiment["crr_accessions_json"]),
                                "samc_accession": experiment.get("samc_accession"),
                                "filename": file_name,
                                "download_url": url,
                                "access_status": "locator_recorded_not_downloaded",
                            }
                        )
                (manifests_dir / f"{merged_row['cncb_bioproject']}.json").write_text(json_dumps({"files": manifest_files}), encoding="utf-8")
                all_experiments.extend(experiments)
                for samc in parsed.get("sample_accessions") or []:
                    sample_html_path = project_dir / f"{samc}.html"
                    if sample_html_path.exists() and not refresh:
                        sample_html = sample_html_path.read_text(encoding="utf-8")
                    else:
                        try:
                            sample_html = self.client.biosample_html(samc)
                        except httpx.HTTPError as exc:
                            logger.info("no CNCB BioSample page for %s: %s", samc, exc)
                            sample_html = ""
                        sample_html_path.write_text(sample_html, encoding="utf-8")
                    sample = parse_biosample_html(samc, sample_html) if sample_html else {"samc_accession": samc}
                    self.db.upsert_sample(sample_row_to_db(merged_row["cncb_bioproject"], cra, sample))
                    all_samples.append(sample)
                confidence, methods = classify_marine(merged_row, all_samples, all_experiments)
                self.db.conn.execute(
                    "UPDATE cncb_projects SET marine_confidence = ?, marine_match_methods_json = ?, last_checked_at = ? WHERE cncb_bioproject = ?",
                    (confidence, json_dumps(methods), utc_iso(), merged_row["cncb_bioproject"]),
                )
                counts["projects_with_metadata"] += 1
            self.db.commit()
            if index % 25 == 0:
                logger.info("CNCB metadata progress projects=%s/%s", index, len(rows))


def write_reports(db: CncbDB, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_specs = {
        "new_projects.csv": "SELECT * FROM cncb_projects WHERE overlap_status = 'net_new_project' ORDER BY cncb_bioproject",
        "existing_projects_enriched.csv": "SELECT * FROM cncb_projects WHERE overlap_status != 'net_new_project' AND overlap_status != 'not_checked' ORDER BY cncb_bioproject",
        "source_overlap.csv": "SELECT cncb_bioproject, cra_accessions_json, primary_doi, insdc_bioprojects_json, overlap_status, overlap_sources_json FROM cncb_projects ORDER BY cncb_bioproject",
        "unresolved_publications.csv": "SELECT * FROM cncb_projects WHERE publication_resolution_status IN ('missing', 'not_yet_processed', 'ambiguous') ORDER BY cncb_bioproject",
        "unresolved_accessions.csv": "SELECT * FROM cncb_projects WHERE json_array_length(cra_accessions_json) = 0 ORDER BY cncb_bioproject",
    }
    for filename, sql in report_specs.items():
        write_rows_csv(reports_dir / filename, db.conn.execute(sql).fetchall())
    write_metadata_completeness(db, reports_dir / "metadata_completeness.csv")


def write_rows_csv(path: Path, rows: list[sqlite3.Row]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_metadata_completeness(db: CncbDB, path: Path) -> None:
    metrics = {
        "projects_scanned": "SELECT count(*) FROM cncb_projects",
        "marine_environment_projects": "SELECT count(*) FROM cncb_projects WHERE marine_confidence IN ('high', 'medium', 'low')",
        "with_doi": "SELECT count(*) FROM cncb_projects WHERE primary_doi IS NOT NULL",
        "with_pmid": "SELECT count(*) FROM cncb_projects WHERE json_array_length(pmids_json) > 0",
        "with_cra": "SELECT count(*) FROM cncb_projects WHERE json_array_length(cra_accessions_json) > 0",
        "with_samc_samples": "SELECT count(DISTINCT cncb_bioproject) FROM cncb_samples",
        "with_crx": "SELECT count(DISTINCT cncb_bioproject) FROM cncb_experiments",
        "with_crr": "SELECT count(DISTINCT cncb_bioproject) FROM cncb_experiments WHERE json_array_length(crr_accessions_json) > 0",
        "with_downloadable_raw_data": "SELECT count(*) FROM cncb_projects WHERE sequence_accessibility_status IN ('gsa_raw_reads_confirmed', 'gsa_files_listed', 'gsa_and_insdc')",
        "with_insdc_bioproject_crosslink": "SELECT count(*) FROM cncb_projects WHERE json_array_length(insdc_bioprojects_json) > 0",
        "net_new": "SELECT count(*) FROM cncb_projects WHERE overlap_status = 'net_new_project'",
        "samples_with_collection_date": "SELECT count(*) FROM cncb_samples WHERE collection_date IS NOT NULL",
        "samples_with_lat_lon": "SELECT count(*) FROM cncb_samples WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
        "samples_with_depth": "SELECT count(*) FROM cncb_samples WHERE depth IS NOT NULL",
        "samples_with_environment_context": "SELECT count(*) FROM cncb_samples WHERE env_broad_scale IS NOT NULL OR env_local_scale IS NOT NULL OR env_medium IS NOT NULL",
        "samples_with_temperature": "SELECT count(*) FROM cncb_samples WHERE temperature IS NOT NULL",
        "samples_with_salinity": "SELECT count(*) FROM cncb_samples WHERE salinity IS NOT NULL",
        "samples_with_ph": "SELECT count(*) FROM cncb_samples WHERE ph IS NOT NULL",
        "samples_with_oxygen": "SELECT count(*) FROM cncb_samples WHERE oxygen IS NOT NULL OR dissolved_oxygen IS NOT NULL",
        "experiments_with_platform": "SELECT count(*) FROM cncb_experiments WHERE platform IS NOT NULL OR instrument_model IS NOT NULL",
        "experiments_with_target_gene": "SELECT count(*) FROM cncb_experiments WHERE target_gene IS NOT NULL",
        "experiments_with_primers": "SELECT count(*) FROM cncb_experiments WHERE primer_forward IS NOT NULL OR primer_reverse IS NOT NULL",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "count"])
        for metric, sql in metrics.items():
            writer.writerow([metric, db.conn.execute(sql).fetchone()[0]])


def write_output_locations(config: CncbConfig, reports_dir: Path, manifests_dir: Path) -> None:
    payload = {
        "database": str(config.db_path),
        "tables": ["cncb_projects", "cncb_samples", "cncb_experiments"],
        "paper_seeds_view": "paper_seeds",
        "raw_data_dir": str((config.data_dir / "raw").resolve()),
        "file_manifest_dir": str(manifests_dir.resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "seed_csv": "cluster/seeds_cncb_gsa.csv",
        "mgnify_database": str(config.mgnify_db_path) if config.mgnify_db_path.exists() else "not available",
        "qiita_database": str(config.qiita_db_path) if config.qiita_db_path.exists() else "not available",
        "gold_database": str(config.gold_db_path) if config.gold_db_path.exists() else "not available",
    }
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "OUTPUT_LOCATIONS.json").write_text(json_dumps(payload), encoding="utf-8")


def print_cncb_report(config: CncbConfig) -> None:
    db = CncbDB(config.db_path)
    try:
        db.initialize()

        def scalar(sql: str) -> int:
            return int(db.conn.execute(sql).fetchone()[0])
        marine_sql = "SELECT count(*) FROM cncb_projects WHERE marine_confidence IN ('high','medium','low')"
        raw_data_sql = (
            "SELECT count(*) FROM cncb_projects WHERE sequence_accessibility_status IN "
            "('gsa_raw_reads_confirmed','gsa_files_listed','gsa_and_insdc')"
        )
        net_new_sql = "SELECT count(*) FROM cncb_projects WHERE overlap_status = 'net_new_project'"

        print("=" * 60)
        print("CNCB / GSA DISCOVERY COMPLETE")
        print("=" * 60)
        print("\nDATABASE")
        print(config.db_path)
        print("\nTABLES")
        print("  cncb_projects")
        print("  cncb_samples")
        print("  cncb_experiments")
        print("\nPAPER SEEDS VIEW")
        print("paper_seeds")
        print("\nRAW SOURCE DATA")
        print((config.data_dir / "raw").resolve())
        print("\nFILE MANIFESTS")
        print((config.data_dir / "file_manifests").resolve())
        print("\nREPORTS")
        print((config.data_dir / "reports").resolve())
        print("\nSEED CSV")
        print("cluster/seeds_cncb_gsa.csv")
        print("\n" + "=" * 60)
        print(f"NATIVE CNCB PROJECTS SCANNED: {scalar('SELECT count(*) FROM cncb_projects')}")
        print(f"MARINE/ENVIRONMENT PROJECTS: {scalar(marine_sql)}")
        print(f"PROJECTS WITH DOI: {scalar('SELECT count(*) FROM cncb_projects WHERE primary_doi IS NOT NULL')}")
        print(f"PROJECTS WITH RAW DATA: {scalar(raw_data_sql)}")
        print(f"PROJECTS WITH PRJCA: {scalar('SELECT count(*) FROM cncb_projects WHERE cncb_bioproject IS NOT NULL')}")
        print(f"PROJECTS WITH CRA: {scalar('SELECT count(*) FROM cncb_projects WHERE json_array_length(cra_accessions_json) > 0')}")
        print(f"PROJECTS WITH SAMC: {scalar('SELECT count(DISTINCT cncb_bioproject) FROM cncb_samples')}")
        print(f"PROJECTS WITH CRX: {scalar('SELECT count(DISTINCT cncb_bioproject) FROM cncb_experiments')}")
        print(f"PROJECTS WITH CRR: {scalar('SELECT count(DISTINCT cncb_bioproject) FROM cncb_experiments WHERE json_array_length(crr_accessions_json) > 0')}")
        print(f"PROJECTS WITH INSDC CROSS-LINK: {scalar('SELECT count(*) FROM cncb_projects WHERE json_array_length(insdc_bioprojects_json) > 0')}")
        for source in ("ena", "mgnify", "qiita", "jgi"):
            overlap_sql = f"SELECT count(*) FROM cncb_projects WHERE overlap_sources_json LIKE '%\"{source}\": [\"%'"
            print(f"OVERLAP {source.upper()}: {scalar(overlap_sql)}")
        print(f"NET-NEW PROJECTS: {scalar(net_new_sql)}")
        print("=" * 60)
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover native CNCB/GSA environmental/marine study seeds.")
    parser.add_argument("--db", default=str(DEFAULT_CNCB_DB))
    parser.add_argument("--data-dir", default=str(DEFAULT_CNCB_DATA_DIR))
    parser.add_argument("--mgnify-db", default=str(DEFAULT_MGNIFY_DB))
    parser.add_argument("--qiita-db", default=str(DEFAULT_QIITA_DB))
    parser.add_argument("--gold-db", default=str(DEFAULT_GOLD_DB))
    parser.add_argument("--phase", choices=("all", "discovery", "metadata", "accessions", "publications", "files", "overlap", "reports"), default="all")
    parser.add_argument("--resume", action="store_true", help="Accepted for parity; cached raw files are reused by default.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-projects", type=int)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages-per-query", type=int, default=int(os.environ.get("CNCB_MAX_PAGES_PER_QUERY", "300")))
    parser.add_argument("--cncb-min-request-interval-seconds", type=float, default=float(os.environ.get("CNCB_MIN_REQUEST_INTERVAL_SECONDS", "2.0")))
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = CncbConfig(
        db_path=Path(args.db),
        data_dir=Path(args.data_dir),
        mgnify_db_path=Path(args.mgnify_db),
        qiita_db_path=Path(args.qiita_db),
        gold_db_path=Path(args.gold_db),
        min_request_interval_seconds=args.cncb_min_request_interval_seconds,
        page_size=args.page_size,
        max_pages_per_query=args.max_pages_per_query,
    )
    runner = CncbGsaDiscoveryRunner(config)
    try:
        runner.run(phase=args.phase, max_projects=args.max_projects, refresh=args.refresh)
    finally:
        runner.close()
    return 0
