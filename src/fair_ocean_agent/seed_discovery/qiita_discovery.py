from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from zipfile import ZipFile

import httpx

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.identity.identifiers import IdentifierError, normalize_doi

logger = logging.getLogger(__name__)

QIITA_BASE_URL = "https://qiita.ucsd.edu"
DEFAULT_QIITA_DB = Path("data/seed_discovery/qiita_paper_seeds.sqlite")
DEFAULT_QIITA_DATA_DIR = Path("data/qiita")
DEFAULT_MGNIFY_DB = Path("data/seed_discovery/mgnify_paper_seeds.sqlite")

STUDY_ID_RE = re.compile(r"_generate_iconFeature\((\d+),")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
PMID_RE = re.compile(r"\b(?:PMID|PubMed(?:\s+ID)?)[:\s]*([0-9]{5,})\b", re.IGNORECASE)
ACCESSION_RE = re.compile(
    r"\b(?:PRJ(?:NA|EB|DB)\d+|ERP\d+|SRP\d+|DRP\d+|[DES]RP\d+|[DES]RR\d+|[DES]RX\d+|SAMN\d+|SAMEA\d+|SAMD\d+)\b",
    re.IGNORECASE,
)
ENA_STUDY_RE = re.compile(r"\b(?:ERP\d+|SRP\d+|DRP\d+|PRJEB\d+|PRJNA\d+|PRJDB\d+)\b", re.IGNORECASE)
BIOPROJECT_RE = re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.IGNORECASE)
BIOSAMPLE_RE = re.compile(r"\b(?:SAMN|SAMEA|SAMD)\d+\b", re.IGNORECASE)

MARINE_TERMS = (
    "marine",
    "ocean",
    "seawater",
    "sea water",
    "coastal",
    "estuary",
    "estuarine",
    "brackish",
    "mangrove",
    "salt marsh",
    "reef",
    "coral",
    "marine sediment",
    "seafloor",
    "pelagic",
    "benthic",
    "intertidal",
    "subtidal",
    "sea ice",
    "hydrothermal",
    "continental shelf",
)

ENVIRONMENT_TERMS = (
    "environmental",
    "soil",
    "sediment",
    "water",
    "freshwater",
    "lake",
    "river",
    "stream",
    "wetland",
    "microbiome",
    "microbial community",
    "metagenom",
    "metatranscript",
    "amplicon",
)

REJECT_TERMS = (
    "human gut",
    "human microbiome",
    "human skin",
    "human oral",
    "human vaginal",
    "human fecal",
    "gut microbiome",
    "obese",
    "lean twins",
    "human stool",
    "built environment",
    "indoor environment",
    "clinical",
    "patient",
    "neonatal",
    "infant",
    "animal gut",
    "animal microbiome",
    "mouse",
    "mice",
    "rat ",
    "swine",
    "cattle",
    "bovine",
)

FIELD_CANDIDATES = {
    "sample_id": ("sample_name", "sample id", "sampleid", "sample"),
    "biosample": ("biosample", "biosample_accession", "ebi_sample_accession", "ncbi_biosample_accession"),
    "collection_date": ("collection_date", "collection date", "sample collection date", "date", "sampling date"),
    "latitude": ("latitude", "lat", "collection_timestamp", "geo_loc_latitude"),
    "longitude": ("longitude", "lon", "lng", "geo_loc_longitude"),
    "depth": ("depth", "sample depth", "water depth", "env_package_depth"),
    "geo_loc_name": ("geo_loc_name", "geographic location", "country", "country/ocean", "location"),
    "env_broad_scale": ("env_broad_scale", "environment biome", "empo_1", "empo_2"),
    "env_local_scale": ("env_local_scale", "environment feature", "empo_3"),
    "env_medium": ("env_medium", "environment material", "sample_type", "qiita_study_type"),
    "sampling_method": ("sampling_method", "sample collection method", "collection method"),
    "size_fraction": ("size_fraction", "filter pore size", "filter_size", "filtration", "size fraction"),
    "temperature": ("temperature", "temp", "water temperature"),
    "salinity": ("salinity",),
    "ph": ("ph", "pH"),
    "oxygen": ("oxygen", "dissolved oxygen"),
    "data_type": ("data_type", "data type", "target_gene", "investigation_type"),
    "platform": ("platform", "instrument", "sequencing platform", "instrument_model"),
    "target_gene": ("target_gene", "target gene", "gene", "target_subfragment"),
    "target_subfragment": ("target_subfragment", "target subfragment", "region", "target_region"),
    "primer_forward": ("primer", "forward_primer", "primer_forward", "fwd_primer", "linkerprimersequence"),
    "primer_reverse": ("reverse_primer", "primer_reverse", "rev_primer"),
    "extraction_method": ("extraction_method", "dna_extraction_method", "nucleic_acid_extraction"),
    "pcr_method": ("pcr_method", "pcr_primers", "pcr_conditions"),
    "library_method": ("library_method", "library_construction_protocol", "library_strategy"),
}


def utc_iso() -> str:
    return utcnow().isoformat()


def normalize_doi_or_none(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return normalize_doi(value.strip().rstrip(".,;)"))
    except IdentifierError:
        return None


def json_dumps(value) -> str:  # noqa: ANN001
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@dataclass
class QiitaConfig:
    db_path: Path = DEFAULT_QIITA_DB
    data_dir: Path = DEFAULT_QIITA_DATA_DIR
    mgnify_db_path: Path = DEFAULT_MGNIFY_DB
    base_url: str = QIITA_BASE_URL
    min_request_interval_seconds: float = 2.0
    request_timeout_seconds: float = 30.0


class QiitaDB:
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
            CREATE TABLE IF NOT EXISTS qiita_studies (
                qiita_study_id TEXT PRIMARY KEY,
                title TEXT,
                alias TEXT,
                abstract TEXT,
                description TEXT,
                pi_name TEXT,
                primary_doi TEXT,
                publication_dois_json TEXT NOT NULL DEFAULT '[]',
                pmids_json TEXT NOT NULL DEFAULT '[]',
                primary_bioproject TEXT,
                ena_study_accessions_json TEXT NOT NULL DEFAULT '[]',
                all_sequence_accessions_json TEXT NOT NULL DEFAULT '[]',
                sequence_accessibility_status TEXT,
                marine_confidence TEXT,
                marine_match_methods_json TEXT NOT NULL DEFAULT '[]',
                sample_count INTEGER DEFAULT 0,
                prep_count INTEGER DEFAULT 0,
                publication_resolution_status TEXT NOT NULL DEFAULT 'not_yet_processed',
                accession_resolution_status TEXT NOT NULL DEFAULT 'not_yet_processed',
                overlaps_mgnify INTEGER NOT NULL DEFAULT 0,
                matched_mgnify_ids_json TEXT NOT NULL DEFAULT '[]',
                source_metadata_json TEXT NOT NULL DEFAULT '{}',
                first_seen_at TEXT NOT NULL,
                last_checked_at TEXT
            );

            CREATE TABLE IF NOT EXISTS qiita_samples (
                qiita_study_id TEXT NOT NULL,
                qiita_sample_id TEXT NOT NULL,
                biosample_accession TEXT,
                collection_date TEXT,
                latitude TEXT,
                longitude TEXT,
                depth TEXT,
                geo_loc_name TEXT,
                env_broad_scale TEXT,
                env_local_scale TEXT,
                env_medium TEXT,
                sampling_method TEXT,
                size_fraction TEXT,
                temperature TEXT,
                salinity TEXT,
                ph TEXT,
                oxygen TEXT,
                source_metadata_json TEXT NOT NULL,
                PRIMARY KEY (qiita_study_id, qiita_sample_id)
            );

            CREATE TABLE IF NOT EXISTS qiita_preparations (
                qiita_study_id TEXT NOT NULL,
                prep_id TEXT NOT NULL,
                artifact_ids_json TEXT NOT NULL DEFAULT '[]',
                data_type TEXT,
                platform TEXT,
                target_gene TEXT,
                target_subfragment TEXT,
                primer_forward TEXT,
                primer_reverse TEXT,
                extraction_method TEXT,
                pcr_method TEXT,
                library_method TEXT,
                raw_sequence_available INTEGER NOT NULL DEFAULT 0,
                raw_download_manifest_json TEXT NOT NULL DEFAULT '{}',
                experiment_accessions_json TEXT NOT NULL DEFAULT '[]',
                run_accessions_json TEXT NOT NULL DEFAULT '[]',
                source_metadata_json TEXT NOT NULL,
                PRIMARY KEY (qiita_study_id, prep_id)
            );

            CREATE VIEW IF NOT EXISTS qiita_faire_sample_enrichment AS
            SELECT
                qiita_study_id,
                qiita_sample_id,
                biosample_accession,
                latitude AS decimalLatitude,
                longitude AS decimalLongitude,
                collection_date AS eventDate,
                depth,
                geo_loc_name,
                env_broad_scale,
                env_local_scale,
                env_medium,
                sampling_method,
                size_fraction
            FROM qiita_samples;

            CREATE VIEW IF NOT EXISTS qiita_faire_experiment_enrichment AS
            SELECT
                qiita_study_id,
                prep_id,
                data_type,
                platform,
                target_gene,
                target_subfragment,
                primer_forward,
                primer_reverse,
                experiment_accessions_json,
                run_accessions_json
            FROM qiita_preparations;

            CREATE VIEW IF NOT EXISTS paper_seeds AS
            SELECT
                'qiita' AS seed_source,
                qiita_study_id AS source_study_id,
                primary_bioproject AS bioproject_accession,
                ena_study_accessions_json AS secondary_study_accessions_json,
                title AS study_title,
                primary_doi,
                CASE
                  WHEN json_array_length(pmids_json) > 0 THEN json_extract(pmids_json, '$[0]')
                  ELSE NULL
                END AS primary_pmid,
                publication_dois_json,
                sequence_accessibility_status,
                marine_confidence,
                overlaps_mgnify,
                matched_mgnify_ids_json,
                publication_resolution_status,
                first_seen_at,
                last_checked_at
            FROM qiita_studies
            WHERE marine_confidence IN ('high', 'medium', 'low');
            """
        )
        self.conn.commit()

    def upsert_study(self, row: dict) -> None:
        now = utc_iso()
        row = dict(row)
        row.setdefault("first_seen_at", now)
        row.setdefault("last_checked_at", now)
        columns = [
            "qiita_study_id", "title", "alias", "abstract", "description", "pi_name",
            "primary_doi", "publication_dois_json", "pmids_json", "primary_bioproject",
            "ena_study_accessions_json", "all_sequence_accessions_json", "sequence_accessibility_status",
            "marine_confidence", "marine_match_methods_json", "sample_count", "prep_count",
            "publication_resolution_status", "accession_resolution_status", "overlaps_mgnify",
            "matched_mgnify_ids_json", "source_metadata_json", "first_seen_at", "last_checked_at",
        ]
        values = [row.get(column) for column in columns]
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"qiita_study_id", "first_seen_at"})
        self.conn.execute(
            f"""
            INSERT INTO qiita_studies({', '.join(columns)})
            VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(qiita_study_id) DO UPDATE SET {assignments}
            """,
            values,
        )

    def upsert_sample(self, row: dict) -> None:
        columns = [
            "qiita_study_id", "qiita_sample_id", "biosample_accession", "collection_date",
            "latitude", "longitude", "depth", "geo_loc_name", "env_broad_scale",
            "env_local_scale", "env_medium", "sampling_method", "size_fraction",
            "temperature", "salinity", "ph", "oxygen", "source_metadata_json",
        ]
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"qiita_study_id", "qiita_sample_id"})
        self.conn.execute(
            f"""
            INSERT INTO qiita_samples({', '.join(columns)})
            VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(qiita_study_id, qiita_sample_id) DO UPDATE SET {assignments}
            """,
            [row.get(column) for column in columns],
        )

    def upsert_preparation(self, row: dict) -> None:
        columns = [
            "qiita_study_id", "prep_id", "artifact_ids_json", "data_type", "platform",
            "target_gene", "target_subfragment", "primer_forward", "primer_reverse",
            "extraction_method", "pcr_method", "library_method", "raw_sequence_available",
            "raw_download_manifest_json", "experiment_accessions_json", "run_accessions_json",
            "source_metadata_json",
        ]
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"qiita_study_id", "prep_id"})
        self.conn.execute(
            f"""
            INSERT INTO qiita_preparations({', '.join(columns)})
            VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(qiita_study_id, prep_id) DO UPDATE SET {assignments}
            """,
            [row.get(column) for column in columns],
        )

    def commit(self) -> None:
        self.conn.commit()

    def count(self, table: str) -> int:
        return int(self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


class QiitaClient:
    def __init__(self, config: QiitaConfig):
        self.config = config
        self.last_request_at = 0.0
        self.client = httpx.Client(timeout=config.request_timeout_seconds, follow_redirects=True)

    def close(self) -> None:
        self.client.close()

    def get(self, url: str) -> httpx.Response:
        elapsed = time.monotonic() - self.last_request_at
        wait = self.config.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        response = self.client.get(url)
        self.last_request_at = time.monotonic()
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(int(retry_after))
                response = self.client.get(url)
                self.last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def get_text(self, url: str) -> str:
        return self.get(url).text

    def enumerate_public_study_ids(self) -> list[str]:
        html = self.get_text(urljoin(self.config.base_url, "/stats/"))
        ids = sorted(set(STUDY_ID_RE.findall(html)), key=lambda value: int(value))
        return ids

    def public_study_html(self, study_id: str) -> str:
        return self.get_text(urljoin(self.config.base_url, f"/public/?study_id={study_id}"))

    def download_public_zip(self, url: str) -> bytes | None:
        try:
            response = self.get(url)
        except httpx.HTTPStatusError as exc:
            logger.info("Qiita download unavailable url=%s status=%s", url, exc.response.status_code)
            return None
        content_type = response.headers.get("content-type", "")
        if "zip" not in content_type and not response.content.startswith(b"PK"):
            return None
        return response.content


def extract_public_study_ids_from_stats_html(html: str) -> list[str]:
    return sorted(set(STUDY_ID_RE.findall(html)), key=lambda value: int(value))


def parse_study_html(study_id: str, html: str) -> dict:
    text = html_to_text(html)
    title = extract_title(html, text)
    dois = sorted({doi for doi in (normalize_doi_or_none(match.group(0)) for match in DOI_RE.finditer(text)) if doi})
    pmids = sorted(set(PMID_RE.findall(text)))
    accessions = sorted({match.group(0).upper() for match in ACCESSION_RE.finditer(text)})
    ena_studies = sorted({match.group(0).upper() for match in ENA_STUDY_RE.finditer(text)})
    bioprojects = sorted({match.group(0).upper() for match in BIOPROJECT_RE.finditer(text)})
    prep_ids = sorted(
        set(re.findall(r"prep(?:aration)?[_\s-]*id[=:;\s]+(\d+)|prep_id=(\d+)", f"{text} {html}", re.IGNORECASE))
    )
    flat_prep_ids = sorted({value for pair in prep_ids for value in pair if value})
    return {
        "qiita_study_id": study_id,
        "title": title,
        "alias": extract_label(text, ("Alias", "Study alias")),
        "abstract": extract_label(text, ("Abstract",)),
        "description": extract_label(text, ("Description", "Study description")),
        "pi_name": extract_label(text, ("PI", "Principal Investigator", "Study PI")),
        "publication_dois": dois,
        "pmids": pmids,
        "all_sequence_accessions": accessions,
        "ena_study_accessions": ena_studies,
        "primary_bioproject": bioprojects[0] if bioprojects else None,
        "prep_ids": flat_prep_ids,
        "source": {"html_text": text[:20000], "html_length": len(html), "public_url": f"https://qiita.ucsd.edu/public/?study_id={study_id}"},
    }


def html_to_text(html: str) -> str:
    cleaned = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", html)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_title(html: str, text: str) -> str | None:
    blocked = {"qiita", "this site only works with the following browsers"}
    study_heading = re.search(r"<h[12][^>]*>(.*?)-\s*ID\s+\d+\s*</h[12]>", html, re.IGNORECASE | re.DOTALL)
    if study_heading:
        title = re.sub(r"\s*-\s*ID\s+\d+\s*$", "", html_to_text(study_heading.group(1))).strip()
        if title:
            return title
    for pattern in (r"<h2[^>]*>(.*?)</h2>", r"<h1[^>]*>(.*?)</h1>", r"<title[^>]*>(.*?)</title>"):
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            title = html_to_text(match.group(1))
            if title and title.casefold() not in blocked:
                return re.sub(r"\s*-\s*ID\s+\d+\s*$", "", title).strip()
    return text[:200] if text else None


def extract_label(text: str, labels: Iterable[str]) -> str | None:
    for label in labels:
        match = re.search(rf"\b{re.escape(label)}\b\s*[:\-]\s*(.{{1,1500}}?)(?=\s+[A-Z][A-Za-z ]{{2,30}}\s*[:\-]|\Z)", text)
        if match:
            return match.group(1).strip()
    return None


def classify_marine(study: dict, sample_rows: list[dict] | None = None) -> tuple[str, list[str]]:
    chunks = [study.get("title"), study.get("abstract"), study.get("description")]
    if sample_rows:
        chunks.extend(" ".join(str(value) for value in row.values() if value not in (None, "")) for row in sample_rows[:100])
    text = " ".join(chunk for chunk in chunks if chunk).casefold()
    methods = []
    for term in MARINE_TERMS:
        if term in text:
            methods.append(f"marine_term:{term}")
    if methods:
        return "high", methods[:20]
    env_hits = [term for term in ENVIRONMENT_TERMS if term in text]
    reject_hits = [term for term in REJECT_TERMS if term in text]
    if env_hits and not reject_hits:
        return "medium", [f"environment_term:{term}" for term in env_hits[:20]]
    if env_hits:
        return "not_marine", [f"environment_term:{term}" for term in env_hits[:10]] + [f"reject_context:{term}" for term in reject_hits[:10]]
    return "not_marine", [f"reject_context:{term}" for term in reject_hits[:10]]


def pick(row: dict, logical_field: str) -> str | None:
    normalized = {normalize_key(key): key for key in row}
    for candidate in FIELD_CANDIDATES.get(logical_field, (logical_field,)):
        key = normalized.get(normalize_key(candidate))
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key]).strip()
    if logical_field == "biosample":
        return find_first(row, BIOSAMPLE_RE)
    return None


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def find_first(row: dict, regex: re.Pattern) -> str | None:
    for value in row.values():
        if value is None:
            continue
        match = regex.search(str(value))
        if match:
            return match.group(0).upper()
    return None


def parse_tsv(text: str) -> list[dict]:
    if not text.strip():
        return []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return [dict(row) for row in reader]


def first_tsv_from_zip(content: bytes) -> tuple[str | None, list[dict]]:
    with ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            if name.lower().endswith((".tsv", ".txt")) and not name.endswith("/"):
                data = archive.read(name).decode("utf-8", errors="replace")
                return name, parse_tsv(data)
    return None, []


def sample_row_to_db(study_id: str, row: dict) -> dict:
    sample_id = pick(row, "sample_id") or row.get("sample_name") or f"row_{abs(hash(json_dumps(row)))}"
    return {
        "qiita_study_id": study_id,
        "qiita_sample_id": sample_id,
        "biosample_accession": pick(row, "biosample"),
        "collection_date": pick(row, "collection_date"),
        "latitude": pick(row, "latitude"),
        "longitude": pick(row, "longitude"),
        "depth": pick(row, "depth"),
        "geo_loc_name": pick(row, "geo_loc_name"),
        "env_broad_scale": pick(row, "env_broad_scale"),
        "env_local_scale": pick(row, "env_local_scale"),
        "env_medium": pick(row, "env_medium"),
        "sampling_method": pick(row, "sampling_method"),
        "size_fraction": pick(row, "size_fraction"),
        "temperature": pick(row, "temperature"),
        "salinity": pick(row, "salinity"),
        "ph": pick(row, "ph"),
        "oxygen": pick(row, "oxygen"),
        "source_metadata_json": json_dumps(row),
    }


def prep_rows_to_db(study_id: str, prep_id: str, rows: list[dict], manifest: dict) -> dict:
    merged: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            if value not in (None, "") and key not in merged:
                merged[key] = value
    text = json_dumps({"rows": rows[:20], "manifest": manifest})
    artifact_ids = sorted(set(re.findall(r"\bartifact[_\s-]*(?:id)?[=:\s]+(\d+)|artifact_id=(\d+)", text, re.IGNORECASE)))
    flat_artifact_ids = sorted({value for pair in artifact_ids for value in pair if value})
    accessions = sorted({match.group(0).upper() for match in ACCESSION_RE.finditer(text)})
    experiments = sorted({value for value in accessions if re.match(r"^[DES]RX\d+$", value)})
    runs = sorted({value for value in accessions if re.match(r"^[DES]RR\d+$", value)})
    return {
        "qiita_study_id": study_id,
        "prep_id": prep_id,
        "artifact_ids_json": json_dumps(flat_artifact_ids),
        "data_type": pick(merged, "data_type"),
        "platform": pick(merged, "platform"),
        "target_gene": pick(merged, "target_gene"),
        "target_subfragment": pick(merged, "target_subfragment"),
        "primer_forward": pick(merged, "primer_forward"),
        "primer_reverse": pick(merged, "primer_reverse"),
        "extraction_method": pick(merged, "extraction_method"),
        "pcr_method": pick(merged, "pcr_method"),
        "library_method": pick(merged, "library_method"),
        "raw_sequence_available": 0,
        "raw_download_manifest_json": json_dumps(manifest),
        "experiment_accessions_json": json_dumps(experiments),
        "run_accessions_json": json_dumps(runs),
        "source_metadata_json": json_dumps({"rows": rows, "manifest": manifest}),
    }


def build_study_db_row(study: dict, source_metadata: dict | None = None) -> dict:
    dois = study.get("publication_dois") or []
    pmids = study.get("pmids") or []
    accessions = study.get("all_sequence_accessions") or []
    ena_studies = study.get("ena_study_accessions") or []
    confidence, methods = classify_marine(study)
    primary_bioproject = study.get("primary_bioproject") or next((acc for acc in accessions if BIOPROJECT_RE.match(acc)), None)
    accession_status = "resolved_bioproject" if primary_bioproject else ("ena_study_only" if ena_studies else "unresolved_sequence_accession")
    sequence_status = "raw_artifact_present_download_unverified" if accessions else "no_sequence_locator_found"
    publication_status = "resolved" if dois else ("pmid_only" if pmids else "missing")
    return {
        "qiita_study_id": str(study["qiita_study_id"]),
        "title": study.get("title"),
        "alias": study.get("alias"),
        "abstract": study.get("abstract"),
        "description": study.get("description"),
        "pi_name": study.get("pi_name"),
        "primary_doi": dois[0] if dois else None,
        "publication_dois_json": json_dumps(dois),
        "pmids_json": json_dumps(pmids),
        "primary_bioproject": primary_bioproject,
        "ena_study_accessions_json": json_dumps(ena_studies),
        "all_sequence_accessions_json": json_dumps(accessions),
        "sequence_accessibility_status": sequence_status,
        "marine_confidence": confidence,
        "marine_match_methods_json": json_dumps(methods),
        "sample_count": 0,
        "prep_count": len(study.get("prep_ids") or []),
        "publication_resolution_status": publication_status,
        "accession_resolution_status": accession_status,
        "overlaps_mgnify": 0,
        "matched_mgnify_ids_json": "[]",
        "source_metadata_json": json_dumps(source_metadata or study),
    }


def update_mgnify_overlap(db: QiitaDB, mgnify_db_path: Path) -> int:
    if not mgnify_db_path.exists():
        return 0
    mgnify = sqlite3.connect(f"file:{mgnify_db_path}?mode=ro", uri=True, timeout=60)
    mgnify.row_factory = sqlite3.Row
    updated = 0
    try:
        mgnify_bioproject = {
            row["bioproject_accession"]: row["mgnify_accession"]
            for row in mgnify.execute("SELECT mgnify_accession, bioproject_accession FROM mgnify_studies WHERE bioproject_accession IS NOT NULL")
        }
        mgnify_doi = {
            row["normalized_doi"] or row["doi"]: row["mgnify_accession"]
            for row in mgnify.execute(
                """
                SELECT pc.normalized_doi, pc.doi, ms.mgnify_accession
                FROM publication_candidates pc
                JOIN mgnify_studies ms ON ms.id = pc.mgnify_study_id
                WHERE pc.doi IS NOT NULL
                """
            )
        }
    finally:
        mgnify.close()
    for row in db.conn.execute("SELECT qiita_study_id, primary_bioproject, publication_dois_json FROM qiita_studies").fetchall():
        matches = set()
        if row["primary_bioproject"] in mgnify_bioproject:
            matches.add(mgnify_bioproject[row["primary_bioproject"]])
        for doi in json.loads(row["publication_dois_json"] or "[]"):
            if doi in mgnify_doi:
                matches.add(mgnify_doi[doi])
        if matches:
            db.conn.execute(
                "UPDATE qiita_studies SET overlaps_mgnify = 1, matched_mgnify_ids_json = ? WHERE qiita_study_id = ?",
                (json_dumps(sorted(matches)), row["qiita_study_id"]),
            )
            updated += 1
    db.commit()
    return updated


class QiitaSeedDiscoveryRunner:
    def __init__(self, config: QiitaConfig):
        self.config = config
        self.db = QiitaDB(config.db_path)
        self.client = QiitaClient(config)

    def close(self) -> None:
        self.client.close()
        self.db.close()

    def run(self, phase: str = "all", max_studies: int | None = None, refresh: bool = False) -> dict:
        self.db.initialize()
        raw_dir = self.config.data_dir / "raw"
        cache_dir = self.config.data_dir / "cache"
        reports_dir = self.config.data_dir / "reports"
        manifests_dir = self.config.data_dir / "file_manifests"
        raw_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        counts: Counter = Counter()
        study_page_phases = {"all", "enumerate", "marine_filter", "study_metadata", "accessions", "publications"}
        if phase in study_page_phases:
            ids = self.client.enumerate_public_study_ids()
            counts["public_studies_scanned"] = len(ids)
            for index, study_id in enumerate(ids[:max_studies] if max_studies else ids, start=1):
                study_raw_dir = raw_dir / study_id
                study_raw_dir.mkdir(parents=True, exist_ok=True)
                html_path = study_raw_dir / "study.html"
                if html_path.exists() and not refresh:
                    html = html_path.read_text(encoding="utf-8")
                else:
                    html = self.client.public_study_html(study_id)
                    html_path.write_text(html, encoding="utf-8")
                study = parse_study_html(study_id, html)
                (study_raw_dir / "study.json").write_text(json_dumps(study), encoding="utf-8")
                self.db.upsert_study(build_study_db_row(study))
                # Commit after every study, not once after the whole loop --
                # this can be thousands of studies against a walltime-limited
                # job; a kill mid-loop previously lost every upsert since the
                # last commit (the SQLite connection just closes without one).
                # Each commit is cheap (page HTML is already cached to disk
                # regardless), so there's no real cost to paying it every
                # iteration rather than batching.
                self.db.commit()
                counts["studies_stored"] += 1
                if index % 100 == 0:
                    logger.info("Qiita enumeration progress %s/%s", index, len(ids))
        if phase in {"all", "sample_metadata", "prep_metadata"}:
            self._download_relevant_metadata(raw_dir, manifests_dir, counts, max_studies=max_studies, refresh=refresh)
        if phase in {"all", "mgnify_overlap"}:
            counts["overlapping_mgnify"] = update_mgnify_overlap(self.db, self.config.mgnify_db_path)
        if phase in {"all", "reports"}:
            write_reports(self.db, reports_dir)
        write_output_locations(self.config, reports_dir, manifests_dir)
        print_qiita_report(self.config, counts)
        return {
            "counts": dict(counts),
            "database": str(self.config.db_path),
            "reports_dir": str(reports_dir),
        }

    def _download_relevant_metadata(self, raw_dir: Path, manifests_dir: Path, counts: Counter, *, max_studies: int | None, refresh: bool) -> None:
        rows = self.db.conn.execute(
            "SELECT * FROM qiita_studies WHERE marine_confidence IN ('high', 'medium', 'low') ORDER BY qiita_study_id"
        ).fetchall()
        if max_studies:
            rows = rows[:max_studies]
        for index, study_row in enumerate(rows, start=1):
            study_id = study_row["qiita_study_id"]
            study_raw_dir = raw_dir / study_id
            study_raw_dir.mkdir(parents=True, exist_ok=True)
            sample_rows = self._download_sample_metadata(study_id, study_raw_dir, refresh=refresh)
            for sample in sample_rows:
                self.db.upsert_sample(sample_row_to_db(study_id, sample))
            prep_ids = self.prep_ids_for_study(study_row)
            prep_count = 0
            for prep_id in prep_ids:
                prep_rows, manifest = self._download_prep_metadata(study_id, prep_id, study_raw_dir, manifests_dir, refresh=refresh)
                if prep_rows or manifest:
                    self.db.upsert_preparation(prep_rows_to_db(study_id, prep_id, prep_rows, manifest))
                    prep_count += 1
            confidence, methods = classify_marine(dict(study_row), sample_rows)
            self.db.conn.execute(
                """
                UPDATE qiita_studies
                SET sample_count = ?, prep_count = ?, marine_confidence = ?, marine_match_methods_json = ?,
                    sequence_accessibility_status = CASE
                      WHEN sequence_accessibility_status = 'no_sequence_locator_found' AND ? > 0 THEN 'raw_artifact_present_download_unverified'
                      ELSE sequence_accessibility_status
                    END,
                    last_checked_at = ?
                WHERE qiita_study_id = ?
                """,
                (len(sample_rows), prep_count, confidence, json_dumps(methods), prep_count, utc_iso(), study_id),
            )
            counts["sample_rows"] += len(sample_rows)
            counts["prep_rows"] += prep_count
            # Commit after every study for the same reason as the
            # enumeration loop above -- this loop does real network I/O
            # per study (sample + every prep's metadata), so it's both the
            # slowest phase and the one most likely to still be running
            # when a walltime cutoff hits.
            self.db.commit()
            if index % 25 == 0:
                logger.info("Qiita metadata progress studies=%s/%s sample_rows=%s prep_rows=%s", index, len(rows), counts["sample_rows"], counts["prep_rows"])

    def prep_ids_for_study(self, study_row: sqlite3.Row) -> list[str]:
        source = json.loads(study_row["source_metadata_json"] or "{}")
        return [str(value) for value in source.get("prep_ids", [])]

    def _download_sample_metadata(self, study_id: str, raw_dir: Path, *, refresh: bool) -> list[dict]:
        path = raw_dir / "sample_information.tsv"
        if path.exists() and not refresh:
            return parse_tsv(path.read_text(encoding="utf-8"))
        url = urljoin(self.config.base_url, f"/public_download/?data=sample_information&study_id={study_id}")
        content = self.client.download_public_zip(url)
        if not content:
            return []
        archive_path = raw_dir / "sample_information.zip"
        archive_path.write_bytes(content)
        name, rows = first_tsv_from_zip(content)
        if name:
            write_tsv_rows(path, rows)
        return rows

    def _download_prep_metadata(self, study_id: str, prep_id: str, raw_dir: Path, manifests_dir: Path, *, refresh: bool) -> tuple[list[dict], dict]:
        prep_path = raw_dir / f"prep_{prep_id}.tsv"
        rows: list[dict] = []
        if prep_path.exists() and not refresh:
            rows = parse_tsv(prep_path.read_text(encoding="utf-8"))
        else:
            url = urljoin(self.config.base_url, f"/public_download/?data=prep_information&prep_id={prep_id}")
            content = self.client.download_public_zip(url)
            if content:
                (raw_dir / f"prep_{prep_id}.zip").write_bytes(content)
                _name, rows = first_tsv_from_zip(content)
                if rows:
                    write_tsv_rows(prep_path, rows)
        manifest = {
            "study_id": study_id,
            "prep_id": prep_id,
            "raw_data_url": urljoin(self.config.base_url, f"/public_download/?data=raw&prep_id={prep_id}"),
            "biom_data_url": urljoin(self.config.base_url, f"/public_download/?data=biom&prep_id={prep_id}"),
            "download_status": "locator_recorded_not_downloaded",
        }
        (manifests_dir / f"{study_id}_{prep_id}.json").write_text(json_dumps(manifest), encoding="utf-8")
        return rows, manifest


def write_reports(db: QiitaDB, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_specs = {
        "new_studies.csv": "SELECT * FROM qiita_studies WHERE overlaps_mgnify = 0 ORDER BY qiita_study_id",
        "mgnify_overlap.csv": "SELECT * FROM qiita_studies WHERE overlaps_mgnify = 1 ORDER BY qiita_study_id",
        "known_studies_enriched.csv": "SELECT * FROM qiita_studies WHERE overlaps_mgnify = 1 AND (sample_count > 0 OR prep_count > 0) ORDER BY qiita_study_id",
        "unresolved_publications.csv": "SELECT * FROM qiita_studies WHERE publication_resolution_status IN ('missing', 'ambiguous', 'not_yet_processed') ORDER BY qiita_study_id",
        "unresolved_accessions.csv": "SELECT * FROM qiita_studies WHERE accession_resolution_status IN ('unresolved_sequence_accession', 'not_yet_processed') ORDER BY qiita_study_id",
    }
    for filename, sql in report_specs.items():
        rows = db.conn.execute(sql).fetchall()
        write_rows_csv(reports_dir / filename, rows)
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


def write_tsv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_completeness(db: QiitaDB, path: Path) -> None:
    metrics = {
        "total_public_qiita_studies": "SELECT count(*) FROM qiita_studies",
        "marine_environment_candidates": "SELECT count(*) FROM qiita_studies WHERE marine_confidence IN ('high', 'medium', 'low')",
        "with_doi": "SELECT count(*) FROM qiita_studies WHERE primary_doi IS NOT NULL",
        "with_pmid": "SELECT count(*) FROM qiita_studies WHERE json_array_length(pmids_json) > 0",
        "with_bioproject": "SELECT count(*) FROM qiita_studies WHERE primary_bioproject IS NOT NULL",
        "with_ena_accession": "SELECT count(*) FROM qiita_studies WHERE json_array_length(ena_study_accessions_json) > 0",
        "with_raw_data_locator": "SELECT count(*) FROM qiita_studies WHERE sequence_accessibility_status = 'raw_artifact_present_download_unverified'",
        "overlapping_mgnify": "SELECT count(*) FROM qiita_studies WHERE overlaps_mgnify = 1",
        "net_new_studies": "SELECT count(*) FROM qiita_studies WHERE overlaps_mgnify = 0 AND marine_confidence IN ('high', 'medium', 'low')",
        "samples_with_lat_lon": "SELECT count(*) FROM qiita_samples WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
        "samples_with_collection_date": "SELECT count(*) FROM qiita_samples WHERE collection_date IS NOT NULL",
        "samples_with_env_medium": "SELECT count(*) FROM qiita_samples WHERE env_medium IS NOT NULL",
        "preps_with_platform": "SELECT count(*) FROM qiita_preparations WHERE platform IS NOT NULL",
        "preps_with_target_gene": "SELECT count(*) FROM qiita_preparations WHERE target_gene IS NOT NULL",
        "preps_with_primers": "SELECT count(*) FROM qiita_preparations WHERE primer_forward IS NOT NULL OR primer_reverse IS NOT NULL",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "count"])
        for metric, sql in metrics.items():
            writer.writerow([metric, db.conn.execute(sql).fetchone()[0]])


def write_output_locations(config: QiitaConfig, reports_dir: Path, manifests_dir: Path) -> None:
    payload = {
        "database": str(config.db_path),
        "tables": ["qiita_studies", "qiita_samples", "qiita_preparations"],
        "paper_seeds_view": "paper_seeds",
        "raw_data_dir": str((config.data_dir / "raw").resolve()),
        "file_manifest_dir": str(manifests_dir.resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "mgnify_database": str(config.mgnify_db_path) if config.mgnify_db_path.exists() else "not available",
    }
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "OUTPUT_LOCATIONS.json").write_text(json_dumps(payload), encoding="utf-8")


def print_qiita_report(config: QiitaConfig, counts: Counter | dict) -> None:
    db = QiitaDB(config.db_path)
    try:
        db.initialize()
        def scalar(sql: str) -> int:
            return int(db.conn.execute(sql).fetchone()[0])
        marine_sql = "SELECT count(*) FROM qiita_studies WHERE marine_confidence IN ('high','medium','low')"
        ena_confirmed_sql = (
            "SELECT count(*) FROM qiita_studies "
            "WHERE sequence_accessibility_status IN ('ena_raw_reads_confirmed','ena_and_qiita_raw_reads')"
        )
        qiita_unverified_sql = (
            "SELECT count(*) FROM qiita_studies "
            "WHERE sequence_accessibility_status = 'raw_artifact_present_download_unverified'"
        )
        net_new_sql = (
            "SELECT count(*) FROM qiita_studies "
            "WHERE overlaps_mgnify = 0 AND marine_confidence IN ('high','medium','low')"
        )
        print("=" * 60)
        print("QIITA INGEST COMPLETE")
        print("=" * 60)
        print("\nQIITA DATABASE")
        print(config.db_path)
        print("\nTABLES")
        print("  qiita_studies")
        print("  qiita_samples")
        print("  qiita_preparations")
        print("\nPAPER SEEDS VIEW")
        print("  paper_seeds")
        print("\nRAW QIITA DATA")
        print((config.data_dir / "raw").resolve())
        print("\nFILE MANIFESTS")
        print((config.data_dir / "file_manifests").resolve())
        print("\nREPORTS")
        print((config.data_dir / "reports").resolve())
        print("\nMGNIFY DATABASE USED FOR OVERLAP")
        print(config.mgnify_db_path if config.mgnify_db_path.exists() else "not available")
        print("\n" + "=" * 60)
        print(f"PUBLIC STUDIES SCANNED: {scalar('SELECT count(*) FROM qiita_studies')}")
        print(f"MARINE/ENVIRONMENT STUDIES: {scalar(marine_sql)}")
        print(f"STUDIES WITH DOI: {scalar('SELECT count(*) FROM qiita_studies WHERE primary_doi IS NOT NULL')}")
        print(f"STUDIES WITH BIOPROJECT: {scalar('SELECT count(*) FROM qiita_studies WHERE primary_bioproject IS NOT NULL')}")
        print(f"STUDIES WITH ENA ACCESSION: {scalar('SELECT count(*) FROM qiita_studies WHERE json_array_length(ena_study_accessions_json) > 0')}")
        print(f"ENA RAW DATA CONFIRMED: {scalar(ena_confirmed_sql)}")
        print(f"QIITA RAW DATA LOCATORS: {scalar(qiita_unverified_sql)}")
        print(f"OVERLAPPING MGNIFY: {scalar('SELECT count(*) FROM qiita_studies WHERE overlaps_mgnify = 1')}")
        print(f"NET-NEW STUDIES: {scalar(net_new_sql)}")
        print("=" * 60)
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover Qiita public environmental/marine study seeds.")
    parser.add_argument("--db", default=str(DEFAULT_QIITA_DB))
    parser.add_argument("--data-dir", default=str(DEFAULT_QIITA_DATA_DIR))
    parser.add_argument("--mgnify-db", default=str(DEFAULT_MGNIFY_DB))
    parser.add_argument(
        "--phase",
        choices=(
            "all",
            "enumerate",
            "marine_filter",
            "study_metadata",
            "sample_metadata",
            "prep_metadata",
            "accessions",
            "publications",
            "mgnify_overlap",
            "reports",
        ),
        default="all",
    )
    parser.add_argument("--max-studies", type=int)
    parser.add_argument("--resume", action="store_true", help="Accepted for parity with other seed crawlers; cached files are reused by default.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--qiita-min-request-interval-seconds",
        type=float,
        default=float(os.environ.get("QIITA_MIN_REQUEST_INTERVAL_SECONDS", "2.0")),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = QiitaConfig(
        db_path=Path(args.db),
        data_dir=Path(args.data_dir),
        mgnify_db_path=Path(args.mgnify_db),
        min_request_interval_seconds=args.qiita_min_request_interval_seconds,
    )
    runner = QiitaSeedDiscoveryRunner(config)
    try:
        runner.run(phase=args.phase, max_studies=args.max_studies, refresh=args.refresh)
    finally:
        runner.close()
    return 0
