from __future__ import annotations

import json

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import EnaRun


ENA_READ_RUN_FIELDS = (
    "study_accession,secondary_study_accession,secondary_project_accession,"
    "run_accession,experiment_accession,sample_accession,secondary_sample_accession,submission_accession,"
    "study_title,project_name,center_name,first_public,"
    "fastq_ftp,fastq_md5,fastq_bytes,submitted_ftp,submitted_md5,submitted_bytes,submitted_format,"
    "sra_ftp,sra_md5,sra_bytes,"
    "library_strategy,library_source,library_selection,library_layout,"
    "instrument_platform,instrument_model,target_gene,"
    "collection_date,lat,lon,depth,country,marine_region,"
    "environment_biome,environment_feature,environment_material,sample_collection,"
    "extraction_protocol,library_construction_protocol,marine"
)


def _first_string(row: dict, *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _bioproject_from_row(row: dict) -> str | None:
    for key in ("study_accession", "secondary_project_accession", "secondary_study_accession"):
        value = _first_string(row, key)
        if value and value.upper().startswith("PRJ"):
            return value.upper()
    return None


def classify_sequence_accessibility(row: dict) -> str:
    fastq = _first_string(row, "fastq_ftp")
    fastq_bytes = _first_string(row, "fastq_bytes")
    submitted = _first_string(row, "submitted_ftp")
    submitted_bytes = _first_string(row, "submitted_bytes")
    sra = _first_string(row, "sra_ftp")
    sra_bytes = _first_string(row, "sra_bytes")
    if fastq and _positive_size_list(fastq_bytes):
        return "fastq_confirmed"
    if submitted and (_positive_size_list(submitted_bytes) or _first_string(row, "submitted_format")):
        return "submitted_reads_confirmed"
    if sra and _positive_size_list(sra_bytes):
        return "sra_archive_confirmed"
    if fastq or submitted or sra:
        return "sequence_locator_present_unverified"
    return "no_downloadable_reads"


def _positive_size_list(value: str | None) -> bool:
    if not value:
        return False
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit() and int(part) > 0:
            return True
    return False


def parse_read_run(row: dict, *, marine_confidence: str, marine_match_methods: str) -> EnaRun | None:
    run_accession = _first_string(row, "run_accession")
    if not run_accession:
        return None
    return EnaRun(
        run_accession=run_accession,
        experiment_accession=_first_string(row, "experiment_accession"),
        sample_accession=_first_string(row, "sample_accession"),
        secondary_sample_accession=_first_string(row, "secondary_sample_accession"),
        study_accession=_first_string(row, "study_accession"),
        secondary_study_accession=_first_string(row, "secondary_study_accession"),
        bioproject_accession=_bioproject_from_row(row),
        submission_accession=_first_string(row, "submission_accession"),
        study_title=_first_string(row, "study_title"),
        project_name=_first_string(row, "project_name"),
        centre_name=_first_string(row, "center_name", "centre_name"),
        first_public=_first_string(row, "first_public"),
        fastq_ftp=_first_string(row, "fastq_ftp"),
        fastq_md5=_first_string(row, "fastq_md5"),
        fastq_bytes=_first_string(row, "fastq_bytes"),
        submitted_ftp=_first_string(row, "submitted_ftp"),
        submitted_md5=_first_string(row, "submitted_md5"),
        submitted_bytes=_first_string(row, "submitted_bytes"),
        submitted_format=_first_string(row, "submitted_format"),
        sra_ftp=_first_string(row, "sra_ftp"),
        sra_md5=_first_string(row, "sra_md5"),
        sra_bytes=_first_string(row, "sra_bytes"),
        library_strategy=_first_string(row, "library_strategy"),
        library_source=_first_string(row, "library_source"),
        library_selection=_first_string(row, "library_selection"),
        library_layout=_first_string(row, "library_layout"),
        instrument_platform=_first_string(row, "instrument_platform"),
        instrument_model=_first_string(row, "instrument_model"),
        target_gene=_first_string(row, "target_gene"),
        collection_date=_first_string(row, "collection_date"),
        lat=_first_string(row, "lat"),
        lon=_first_string(row, "lon"),
        depth=_first_string(row, "depth"),
        country=_first_string(row, "country"),
        marine_region=_first_string(row, "marine_region"),
        environment_biome=_first_string(row, "environment_biome", "broad_scale_environmental_context"),
        environment_feature=_first_string(row, "environment_feature", "local_environmental_context"),
        environment_material=_first_string(row, "environment_material", "environmental_medium"),
        sample_collection=_first_string(row, "sample_collection"),
        extraction_protocol=_first_string(row, "extraction_protocol"),
        library_construction_protocol=_first_string(row, "library_construction_protocol"),
        marine_tag=_first_string(row, "marine"),
        marine_confidence=marine_confidence,
        marine_match_methods=marine_match_methods,
        sequence_accessibility_status=classify_sequence_accessibility(row),
        raw_json=json.dumps(row, sort_keys=True),
    )


class EnaPortalClient:
    source = "ena_portal"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def search_read_runs(
        self,
        query: str,
        *,
        marine_confidence: str,
        marine_match_methods: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[EnaRun]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.ena_portal_base_url.rstrip('/')}/search",
            params={
                "result": "read_run",
                "query": query,
                "fields": ENA_READ_RUN_FIELDS,
                "format": "json",
                "limit": limit or self.config.ena_page_size,
                "offset": offset,
            },
        )
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        runs: list[EnaRun] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed = parse_read_run(row, marine_confidence=marine_confidence, marine_match_methods=marine_match_methods)
            if parsed is not None:
                runs.append(parsed)
        return runs

