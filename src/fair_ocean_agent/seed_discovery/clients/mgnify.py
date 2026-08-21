from __future__ import annotations

import json
import re
from typing import Iterator

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MatchConfidence, MgnifyStudy, PublicationCandidate

_BIOPROJECT_RE = re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.IGNORECASE)
_SECONDARY_STUDY_RE = re.compile(r"\b(?:SRP|ERP|DRP)\d+\b", re.IGNORECASE)


def _first_string(payload: dict, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _nested_string(payload: dict, *paths: tuple[str, ...]) -> str | None:
    for path in paths:
        current = payload
        for part in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if isinstance(current, str) and current.strip():
            return current.strip()
    return None


def _all_text(payload: dict) -> str:
    return json.dumps(payload, default=str)


def extract_insdc_identifiers(payload: dict) -> tuple[str | None, str | None]:
    text = _all_text(payload)
    bioproject = _first_string(payload, "bioproject", "bioproject_accession", "project_accession", "secondary_accession")
    if not bioproject or not _BIOPROJECT_RE.fullmatch(bioproject):
        match = _BIOPROJECT_RE.search(text)
        bioproject = match.group(0).upper() if match else None
    secondary = _first_string(payload, "ena_study_accession", "secondary_study_accession", "study_accession")
    if not secondary or not _SECONDARY_STUDY_RE.fullmatch(secondary):
        match = _SECONDARY_STUDY_RE.search(text)
        secondary = match.group(0).upper() if match else None
    return bioproject, secondary


def parse_study(payload: dict) -> MgnifyStudy:
    accession = _first_string(payload, "accession", "id", "study_accession")
    if accession is None:
        raise ValueError("MGnify study payload has no accession")
    bioproject, secondary = extract_insdc_identifiers(payload)
    # MGnify v2's real /studies/ payload nests biome as an object
    # ({"biome_name": "Soil", "lineage": "root:Environmental:Terrestrial:Soil"}),
    # not a flat string -- confirmed live against a real cached response.
    # _first_string only ever matches a plain string value, so this
    # always silently found nothing and every study's biome stayed
    # empty, which in turn starved is_marine_study's own filtering (see
    # its own comment) of its main intended signal. lineage is the more
    # useful half for filtering (a stable, structured taxonomy path
    # rather than free text), so it's kept alongside the human-readable
    # name in this same field rather than added as a new column that a
    # database created before this fix would need a migration to gain
    # (this module's initialize() is a plain CREATE TABLE IF NOT EXISTS,
    # same drift risk as the main pipeline's init-db/create_all()).
    biome_field = payload.get("biome")
    if isinstance(biome_field, dict):
        biome_name = _first_string(biome_field, "biome_name", "name")
        biome_lineage = _first_string(biome_field, "lineage")
        biome = f"{biome_name} ({biome_lineage})" if biome_name and biome_lineage else (biome_name or biome_lineage)
    else:
        biome = _first_string(payload, "biome", "biomes") or _nested_string(payload, ("relationships", "biome", "data", "id"))
    experiment_types = payload.get("experiment_types") or payload.get("experiment_type") or payload.get("analysis_types")
    if isinstance(experiment_types, (list, tuple)):
        experiment_types_value = " | ".join(str(v) for v in experiment_types if v)
    elif experiment_types is None:
        experiment_types_value = None
    else:
        experiment_types_value = str(experiment_types)
    sample_count = payload.get("sample_count") or payload.get("samples_count") or payload.get("num_samples")
    return MgnifyStudy(
        mgnify_accession=accession,
        bioproject_accession=bioproject,
        secondary_study_accession=secondary,
        study_name=_first_string(payload, "study_name", "name", "title"),
        study_abstract=_first_string(payload, "study_abstract", "abstract", "description"),
        centre_name=_first_string(payload, "centre_name", "center_name", "submitter"),
        public_release_date=_first_string(payload, "public_release_date", "release_date", "first_public"),
        sample_count=int(sample_count) if str(sample_count or "").isdigit() else None,
        biome=biome,
        experiment_types=experiment_types_value,
        mgnify_last_updated=_first_string(payload, "last_updated", "updated_at", "last_update"),
        raw_json=json.dumps(payload, sort_keys=True),
    )


def parse_publication(payload: dict) -> PublicationCandidate:
    doi = _first_string(payload, "doi", "DOI")
    pmid = _first_string(payload, "pmid", "pubmed_id", "pubmedId", "id")
    pmcid = _first_string(payload, "pmcid", "pmc_id")
    year = payload.get("publication_year") or payload.get("year")
    return PublicationCandidate(
        doi=doi,
        pmid=str(pmid) if pmid else None,
        pmcid=pmcid,
        title=_first_string(payload, "title", "article_title", "name"),
        publication_date=_first_string(payload, "publication_date", "published", "date"),
        publication_year=int(year) if str(year or "").isdigit() else None,
        publication_type=_first_string(payload, "publication_type", "type"),
        match_method="mgnify_publication",
        matched_identifier=pmid or doi,
        match_confidence=MatchConfidence.VERY_HIGH,
        match_score=100.0,
        raw_json=json.dumps(payload, sort_keys=True),
    )


class MgnifyClient:
    source = "mgnify"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def list_studies_page(self, page: int) -> dict:
        return self.http.get_json(
            self.source,
            f"{self.config.mgnify_base_url.rstrip('/')}/studies/",
            params={"page": page, "page_size": self.config.page_size},
        )

    def iter_study_payloads(self, *, start_page: int = 1, max_pages: int | None = None) -> Iterator[tuple[int, dict]]:
        page = start_page
        pages_seen = 0
        while True:
            payload = self.list_studies_page(page)
            items = payload.get("items") if isinstance(payload, dict) else None
            if not items:
                break
            for item in items:
                if isinstance(item, dict):
                    yield page, item
            pages_seen += 1
            if max_pages is not None and pages_seen >= max_pages:
                break
            page += 1

    def publications(self, mgnify_accession: str) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.mgnify_base_url.rstrip('/')}/studies/{mgnify_accession}/publications/",
        )
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        return [parse_publication(item) for item in items if isinstance(item, dict)]

