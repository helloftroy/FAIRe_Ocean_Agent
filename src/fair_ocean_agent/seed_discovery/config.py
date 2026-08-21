from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from fair_ocean_agent.config import REPO_ROOT

# Same env var the main pipeline's own retrieval.user_agent reads
# (config.py's FAIR_OCEAN_CONTACT_EMAIL) -- one place to update the
# contact address used in outbound requests, not two. Unlike the main
# pipeline's own load_config(), nothing on this module's own import path
# was loading .env at all -- confirmed live, os.environ.get returned the
# literal fallback below even with FAIR_OCEAN_CONTACT_EMAIL set in .env,
# since nothing had ever read that file into the process environment for
# a run that only imports seed_discovery.*, never fair_ocean_agent.config.
# override=False matches load_config()'s own contract: a real shell
# export always wins over .env, .env only fills in what's otherwise unset.
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

_DEFAULT_CONTACT_EMAIL = "parkmhelen@gmail.com"


def _contact_email() -> str:
    return os.environ.get("FAIR_OCEAN_CONTACT_EMAIL", _DEFAULT_CONTACT_EMAIL)


@dataclass(frozen=True)
class SeedDiscoveryConfig:
    db_path: Path = Path("data/seed_discovery/mgnify_paper_seeds.sqlite")
    page_size: int = 100
    request_timeout_seconds: float = 30.0
    min_request_interval_seconds: float = 1.0
    source_min_request_interval_seconds: dict[str, float] | None = None
    max_retries: int = 5
    retry_base_seconds: float = 1.0
    # Discovery defaults are intentionally broad: keep environmental records
    # unless the biome/title/abstract is clearly human, clinical, built-
    # environment, or animal-host-associated. The detailed downstream pipeline
    # can narrow marine relevance later with richer context.
    accepted_biome_terms: tuple[str, ...] = ()
    rejected_biome_terms: tuple[str, ...] = (
        "human",
        "human gut",
        "human skin",
        "host-associated:human",
        "built environment",
        "built-environment",
        "indoor",
        "household",
        "hospital",
        "healthcare",
        "clinical",
        "patient",
        "disease",
        "oral",
        "skin",
        "gut",
        "fecal",
        "faecal",
        "stool",
        "vaginal",
        "animal",
        "host-associated:animal",
        "mammal",
        "mouse",
        "murine",
        "rat",
        "bovine",
        "cattle",
        "cow",
        "porcine",
        "pig",
        "swine",
        "chicken",
        "poultry",
        "rumen",
        "livestock",
        "companion animal",
        "canine",
        "dog",
        "feline",
        "cat",
    )
    accepted_experiment_types: tuple[str, ...] = (
        "amplicon",
        "metagenomic",
        "metatranscriptomic",
    )
    metadata_search_enabled: bool = False
    metadata_search_year_window: int = 3
    mgnify_base_url: str = "https://www.ebi.ac.uk/metagenomics/api/v2"
    ena_xref_base_url: str = "https://www.ebi.ac.uk/ena/xref/rest/json/search"
    ena_portal_base_url: str = "https://www.ebi.ac.uk/ena/portal/api"
    europepmc_base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    crossref_base_url: str = "https://api.crossref.org"
    openalex_enabled: bool = True
    openalex_base_url: str = "https://api.openalex.org"
    openalex_mailto: str | None = field(default_factory=_contact_email)
    openalex_api_key: str | None = None
    ncbi_eutils_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ncbi_api_key: str | None = None
    user_agent: str = field(default_factory=lambda: f"fair-ocean-agent-mgnify-seed-discovery/0.1 (mailto:{_contact_email()})")
    ena_marine_tags_primary: tuple[str, ...] = (
        "marine:high_confidence",
        "marine:medium_confidence",
        "coastal_brackish:high_confidence",
        "coastal_brackish:medium_confidence",
    )
    ena_marine_tags_secondary: tuple[str, ...] = (
        "marine:low_confidence",
        "coastal_brackish:low_confidence",
    )
    ena_marine_tax_ids: tuple[str, ...] = ("408172", "1561972", "412755")
    ena_marine_terms: tuple[str, ...] = (
        "marine",
        "ocean",
        "oceanic",
        "seawater",
        "sea water",
        "coastal",
        "estuary",
        "estuarine",
        "brackish",
        "mangrove",
        "salt marsh",
        "coral reef",
        "reef",
        "pelagic",
        "benthic",
        "intertidal",
        "subtidal",
        "continental shelf",
        "deep sea",
        "deep-sea",
        "hydrothermal vent",
        "marine sediment",
        "seafloor",
        "sea ice",
        "sea-ice",
    )
    ena_page_size: int = 100

    def request_interval_for_source(self, source: str) -> float:
        source_intervals = self.source_min_request_interval_seconds or {
            "openalex": 5.0,
            "europepmc": 1.0,
            "crossref": 1.0,
            "ncbi": 1.0,
            "ena_portal": 1.0,
            "ena_xref": 1.0,
            "mgnify": 1.0,
        }
        return source_intervals.get(source, self.min_request_interval_seconds)


@dataclass(frozen=True)
class RunLimits:
    max_pages: int | None = None
    max_studies: int | None = None
    resolve_only: bool = False
    refresh: bool = False
    resume: bool = True
    accession: str | None = None
    phase: str = "all"
    include_secondary: bool = False
