from __future__ import annotations

import json

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


def parse_europepmc_result(item: dict, *, match_method: str, matched_identifier: str,
                           confidence: MatchConfidence) -> PublicationCandidate:
    year = item.get("pubYear") or item.get("journalInfo", {}).get("yearOfPublication")
    return PublicationCandidate(
        doi=item.get("doi"),
        pmid=str(item.get("pmid")) if item.get("pmid") else None,
        pmcid=item.get("pmcid"),
        title=item.get("title"),
        publication_date=item.get("firstPublicationDate") or item.get("electronicPublicationDate") or item.get("printPublicationDate"),
        publication_year=int(year) if str(year or "").isdigit() else None,
        publication_type=item.get("pubType") or item.get("publicationType"),
        match_method=match_method,
        matched_identifier=matched_identifier,
        match_confidence=confidence,
        match_score=90.0 if confidence == MatchConfidence.HIGH else 70.0,
        raw_json=json.dumps(item, sort_keys=True),
    )


class EuropePmcSeedClient:
    source = "europepmc"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def search(self, query: str, *, match_method: str, matched_identifier: str,
               confidence: MatchConfidence, limit: int = 25) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.europepmc_base_url.rstrip('/')}/search",
            params={"query": query, "format": "json", "pageSize": limit},
        )
        items = ((payload.get("resultList") or {}).get("result") or []) if isinstance(payload, dict) else []
        return [
            parse_europepmc_result(item, match_method=match_method, matched_identifier=matched_identifier, confidence=confidence)
            for item in items
            if isinstance(item, dict)
        ]

    def resolve_pmid(self, pmid: str) -> PublicationCandidate | None:
        results = self.search(f"EXT_ID:{pmid} AND SRC:MED", match_method="pmid_doi_resolution",
                              matched_identifier=pmid, confidence=MatchConfidence.VERY_HIGH, limit=1)
        return results[0] if results else None

    def accession_search(self, accession: str) -> list[PublicationCandidate]:
        return self.search(f'"{accession}"', match_method="europepmc_accession",
                           matched_identifier=accession, confidence=MatchConfidence.HIGH)

    def title_search(self, title: str) -> list[PublicationCandidate]:
        safe_title = title.replace('"', " ")
        return self.search(f'TITLE:"{safe_title}"', match_method="europepmc_title_search",
                           matched_identifier=title, confidence=MatchConfidence.MEDIUM, limit=10)
