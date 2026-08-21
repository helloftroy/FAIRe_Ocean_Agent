from __future__ import annotations

import json

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


def parse_openalex_work(item: dict, *, match_method: str, matched_identifier: str,
                        confidence: MatchConfidence) -> PublicationCandidate:
    doi = item.get("doi")
    if isinstance(doi, str):
        doi = doi.replace("https://doi.org/", "")
    work_id = item.get("id")
    return PublicationCandidate(
        doi=doi,
        openalex_id=work_id.rsplit("/", 1)[-1] if isinstance(work_id, str) else None,
        title=item.get("title") or item.get("display_name"),
        publication_date=item.get("publication_date"),
        publication_year=item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
        publication_type=item.get("type"),
        match_method=match_method,
        matched_identifier=matched_identifier,
        match_confidence=confidence,
        match_score=85.0 if confidence == MatchConfidence.HIGH else 50.0,
        raw_json=json.dumps(item, sort_keys=True),
    )


class OpenAlexSeedClient:
    source = "openalex"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def _params(self, params: dict) -> dict:
        params = dict(params)
        if self.config.openalex_mailto:
            params["mailto"] = self.config.openalex_mailto
        if self.config.openalex_api_key:
            params["api_key"] = self.config.openalex_api_key
        return params

    def accession_search(self, accession: str) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.openalex_base_url.rstrip('/')}/works",
            params=self._params({"search.exact": accession, "per-page": 25}),
        )
        items = payload.get("results", []) if isinstance(payload, dict) else []
        candidates: list[PublicationCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = json.dumps(item, default=str)
            if accession.casefold() not in text.casefold():
                continue
            candidates.append(
                parse_openalex_work(
                    item,
                    match_method="openalex_accession",
                    matched_identifier=accession,
                    confidence=MatchConfidence.HIGH if not accession.startswith("MGYS") else MatchConfidence.MEDIUM,
                )
            )
        return candidates

    def title_search(self, title: str) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.openalex_base_url.rstrip('/')}/works",
            params=self._params({"search": title, "per-page": 10}),
        )
        items = payload.get("results", []) if isinstance(payload, dict) else []
        return [
            parse_openalex_work(item, match_method="openalex_title_search", matched_identifier=title,
                                confidence=MatchConfidence.MEDIUM)
            for item in items
            if isinstance(item, dict)
        ]

    def metadata_search(self, query: str) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.openalex_base_url.rstrip('/')}/works",
            params=self._params({"search": query, "per-page": 10}),
        )
        items = payload.get("results", []) if isinstance(payload, dict) else []
        return [
            parse_openalex_work(item, match_method="metadata_search", matched_identifier=query, confidence=MatchConfidence.LOW)
            for item in items
            if isinstance(item, dict)
        ]
