from __future__ import annotations

import json

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


def _first(value) -> str | None:  # noqa: ANN001
    if isinstance(value, list) and value:
        return str(value[0])
    if value:
        return str(value)
    return None


def parse_crossref_work(item: dict, *, matched_identifier: str, confidence: MatchConfidence) -> PublicationCandidate:
    issued = item.get("published-print") or item.get("published-online") or item.get("issued") or {}
    parts = (issued.get("date-parts") or [[None]])[0]
    year = parts[0] if parts and isinstance(parts[0], int) else None
    date = "-".join(f"{part:02d}" if i else str(part) for i, part in enumerate(parts) if isinstance(part, int)) or None
    return PublicationCandidate(
        doi=item.get("DOI"),
        title=_first(item.get("title")),
        publication_date=date,
        publication_year=year,
        publication_type=item.get("type"),
        match_method="crossref_metadata",
        matched_identifier=matched_identifier,
        match_confidence=confidence,
        match_score=80.0 if confidence == MatchConfidence.HIGH else 60.0,
        raw_json=json.dumps(item, sort_keys=True),
    )


class CrossrefSeedClient:
    source = "crossref"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def title_search(self, title: str, *, limit: int = 10) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            f"{self.config.crossref_base_url.rstrip('/')}/works",
            params={"query.title": title, "rows": limit},
        )
        items = ((payload.get("message") or {}).get("items") or []) if isinstance(payload, dict) else []
        return [
            parse_crossref_work(item, matched_identifier=title, confidence=MatchConfidence.MEDIUM)
            for item in items
            if isinstance(item, dict)
        ]

