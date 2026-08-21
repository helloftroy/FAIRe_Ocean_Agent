from __future__ import annotations

import json

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


class EnaXrefClient:
    source = "ena_xref"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def publications_for_accession(self, accession: str) -> list[PublicationCandidate]:
        payload = self.http.get_json(
            self.source,
            self.config.ena_xref_base_url,
            params={"accession": accession},
        )
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        candidates: list[PublicationCandidate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = str(row.get("Source") or row.get("source") or "")
            if "europepmc" not in source.casefold() and "pubmed" not in source.casefold():
                continue
            pmcid = row.get("Source primary accession") or row.get("sourcePrimaryAccession") or row.get("source_primary_accession")
            pmid = row.get("Source secondary accession") or row.get("sourceSecondaryAccession") or row.get("source_secondary_accession")
            candidates.append(
                PublicationCandidate(
                    pmid=str(pmid) if pmid else None,
                    pmcid=str(pmcid) if pmcid else None,
                    match_method="ena_xref",
                    matched_identifier=accession,
                    match_confidence=MatchConfidence.VERY_HIGH,
                    match_score=95.0,
                    raw_json=json.dumps(row, sort_keys=True),
                )
            )
        return candidates

