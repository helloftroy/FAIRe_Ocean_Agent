from __future__ import annotations

import json

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


class NcbiPublicationClient:
    source = "ncbi"

    def __init__(self, http: CachedHttpClient, config: SeedDiscoveryConfig):
        self.http = http
        self.config = config

    def _params(self, params: dict) -> dict:
        if self.config.ncbi_api_key:
            params = {**params, "api_key": self.config.ncbi_api_key}
        return params

    def _bioproject_uid(self, accession: str) -> str | None:
        payload = self.http.get_json(
            self.source,
            f"{self.config.ncbi_eutils_base_url.rstrip('/')}/esearch.fcgi",
            params=self._params({"db": "bioproject", "term": accession, "retmode": "json", "retmax": 5}),
        )
        ids = (((payload.get("esearchresult") or {}).get("idlist")) or []) if isinstance(payload, dict) else []
        return str(ids[0]) if ids else None

    def pubmed_for_bioproject(self, accession: str) -> list[PublicationCandidate]:
        uid = self._bioproject_uid(accession)
        if not uid:
            return []
        payload = self.http.get_json(
            self.source,
            f"{self.config.ncbi_eutils_base_url.rstrip('/')}/elink.fcgi",
            params=self._params({"dbfrom": "bioproject", "db": "pubmed", "id": uid, "retmode": "json"}),
        )
        linksets = payload.get("linksets", []) if isinstance(payload, dict) else []
        pmids: list[str] = []
        for linkset in linksets:
            for linkdb in linkset.get("linksetdbs", []) or []:
                for link in linkdb.get("links", []) or []:
                    pmids.append(str(link))
        return [
            PublicationCandidate(
                pmid=pmid,
                match_method="ncbi_link",
                matched_identifier=accession,
                match_confidence=MatchConfidence.VERY_HIGH,
                match_score=95.0,
                raw_json=json.dumps({"bioproject_uid": uid, "pmid": pmid}, sort_keys=True),
            )
            for pmid in sorted(set(pmids))
        ]
