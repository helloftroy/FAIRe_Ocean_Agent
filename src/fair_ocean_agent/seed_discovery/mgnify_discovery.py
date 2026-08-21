from __future__ import annotations

import logging
import signal
from collections import Counter

from fair_ocean_agent.seed_discovery.clients.ena import EnaXrefClient
from fair_ocean_agent.seed_discovery.clients.europepmc import EuropePmcSeedClient
from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.clients.mgnify import MgnifyClient, parse_study
from fair_ocean_agent.seed_discovery.clients.ncbi import NcbiPublicationClient
from fair_ocean_agent.seed_discovery.clients.openalex import OpenAlexSeedClient
from fair_ocean_agent.seed_discovery.config import RunLimits, SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.filters import experiment_type_status, is_marine_study
from fair_ocean_agent.seed_discovery.publication_resolver import PublicationResolver

logger = logging.getLogger(__name__)


class StopRequested(Exception):
    pass


class MgnifySeedDiscoveryRunner:
    def __init__(self, config: SeedDiscoveryConfig):
        self.config = config
        self.db = SeedDiscoveryDB(config.db_path)
        self.http = CachedHttpClient(config, self.db)
        self.stop_requested = False

    def close(self) -> None:
        self.http.close()
        self.db.close()

    def install_signal_handlers(self) -> None:
        def _request_stop(signum, frame) -> None:  # noqa: ANN001
            logger.warning("received signal %s; stopping after current study", signum)
            self.stop_requested = True

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    def run(self, limits: RunLimits) -> Counter:
        self.db.initialize()
        mgnify = MgnifyClient(self.http, self.config)
        resolver = PublicationResolver(
            self.db,
            self.config,
            mgnify=mgnify,
            ena=EnaXrefClient(self.http, self.config),
            ncbi=NcbiPublicationClient(self.http, self.config),
            openalex=OpenAlexSeedClient(self.http, self.config),
            europepmc=EuropePmcSeedClient(self.http, self.config),
        )
        counts: Counter = Counter()
        self.db.mark_run_started("mgnify_studies")
        if limits.refresh:
            logger.info("refresh requested; clearing API response cache while preserving discovered studies/candidates")
            self.db.clear_api_cache()
        try:
            if not limits.resolve_only:
                self._discover_studies(mgnify, limits, counts)
            self._resolve_publications(resolver, limits, counts)
            self.db.update_crawl_state("mgnify_studies", status="completed", completed=True)
        except Exception as exc:
            self.db.update_crawl_state("mgnify_studies", status="error", error=str(exc))
            raise
        return counts

    def _discover_studies(self, mgnify: MgnifyClient, limits: RunLimits, counts: Counter) -> None:
        if limits.accession:
            payload = self.http.get_json(
                "mgnify",
                f"{self.config.mgnify_base_url.rstrip('/')}/studies/{limits.accession}",
                use_cache=not limits.refresh,
            )
            study = parse_study(payload)
            counts["mgnify_studies_scanned"] += 1
            if is_marine_study(study, self.config):
                self.db.upsert_study(study)
                counts["marine_studies_accepted"] += 1
            return

        cursor = self.db.crawl_cursor("mgnify_studies") if limits.resume else None
        start_page = int(cursor) if cursor and str(cursor).isdigit() else 1
        accepted = 0
        for page, payload in mgnify.iter_study_payloads(start_page=start_page, max_pages=limits.max_pages):
            if self.stop_requested:
                raise StopRequested()
            counts["mgnify_studies_scanned"] += 1
            study = parse_study(payload)
            if is_marine_study(study, self.config):
                self.db.upsert_study(study)
                counts["marine_studies_accepted"] += 1
                counts[experiment_type_status(study, self.config)] += 1
                accepted += 1
                if limits.max_studies is not None and accepted >= limits.max_studies:
                    break
            self.db.update_crawl_state("mgnify_studies", cursor=str(page), status="running")

    def _resolve_publications(self, resolver: PublicationResolver, limits: RunLimits, counts: Counter) -> None:
        rows = self.db.studies_for_resolution(refresh=limits.refresh, limit=limits.max_studies)
        for row in rows:
            if self.stop_requested:
                raise StopRequested()
            status = resolver.resolve_study(row)
            counts[f"publication_status_{status.value}"] += 1
            logger.info("resolved %s -> %s", row["mgnify_accession"], status.value)
        counts["mgnify_studies_total"] = self.db.count("mgnify_studies")
        counts["publication_candidates_total"] = self.db.count("publication_candidates")
