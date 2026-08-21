from __future__ import annotations

import os

import pytest

from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.clients.mgnify import MgnifyClient, parse_study
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB


@pytest.mark.skipif(
    os.environ.get("FAIR_OCEAN_RUN_MGNIFY_INTEGRATION") != "1",
    reason="set FAIR_OCEAN_RUN_MGNIFY_INTEGRATION=1 to query live MGnify",
)
def test_live_mgnify_v2_returns_parseable_studies(tmp_path):
    db = SeedDiscoveryDB(tmp_path / "live_mgnify.sqlite")
    db.initialize()
    config = SeedDiscoveryConfig(db_path=tmp_path / "live_mgnify.sqlite", page_size=5)
    http = CachedHttpClient(config, db)
    try:
        payloads = [payload for _page, payload in MgnifyClient(http, config).iter_study_payloads(max_pages=1)]
    finally:
        http.close()
        db.close()

    assert 1 <= len(payloads) <= 5
    studies = [parse_study(payload) for payload in payloads]
    assert all(study.mgnify_accession for study in studies)
