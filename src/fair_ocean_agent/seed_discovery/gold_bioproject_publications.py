from __future__ import annotations

import logging
import signal
from collections import Counter
from dataclasses import replace

import httpx

from fair_ocean_agent.seed_discovery.clients.europepmc import EuropePmcSeedClient
from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.clients.ncbi import NcbiPublicationClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB, utc_iso
from fair_ocean_agent.seed_discovery.models import PublicationCandidate

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES = 5


class StopRequested(Exception):
    pass


def is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def _plausible(candidate: PublicationCandidate) -> bool:
    return bool(candidate.doi or candidate.pmid or candidate.pmcid or candidate.title)


class GoldBioprojectPublicationSearchRunner:
    """Searches Europe PMC + NCBI for the paper behind each real NCBI
    BioProject accession GOLD already recorded (gold_sequencing_projects.
    ncbi_bioproject_accession), instead of trusting GOLD's own bulk
    publication field -- see resolve_gold_primary_publications's own
    docstring for why that field alone isn't reliable (mostly OSTI data-
    report DOIs, not source papers).

    Deliberately skips OpenAlex entirely, per an explicit user request:
    a separate DISCOVER_IDENTIFIERS run against ~3000 papers is already
    hitting the main pipeline's own OpenAlex adapter (5 req/s configured)
    concurrently -- adding a second, independent OpenAlex-calling process
    on top of that would double the load on the one source that's already
    the tightest-margin dependency this pipeline has. Europe PMC full-text
    accession search and NCBI's own BioProject->PubMed ELink (a direct,
    structured cross-reference, not a fuzzy match -- VERY_HIGH confidence)
    are both real, independent, sufficient primary paths; MGnify/ENA's own
    resolver already treats OpenAlex as a fallback, not the primary
    method. Re-run with an OpenAlex pass added once the OpenAlex-heavy job
    is no longer running, for whatever this pass leaves unresolved.

    A candidate found for a bioproject_accession is attached to every
    (gold_study_id, gold_project_id) pair that accession belongs to in
    gold_sequencing_projects (normally exactly one), written into the same
    gold_study_publications table GOLD's own bulk field populates -- so it
    feeds straight into resolve_gold_primary_publications's existing
    fanout-based primary-selection pass with no further wiring.

    Resumable like every other seed-discovery runner: each bioproject
    accession's outcome is recorded in gold_bioproject_publication_search,
    so a re-run only processes accessions it hasn't already checked unless
    refresh=True."""

    def __init__(self, config: SeedDiscoveryConfig, db: SeedDiscoveryDB):
        self.config = config
        self.db = db
        self.http = CachedHttpClient(config, db)
        self.europepmc = EuropePmcSeedClient(self.http, config)
        self.ncbi = NcbiPublicationClient(self.http, config)
        self.stop_requested = False

    def close(self) -> None:
        self.http.close()

    def install_signal_handlers(self) -> None:
        def _request_stop(signum, frame) -> None:  # noqa: ANN001
            logger.warning("received signal %s; stopping after current bioproject", signum)
            self.stop_requested = True

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    def run(
        self,
        *,
        limit: int | None = None,
        refresh: bool = False,
        max_consecutive_rate_limit_failures: int | None = DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
    ) -> dict[str, int]:
        self.db.initialize()
        accessions = self._pending_accessions(refresh=refresh, limit=limit)
        counts: Counter = Counter()
        counts["accessions_to_check"] = len(accessions)
        consecutive_rate_limit_failures = 0
        stopped_reason: str | None = None

        for accession in accessions:
            if self.stop_requested:
                stopped_reason = "stop requested"
                break
            try:
                candidates = self.europepmc.accession_search(accession)
                candidates.extend(self.ncbi.pubmed_for_bioproject(accession))
                candidates = self._enrich_pmids(candidates)
            except Exception as exc:
                counts["errored"] += 1
                logger.warning("bioproject publication search failed for %s: %s", accession, exc)
                self.db.conn.execute(
                    """
                    INSERT INTO gold_bioproject_publication_search(bioproject_accession, status, candidates_found, error, checked_at)
                    VALUES (?, 'error', 0, ?, ?)
                    ON CONFLICT(bioproject_accession) DO UPDATE SET status='error', candidates_found=0, error=excluded.error, checked_at=excluded.checked_at
                    """,
                    (accession, str(exc), utc_iso()),
                )
                self.db.conn.commit()
                consecutive_rate_limit_failures = consecutive_rate_limit_failures + 1 if is_rate_limit_error(exc) else 0
                if max_consecutive_rate_limit_failures and consecutive_rate_limit_failures >= max_consecutive_rate_limit_failures:
                    stopped_reason = (
                        f"{consecutive_rate_limit_failures} consecutive bioprojects failed with 429 Too Many "
                        "Requests -- stopping to avoid hammering a source that's actively rate-limiting/"
                        "blocking this machine. Wait a while before retrying."
                    )
                    logger.warning(stopped_reason)
                    break
                continue
            consecutive_rate_limit_failures = 0

            plausible = [c for c in candidates if _plausible(c)]
            counts["accessions_checked"] += 1
            if plausible:
                counts["accessions_with_candidates"] += 1
                self._store_candidates(accession, plausible)
                counts["candidates_stored"] += len(plausible)
            self.db.conn.execute(
                """
                INSERT INTO gold_bioproject_publication_search(bioproject_accession, status, candidates_found, checked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(bioproject_accession) DO UPDATE SET status=excluded.status, candidates_found=excluded.candidates_found, error=NULL, checked_at=excluded.checked_at
                """,
                (accession, "found" if plausible else "not_found", len(plausible), utc_iso()),
            )
            self.db.conn.commit()

        counts["stopped_reason"] = stopped_reason  # type: ignore[assignment]
        return dict(counts)

    def _pending_accessions(self, *, refresh: bool, limit: int | None) -> list[str]:
        if refresh:
            query = """
                SELECT DISTINCT ncbi_bioproject_accession FROM gold_sequencing_projects
                WHERE ncbi_bioproject_accession IS NOT NULL AND ncbi_bioproject_accession != ''
                ORDER BY ncbi_bioproject_accession
            """
        else:
            query = """
                SELECT DISTINCT sp.ncbi_bioproject_accession FROM gold_sequencing_projects sp
                LEFT JOIN gold_bioproject_publication_search s ON s.bioproject_accession = sp.ncbi_bioproject_accession
                WHERE sp.ncbi_bioproject_accession IS NOT NULL AND sp.ncbi_bioproject_accession != ''
                  AND s.bioproject_accession IS NULL
                ORDER BY sp.ncbi_bioproject_accession
            """
        rows = self.db.conn.execute(query).fetchall()
        accessions = [row[0] for row in rows]
        return accessions[:limit] if limit is not None else accessions

    def _enrich_pmids(self, candidates: list[PublicationCandidate]) -> list[PublicationCandidate]:
        enriched: list[PublicationCandidate] = []
        for candidate in candidates:
            if candidate.pmid and not candidate.doi:
                resolved = self.europepmc.resolve_pmid(candidate.pmid)
                if resolved is not None:
                    candidate = replace(
                        candidate,
                        doi=resolved.doi or candidate.doi,
                        pmcid=resolved.pmcid or candidate.pmcid,
                        title=resolved.title or candidate.title,
                    )
            enriched.append(candidate)
        return enriched

    def _store_candidates(self, bioproject_accession: str, candidates: list[PublicationCandidate]) -> None:
        owners = self.db.conn.execute(
            "SELECT gold_study_id, gold_project_id FROM gold_sequencing_projects WHERE ncbi_bioproject_accession = ?",
            (bioproject_accession,),
        ).fetchall()
        now = utc_iso()
        for gold_study_id, gold_project_id in owners:
            for candidate in candidates:
                existing = self.db.conn.execute(
                    """
                    SELECT id FROM gold_study_publications
                    WHERE COALESCE(gold_study_id, '') = ? AND COALESCE(gold_project_id, '') = ?
                      AND COALESCE(doi, '') = ? AND COALESCE(pmid, '') = ? AND COALESCE(pmcid, '') = ?
                      AND match_method = ?
                    """,
                    (
                        gold_study_id or "",
                        gold_project_id or "",
                        candidate.doi or "",
                        candidate.pmid or "",
                        candidate.pmcid or "",
                        candidate.match_method,
                    ),
                ).fetchone()
                if existing:
                    continue
                self.db.conn.execute(
                    """
                    INSERT INTO gold_study_publications(
                        gold_study_id, gold_project_id, doi, pmid, pmcid, title, match_method,
                        matched_identifier, match_confidence, match_score, is_primary, source_snapshot_date,
                        raw_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        gold_study_id,
                        gold_project_id,
                        candidate.doi,
                        candidate.pmid,
                        candidate.pmcid,
                        candidate.title,
                        candidate.match_method,
                        candidate.matched_identifier,
                        candidate.match_confidence.value,
                        candidate.match_score,
                        "bioproject_search",
                        candidate.raw_json,
                        now,
                        now,
                    ),
                )
