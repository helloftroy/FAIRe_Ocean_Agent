from __future__ import annotations

import logging
import json
import re
from dataclasses import replace
from difflib import SequenceMatcher

import httpx

from fair_ocean_agent.seed_discovery.clients.ena import EnaXrefClient
from fair_ocean_agent.seed_discovery.clients.europepmc import EuropePmcSeedClient
from fair_ocean_agent.seed_discovery.clients.mgnify import MgnifyClient
from fair_ocean_agent.seed_discovery.clients.ncbi import NcbiPublicationClient
from fair_ocean_agent.seed_discovery.clients.openalex import OpenAlexSeedClient
from fair_ocean_agent.seed_discovery.clients.crossref import CrossrefSeedClient
from fair_ocean_agent.seed_discovery.config import SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB, choose_primary_candidate, normalize_doi_for_seed
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate, ResolutionStatus

logger = logging.getLogger(__name__)

_NON_PRIMARY_PUBLICATION_TYPES = {
    "review",
    "editorial",
    "correction",
    "commentary",
    "letter",
}
_PUBLICATION_LIKE_TITLE_RE = re.compile(
    r"\b(?:analysis|diversity|community|communities|microbiome|metagenom|metatranscriptom|"
    r"microbial|bacterial|archaeal|fungal|biogeography|composition|structure|ecology|"
    r"distribution|abundance|sediment|water|marine|soil|biofilm|revealed|using|approach)\b",
    re.IGNORECASE,
)


class OpenAlexRateLimitError(RuntimeError):
    """Raised when OpenAlex rate-limits a seed discovery run."""


def _openalex_rate_limit_error(exc: httpx.HTTPStatusError) -> OpenAlexRateLimitError:
    return OpenAlexRateLimitError(
        "OpenAlex returned 429 Too Many Requests; exiting immediately. "
        "Rerun later, or use --no-openalex to continue discovery without OpenAlex."
    )


class PublicationResolver:
    def __init__(
        self,
        db: SeedDiscoveryDB,
        config: SeedDiscoveryConfig,
        *,
        mgnify: MgnifyClient,
        ena: EnaXrefClient,
        ncbi: NcbiPublicationClient,
        openalex: OpenAlexSeedClient,
        europepmc: EuropePmcSeedClient,
        crossref: CrossrefSeedClient | None = None,
    ):
        self.db = db
        self.config = config
        self.mgnify = mgnify
        self.ena = ena
        self.ncbi = ncbi
        self.openalex = openalex
        self.europepmc = europepmc
        self.crossref = crossref

    def resolve_study(self, study_row) -> ResolutionStatus:
        study_id = int(study_row["id"])
        accessions = self._accessions(study_row)
        try:
            candidates: list[PublicationCandidate] = []
            openalex_attempted = False
            openalex_retryable = False
            openalex_skipped = False
            candidates.extend(self.mgnify.publications(study_row["mgnify_accession"]))

            for accession in accessions:
                candidates.extend(self.ena.publications_for_accession(accession))
            for accession in accessions:
                if accession.startswith("PRJ"):
                    candidates.extend(self.ncbi.pubmed_for_bioproject(accession))

            candidates = self._enrich_pmids(candidates)

            if not self._has_high_confidence_identifier_candidate(candidates):
                for accession in accessions:
                    if self.config.openalex_enabled:
                        openalex_attempted = True
                        try:
                            candidates.extend(self.openalex.accession_search(accession))
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 429:
                                raise _openalex_rate_limit_error(exc) from exc
                            else:
                                raise
                    else:
                        openalex_skipped = True
                    candidates.extend(self.europepmc.accession_search(accession))

            if not self._has_high_confidence_identifier_candidate(candidates):
                title = str(study_row["study_name"] or "").strip()
                if self._publication_like_title(title):
                    title_candidates: list[PublicationCandidate] = []
                    if self.config.openalex_enabled:
                        openalex_attempted = True
                        try:
                            title_candidates.extend(self.openalex.title_search(title))
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 429:
                                raise _openalex_rate_limit_error(exc) from exc
                            else:
                                raise
                    else:
                        openalex_skipped = True
                        candidates.append(self._title_only_candidate(title, source="mgnify"))
                    title_candidates.extend(self.europepmc.title_search(title))
                    candidates.extend(self._validated_title_candidates(title, title_candidates))

            if not candidates and self.config.metadata_search_enabled:
                query = self._metadata_query(study_row)
                if query:
                    if self.config.openalex_enabled:
                        openalex_attempted = True
                        try:
                            candidates.extend(self.openalex.metadata_search(query))
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 429:
                                raise _openalex_rate_limit_error(exc) from exc
                            else:
                                raise
                    else:
                        openalex_skipped = True

            candidates = [candidate for candidate in candidates if self._plausible_publication(candidate)]
            for candidate in candidates:
                self.db.upsert_publication_candidate(study_id, candidate)
            primary_id, status, reason = choose_primary_candidate(self.db.candidates_for_study(study_id))
            if primary_id is None and (openalex_retryable or openalex_attempted or openalex_skipped):
                status = ResolutionStatus.OPENALEX_REPROCESS
                reason = "OpenAlex was disabled; reprocess later." if openalex_skipped and not openalex_attempted else (
                    "OpenAlex did not produce a resolvable publication candidate; "
                    "Europe PMC fallback was attempted when possible. Reprocess later."
                )
            self.db.set_primary(study_id, primary_id, status, reason)
            return status
        except OpenAlexRateLimitError:
            raise
        except Exception as exc:
            logger.exception("publication resolution failed for %s", study_row["mgnify_accession"])
            self.db.set_primary(study_id, None, ResolutionStatus.API_ERROR, str(exc))
            return ResolutionStatus.API_ERROR

    def resolve_ena_study(self, study_row) -> ResolutionStatus:
        study_id = int(study_row["id"])
        accessions = self._ena_accessions(study_row)
        try:
            candidates: list[PublicationCandidate] = []
            openalex_attempted = False
            openalex_retryable = False
            openalex_skipped = False

            for accession in accessions:
                candidates.extend(self.ena.publications_for_accession(accession))
            for accession in accessions:
                if accession.startswith("PRJ"):
                    candidates.extend(self.ncbi.pubmed_for_bioproject(accession))

            candidates = self._enrich_pmids(candidates)

            if not self._has_high_confidence_identifier_candidate(candidates):
                for accession in accessions:
                    candidates.extend(self.europepmc.accession_search(accession))

            title = str(study_row["study_title"] or study_row["project_name"] or "").strip()
            if not self._has_high_confidence_identifier_candidate(candidates) and self._publication_like_title(title):
                title_candidates: list[PublicationCandidate] = []
                title_candidates.extend(self.europepmc.title_search(title))
                if self.crossref is not None:
                    title_candidates.extend(self.crossref.title_search(title))
                candidates.extend(self._validated_title_candidates(title, title_candidates))

            if not self._has_high_confidence_identifier_candidate(candidates):
                if self.config.openalex_enabled:
                    for accession in accessions:
                        openalex_attempted = True
                        try:
                            candidates.extend(self.openalex.accession_search(accession))
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 429:
                                raise _openalex_rate_limit_error(exc) from exc
                            raise
                    if not openalex_retryable and self._publication_like_title(title):
                        openalex_attempted = True
                        try:
                            candidates.extend(self._validated_title_candidates(title, self.openalex.title_search(title)))
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 429:
                                raise _openalex_rate_limit_error(exc) from exc
                            else:
                                raise
                else:
                    openalex_skipped = True
                    if self._publication_like_title(title):
                        candidates.append(self._title_only_candidate(title, source="ena"))

            candidates = [candidate for candidate in candidates if self._plausible_publication(candidate)]
            for candidate in candidates:
                self.db.upsert_ena_publication_candidate(study_id, candidate)
            primary_id, status, reason = choose_primary_candidate(self.db.candidates_for_ena_study(study_id))
            if primary_id is None and (openalex_retryable or openalex_attempted or openalex_skipped):
                status = ResolutionStatus.OPENALEX_REPROCESS
                reason = "OpenAlex was disabled; reprocess later." if openalex_skipped and not openalex_attempted else (
                    "OpenAlex did not produce a resolvable publication candidate; "
                    "reprocess deferred OpenAlex work later."
                )
            self.db.set_ena_primary(study_id, primary_id, status, reason)
            return status
        except OpenAlexRateLimitError:
            raise
        except Exception as exc:
            logger.exception("ENA publication resolution failed for %s", study_row["canonical_dataset_id"])
            self.db.set_ena_primary(study_id, None, ResolutionStatus.API_ERROR, str(exc))
            return ResolutionStatus.API_ERROR

    def _accessions(self, study_row) -> list[str]:
        values = [
            study_row["bioproject_accession"],
            study_row["secondary_study_accession"],
            study_row["mgnify_accession"],
        ]
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if not value:
                continue
            value = str(value).upper()
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def _ena_accessions(self, study_row) -> list[str]:
        values = [
            study_row["bioproject_accession"],
            study_row["secondary_study_accession"],
            study_row["ena_study_accession"],
            study_row["canonical_dataset_id"],
        ]
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if not value:
                continue
            value = str(value).upper()
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

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
                        publication_date=resolved.publication_date or candidate.publication_date,
                        publication_year=resolved.publication_year or candidate.publication_year,
                        publication_type=resolved.publication_type or candidate.publication_type,
                    )
            enriched.append(candidate)
        return enriched

    def _has_high_confidence_identifier_candidate(self, candidates: list[PublicationCandidate]) -> bool:
        for candidate in candidates:
            if candidate.doi or candidate.pmid or candidate.pmcid:
                if candidate.match_confidence in {MatchConfidence.VERY_HIGH, MatchConfidence.HIGH}:
                    return True
        return False

    def _metadata_query(self, study_row) -> str | None:
        parts = [study_row["study_name"], study_row["centre_name"]]
        year = str(study_row["public_release_date"] or "")[:4]
        if year.isdigit():
            parts.append(year)
        query = " ".join(str(part) for part in parts if part)
        return query or None

    def _publication_like_title(self, title: str) -> bool:
        if len(title.split()) < 5:
            return False
        return bool(_PUBLICATION_LIKE_TITLE_RE.search(title))

    def _title_only_candidate(self, title: str, *, source: str) -> PublicationCandidate:
        return PublicationCandidate(
            title=title,
            match_method=f"{source}_publication_like_title",
            matched_identifier=title,
            match_confidence=MatchConfidence.LOW,
            match_score=1.0,
            raw_json=json.dumps(
                {
                    "note": (
                        f"{source} study title looked publication-like; "
                        "OpenAlex disabled, queued for later title resolution."
                    )
                },
                sort_keys=True,
            ),
        )

    def _validated_title_candidates(
        self,
        study_title: str,
        candidates: list[PublicationCandidate],
    ) -> list[PublicationCandidate]:
        validated: list[PublicationCandidate] = []
        for candidate in candidates:
            if not candidate.title:
                continue
            score = title_similarity(study_title, candidate.title)
            if score < 0.92:
                continue
            confidence = MatchConfidence.HIGH if score >= 0.98 else MatchConfidence.MEDIUM
            validated.append(
                replace(
                    candidate,
                    match_confidence=confidence,
                    match_score=max(candidate.match_score, score * 100),
                )
            )
        return validated

    def _plausible_publication(self, candidate: PublicationCandidate) -> bool:
        if not (candidate.doi or candidate.pmid or candidate.pmcid or candidate.openalex_id or candidate.title):
            return False
        pub_type = (candidate.publication_type or "").casefold()
        if pub_type in _NON_PRIMARY_PUBLICATION_TYPES and candidate.match_confidence == MatchConfidence.LOW:
            return False
        if candidate.doi and normalize_doi_for_seed(candidate.doi) is None:
            return False
        return True


def normalize_title_for_match(title: str | None) -> str:
    value = (title or "").casefold()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(?:a|an|the)\b", " ", value)
    return " ".join(value.split())


def title_similarity(left: str | None, right: str | None) -> float:
    left_norm = normalize_title_for_match(left)
    right_norm = normalize_title_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()
