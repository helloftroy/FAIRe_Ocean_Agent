from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.models import MatchConfidence, PublicationCandidate


_ACCESSION_EDGE_RE = re.compile(r"^[\s\(\[\{<\"']+|[\s\)\]\}>\"'.,;:]+$")


def normalize_epmc_accession(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _ACCESSION_EDGE_RE.sub("", str(value).strip()).upper()
    return normalized or None


def accession_evidence_level(accession: str, database_name: str | None = None) -> str:
    value = accession.upper()
    database = (database_name or "").casefold()
    if database == "bioproject" or value.startswith(("PRJNA", "PRJEB", "PRJDB")):
        return "bioproject"
    if value.startswith(("SRP", "ERP", "DRP")):
        return "study"
    if value.startswith("MGYS") or database == "metagenomics":
        return "mgnify"
    if value.startswith(("SAMN", "SAMEA", "SAMD", "SRS", "ERS", "DRS")) or database == "biosample":
        return "biosample"
    if value.startswith(("SRX", "ERX", "DRX")):
        return "experiment"
    if value.startswith(("SRR", "ERR", "DRR")):
        return "run"
    if value.startswith(("GCA", "GCF")) or database == "gca":
        return "assembly"
    return "other"


_LEVEL_WEIGHTS = {
    "bioproject": 10.0,
    "study": 9.0,
    "mgnify": 7.0,
    "biosample": 3.0,
    "experiment": 3.0,
    "run": 2.0,
    "assembly": 4.0,
    "other": 1.0,
}
_LEVEL_CAPS = {
    "biosample": 12.0,
    "experiment": 9.0,
    "run": 10.0,
}


@dataclass
class DatasetAccessions:
    bioproject: str | None = None
    study_accessions: list[str] = field(default_factory=list)
    biosamples: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    assemblies: list[str] = field(default_factory=list)
    mgnify_accession: str | None = None

    def normalized_by_level(self) -> dict[str, list[str]]:
        values = {
            "bioproject": [self.bioproject] if self.bioproject else [],
            "study": self.study_accessions,
            "biosample": self.biosamples,
            "experiment": self.experiments,
            "run": self.runs,
            "assembly": self.assemblies,
            "mgnify": [self.mgnify_accession] if self.mgnify_accession else [],
        }
        out: dict[str, list[str]] = {}
        for level, raw_values in values.items():
            seen: set[str] = set()
            for raw_value in raw_values:
                normalized = normalize_epmc_accession(raw_value)
                if normalized and normalized not in seen:
                    seen.add(normalized)
            out[level] = sorted(seen)
        return out


def _article_key(row) -> tuple[str, str]:  # noqa: ANN001
    doi = row["mapped_normalized_doi"]
    if doi:
        return ("doi", str(doi))
    pmid = row["mapped_pmid"] or (row["article_external_id"] if row["article_source"] == "MED" else None)
    if pmid:
        return ("pmid", str(pmid))
    pmcid = row["mapped_pmcid"] or row["pmcid"]
    if pmcid:
        return ("pmcid", str(pmcid))
    return (str(row["article_source"] or ""), str(row["article_external_id"] or ""))


def _confidence(score: float, matched: dict[str, list[str]]) -> MatchConfidence:
    if matched.get("bioproject") or matched.get("study") or score >= 12:
        return MatchConfidence.VERY_HIGH
    if matched.get("mgnify") or matched.get("biosample") or matched.get("experiment") or matched.get("run") or score >= 7:
        return MatchConfidence.HIGH
    return MatchConfidence.MEDIUM


class LocalEuropePmcResolver:
    def __init__(self, db: SeedDiscoveryDB):
        self.db = db

    def resolve_publications_from_accessions(self, accessions: list[str]) -> list[PublicationCandidate]:
        dataset = DatasetAccessions()
        by_level: dict[str, list[str]] = defaultdict(list)
        for accession in accessions:
            normalized = normalize_epmc_accession(accession)
            if not normalized:
                continue
            by_level[accession_evidence_level(normalized)].append(normalized)
        dataset.bioproject = next(iter(by_level["bioproject"]), None)
        dataset.study_accessions = by_level["study"]
        dataset.biosamples = by_level["biosample"]
        dataset.experiments = by_level["experiment"]
        dataset.runs = by_level["run"]
        dataset.assemblies = by_level["assembly"]
        dataset.mgnify_accession = next(iter(by_level["mgnify"]), None)
        return self.resolve_publication_for_dataset(dataset)

    def resolve_publication_for_dataset(self, dataset: DatasetAccessions) -> list[PublicationCandidate]:
        by_level = dataset.normalized_by_level()
        accession_to_level = {
            accession: level
            for level, accessions in by_level.items()
            for accession in accessions
        }
        rows = self.db.epmc_links_for_accessions(accession_to_level.keys())
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for row in rows:
            grouped[_article_key(row)].append(row)

        candidates: list[PublicationCandidate] = []
        for rows_for_article in grouped.values():
            matched: dict[str, set[str]] = defaultdict(set)
            article_sources: set[str] = set()
            source_files: set[str] = set()
            doi = None
            pmid = None
            pmcid = None
            for row in rows_for_article:
                accession = str(row["normalized_accession"])
                level = accession_to_level.get(accession) or accession_evidence_level(accession, row["database_name"])
                matched[level].add(accession)
                article_sources.add(str(row["article_source"] or ""))
                source_files.add(str(row["source_file"] or ""))
                doi = doi or row["mapped_doi"]
                pmid = pmid or row["mapped_pmid"] or (row["article_external_id"] if row["article_source"] == "MED" else None)
                pmcid = pmcid or row["mapped_pmcid"] or row["pmcid"]

            score = _score_matches(matched)
            matched_lists = {level: sorted(values) for level, values in matched.items()}
            raw = {
                "resolution_source": "europe_pmc_bulk_accessions",
                "matched_accessions": matched_lists,
                "article_sources": sorted(value for value in article_sources if value),
                "source_files": sorted(value for value in source_files if value),
                "score": score,
            }
            candidates.append(
                PublicationCandidate(
                    doi=doi,
                    pmid=str(pmid) if pmid else None,
                    pmcid=pmcid,
                    match_method="europe_pmc_bulk_accessions",
                    matched_identifier="|".join(
                        accession
                        for values in matched_lists.values()
                        for accession in values
                    ),
                    match_confidence=_confidence(score, matched_lists),
                    match_score=score,
                    raw_json=json.dumps(raw, sort_keys=True),
                )
            )
        return sorted(candidates, key=lambda candidate: (-candidate.match_score, candidate.doi or candidate.pmid or ""))


def _score_matches(matched: dict[str, set[str]]) -> float:
    score = 0.0
    for level, accessions in matched.items():
        contribution = _LEVEL_WEIGHTS.get(level, 1.0) * len(accessions)
        if level in _LEVEL_CAPS:
            contribution = min(contribution, _LEVEL_CAPS[level])
        score += contribution
    return score
