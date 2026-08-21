from __future__ import annotations

import json
import logging
import signal
from collections import Counter
from dataclasses import dataclass

from fair_ocean_agent.seed_discovery.clients.crossref import CrossrefSeedClient
from fair_ocean_agent.seed_discovery.clients.ena import EnaXrefClient
from fair_ocean_agent.seed_discovery.clients.ena_portal import EnaPortalClient
from fair_ocean_agent.seed_discovery.clients.europepmc import EuropePmcSeedClient
from fair_ocean_agent.seed_discovery.clients.http import CachedHttpClient
from fair_ocean_agent.seed_discovery.clients.mgnify import MgnifyClient
from fair_ocean_agent.seed_discovery.clients.ncbi import NcbiPublicationClient
from fair_ocean_agent.seed_discovery.clients.openalex import OpenAlexSeedClient
from fair_ocean_agent.seed_discovery.config import RunLimits, SeedDiscoveryConfig
from fair_ocean_agent.seed_discovery.db import SeedDiscoveryDB
from fair_ocean_agent.seed_discovery.models import EnaStudy
from fair_ocean_agent.seed_discovery.publication_resolver import PublicationResolver

logger = logging.getLogger(__name__)


class StopRequested(Exception):
    pass


@dataclass(frozen=True)
class EnaQueryPartition:
    name: str
    query: str
    marine_confidence: str
    marine_match_methods: str


_ACCESSIBILITY_RANK = {
    "fastq_confirmed": 4,
    "submitted_reads_confirmed": 3,
    "sra_archive_confirmed": 2,
    "sequence_locator_present_unverified": 1,
    "no_downloadable_reads": 0,
}
_MARINE_RANK = {"high": 3, "medium": 2, "low": 1}


def build_ena_query_partitions(config: SeedDiscoveryConfig, *, include_secondary: bool = False) -> list[EnaQueryPartition]:
    partitions: list[EnaQueryPartition] = []
    tags = list(config.ena_marine_tags_primary)
    if include_secondary:
        tags.extend(config.ena_marine_tags_secondary)
    for tag in tags:
        if "high_confidence" in tag:
            confidence = "high"
        elif "medium_confidence" in tag:
            confidence = "medium"
        else:
            confidence = "low"
        safe = tag.replace(":", "_")
        partitions.append(
            EnaQueryPartition(
                name=f"tag_{safe}",
                query=f'marine="{tag}"',
                marine_confidence=confidence,
                marine_match_methods=f"ena_marine_tag:{tag}",
            )
        )
    if include_secondary:
        for tax_id in config.ena_marine_tax_ids:
            partitions.append(
                EnaQueryPartition(
                    name=f"tax_tree_{tax_id}",
                    query=f"tax_tree({tax_id})",
                    marine_confidence="medium",
                    marine_match_methods=f"marine_taxonomy:{tax_id}",
                )
            )
        for term in config.ena_marine_terms:
            escaped = term.replace('"', " ")
            slug = "_".join(escaped.casefold().split())[:40]
            partitions.append(
                EnaQueryPartition(
                    name=f"term_{slug}",
                    query=(
                        f'study_title="{escaped}" OR sample_title="{escaped}" OR '
                        f'isolation_source="{escaped}" OR environment_biome="{escaped}" OR '
                        f'environment_feature="{escaped}" OR environment_material="{escaped}"'
                    ),
                    marine_confidence="low",
                    marine_match_methods=f"marine_keyword:{term}",
                )
            )
    return partitions


def _first_nonempty(rows, key: str) -> str | None:  # noqa: ANN001
    for row in rows:
        value = row[key]
        if value not in (None, ""):
            return str(value)
    return None


def _unique_count(rows, key: str) -> int:  # noqa: ANN001
    return len({str(row[key]) for row in rows if row[key] not in (None, "")})


def _count_present(rows, key: str) -> int:  # noqa: ANN001
    return sum(1 for row in rows if row[key] not in (None, ""))


def _pct(count: int, total: int) -> float:
    return round((count / total) * 100, 2) if total else 0.0


def _bytes_total(rows, key: str) -> int:  # noqa: ANN001
    total = 0
    for row in rows:
        value = row[key]
        if not value:
            continue
        for part in str(value).replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit():
                total += int(part)
    return total


def _best_accessibility(rows) -> str:  # noqa: ANN001
    statuses = [str(row["sequence_accessibility_status"]) for row in rows]
    return max(statuses, key=lambda status: _ACCESSIBILITY_RANK.get(status, 0), default="no_downloadable_reads")


def _best_marine_confidence(rows) -> str:  # noqa: ANN001
    values = [str(row["marine_confidence"] or "low") for row in rows]
    return max(values, key=lambda value: _MARINE_RANK.get(value, 0), default="low")


def _metadata_completeness(rows) -> tuple[dict, int]:  # noqa: ANN001
    sample_count = _unique_count(rows, "sample_accession") or _unique_count(rows, "secondary_sample_accession")
    denominator = sample_count or len(rows)
    metrics = {
        "sample_count": sample_count,
        "run_count": len(rows),
        "samples_with_collection_date": _count_present(rows, "collection_date"),
        "pct_samples_with_collection_date": _pct(_count_present(rows, "collection_date"), denominator),
        "samples_with_lat_lon": sum(1 for row in rows if row["lat"] not in (None, "") and row["lon"] not in (None, "")),
        "pct_samples_with_lat_lon": _pct(sum(1 for row in rows if row["lat"] not in (None, "") and row["lon"] not in (None, "")), denominator),
        "samples_with_depth": _count_present(rows, "depth"),
        "pct_samples_with_depth": _pct(_count_present(rows, "depth"), denominator),
        "samples_with_environment_biome": _count_present(rows, "environment_biome"),
        "samples_with_environment_feature": _count_present(rows, "environment_feature"),
        "samples_with_environment_material": _count_present(rows, "environment_material"),
        "samples_with_sample_collection": _count_present(rows, "sample_collection"),
        "experiments_with_target_gene": _count_present(rows, "target_gene"),
        "experiments_with_extraction_protocol": _count_present(rows, "extraction_protocol"),
        "experiments_with_library_protocol": _count_present(rows, "library_construction_protocol"),
    }
    score = 0
    score += int(metrics["samples_with_collection_date"] > 0)
    score += int(metrics["samples_with_lat_lon"] > 0)
    score += int(metrics["samples_with_depth"] > 0)
    score += int(any(metrics[key] > 0 for key in ("samples_with_environment_biome", "samples_with_environment_feature", "samples_with_environment_material")))
    score += int(metrics["samples_with_sample_collection"] > 0)
    score += int(any(_count_present(rows, key) > 0 for key in ("library_strategy", "library_source")))
    score += int(metrics["experiments_with_target_gene"] > 0)
    score += int(any(metrics[key] > 0 for key in ("experiments_with_extraction_protocol", "experiments_with_library_protocol")))
    return metrics, score


def aggregate_ena_study(rows) -> EnaStudy:  # noqa: ANN001
    bioproject = _first_nonempty(rows, "bioproject_accession")
    secondary = _first_nonempty(rows, "secondary_study_accession")
    study_accession = _first_nonempty(rows, "study_accession")
    canonical = bioproject or secondary or study_accession
    if canonical is None:
        raise ValueError("cannot aggregate ENA study without a study/project accession")
    metadata, score = _metadata_completeness(rows)
    downloadable = sum(1 for row in rows if _ACCESSIBILITY_RANK.get(str(row["sequence_accessibility_status"]), 0) >= 1)
    fastq_runs = sum(1 for row in rows if row["sequence_accessibility_status"] == "fastq_confirmed")
    methods = sorted({str(row["marine_match_methods"]) for row in rows if row["marine_match_methods"]})
    tags = sorted({str(row["marine_tag"]) for row in rows if row["marine_tag"]})
    raw = {
        "run_accessions": [row["run_accession"] for row in rows],
        "sequence_accessibility_statuses": sorted({row["sequence_accessibility_status"] for row in rows}),
    }
    return EnaStudy(
        canonical_dataset_id=str(canonical),
        ena_study_accession=study_accession,
        secondary_study_accession=secondary,
        bioproject_accession=bioproject,
        bioproject_resolution_method="ena_direct" if bioproject else None,
        bioproject_status="resolved" if bioproject else "unresolved",
        ncbi_bioproject_verified=False,
        study_title=_first_nonempty(rows, "study_title"),
        project_name=_first_nonempty(rows, "project_name"),
        centre_name=_first_nonempty(rows, "centre_name"),
        first_public=_first_nonempty(rows, "first_public"),
        marine_confidence=_best_marine_confidence(rows),
        marine_match_methods="|".join(methods) if methods else None,
        marine_tags="|".join(tags) if tags else None,
        sample_count=metadata["sample_count"],
        run_count=len(rows),
        downloadable_run_count=downloadable,
        fastq_run_count=fastq_runs,
        fastq_bytes_total=_bytes_total(rows, "fastq_bytes"),
        sequence_accessibility_status=_best_accessibility(rows),
        metadata_completeness_json=json.dumps(metadata, sort_keys=True),
        metadata_usefulness_score=score,
        raw_json=json.dumps(raw, sort_keys=True),
    )


class EnaSeedDiscoveryRunner:
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
            logger.warning("received signal %s; stopping after current ENA page", signum)
            self.stop_requested = True

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    def run(self, limits: RunLimits) -> Counter:
        self.db.initialize()
        counts: Counter = Counter()
        self.db.mark_run_started("ena_read_run")
        try:
            if limits.refresh:
                logger.info("refresh requested; clearing API response cache while preserving discovered ENA studies/candidates")
                self.db.clear_api_cache()
            if limits.phase in {"all", "discovery"}:
                self._discover_runs(limits, counts)
                self._aggregate_studies(counts)
            if limits.phase in {"all", "publications"}:
                self._resolve_publications(limits, counts)
            self.db.update_crawl_state("ena_read_run", status="completed", completed=True)
        except Exception as exc:
            self.db.update_crawl_state("ena_read_run", status="error", error=str(exc))
            raise
        return counts

    def _discover_runs(self, limits: RunLimits, counts: Counter) -> None:
        client = EnaPortalClient(self.http, self.config)
        pages_seen = 0
        for partition in build_ena_query_partitions(self.config, include_secondary=limits.include_secondary):
            state_key = f"ena_read_run:{partition.name}"
            offset = 0
            cursor = self.db.crawl_cursor(state_key) if limits.resume else None
            if cursor and str(cursor).isdigit():
                offset = int(cursor)
            while True:
                if self.stop_requested:
                    raise StopRequested()
                if limits.max_pages is not None and pages_seen >= limits.max_pages:
                    return
                runs = client.search_read_runs(
                    partition.query,
                    marine_confidence=partition.marine_confidence,
                    marine_match_methods=partition.marine_match_methods,
                    limit=self.config.ena_page_size,
                    offset=offset,
                )
                pages_seen += 1
                counts["ena_pages_scanned"] += 1
                counts[f"ena_pages_{partition.name}"] += 1
                if not runs:
                    self.db.update_crawl_state(state_key, cursor=str(offset), status="completed", completed=True)
                    break
                for run in runs:
                    self.db.upsert_ena_run(run)
                    counts["ena_read_runs_scanned"] += 1
                    counts[f"ena_runs_{run.sequence_accessibility_status}"] += 1
                offset += len(runs)
                self.db.update_crawl_state(state_key, cursor=str(offset), status="running")
                if len(runs) < self.config.ena_page_size:
                    self.db.update_crawl_state(state_key, cursor=str(offset), status="completed", completed=True)
                    break

    def _aggregate_studies(self, counts: Counter) -> None:
        for _key, rows in self.db.ena_run_groups().items():
            study = aggregate_ena_study(rows)
            self.db.upsert_ena_study(study)
            counts["ena_candidate_studies"] += 1
            counts[f"ena_studies_{study.sequence_accessibility_status}"] += 1

    def _resolve_publications(self, limits: RunLimits, counts: Counter) -> None:
        resolver = PublicationResolver(
            self.db,
            self.config,
            mgnify=MgnifyClient(self.http, self.config),
            ena=EnaXrefClient(self.http, self.config),
            ncbi=NcbiPublicationClient(self.http, self.config),
            openalex=OpenAlexSeedClient(self.http, self.config),
            europepmc=EuropePmcSeedClient(self.http, self.config),
            crossref=CrossrefSeedClient(self.http, self.config),
        )
        rows = self.db.ena_studies_for_resolution(refresh=limits.refresh, limit=limits.max_studies)
        for row in rows:
            if self.stop_requested:
                raise StopRequested()
            status = resolver.resolve_ena_study(row)
            counts[f"ena_publication_status_{status.value}"] += 1
            logger.info("resolved ENA %s -> %s", row["canonical_dataset_id"], status.value)
        counts["ena_studies_total"] = self.db.count("ena_studies")
        counts["ena_runs_total"] = self.db.count("ena_runs")
        counts["publication_candidates_total"] = self.db.count("publication_candidates")

