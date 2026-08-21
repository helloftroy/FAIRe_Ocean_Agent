"""ENA (European Nucleotide Archive) Portal API adapter: study/project
metadata plus read_run listing (library strategy/source/selection,
platform, file accession/size/checksum -- never the file itself, per
section 1's "raw data should not be downloaded").

Also serves as this pipeline's "NCBI SRA" equivalent: ENA mirrors the same
INSDC-shared run records NCBI SRA has, via a much simpler JSON API than
NCBI's SRA XML. See sources/ncbi.py's module docstring.
"""
from __future__ import annotations

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import (
    EntityLevel,
    EntityRelationshipType,
    IdentifierType,
    RelationshipType,
)
from fair_ocean_agent.identity.identifiers import guess_identifier_type
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import (
    EntityLinkCandidate,
    RawFactCandidate,
    RelatedIdentifier,
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    SourceRecordNotFoundError,
    hash_payload,
)

logger = get_logger(__name__)

STUDY_FIELDS = "study_accession,secondary_study_accession,study_title,study_description,center_name,first_public"
RUN_FIELDS = (
    "run_accession,sample_accession,experiment_accession,experiment_alias,experiment_title,experiment_target,"
    "library_name,library_strategy,library_source,library_selection,"
    "library_construction_protocol,instrument_platform,instrument_model,"
    "base_count,read_count,fastq_bytes,fastq_md5,fastq_ftp,first_public"
)

# Same rationale as NCBI's MAX_SAMPLES_PER_PROJECT: bound worst-case work per
# task against very large run collections; truncation is logged, not silent.
MAX_RUNS_PER_STUDY = 500


def _split_fastq_urls(fastq_ftp: str | None) -> list[str]:
    if not fastq_ftp:
        return []
    return [part.strip() for part in fastq_ftp.split(";") if part.strip()]


def _fastq_check_url(url: str) -> str:
    if url.startswith("ftp://"):
        return "https://" + url.removeprefix("ftp://")
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


class EnaAdapter(SourceAdapter):
    name = "ena"

    def fetch_record(self, identifier: str) -> SourceRecord:
        study_rows, _ = self.http.get_json(
            f"{self.config.base_url}/search",
            params={
                "result": "study",
                "query": f'study_accession="{identifier}" OR secondary_study_accession="{identifier}"',
                "fields": STUDY_FIELDS,
                "format": "json",
            },
        )
        if not study_rows:
            raise SourceRecordNotFoundError(f"No ENA study record for {identifier}")
        study = study_rows[0]

        run_rows, _ = self.http.get_json(
            f"{self.config.base_url}/search",
            params={
                "result": "read_run",
                "query": f'study_accession="{study["study_accession"]}"',
                "fields": RUN_FIELDS,
                "format": "json",
                "limit": MAX_RUNS_PER_STUDY + 1,
            },
        )
        total_runs = len(run_rows)
        truncated = total_runs > MAX_RUNS_PER_STUDY
        if truncated:
            logger.warning(
                "ENA study %s has more than %d runs; processing only the first %d "
                "(MAX_RUNS_PER_STUDY) -- not a silent drop, see raw_facts for the count.",
                identifier, MAX_RUNS_PER_STUDY, MAX_RUNS_PER_STUDY,
            )
        run_rows = run_rows[:MAX_RUNS_PER_STUDY]
        for run in run_rows:
            fastq_urls = _split_fastq_urls(run.get("fastq_ftp"))
            if not fastq_urls:
                run["fastq_access_status"] = "no_fastq_url"
                continue
            checked_urls = [_fastq_check_url(url) for url in fastq_urls]
            accessible = [self.http.url_accessible(url) for url in checked_urls]
            run["fastq_access_status"] = "accessible" if all(accessible) else "not_accessible"
            run["fastq_access_checked_urls"] = ";".join(checked_urls)

        raw = {"study": study, "runs": run_rows, "truncated": truncated, "total_runs_seen": total_runs}

        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=f"https://www.ebi.ac.uk/ena/browser/view/{study['study_accession']}",
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=hash_payload(raw),
        )

    def resolve_read_accession_to_study_accession(self, accession: str) -> str | None:
        """A sample- or run-level accession (SRS/ERS/DRS or SRR/ERR/DRR --
        cited directly in a paper's Data Availability statement far more
        often than the containing study accession, confirmed live:
        10.1073/pnas.2103275118 cites "SRS7105074 - SRS7105095" and never
        states its own study accession at all) resolves 1:1 to a real
        parent ENA study, but fetch_record above only understands
        study-level accessions. Used to translate a sample/run accession
        found by discovery/text_identifiers.py's identifier mining into the
        study accession this pipeline's existing resolution machinery
        already knows how to expand into the full sibling sample set,
        rather than adding a whole new identifier type and downstream
        handling path for two more accession flavors.

        Confirmed live that /filereport's accession= parameter resolves a
        run accession (e.g. SRR12335159 -> PRJNA649058) exactly the same
        way it resolves a sample accession -- it isn't sample-specific by
        construction, so one method serves both."""
        # confirmed live: /search with query=sample_accession="..." returns
        # an empty result set for a real, existing sample accession
        # (SRS7105074) -- that query field only indexes read_run-linked
        # searches by other fields, not a direct sample-accession lookup.
        # /filereport's own accession= parameter (built for exactly this:
        # looking up a specific accession's own records) resolves it
        # correctly.
        rows, _ = self.http.get_json(
            f"{self.config.base_url}/filereport",
            params={
                "accession": accession,
                "result": "read_run",
                "fields": "study_accession",
                "format": "json",
            },
        )
        if not rows:
            return None
        return rows[0].get("study_accession") or None

    def search(self, query: SearchQuery) -> SearchPage:
        rows, _ = self.http.get_json(
            f"{self.config.base_url}/search",
            params={"result": "study", "query": query.query, "fields": STUDY_FIELDS, "format": "json", "limit": query.limit},
        )
        records = []
        for row in rows:
            try:
                records.append(self.fetch_record(row["study_accession"]))
            except SourceRecordNotFoundError:
                continue
        return SearchPage(records=records, total_count=len(rows))

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        study = record.raw["study"]
        facts: list[RawFactCandidate] = []

        def add_study_fact(field: str) -> None:
            value = study.get(field)
            if value in (None, ""):
                return
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate=field,
                    raw_field_name=field,
                    raw_value=str(value),
                    source_locator=f"ena.study.{field}",
                )
            )

        for field in ("study_title", "study_description", "center_name", "first_public", "secondary_study_accession"):
            add_study_fact(field)

        if record.raw.get("truncated"):
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.PROJECT,
                    fact_type_candidate="run_coverage_note",
                    raw_field_name="run_coverage_note",
                    raw_value=(
                        f"Only the first {MAX_RUNS_PER_STUDY} of {record.raw['total_runs_seen']}+ "
                        f"runs were processed (MAX_RUNS_PER_STUDY cap)."
                    ),
                    source_locator="ena.truncation_note",
                )
            )

        for run in record.raw.get("runs", []):
            run_accession = run.get("run_accession")
            if not run_accession:
                continue
            for field in (
                "run_accession", "sample_accession", "library_strategy", "library_source", "library_selection",
                "library_construction_protocol", "instrument_platform", "instrument_model",
                "base_count", "read_count", "fastq_bytes", "fastq_md5", "fastq_ftp",
                "fastq_access_status", "fastq_access_checked_urls", "first_public",
            ):
                value = run.get(field)
                if value in (None, ""):
                    continue
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SEQUENCING_RUN,
                        fact_type_candidate=field,
                        raw_field_name=field,
                        raw_value=str(value),
                        source_locator=f"ena.read_run.{run_accession}.{field}",
                        entity_external_id=run_accession,
                        entity_label=run_accession,
                    )
                )

            # ENA's experiment is the library-producing instance; the run
            # is the physical sequencing event/file set. Keep the original
            # sequencing_run facts above for inventory and provenance, and
            # attach FAIRe row-shaped facts to a distinct experiment_run.
            experiment_accession = run.get("experiment_accession")
            library_name = run.get("library_name")
            experiment_id = experiment_accession or library_name or f"internal:ena:{run_accession}"
            sample_accession = run.get("sample_accession")
            links = [
                EntityLinkCandidate(
                    entity_level=EntityLevel.SEQUENCING_RUN,
                    external_identifier=run_accession,
                    relationship_type=EntityRelationshipType.SEQUENCED_IN_RUN,
                    label=run_accession,
                )
            ]
            if sample_accession:
                links.append(
                    EntityLinkCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        external_identifier=sample_accession,
                        relationship_type=EntityRelationshipType.DERIVED_FROM_SAMPLE,
                        label=sample_accession,
                    )
                )

            experiment_facts: list[tuple[str, str, str]] = []
            if library_name or experiment_accession:
                # ENA experiment accessions are archive-unique; submitter
                # library names are not guaranteed unique. Prefer the
                # accession for FAIRe's required unique lib_id and retain
                # library_name separately as source-native metadata.
                chosen_field = "experiment_accession" if experiment_accession else "library_name"
                experiment_facts.append(("lib_id", chosen_field, str(experiment_accession or library_name)))
            if sample_accession:
                experiment_facts.append(("samp_name", "sample_accession", str(sample_accession)))
            experiment_facts.extend(
                (
                    ("seq_run_id", "run_accession", str(run_accession)),
                    ("associatedSequences", "run_accession", str(run_accession)),
                )
            )
            for field in (
                "experiment_accession",
                "experiment_alias",
                "experiment_title",
                "experiment_target",
                "library_name",
                "library_strategy",
                "library_source",
                "library_selection",
                "library_construction_protocol",
                "instrument_platform",
                "instrument_model",
                "base_count",
                "read_count",
                "fastq_bytes",
                "fastq_md5",
                "fastq_ftp",
                "fastq_access_status",
                "fastq_access_checked_urls",
            ):
                value = run.get(field)
                if value not in (None, ""):
                    experiment_facts.append((field, field, str(value)))

            for fact_type, raw_field, value in experiment_facts:
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.EXPERIMENT_RUN,
                        fact_type_candidate=fact_type,
                        raw_field_name=raw_field,
                        raw_value=value,
                        source_locator=f"ena.read_run.{run_accession}.{raw_field}",
                        entity_external_id=str(experiment_id),
                        entity_label=str(library_name or experiment_accession or f"Library for {run_accession}"),
                        entity_links=links,
                    )
                )
        return facts

    def find_related(self, record: SourceRecord) -> list[RelatedIdentifier]:
        study = record.raw["study"]
        related: list[RelatedIdentifier] = []

        study_accession = study.get("study_accession")
        if study_accession and study_accession != record.external_identifier:
            related.append(
                RelatedIdentifier(
                    identifier_type=guess_identifier_type(study_accession) or IdentifierType.OTHER,
                    value=study_accession,
                    relationship_type=RelationshipType.RELATED_TO,
                    source=self.name,
                )
            )

        # secondary_study_accession is SRP/ERP/DRP (SRA-namespace) for most
        # studies -- guess_identifier_type disambiguates rather than
        # hardcoding one, since both SRA_STUDY_ACCESSION and
        # ENA_STUDY_ACCESSION overlap on the ERP prefix.
        secondary = study.get("secondary_study_accession")
        if secondary and secondary != record.external_identifier:
            related.append(
                RelatedIdentifier(
                    identifier_type=guess_identifier_type(secondary) or IdentifierType.OTHER,
                    value=secondary,
                    relationship_type=RelationshipType.RELATED_TO,
                    source=self.name,
                )
            )

        for run in record.raw.get("runs", []):
            sample_accession = run.get("sample_accession")
            if sample_accession:
                related.append(
                    RelatedIdentifier(
                        identifier_type=IdentifierType.BIOSAMPLE_ACCESSION,
                        value=sample_accession,
                        relationship_type=RelationshipType.CONTAINS_SAMPLES_FROM,
                        source=self.name,
                    )
                )
        return related
