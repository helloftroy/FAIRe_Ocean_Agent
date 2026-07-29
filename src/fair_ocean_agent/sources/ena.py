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
from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, RelationshipType
from fair_ocean_agent.identity.identifiers import guess_identifier_type
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import (
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
    "run_accession,sample_accession,library_strategy,library_source,library_selection,"
    "library_layout,instrument_platform,instrument_model,base_count,read_count,fastq_bytes,fastq_md5,"
    "fastq_ftp,first_public"
)

# Same rationale as NCBI's MAX_SAMPLES_PER_PROJECT: bound worst-case work per
# task against very large run collections; truncation is logged, not silent.
MAX_RUNS_PER_STUDY = 500


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

        raw = {"study": study, "runs": run_rows, "truncated": truncated, "total_runs_seen": total_runs}

        return SourceRecord(
            source_name=self.name,
            external_identifier=identifier,
            url=f"https://www.ebi.ac.uk/ena/browser/view/{study['study_accession']}",
            raw=raw,
            retrieved_at=utcnow(),
            content_hash=hash_payload(raw),
        )

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
                "sample_accession", "library_strategy", "library_source", "library_selection",
                "library_layout", "instrument_platform", "instrument_model", "base_count", "read_count",
                "fastq_bytes", "fastq_md5", "fastq_ftp", "first_public",
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
