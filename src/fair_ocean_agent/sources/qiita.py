"""Qiita metadata adapter -- lightweight, deliberately narrow scope per an
explicit user request: "just want to indicate data is there and
downloadable, and know how many samples there are."

For a study that never got a real BioProject/ENA accession (see
scripts/qiita_seeds_to_csv.py's "qiita-only" seed case: real DOI + real
sequence data, genuinely never mirrored to ENA/SRA), this is the only way
DISCOVER_IDENTIFIERS can attach any sample-level structure at all --
without it, such a study would only ever get whatever its own paper's
full text happens to mention.

Unlike Zenodo/Dryad/Figshare/OSF (sources/zenodo.py etc.), whose records
carry no structured per-sample breakdown at all (hence
sequence_file_heuristics.py's single synthesized placeholder sample+run),
Qiita's own public sample_information export *is* a real per-sample
table -- one real row per real sample name. This emits one real SAMPLE +
EXPERIMENT_RUN pair per row instead of a single placeholder, so
"how many samples" is an actual count, not an inferred "unresolved
sample(s)" stand-in.

Deliberately light: one network call (the study's public
sample_information download), no prep-level scraping, no per-file
listing, no download of the raw sequence files themselves -- it marks
each sample's data as available at Qiita's own study page, not a
verified per-prep raw-data URL. If a richer per-prep resolution is ever
wanted later, seed_discovery/qiita_discovery.py already does that
scraping for its own reporting purposes; this adapter intentionally
doesn't duplicate it.

Physicochemical/environmental columns already present in the same
sample_information row (temperature/salinity/ph/oxygen/depth/lat-lon/
collection date/env_broad_scale/env_local_scale/env_medium/size
fraction) are emitted too, using the same fact_type_candidate names
mapping/rules.py's existing MappingRules already expect (temp/salinity/
ph/diss_oxygen/depth/latitude/longitude/collection_date/env_broad_scale/
env_local_scale/env_medium/size_frac) -- essentially free, since the row
is already being parsed for the sample name anyway. See
scripts/apply_gold_physicochemical_enrichment.py for the GOLD equivalent
of this same reasoning.
"""
from __future__ import annotations

from fair_ocean_agent.clock import utcnow
from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, SupportType
from fair_ocean_agent.seed_discovery.qiita_discovery import BIOSAMPLE_RE, first_tsv_from_zip, pick
from fair_ocean_agent.sources.base import (
    EntityLinkCandidate,
    RawFactCandidate,
    SearchPage,
    SearchQuery,
    SourceAdapter,
    SourceRecord,
    SourceRecordNotFoundError,
    hash_payload,
)

# (sample_information column via qiita_discovery.py's own FIELD_CANDIDATES,
# fact_type_candidate expected by an existing mapping/rules.py MappingRule
# at EntityLevel.SAMPLE)
QIITA_FIELD_TO_FACT_TYPE = (
    ("collection_date", "collection_date"),
    ("latitude", "latitude"),
    ("longitude", "longitude"),
    ("depth", "depth"),
    ("env_broad_scale", "env_broad_scale"),
    ("env_local_scale", "env_local_scale"),
    ("env_medium", "env_medium"),
    ("size_fraction", "size_frac"),
    ("temperature", "temp"),
    ("salinity", "salinity"),
    ("ph", "ph"),
    ("oxygen", "diss_oxygen"),
)


class QiitaAdapter(SourceAdapter):
    name = "qiita"

    def search(self, query: SearchQuery) -> SearchPage:
        # No search surface used for Qiita -- studies arrive already
        # identified via scripts/qiita_seeds_to_csv.py's own seed export,
        # same as every other repository-native adapter here.
        return SearchPage(records=[], total_count=0)

    def fetch_record(self, identifier: str) -> SourceRecord:
        study_id = identifier.strip()
        content, from_cache = self.http.get_binary(
            f"{self.config.base_url.rstrip('/')}/public_download/",
            params={"data": "sample_information", "study_id": study_id},
        )
        try:
            _name, sample_rows = first_tsv_from_zip(content)
        except Exception as exc:
            # Not a real zip (e.g. an HTML error/login page served with a
            # 200) -- treated the same as "no sample data", never a crash.
            raise SourceRecordNotFoundError(f"No Qiita sample_information for study {study_id} ({exc})") from exc
        if not sample_rows:
            raise SourceRecordNotFoundError(f"No Qiita sample_information for study {study_id}")

        payload = {"study_id": study_id, "sample_rows": sample_rows}
        return SourceRecord(
            source_name=self.name,
            external_identifier=study_id,
            url=f"{self.config.base_url.rstrip('/')}/study/description/{study_id}",
            raw=payload,
            retrieved_at=utcnow(),
            content_hash=hash_payload(payload),
            from_cache=from_cache,
        )

    def extract_structured_facts(self, record: SourceRecord) -> list[RawFactCandidate]:
        study_id = record.external_identifier
        sample_rows = record.raw.get("sample_rows") or []
        study_url = record.url
        facts: list[RawFactCandidate] = []

        for index, row in enumerate(sample_rows):
            sample_name = pick(row, "sample_id") or f"sample_{index}"
            biosample = pick(row, "biosample")
            # A real, well-formed NCBI/ENA/DDBJ BioSample accession is a
            # genuine shareable identifier (see Entity's own docstring on
            # SAMPLE/EXPERIMENT_RUN/SEQUENCING_RUN entities being
            # get-or-create-by-accession across studies) -- use it as this
            # sample's real external_identifier so it merges with the same
            # sample if it's ever also discovered via the normal ENA/NCBI
            # path, instead of creating a disconnected duplicate under a
            # synthetic internal id.
            sample_id = biosample if biosample and BIOSAMPLE_RE.fullmatch(biosample) else f"internal:qiita:{study_id}:{sample_name}"
            run_id = f"{sample_id}:run"
            run_links = [
                EntityLinkCandidate(
                    entity_level=EntityLevel.SAMPLE,
                    external_identifier=sample_id,
                    relationship_type=EntityRelationshipType.DERIVED_FROM_SAMPLE,
                    label=sample_name,
                )
            ]

            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.SAMPLE,
                    fact_type_candidate="samp_name",
                    raw_field_name="sample_id",
                    raw_value=sample_name,
                    source_locator=f"qiita.{study_id}.sample_information.{sample_name}",
                    entity_external_id=sample_id,
                    entity_label=sample_name,
                )
            )
            for qiita_field, fact_type in QIITA_FIELD_TO_FACT_TYPE:
                value = pick(row, qiita_field)
                if not value:
                    continue
                facts.append(
                    RawFactCandidate(
                        entity_level=EntityLevel.SAMPLE,
                        fact_type_candidate=fact_type,
                        raw_field_name=qiita_field,
                        raw_value=value,
                        source_locator=f"qiita.{study_id}.sample_information.{sample_name}.{qiita_field}",
                        entity_external_id=sample_id,
                        entity_label=sample_name,
                    )
                )

            # The "data is there and downloadable" marker -- deliberately
            # not a verified per-file listing (see the module docstring):
            # Qiita's own sample_information export existing at all for a
            # real sample name is itself the confirmation that this
            # sample's raw data is registered and downloadable from the
            # study's public page, support_type=INFERRED since no specific
            # file was individually checked.
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.EXPERIMENT_RUN,
                    fact_type_candidate="sequence_data_status",
                    raw_field_name="sequence_data_status",
                    raw_value="likely",
                    source_locator=f"qiita.{study_id}.sample_information_present",
                    support_type=SupportType.INFERRED,
                    entity_external_id=run_id,
                    entity_label=f"{sample_name} (Qiita raw data)",
                    entity_links=run_links,
                )
            )
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.EXPERIMENT_RUN,
                    fact_type_candidate="associatedSequences",
                    raw_field_name="qiita_study_url",
                    raw_value=study_url,
                    source_locator=f"qiita.{study_id}.study_url",
                    support_type=SupportType.INFERRED,
                    entity_external_id=run_id,
                    entity_label=f"{sample_name} (Qiita raw data)",
                )
            )
        return facts
