"""Offline tests for extract_structured_facts/parse_publication_fields/
find_related given hand-built `raw` dicts matching what fetch_record
produces -- same pattern as test_source_adapters_parsing.py for Milestone
2. HTTP+XML-parsing itself (fetch_record) is covered separately in
test_ncbi_ena_http.py via httpx.MockTransport.
"""
from datetime import datetime, timezone

import pytest

from fair_ocean_agent.database.enums import EntityLevel, IdentifierType
from fair_ocean_agent.sources.base import SourceConfig, SourceRecord
from fair_ocean_agent.sources.ena import EnaAdapter
from fair_ocean_agent.sources.ncbi import NcbiBioProjectAdapter, NcbiBioSampleAdapter


def _record(source_name: str, raw: dict, external_identifier: str = "PRJNA1425045") -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier=external_identifier,
        raw=raw,
        retrieved_at=datetime.now(timezone.utc),
        content_hash="deadbeef",
    )


@pytest.fixture
def bioproject_adapter(retrieval_config):
    adapter = NcbiBioProjectAdapter(
        SourceConfig(name="ncbi_bioproject", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils"),
        retrieval_config,
    )
    yield adapter
    adapter.close()


@pytest.fixture
def biosample_adapter(retrieval_config):
    adapter = NcbiBioSampleAdapter(
        SourceConfig(name="ncbi_biosample", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils"),
        retrieval_config,
    )
    yield adapter
    adapter.close()


@pytest.fixture
def ena_adapter(retrieval_config):
    adapter = EnaAdapter(
        SourceConfig(name="ena", enabled=True, base_url="https://www.ebi.ac.uk/ena/portal/api"), retrieval_config
    )
    yield adapter
    adapter.close()


def test_bioproject_extract_structured_facts(bioproject_adapter):
    raw = {
        "uid": "1425045",
        "accession": "PRJNA1425045",
        "name": None,
        "title": "SF Bay 18S Metabarcoding Monitoring",
        "description": "Results from a survey of filtered seawater samples.",
        "organism": None,
        "submitted": "2026-02-17",
    }
    facts = bioproject_adapter.extract_structured_facts(_record("ncbi_bioproject", raw))

    field_names = {f.fact_type_candidate for f in facts}
    assert field_names == {"title", "description", "submitted"}
    assert all(f.entity_level == EntityLevel.PROJECT for f in facts)
    assert all(f.entity_external_id is None for f in facts)  # project-level, not a sub-entity


def test_biosample_extract_structured_facts_creates_per_sample_entities(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN55404186",
                "title": "MIMS Environmental sample",
                "attributes": {"collection_date": "2023-12-06", "depth": "1", "lat_lon": "38.03 N 122.15 W"},
            },
            {
                "accession": "SAMN55404185",
                "title": "MIMS Environmental sample",
                "attributes": {"collection_date": "2023-12-07"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    assert len(facts) == 4  # 3 attrs for sample 1 + 1 for sample 2
    assert all(f.entity_level == EntityLevel.SAMPLE for f in facts)
    sample_1_facts = [f for f in facts if f.entity_external_id == "SAMN55404186"]
    assert len(sample_1_facts) == 3
    assert {f.fact_type_candidate for f in sample_1_facts} == {"collection_date", "depth", "lat_lon"}


def test_biosample_extract_structured_facts_notes_truncation(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 837,
        "truncated": True,
        "samples": [{"accession": "SAMN1", "title": None, "attributes": {"depth": "1"}}],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    note = next(f for f in facts if f.fact_type_candidate == "biosample_coverage_note")
    assert "837" in note.raw_value
    assert note.entity_external_id is None  # a project-level note, not tied to one sample


def test_biosample_find_related_returns_biosample_accessions(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {"accession": "SAMN1", "title": None, "attributes": {}},
            {"accession": "SAMN2", "title": None, "attributes": {}},
        ],
    }
    related = biosample_adapter.find_related(_record("ncbi_biosample", raw))
    assert {r.value for r in related} == {"SAMN1", "SAMN2"}
    assert all(r.identifier_type == IdentifierType.BIOSAMPLE_ACCESSION for r in related)


def test_ena_extract_structured_facts_splits_project_and_run_level(ena_adapter):
    raw = {
        "study": {
            "study_accession": "PRJNA1425045",
            "secondary_study_accession": "SRP677779",
            "study_title": "SF Bay 18S Metabarcoding Monitoring",
            "study_description": "desc",
            "center_name": "SFEI",
            "first_public": "2026-02-19",
        },
        "runs": [
            {
                "run_accession": "SRR1",
                "sample_accession": "SAMN1",
                "library_strategy": "AMPLICON",
                "library_source": "METAGENOMIC",
                "fastq_ftp": "ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1.fastq.gz",
                "fastq_bytes": "12345",
            }
        ],
        "truncated": False,
        "total_runs_seen": 1,
    }
    facts = ena_adapter.extract_structured_facts(_record("ena", raw))

    project_facts = [f for f in facts if f.entity_level == EntityLevel.PROJECT]
    run_facts = [f for f in facts if f.entity_level == EntityLevel.SEQUENCING_RUN]
    assert {f.fact_type_candidate for f in project_facts} == {
        "study_title", "study_description", "center_name", "first_public", "secondary_study_accession",
    }
    assert all(f.entity_external_id == "SRR1" for f in run_facts)
    assert {f.fact_type_candidate for f in run_facts} == {
        "sample_accession", "library_strategy", "library_source", "fastq_ftp", "fastq_bytes",
    }


def test_ena_find_related_disambiguates_secondary_accession_type(ena_adapter):
    raw = {
        "study": {
            "study_accession": "PRJNA1425045",
            "secondary_study_accession": "SRP677779",
        },
        "runs": [{"run_accession": "SRR1", "sample_accession": "SAMN1"}],
    }
    related = ena_adapter.find_related(_record("ena", raw, external_identifier="PRJNA1425045"))

    by_value = {r.value: r for r in related}
    assert by_value["SRP677779"].identifier_type == IdentifierType.SRA_STUDY_ACCESSION
    assert by_value["SAMN1"].identifier_type == IdentifierType.BIOSAMPLE_ACCESSION
    # study_accession matches the identifier we queried with -- not re-added
    assert "PRJNA1425045" not in by_value


def test_ena_find_related_adds_bioproject_accession_when_queried_by_ena_accession(ena_adapter):
    raw = {"study": {"study_accession": "PRJNA1425045", "secondary_study_accession": "SRP677779"}, "runs": []}
    related = ena_adapter.find_related(_record("ena", raw, external_identifier="SRP677779"))

    by_value = {r.value: r for r in related}
    assert by_value["PRJNA1425045"].identifier_type == IdentifierType.BIOPROJECT_ACCESSION
    assert "SRP677779" not in by_value  # matches the identifier we queried with
