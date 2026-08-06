"""Offline tests for extract_structured_facts/parse_publication_fields/
find_related given hand-built `raw` dicts matching what fetch_record
produces -- same pattern as test_source_adapters_parsing.py for Milestone
2. HTTP+XML-parsing itself (fetch_record) is covered separately in
test_ncbi_ena_http.py via httpx.MockTransport.
"""
from datetime import datetime, timezone

import pytest

from fair_ocean_agent.database.enums import EntityLevel, EntityRelationshipType, IdentifierType, SupportType
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


def test_biosample_extract_structured_facts_normalizes_location_and_host_aliases(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA242644",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN02700077",
                "title": "cue sample",
                "organism": {"taxonomy_name": "coral metagenome", "taxonomy_id": "496922"},
                "attributes": {
                    "geographic location": "USA: Florida Keys",
                    "cultivar": "crustose coralline algae (CCA)",
                },
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))
    values = {fact.fact_type_candidate: fact for fact in facts}

    assert values["geographic location"].raw_value == "USA: Florida Keys"
    assert values["geo_loc_name"].raw_value == "USA: Florida Keys"
    assert values["cultivar"].raw_value == "crustose coralline algae (CCA)"
    assert values["host_species"].raw_value == "crustose coralline algae (CCA)"
    assert values["organism"].raw_value == "coral metagenome"
    assert values["geo_loc_name"].source_locator.endswith("Attributes.geo_loc_name")


def test_biosample_extract_structured_facts_maps_owner_to_recorded_by(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA994076",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN36415090",
                "title": "pond sample",
                "owner": {"name": "University of Konstanz, Corentin Fournier"},
                "attributes": {"collection_date": "2022-04-11"},
            },
            {
                "accession": "SAMN36415091",
                "title": "pond sample",
                "owner": {"name": "University of Konstanz, Corentin Fournier"},
                "attributes": {"collection_date": "2022-04-11"},
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA994076"))

    recorded_by = [f for f in facts if f.fact_type_candidate == "recordedBy"]
    assert len(recorded_by) == 1
    assert recorded_by[0].entity_level == EntityLevel.STUDY
    assert recorded_by[0].raw_value == "University of Konstanz, Corentin Fournier"
    assert recorded_by[0].source_locator == "ncbi_biosample.SAMN36415090.Owner.Name"


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


def test_biosample_extract_structured_facts_excludes_mag_records_by_package(biosample_adapter):
    """Regression guard for a real finding (ISME J-adjacent marine-sediment
    audit, BioProject PRJNA529480): confirmed live against a real efetch
    that a metagenome-assembled-genome BioSample carries package="MIMAG.*"
    and attributes (assembly software, completeness score) a raw
    environmental sample never has -- these must never become SAMPLE
    entities alongside real samples."""
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_MAG_1",
                "title": "Metagenome-assembled genome: STUDY_SAMN123_MAG_00000024",
                "package": "MIMAG.host-associated.6.0",
                "attributes": {"assembly software": "metaSPAdes 3.14", "completeness score": "95.77"},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    assert facts == []


def test_biosample_extract_structured_facts_excludes_mag_records_by_title_fallback(biosample_adapter):
    """No package/Models field on this record -- falls back to the title
    text, which still says "Metagenome-assembled genome"."""
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_MAG_2",
                "title": "Metagenome-assembled genome: STUDY_SAMN124_MAG_00000036",
                "package": None,
                "attributes": {"assembly software": "metaSPAdes 3.14"},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    assert facts == []


def test_biosample_extract_structured_facts_keeps_a_real_raw_sample_unaffected(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_REAL",
                "title": "MIMS Environmental sample",
                "package": "Generic.1.0",
                "attributes": {"collection_date": "2019-07-02"},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    assert {f.fact_type_candidate for f in facts} == {"collection_date"}


def test_biosample_extract_structured_facts_derives_filter_facts_from_samp_mat_process(biosample_adapter):
    """Regression guard for a real gap (PeerJ-adjacent Indian Ocean
    prokaryote audit): samp_mat_process's free text carries a pore size
    ("0.22 um") that must land in size_frac, NEVER filter_diameter (a
    different concept and unit -- physical filter disc diameter in mm,
    per the real FAIRe schema)."""
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_WATER",
                "title": "MIMS Environmental sample",
                "attributes": {"samp_mat_process": "0.22 um cartridge filtration followed by DNA extraction"},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["size_frac"].raw_value == "0.22 um"
    assert by_type["filter_passive_active_0_1"].raw_value == "1"
    assert "filter_diameter" not in by_type
    assert by_type["size_frac"].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert by_type["size_frac"].raw_field_name == "samp_mat_process"


def test_biosample_extract_structured_facts_no_filter_facts_when_nothing_stated(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_SEDIMENT",
                "title": "MIMS Environmental sample",
                "attributes": {"samp_mat_process": "DNA extraction from sediment samples"},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    filter_fields = {"size_frac", "filter_diameter", "filter_material", "filter_name", "filter_passive_active_0_1"}
    assert not filter_fields & {f.fact_type_candidate for f in facts}


def test_biosample_extract_structured_facts_derives_filter_diameter_material_and_name(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_FILTER",
                "title": "MIMS Environmental sample",
                "attributes": {
                    "samp_mat_process": "Filtered through a 47mm cellulose ester filter (Merck Millipore)"
                },
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["filter_diameter"].raw_value == "47"
    assert by_type["filter_material"].raw_value == "cellulose ester"
    assert by_type["filter_name"].raw_value == "Millipore"


def test_biosample_extract_structured_facts_derives_depth_from_source_material_id(biosample_adapter):
    """Regression guard for a real gap: this submitter's source_material_id
    (generically "an identifier for the source material", not inherently
    about depth) embeds per-sample depth as a leading number, e.g.
    "3500 m V3-V4" -- confirmed real per-sample depths vary across samples
    that a study-wide broadcast fallback was incorrectly uniforming."""
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_3500M",
                "title": "MIMS Environmental sample",
                "attributes": {"source_material_id": "3500 m V3-V4"},
            },
            {
                "accession": "SAMN_SURFACE",
                "title": "MIMS Environmental sample",
                "attributes": {"source_material_id": "Overlaying water V3-V4"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    by_entity = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "depth"}
    assert by_entity["SAMN_3500M"].raw_value == "3500 m"
    assert by_entity["SAMN_3500M"].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert "SAMN_SURFACE" not in by_entity


def test_biosample_extract_structured_facts_detects_biological_rep_relation_from_sample_name_attribute(
    biosample_adapter,
):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "MIMS Environmental sample",
                "attributes": {"sample_name": "Site_A_rep1"},
            },
            {
                "accession": "SAMN2",
                "title": "MIMS Environmental sample",
                "attributes": {"sample_name": "Site_A_rep2"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert set(rep_facts) == {"SAMN1", "SAMN2"}
    assert rep_facts["SAMN1"].raw_value == "SAMN1 | SAMN2"  # accessions, never the sample_name text
    assert rep_facts["SAMN1"].raw_field_name == "sample_name"
    assert rep_facts["SAMN1"].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert rep_facts["SAMN1"].confidence_metadata == {
        "replicate_detection_signal": "explicit_rep_marker",
        "replicate_group_size": 2,
    }


def test_biosample_extract_structured_facts_uses_explicit_replicate_attribute_first(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA994076",
        "total_linked_samples": 4,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "MIMS Environmental sample",
                "attributes": {"sample_name": "Site_A_rep1", "replicate": "a"},
            },
            {
                "accession": "SAMN2",
                "title": "MIMS Environmental sample",
                "attributes": {"sample_name": "Site_A_rep2", "replicate": "a"},
            },
            {
                "accession": "SAMN3",
                "title": "MIMS Environmental sample",
                "attributes": {"sample_name": "Site_B_rep1", "replicate": "b"},
            },
            {
                "accession": "SAMN4",
                "title": "MIMS Environmental sample",
                "attributes": {"sample_name": "Site_B_rep2", "replicate": "b"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA994076"))

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert set(rep_facts) == {"SAMN1", "SAMN2", "SAMN3", "SAMN4"}
    assert rep_facts["SAMN1"].raw_value == "SAMN1 | SAMN2"
    assert rep_facts["SAMN3"].raw_value == "SAMN3 | SAMN4"
    assert all(f.raw_field_name == "replicate" for f in rep_facts.values())
    assert rep_facts["SAMN1"].confidence_metadata == {
        "replicate_detection_signal": "explicit_biosample_replicate_attribute",
        "replicate_group_size": 2,
        "replicate_value": "a",
    }

    biological_rep = [f for f in facts if f.fact_type_candidate == "biological_rep_presence"]
    assert len(biological_rep) == 1
    assert biological_rep[0].entity_level == EntityLevel.STUDY
    assert biological_rep[0].raw_value == "TRUE"
    assert biological_rep[0].raw_field_name == "replicate"


def test_biosample_extract_structured_facts_falls_back_to_title_for_replicate_detection(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {"accession": "SAMN1", "title": "Site_A_rep1", "attributes": {}},
            {"accession": "SAMN2", "title": "Site_A_rep2", "attributes": {}},
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    rep_facts = [f for f in facts if f.fact_type_candidate == "biological_rep_relation"]
    assert len(rep_facts) == 2
    assert rep_facts[0].raw_field_name == "title"


def test_biosample_extract_structured_facts_no_biological_rep_relation_without_sibling(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [{"accession": "SAMN1", "title": "Site_A_rep1", "attributes": {}}],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    assert not any(f.fact_type_candidate == "biological_rep_relation" for f in facts)


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
                "experiment_accession": "SRX1",
                "library_name": "LIB1",
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
    experiment_facts = [f for f in facts if f.entity_level == EntityLevel.EXPERIMENT_RUN]
    assert {f.fact_type_candidate for f in project_facts} == {
        "study_title", "study_description", "center_name", "first_public", "secondary_study_accession",
    }
    assert all(f.entity_external_id == "SRR1" for f in run_facts)
    assert {f.fact_type_candidate for f in run_facts} == {
        "run_accession", "sample_accession", "library_strategy", "library_source", "fastq_ftp", "fastq_bytes",
    }
    assert all(f.entity_external_id == "SRX1" for f in experiment_facts)
    by_type = {fact.fact_type_candidate: fact for fact in experiment_facts}
    assert by_type["lib_id"].raw_value == "SRX1"
    assert by_type["library_name"].raw_value == "LIB1"
    assert by_type["samp_name"].raw_value == "SAMN1"
    assert by_type["seq_run_id"].raw_value == "SRR1"
    assert {
        (link.entity_level, link.external_identifier, link.relationship_type)
        for link in by_type["lib_id"].entity_links
    } == {
        (EntityLevel.SAMPLE, "SAMN1", EntityRelationshipType.DERIVED_FROM_SAMPLE),
        (EntityLevel.SEQUENCING_RUN, "SRR1", EntityRelationshipType.SEQUENCED_IN_RUN),
    }


def test_ena_extract_structured_facts_includes_library_layout(ena_adapter):
    """library_layout (PAIRED/SINGLE) feeds mapping/rules.py's lib_layout
    rule -- added alongside that rule since neither did anything without
    the other."""
    raw = {
        "study": {"study_accession": "PRJNA1", "study_title": "t"},
        "runs": [{"run_accession": "SRR1", "sample_accession": "SAMN1", "library_layout": "PAIRED"}],
        "truncated": False,
        "total_runs_seen": 1,
    }
    facts = ena_adapter.extract_structured_facts(_record("ena", raw))
    run_facts = {f.fact_type_candidate: f.raw_value for f in facts if f.entity_level == EntityLevel.SEQUENCING_RUN}
    assert run_facts["library_layout"] == "PAIRED"


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
