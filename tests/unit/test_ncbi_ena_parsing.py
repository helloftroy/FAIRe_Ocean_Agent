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

    # 3 attrs for sample 1 + 1 for sample 2. No study-level
    # biological_rep_presence fact anymore -- projectMetadata.biological_rep
    # is now derived entirely at map-time from biological_rep_relation
    # facts (mapping/faire.py::_apply_biological_rep_from_relations), not
    # emitted here.
    assert len(facts) == 4
    sample_facts = [f for f in facts if f.entity_level == EntityLevel.SAMPLE]
    assert len(sample_facts) == 4
    sample_1_facts = [f for f in facts if f.entity_external_id == "SAMN55404186"]
    assert len(sample_1_facts) == 3
    assert {f.fact_type_candidate for f in sample_1_facts} == {"collection_date", "depth", "lat_lon"}


def test_biosample_extract_structured_facts_logs_useful_title_as_samp_category(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA295",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN32539297",
                "title": "LM 7",
                "attributes": {"collection_date": "2023-12-06"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA295"))

    by_type = {f.fact_type_candidate: f for f in facts if f.entity_external_id == "SAMN32539297"}
    assert by_type["samp_category"].raw_value == "LM 7"
    assert by_type["samp_category"].raw_field_name == "title"
    assert by_type["samp_category"].source_locator == "ncbi_biosample.SAMN32539297.title"


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
                    "isolation source": "coral cue material",
                    "cultivar": "crustose coralline algae (CCA)",
                },
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))
    values = {fact.fact_type_candidate: fact for fact in facts}

    assert values["geographic location"].raw_value == "USA: Florida Keys"
    assert values["geo_loc_name"].raw_value == "USA: Florida Keys"
    assert values["isolation source"].raw_value == "coral cue material"
    assert values["isolation_source"].raw_value == "coral cue material"
    assert values["cultivar"].raw_value == "crustose coralline algae (CCA)"
    assert values["host_species"].raw_value == "crustose coralline algae (CCA)"
    assert values["organism"].raw_value == "coral metagenome"
    assert values["geo_loc_name"].source_locator.endswith("Attributes.geo_loc_name")
    assert values["isolation_source"].source_locator.endswith("Attributes.isolation_source")


def test_biosample_extract_structured_facts_uses_isolate_as_host_species_fallback(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA999999",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN39525774",
                "title": "Aurelia microbiome sample",
                "organism": {"taxonomy_name": "metagenome", "taxonomy_id": "256318"},
                "attributes": {
                    "isolate": "Aurelia coerulea",
                    "isolation source": "Aurelia coerulea microbiome",
                },
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))
    values = {fact.fact_type_candidate: fact for fact in facts}

    assert values["isolate"].raw_value == "Aurelia coerulea"
    assert values["host_species"].raw_value == "Aurelia coerulea"


def test_biosample_extract_structured_facts_host_beats_isolate_for_host_species(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA999999",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "host-associated sample",
                "organism": {"taxonomy_name": "metagenome", "taxonomy_id": "256318"},
                "attributes": {
                    "host": "Aurelia aurita",
                    "isolate": "Aurelia coerulea",
                },
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))
    values = {fact.fact_type_candidate: fact for fact in facts}

    assert values["host_species"].raw_value == "Aurelia aurita"


def test_biosample_extract_structured_facts_maps_owner_contact_to_recorded_by(biosample_adapter):
    """recordedBy comes from the real per-sample submitter PERSON
    (Owner/Contacts/Contact/Name), confirmed against real cached BioSample
    XML -- Owner/Name alone is the submitting INSTITUTION's own name
    (see the regression guard below), never a person."""
    raw = {
        "bioproject_accession": "PRJNA994076",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN36415090",
                "title": "pond sample",
                "owner": {"name": "University of Konstanz", "contact_name": "Corentin Fournier"},
                "attributes": {"collection_date": "2022-04-11"},
            },
            {
                "accession": "SAMN36415091",
                "title": "pond sample",
                "owner": {"name": "University of Konstanz", "contact_name": "Corentin Fournier"},
                "attributes": {"collection_date": "2022-04-11"},
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA994076"))

    recorded_by = [f for f in facts if f.fact_type_candidate == "recordedBy"]
    assert len(recorded_by) == 1
    assert recorded_by[0].entity_level == EntityLevel.STUDY
    assert recorded_by[0].raw_value == "Corentin Fournier"
    assert recorded_by[0].source_locator == "ncbi_biosample.SAMN36415090.Owner.Contacts.Contact.Name"


def test_biosample_extract_structured_facts_institution_only_owner_never_becomes_recorded_by(biosample_adapter):
    """Regression guard for a real bug: Owner/Name (the submitting
    institution, e.g. "San Francisco Estuary Institute" in a real cached
    BioSample record) used to be piped directly into recordedBy alongside
    real person names -- confirmed wrong since Owner/Name is never a
    person. A sample with no Contact name at all must produce zero
    recordedBy facts, not fall back to the institution name."""
    raw = {
        "bioproject_accession": "PRJNA1",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "sample",
                "owner": {"name": "San Francisco Estuary Institute"},
                "attributes": {"collection_date": "2023-03-15"},
            },
        ],
    }

    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1"))

    assert [f for f in facts if f.fact_type_candidate == "recordedBy"] == []


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
    per the real FAIRe schema). Cartridge filtration alone is filter
    evidence, not proof of active pressure/pump-driven filtration."""
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
    assert by_type["filter_passive_active_0_1"].raw_value == "0"
    assert "filter_diameter" not in by_type
    assert by_type["size_frac"].support_type == SupportType.DETERMINISTICALLY_DERIVED
    assert by_type["size_frac"].raw_field_name == "samp_mat_process"


def test_biosample_extract_structured_facts_treats_sterivex_as_active_filter(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_STERIVEX",
                "title": "MIMS Environmental sample",
                "attributes": {"samp_mat_process": "0.22 um Sterivex filtration followed by DNA extraction"},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["filter_name"].raw_value == "Sterivex"
    assert by_type["filter_passive_active_0_1"].raw_value == "1"


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


def test_biosample_extract_structured_facts_emits_biosample_submission_date(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_SUBMITTED",
                "title": "MIMS Environmental sample",
                "submitted": "2024-02-03",
                "attributes": {},
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    by_type = {f.fact_type_candidate: f for f in facts}
    assert by_type["eventDate_submitted"].raw_value == "2024-02-03"
    assert by_type["eventDate_submitted"].raw_field_name == "submission_date"
    assert by_type["eventDate_submitted"].source_locator == "ncbi_biosample.SAMN_SUBMITTED.submission_date"


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


def test_biosample_extract_structured_facts_derives_depth_from_hyphenated_source_material_id(biosample_adapter):
    """Regression guard for a real gap found live: BioSample attribute_name
    spelling is submitter-controlled and varies -- a real dataset stored
    this exact MIxS attribute as "source-material-id" (hyphenated), which
    the dict-exact-key lookup silently never matched at all, so depth
    derivation never fired for that dataset despite the attribute being
    present. _get_attribute's normalized fallback must catch this."""
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_HYPHENATED",
                "title": "MIMS Environmental sample",
                "attributes": {"source-material-id": "10 m V3-V4"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    depth_facts = [f for f in facts if f.fact_type_candidate == "depth"]
    assert len(depth_facts) == 1
    assert depth_facts[0].entity_external_id == "SAMN_HYPHENATED"
    assert depth_facts[0].raw_value == "10 m"


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


def test_biosample_extract_structured_facts_uses_source_material_id_as_sample_name_fallback(
    biosample_adapter,
):
    raw = {
        "bioproject_accession": "PRJNA529480",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN11268162",
                "title": "Metagenome or environmental sample from marine sediment metagenome",
                "attributes": {"source-material-id": "GS16-GC05-20"},
            },
            {
                "accession": "SAMN11268163",
                "title": "Metagenome or environmental sample from marine sediment metagenome",
                "attributes": {"source-material-id": "GS16-GC05-21"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA529480"))

    categories = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "samp_category"}
    assert categories["SAMN11268162"].raw_value == "GS16_GC05_20"
    assert categories["SAMN11268162"].raw_field_name == "source_material_id"

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert set(rep_facts) == {"SAMN11268162", "SAMN11268163"}
    assert rep_facts["SAMN11268162"].raw_value == "SAMN11268162 | SAMN11268163"
    assert rep_facts["SAMN11268162"].raw_field_name == "source_material_id"
    assert rep_facts["SAMN11268162"].confidence_metadata == {
        "replicate_detection_signal": "trailing_number_suffix",
        "replicate_group_size": 2,
    }


def test_biosample_extract_structured_facts_source_material_id_wins_over_sample_name(biosample_adapter):
    """Explicit user instruction: "source material id... should stay the
    default" -- when a BioSample carries BOTH source_material_id and a
    submitted sample_name, source_material_id is preferred for both
    samp_category and replicate detection."""
    raw = {
        "bioproject_accession": "PRJNA529480",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN11268098",
                "title": "MIMS sample from marine sediment metagenome",
                "attributes": {"source-material-id": "GS14-GC08-1", "sample_name": "GS14_GC08_1"},
            },
            {
                "accession": "SAMN11268099",
                "title": "MIMS sample from marine sediment metagenome",
                "attributes": {"source-material-id": "GS14-GC08-2", "sample_name": "GS14_GC08_2"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA529480"))

    categories = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "samp_category"}
    assert categories["SAMN11268098"].raw_field_name == "source_material_id"

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert rep_facts["SAMN11268098"].raw_field_name == "source_material_id"
    assert rep_facts["SAMN11268098"].raw_value == "SAMN11268098 | SAMN11268099"


def test_biosample_extract_structured_facts_falls_back_to_sample_name_when_no_source_material_id(
    biosample_adapter,
):
    """Real live gap (SAMN29179945, STUDY-4fbd530e2a5e|STUDY-e97980f33de0):
    this record has no source_material_id attribute at all, only a
    submitted sample_name ("GS16-GC05-1", hyphenated) -- confirms the
    fallback fires and the hyphen gets normalized to "_" the same way
    source_material_id already is, per an explicit user instruction ("a
    fix was just put in for this behavior for source_material_id"),
    preventing a real sibling record whose sample_name already uses
    underscores ("GS16_GC05_2") from failing to group as a replicate
    purely over a hyphen-vs-underscore naming difference."""
    raw = {
        "bioproject_accession": "PRJNA529480",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN29179945",
                "title": "MIMS Environmental/Metagenome sample from marine sediment metagenome",
                "attributes": {"sample_name": "GS16-GC05-1"},
            },
            {
                "accession": "SAMN29179946",
                "title": "MIMS Environmental/Metagenome sample from marine sediment metagenome",
                "attributes": {"sample_name": "GS16_GC05_2"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA529480"))

    categories = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "samp_category"}
    assert categories["SAMN29179945"].raw_value == "GS16_GC05_1"
    assert categories["SAMN29179945"].raw_field_name == "sample_name"

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert set(rep_facts) == {"SAMN29179945", "SAMN29179946"}
    assert rep_facts["SAMN29179945"].raw_value == "SAMN29179945 | SAMN29179946"
    assert rep_facts["SAMN29179945"].raw_field_name == "sample_name"


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


def test_biosample_extract_structured_facts_groups_title_number_suffix_by_exact_prefix(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA295",
        "total_linked_samples": 4,
        "truncated": False,
        "samples": [
            {"accession": "SAMN_LM_6", "title": "LM 6", "attributes": {}},
            {"accession": "SAMN_LM_7", "title": "LM 7", "attributes": {}},
            {"accession": "SAMN_LMM_6", "title": "LMM 6", "attributes": {}},
            {"accession": "SAMN_LMM_7", "title": "LMM 7", "attributes": {}},
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA295"))

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert rep_facts["SAMN_LM_6"].raw_value == "SAMN_LM_6 | SAMN_LM_7"
    assert rep_facts["SAMN_LM_7"].raw_value == "SAMN_LM_6 | SAMN_LM_7"
    assert rep_facts["SAMN_LMM_6"].raw_value == "SAMN_LMM_6 | SAMN_LMM_7"
    assert rep_facts["SAMN_LMM_7"].raw_value == "SAMN_LMM_6 | SAMN_LMM_7"
    assert rep_facts["SAMN_LM_6"].raw_field_name == "title"
    assert rep_facts["SAMN_LM_6"].confidence_metadata == {
        "replicate_detection_signal": "trailing_number_suffix",
        "replicate_group_size": 2,
    }


def test_biosample_extract_structured_facts_no_biological_rep_relation_without_sibling(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1425045",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [{"accession": "SAMN1", "title": "Site_A_rep1", "attributes": {}}],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw))

    assert not any(f.fact_type_candidate == "biological_rep_relation" for f in facts)


def test_biosample_extract_structured_facts_groups_replicates_by_shared_metadata(biosample_adapter):
    """Lowest-priority replicate signal (after explicit attribute and
    name-pattern), per an explicit user request: same coordinates,
    collection date, and depth -- with no per-sample assay attribute at
    all, so the assay dimension is assumed the same for every sample,
    matching the common real-world case."""
    raw = {
        "bioproject_accession": "PRJNA1",
        "total_linked_samples": 3,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "sample 1",
                "attributes": {
                    "lat_lon": "38.03 N 122.15 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                },
            },
            {
                "accession": "SAMN2",
                "title": "sample 2",
                "attributes": {
                    "lat_lon": "38.03 N 122.15 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                },
            },
            {
                "accession": "SAMN3",
                "title": "sample 3, different site",
                "attributes": {
                    "lat_lon": "40.00 N 120.00 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                },
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1"))

    rep_facts = {f.entity_external_id: f for f in facts if f.fact_type_candidate == "biological_rep_relation"}
    assert set(rep_facts) == {"SAMN1", "SAMN2"}
    assert rep_facts["SAMN1"].raw_value == "SAMN1 | SAMN2"
    assert rep_facts["SAMN1"].confidence_metadata["replicate_detection_signal"] == "biosample_metadata_match"


def test_biosample_extract_structured_facts_metadata_match_excludes_mag_biosamples(biosample_adapter):
    """Regression guard for a real bug found live (BioProject PRJNA529480):
    MAG (metagenome-assembled-genome) BioSamples are excluded from the
    main per-attribute loop and from every replicate-relation tier (they
    stay out of non_mag_samples, which is what those tiers read), but they
    DO still get a curated, safe subset of environmental-context attributes
    (lat_lon/collection_date/depth/etc.) via a separate, narrow loop -- a
    second real bug found live (SAMN42764696, a MIMAG.sediment-packaged
    record) confirmed a MAG record can genuinely carry these directly,
    contradicting the earlier assumption that it never does. Assembly-
    specific attributes (e.g. "assembly software") never get through that
    loop. Two real (non-MAG) samples here do NOT share matching metadata,
    so the only reason a biological_rep_relation fact could wrongly appear
    for them is if the MAG trio below leaked into the metadata-match tier."""
    raw = {
        "bioproject_accession": "PRJNA1",
        "total_linked_samples": 5,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN_REAL_1",
                "title": "sample 1",
                "attributes": {"lat_lon": "38.03 N 122.15 W", "collection_date": "2023-03-15", "depth": "1 m"},
            },
            {
                "accession": "SAMN_REAL_2",
                "title": "sample 2, different site",
                "attributes": {"lat_lon": "40.00 N 120.00 W", "collection_date": "2023-03-16", "depth": "2 m"},
            },
            {
                "accession": "SAMN_MAG_1",
                "title": "Metagenome-assembled genome: STUDY_SAMN_MAG_00001",
                "package": "MIMAG.host-associated.6.0",
                "attributes": {
                    "lat_lon": "38.03 N 122.15 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                    "assembly software": "metaSPAdes 3.14",
                },
            },
            {
                "accession": "SAMN_MAG_2",
                "title": "Metagenome-assembled genome: STUDY_SAMN_MAG_00002",
                "package": "MIMAG.host-associated.6.0",
                "attributes": {
                    "lat_lon": "38.03 N 122.15 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                    "assembly software": "metaSPAdes 3.14",
                },
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1"))

    assert not [f for f in facts if f.fact_type_candidate == "biological_rep_relation"]
    # The MAG entries still contribute their safe environmental attributes...
    mag_facts = [f for f in facts if f.entity_external_id in ("SAMN_MAG_1", "SAMN_MAG_2")]
    mag_fact_types = {f.fact_type_candidate for f in mag_facts}
    assert mag_fact_types == {"lat_lon", "collection_date", "depth"}
    # ...but never their assembly-specific attributes.
    assert "assembly software" not in mag_fact_types


def test_biosample_extract_structured_facts_mag_with_full_environmental_attributes(biosample_adapter):
    """Grounded in the real live BioSample that exposed this gap
    (SAMN42764696, 10.1038/s42003-024-06136-2's BioProject PRJNA529480): a
    MIMAG.sediment-packaged MAG record that directly carries the same kind
    of environmental attributes a real environmental sample would (isolate,
    collection_date, depth, elev, env_broad_scale, env_local_scale,
    env_medium, geo_loc_name, isolation_source, lat_lon) -- previously
    dropped entirely, leaving this sample's exported row completely blank
    even though the live NCBI record plainly has the data. `isolate` isn't
    in the safe allowlist (it names the assembled bin, e.g. "Bin_040", not
    the environment) and correctly still doesn't come through."""
    raw = {
        "bioproject_accession": "PRJNA529480",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN42764696",
                "title": "MIMAG Metagenome-assembled Genome sample from Candidatus Scalindua sp.",
                "submitted": "2024-01-22",
                "package": "MIMAG.sediment.6.0",
                "model": "MIMAG.sediment",
                "organism": {"taxonomy_name": "Candidatus Scalindua sp."},
                "attributes": {
                    "isolate": "Bin_040",
                    "collection_date": "2014-07-22",
                    "depth": "2.5",
                    "elev": "-2476",
                    "env_broad_scale": "sediment microbiome",
                    "env_local_scale": "not applicable",
                    "env_medium": "marine sediment",
                    "geo_loc_name": "Atlantic Ocean",
                    "isolation_source": "Pelagic sediments on the east flank of Mohns Ridge",
                    "lat_lon": "72.00 N 0.10 E",
                },
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA529480"))
    by_type = {f.fact_type_candidate: f.raw_value for f in facts if f.entity_external_id == "SAMN42764696"}
    assert by_type == {
        "eventDate_submitted": "2024-01-22",
        "collection_date": "2014-07-22",
        "depth": "2.5",
        "elev": "-2476",
        "env_broad_scale": "sediment microbiome",
        "env_local_scale": "not applicable",
        "env_medium": "marine sediment",
        "geo_loc_name": "Atlantic Ocean",
        "isolation_source": "Pelagic sediments on the east flank of Mohns Ridge",
        "lat_lon": "72.00 N 0.10 E",
    }
    assert "isolate" not in by_type
    assert "organism" not in by_type
    assert "host_species" not in by_type


def test_biosample_extract_structured_facts_host_associated_mag_keeps_submission_date_and_isolate_host(
    biosample_adapter,
):
    raw = {
        "bioproject_accession": "PRJNA1067395",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN39525748",
                "title": "MIMAG Metagenome-assembled Genome sample from Vibrio sp.",
                "submitted": "2024-01-22T03:00:12.820",
                "package": "MIMAG.host-associated.6.0",
                "model": "MIMAG.host-associated",
                "organism": {"taxonomy_name": "Vibrio sp."},
                "attributes": {
                    "isolate": "Aurelia coerulea",
                    "collection_date": "not provided",
                    "env_broad_scale": "ENVO:00000428",
                    "env_local_scale": "ENVO:00000486",
                    "env_medium": "ENVO:00010483",
                    "geo_loc_name": "China: Yantai",
                    "isolation_source": "Strobilation process",
                    "lat_lon": "37.52 N 121.45 E",
                },
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1067395"))
    by_type = {f.fact_type_candidate: f for f in facts if f.entity_external_id == "SAMN39525748"}

    assert by_type["eventDate_submitted"].raw_value == "2024-01-22T03:00:12.820"
    assert by_type["host_species"].raw_value == "Aurelia coerulea"
    assert by_type["host_species"].raw_field_name == "isolate"


def test_biosample_extract_structured_facts_mag_derived_from_extracts_parent_accession(biosample_adapter):
    """Grounded in the real live BioSample SAMN12415826 (a MIMAG.sediment
    MAG record): its own "derived-from" attribute is a full sentence --
    "This BioSample is a metagenomic assembly obtained from the marine
    sediment metagenome BioSample: SAMN11268106" -- not a bare accession,
    so the parent BioSample's own accession is extracted out of it into
    sample_derived_from rather than storing the whole sentence. Per an
    explicit user request: "'derived from' = sample_derived_from"."""
    raw = {
        "bioproject_accession": "PRJNA1",
        "total_linked_samples": 1,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN12415826",
                "title": "MIMAG Metagenome-assembled Genome sample from Candidatus Scalindua sp.",
                "package": "MIMAG.sediment.6.0",
                "model": "MIMAG.sediment",
                "organism": {"taxonomy_name": "Candidatus Scalindua sp."},
                "attributes": {
                    "geo_loc_name": "Atlantic Ocean",
                    "derived-from": (
                        "This BioSample is a metagenomic assembly obtained from the marine "
                        "sediment metagenome BioSample: SAMN11268106"
                    ),
                },
            }
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1"))
    by_type = {f.fact_type_candidate: f.raw_value for f in facts if f.entity_external_id == "SAMN12415826"}
    assert by_type["sample_derived_from"] == "SAMN11268106"
    assert by_type["geo_loc_name"] == "Atlantic Ocean"


def test_biosample_extract_structured_facts_metadata_match_requires_matching_assay_when_reported(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "sample 1",
                "attributes": {
                    "lat_lon": "38.03 N 122.15 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                    "assay": "16S",
                },
            },
            {
                "accession": "SAMN2",
                "title": "sample 2",
                "attributes": {
                    "lat_lon": "38.03 N 122.15 W",
                    "collection_date": "2023-03-15",
                    "depth": "1 m",
                    "assay": "18S",
                },
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1"))

    assert not any(f.fact_type_candidate == "biological_rep_relation" for f in facts)


def test_biosample_extract_structured_facts_metadata_match_skips_samples_missing_depth(biosample_adapter):
    raw = {
        "bioproject_accession": "PRJNA1",
        "total_linked_samples": 2,
        "truncated": False,
        "samples": [
            {
                "accession": "SAMN1",
                "title": "sample 1",
                "attributes": {"lat_lon": "38.03 N 122.15 W", "collection_date": "2023-03-15"},
            },
            {
                "accession": "SAMN2",
                "title": "sample 2",
                "attributes": {"lat_lon": "38.03 N 122.15 W", "collection_date": "2023-03-15"},
            },
        ],
    }
    facts = biosample_adapter.extract_structured_facts(_record("ncbi_biosample", raw, "PRJNA1"))

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
                "fastq_access_status": "accessible",
                "fastq_access_checked_urls": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1.fastq.gz",
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
        "fastq_access_status", "fastq_access_checked_urls",
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
