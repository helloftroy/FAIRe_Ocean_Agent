"""Offline tests: construct a SourceRecord directly from a saved fixture and
check extract_structured_facts/parse_publication_fields/find_related -- no
network, no live API dependency. HTTP/caching/retry behavior is covered
separately in test_source_adapters_http.py.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from fair_ocean_agent.database.enums import AccessStatus, IdentifierType
from fair_ocean_agent.sources.base import SourceConfig, SourceRecord, hash_payload
from fair_ocean_agent.sources.bcodmo import BcoDmoAdapter
from fair_ocean_agent.sources.crossref import CrossrefAdapter
from fair_ocean_agent.sources.datacite import DataCiteAdapter
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.sources.gbif import GbifAdapter
from fair_ocean_agent.sources.ena import EnaAdapter
from fair_ocean_agent.sources.obis import ObisAdapter
from fair_ocean_agent.sources.openalex import OpenAlexAdapter
from fair_ocean_agent.sources.pangaea import PangaeaAdapter

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _record(source_name: str, raw: dict, external_identifier: str = "10.1002/2015jg003300") -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier=external_identifier,
        raw=raw,
        retrieved_at=datetime.now(timezone.utc),
        content_hash=hash_payload(raw),
    )


@pytest.fixture
def crossref_adapter(retrieval_config):
    adapter = CrossrefAdapter(
        SourceConfig(name="crossref", enabled=True, base_url="https://api.crossref.org"), retrieval_config
    )
    yield adapter
    adapter.close()


@pytest.fixture
def europe_pmc_adapter(retrieval_config):
    adapter = EuropePmcAdapter(
        SourceConfig(name="europe_pmc", enabled=True, base_url="https://www.ebi.ac.uk/europepmc/webservices/rest"),
        retrieval_config,
    )
    yield adapter
    adapter.close()


@pytest.fixture
def openalex_adapter(retrieval_config):
    adapter = OpenAlexAdapter(
        SourceConfig(name="openalex", enabled=True, base_url="https://api.openalex.org"), retrieval_config
    )
    yield adapter
    adapter.close()


def test_crossref_parse_publication_fields(crossref_adapter):
    record = _record("crossref", load_fixture("crossref_work.json"))
    fields = crossref_adapter.parse_publication_fields(record)

    assert fields["doi"] == "10.1002/2015jg003300"
    assert fields["title"].startswith("Shifts in the community structure")
    assert fields["publication_year"] == 2016
    assert fields["journal"] == "Journal of Geophysical Research: Biogeosciences"
    assert fields["authors"] == [
        {"given": "Jane", "family": "Doe"},
        {"given": "John", "family": "Smith"},
    ]


def test_crossref_extract_structured_facts_have_evidence_locators(crossref_adapter):
    record = _record("crossref", load_fixture("crossref_work.json"))
    facts = crossref_adapter.extract_structured_facts(record)

    assert len(facts) > 0
    assert all(f.source_locator.startswith("crossref.message.") for f in facts)
    title_fact = next(f for f in facts if f.fact_type_candidate == "title")
    assert "Shifts in the community structure" in title_fact.raw_value


def test_crossref_extract_structured_facts_skips_absent_fields(crossref_adapter):
    record = _record("crossref", {"DOI": "10.1/x", "title": []})
    facts = crossref_adapter.extract_structured_facts(record)
    assert all(f.fact_type_candidate != "title" for f in facts)


def test_crossref_title_fact_is_a_plain_string_not_a_json_encoded_list(crossref_adapter):
    """Regression test: a real live-validation cross-source comparison
    (validation/cross_source.py) found crossref's "title" and
    "container-title" facts stored as '["Some Title"]' -- Crossref's API
    returns these as lists-of-one-string, and the raw fact was storing the
    JSON-encoded list instead of unwrapping it like
    parse_publication_fields already did. This made every crossref title
    compare as "different" from Europe PMC/OpenAlex's plain-string titles,
    even when they were the same title -- a false cross-source conflict."""
    record = _record("crossref", load_fixture("crossref_work.json"))
    facts = crossref_adapter.extract_structured_facts(record)

    title_fact = next(f for f in facts if f.fact_type_candidate == "title")
    container_fact = next(f for f in facts if f.fact_type_candidate == "container-title")

    assert not title_fact.raw_value.startswith("[")
    assert title_fact.raw_value == "Shifts in the community structure of anaerobic ammonium oxidation bacteria"
    assert not container_fact.raw_value.startswith("[")
    assert container_fact.raw_value == "Journal of Geophysical Research: Biogeosciences"


def test_europe_pmc_parse_publication_fields_open_access(europe_pmc_adapter):
    record = _record("europe_pmc", load_fixture("europe_pmc_result.json"))
    fields = europe_pmc_adapter.parse_publication_fields(record)

    assert fields["pmid"] == "27000000"
    assert fields["pmcid"] == "PMC5000000"
    assert fields["open_access_status"] == AccessStatus.OPEN.value
    assert fields["fulltext_available"] is True


def test_europe_pmc_parse_publication_fields_not_open_access(europe_pmc_adapter):
    record = _record("europe_pmc", {"pmid": "1", "isOpenAccess": "N", "inEPMC": "N"})
    fields = europe_pmc_adapter.parse_publication_fields(record)

    assert fields["open_access_status"] == AccessStatus.UNKNOWN.value
    assert fields["fulltext_available"] is False


def test_europe_pmc_find_related_returns_pmid_and_pmcid(europe_pmc_adapter):
    record = _record("europe_pmc", load_fixture("europe_pmc_result.json"))
    related = europe_pmc_adapter.find_related(record)

    types = {r.identifier_type for r in related}
    assert types == {IdentifierType.PMID, IdentifierType.PMCID}


def test_openalex_parse_publication_fields(openalex_adapter):
    record = _record("openalex", load_fixture("openalex_work.json"))
    fields = openalex_adapter.parse_publication_fields(record)

    assert fields["openalex_id"] == "W2413577766"
    assert fields["publication_year"] == 2016
    assert fields["open_access_status"] == AccessStatus.OPEN.value


def test_openalex_find_related_strips_full_url(openalex_adapter):
    record = _record("openalex", load_fixture("openalex_work.json"))
    related = openalex_adapter.find_related(record)

    assert len(related) == 1
    assert related[0].identifier_type == IdentifierType.OPENALEX_ID
    assert related[0].value == "W2413577766"


def test_datacite_extracts_dataset_metadata_and_related_identifiers(retrieval_config):
    adapter = DataCiteAdapter(SourceConfig(name="datacite", enabled=True, base_url="https://api.datacite.org"), retrieval_config)
    raw = {
        "data": {
            "id": "10.1594/pangaea.923577",
            "attributes": {
                "doi": "10.1594/pangaea.923577",
                "titles": [{"title": "Marine dataset"}],
                "publisher": "PANGAEA",
                "publicationYear": 2024,
                "relatedIdentifiers": [
                    {"relatedIdentifier": "10.1000/paper", "relatedIdentifierType": "DOI", "relationType": "References"}
                ],
            },
        }
    }
    record = _record("datacite", raw, external_identifier="10.1594/pangaea.923577")

    facts = adapter.extract_structured_facts(record)
    related = adapter.find_related(record)

    assert any(f.fact_type_candidate == "titles" for f in facts)
    assert any(r.identifier_type == IdentifierType.DATASET_DOI for r in related)
    assert any(r.identifier_type == IdentifierType.DOI and r.value == "10.1000/paper" for r in related)
    adapter.close()


def test_bcodmo_extracts_metadata_without_downloading_data(retrieval_config):
    adapter = BcoDmoAdapter(SourceConfig(name="bcodmo", enabled=True, base_url="https://www.bco-dmo.org/api"), retrieval_config)
    record = _record(
        "bcodmo",
        {
            "id": "765432",
            "title": "BCO-DMO dataset",
            "doi": "10.26008/1912/bco-dmo.765432.1",
            "parameters": [{"name": "temperature"}],
            "files": [{"name": "data.csv", "size": "1200"}],
        },
        external_identifier="765432",
    )

    facts = adapter.extract_structured_facts(record)
    related = adapter.find_related(record)

    assert {f.fact_type_candidate for f in facts} >= {"dataset_id", "title", "doi", "parameters", "files"}
    assert any(r.identifier_type == IdentifierType.BCODMO_DATASET_ID for r in related)
    assert any(r.identifier_type == IdentifierType.DATASET_DOI for r in related)
    adapter.close()


def test_bcodmo_extracts_jsonld_metadata_shape(retrieval_config):
    adapter = BcoDmoAdapter(SourceConfig(name="bcodmo", enabled=True, base_url="https://www.bco-dmo.org/api"), retrieval_config)
    raw = {
        "@graph": [
            {
                "http://lod.bco-dmo.org/id/dataset/489471": {
                    "@id": "http://lod.bco-dmo.org/id/dataset/489471",
                    "@type": ["http://www.w3.org/ns/dcat#Dataset", "http://ocean-data.org/schema/Dataset"],
                    "http://purl.org/dc/terms/identifier": [{"@value": "489471"}],
                    "http://ocean-data.org/schema/datasetTitle": [{"@value": "Pteropod shell dissolution"}],
                    "http://ocean-data.org/schema/abstract": [{"@value": "A dataset abstract"}],
                    "http://purl.org/ontology/bibo/doi": [{"@value": "10.1575/1912/bco-dmo.489471.1"}],
                    "http://ocean-data.org/schema/storesValuesFor": [
                        {"@id": "http://lod.bco-dmo.org/id/dataset-parameter/489499"}
                    ],
                    "http://xmlns.com/foaf/0.1/homepage": [{"@id": "https://osprey.bco-dmo.org/dataset/489471"}],
                }
            }
        ]
    }
    record = _record("bcodmo", raw, external_identifier="489471")

    facts = adapter.extract_structured_facts(record)
    related = adapter.find_related(record)
    values = {f.fact_type_candidate: f.raw_value for f in facts}

    assert values["dataset_id"] == "489471"
    assert values["title"] == "Pteropod shell dissolution"
    assert values["abstract"] == "A dataset abstract"
    assert values["doi"] == "10.1575/1912/bco-dmo.489471.1"
    assert "dataset-parameter/489499" in values["parameters"]
    assert any(r.identifier_type == IdentifierType.DATASET_DOI for r in related)
    adapter.close()


def test_pangaea_extracts_jsonld_metadata_and_related_identifiers(retrieval_config):
    adapter = PangaeaAdapter(SourceConfig(name="pangaea", enabled=True, base_url="https://doi.pangaea.de"), retrieval_config)
    record = _record(
        "pangaea",
        {
            "@id": "https://doi.org/10.1594/PANGAEA.923577",
            "name": "PANGAEA dataset",
            "spatialCoverage": {"geo": {"latitude": 54.0, "longitude": 10.0}},
            "variableMeasured": [{"name": "salinity"}],
        },
        external_identifier="10.1594/PANGAEA.923577",
    )

    facts = adapter.extract_structured_facts(record)
    related = adapter.find_related(record)

    assert any(f.fact_type_candidate == "spatialCoverage" for f in facts)
    assert any(r.identifier_type == IdentifierType.PANGAEA_ID for r in related)
    assert any(r.identifier_type == IdentifierType.DATASET_DOI for r in related)
    adapter.close()


def test_pangaea_decomposes_real_variable_measured_shape(retrieval_config):
    """Regression fixture from real PANGAEA eDNA JSON-LD:
    10.1594/PANGAEA.935870 exposes parameters such as DATE/TIME, DEPTH,
    water, Water volume filtered, Sample ID, accession numbers, and OTU
    columns via schema.org variableMeasured PropertyValue objects. These are
    parameter definitions, not per-sample measured values, so the adapter
    emits decomposed PANGAEA-specific raw facts rather than pretending they
    are present FAIRe sample values."""
    adapter = PangaeaAdapter(SourceConfig(name="pangaea", enabled=True, base_url="https://doi.pangaea.de"), retrieval_config)
    record = _record(
        "pangaea",
        {
            "@id": "https://doi.org/10.1594/PANGAEA.935870",
            "name": "Operational taxonomic units of deep-sea fishes from environmental DNA",
            "variableMeasured": [
                {"@type": "PropertyValue", "name": "Sample ID"},
                {
                    "@type": "PropertyValue",
                    "name": "DEPTH, water",
                    "unitText": "m",
                    "subjectOf": {
                        "@type": "DefinedTermSet",
                        "hasDefinedTerm": [
                            {
                                "@id": "urn:obo:pato:term:0001595",
                                "@type": "DefinedTerm",
                                "identifier": "urn:obo:pato:term:0001595",
                                "name": "depth",
                                "url": "http://purl.obolibrary.org/obo/PATO_0001595",
                            },
                            {
                                "@id": "http://purl.obolibrary.org/obo/CHEBI_15377",
                                "@type": "DefinedTerm",
                                "identifier": "http://purl.obolibrary.org/obo/CHEBI_15377",
                                "name": "water",
                                "url": "http://purl.obolibrary.org/obo/CHEBI_15377",
                            },
                        ],
                    },
                },
                {
                    "@type": "PropertyValue",
                    "name": "Water volume, filtered",
                    "unitText": "m**3",
                    "subjectOf": {
                        "@type": "DefinedTermSet",
                        "hasDefinedTerm": {
                            "@id": "http://qudt.org/1.1/vocab/quantity#Volume",
                            "@type": "DefinedTerm",
                            "identifier": "http://qudt.org/1.1/vocab/quantity#Volume",
                            "name": "Volume",
                        },
                    },
                },
                {
                    "@type": "PropertyValue",
                    "name": "Thunnus sp., operational taxonomic unit",
                    "unitText": "#",
                    "measurementTechnique": "Metabarcoding",
                    "subjectOf": {
                        "@type": "DefinedTermSet",
                        "hasDefinedTerm": [
                            {
                                "@id": "urn:lsid:marinespecies.org:taxname:126065",
                                "@type": "DefinedTerm",
                                "identifier": "urn:lsid:marinespecies.org:taxname:126065",
                                "name": "Thunnus",
                                "url": "https://www.marinespecies.org/aphia.php?p=taxdetails&id=126065",
                            },
                            {"@type": "DefinedTerm", "name": "operational taxonomic unit", "alternateName": "OTU"},
                        ],
                    },
                },
            ],
        },
        external_identifier="10.1594/PANGAEA.935870",
    )

    facts = adapter.extract_structured_facts(record)
    values_by_type = {}
    for fact in facts:
        values_by_type.setdefault(fact.fact_type_candidate, []).append(fact.raw_value)

    assert "DEPTH, water" in values_by_type["pangaea_variable_name"]
    assert "Water volume, filtered" in values_by_type["pangaea_variable_name"]
    assert "Thunnus sp., operational taxonomic unit" in values_by_type["pangaea_variable_name"]
    assert "m" in values_by_type["pangaea_variable_unit"]
    assert "m**3" in values_by_type["pangaea_variable_unit"]
    assert "Metabarcoding" in values_by_type["pangaea_variable_measurement_technique"]
    assert "urn:obo:pato:term:0001595" in values_by_type["pangaea_variable_defined_term_identifier"]
    assert "urn:lsid:marinespecies.org:taxname:126065" in values_by_type["pangaea_variable_defined_term_identifier"]

    depth_fact = next(
        fact
        for fact in facts
        if fact.fact_type_candidate == "pangaea_variable_unit" and fact.raw_value == "m"
    )
    assert depth_fact.source_locator == "pangaea.jsonld.variableMeasured[1].unitText"
    assert depth_fact.confidence_metadata == {"variable_index": 1, "variable_name": "DEPTH, water"}
    adapter.close()


def test_ena_extracts_run_accession_and_library_construction_protocol(retrieval_config):
    adapter = EnaAdapter(
        SourceConfig(name="ena", enabled=True, base_url="https://www.ebi.ac.uk/ena/portal/api"),
        retrieval_config,
    )
    record = _record(
        "ena",
        {
            "study": {
                "study_accession": "PRJEB1",
                "secondary_study_accession": "ERP1",
                "study_title": "ENA study",
            },
            "runs": [
                {
                    "run_accession": "ERR123",
                    "sample_accession": "SAMEA1",
                    "library_layout": "PAIRED",
                    "library_construction_protocol": "PCR amplified the V4 region with 515F/806R primers.",
                }
            ],
            "truncated": False,
            "total_runs_seen": 1,
        },
        external_identifier="PRJEB1",
    )

    facts = adapter.extract_structured_facts(record)
    values = {f.fact_type_candidate: f for f in facts}

    assert values["run_accession"].raw_value == "ERR123"
    assert values["run_accession"].entity_external_id == "ERR123"
    assert values["library_construction_protocol"].raw_value.startswith("PCR amplified")
    assert values["library_construction_protocol"].source_locator == "ena.read_run.ERR123.library_construction_protocol"
    adapter.close()


def test_obis_extracts_dataset_metadata_and_dataset_uuid(retrieval_config):
    adapter = ObisAdapter(SourceConfig(name="obis", enabled=True, base_url="https://api.obis.org"), retrieval_config)
    record = _record(
        "obis",
        {"id": "11111111-2222-3333-4444-555555555555", "title": "OBIS dataset", "records": 123},
        external_identifier="11111111-2222-3333-4444-555555555555",
    )

    facts = adapter.extract_structured_facts(record)
    related = adapter.find_related(record)

    assert any(f.fact_type_candidate == "records" for f in facts)
    assert related[0].identifier_type == IdentifierType.OBIS_DATASET_UUID
    adapter.close()


def test_obis_fetch_record_unwraps_live_dataset_result_shape(retrieval_config):
    dataset_uuid = "11111111-2222-3333-4444-555555555555"
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == f"/dataset/{dataset_uuid}":
            return httpx.Response(200, json={"total": 1, "results": [{"id": dataset_uuid, "title": "OBIS dataset"}]})
        if request.url.path == "/occurrence":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    adapter = ObisAdapter(
        SourceConfig(name="obis", enabled=True, base_url="https://api.obis.org"),
        retrieval_config,
        transport=httpx.MockTransport(handler),
    )

    record = adapter.fetch_record(dataset_uuid)

    assert record.raw["id"] == dataset_uuid
    assert record.raw["title"] == "OBIS dataset"
    assert record.raw["_occurrence_preview"] == {"results": []}
    adapter.close()


def test_obis_extracts_dna_derived_occurrence_fields(retrieval_config):
    adapter = ObisAdapter(SourceConfig(name="obis", enabled=True, base_url="https://api.obis.org"), retrieval_config)
    record = _record(
        "obis",
        {
            "id": "11111111-2222-3333-4444-555555555555",
            "title": "OBIS dataset",
            "_occurrence_preview": {
                "results": [
                    {
                        "associatedSequences": "ENA:ERR123",
                        "extensions": {
                            "dnaDerivedData": [
                                {
                                    "pcrPrimerForward": "GTGYCAGCMGCCGCGGTAA",
                                    "pcrPrimerReverse": "GGACTACNVGGGTWTCTAAT",
                                    "targetGene": "16S rRNA",
                                }
                            ]
                        },
                    }
                ]
            },
        },
        external_identifier="11111111-2222-3333-4444-555555555555",
    )

    facts = adapter.extract_structured_facts(record)
    values = {f.fact_type_candidate: f for f in facts}

    assert values["associatedSequences"].raw_value == "ENA:ERR123"
    assert values["pcr_primer_forward"].raw_value == "GTGYCAGCMGCCGCGGTAA"
    assert values["pcr_primer_reverse"].source_locator.endswith("pcrPrimerReverse")
    assert values["target_gene"].raw_field_name == "targetGene"
    adapter.close()


def test_gbif_extracts_dataset_metadata_and_dataset_key(retrieval_config):
    adapter = GbifAdapter(SourceConfig(name="gbif", enabled=True, base_url="https://api.gbif.org/v1"), retrieval_config)
    record = _record(
        "gbif",
        {
            "key": "11111111-2222-3333-4444-555555555555",
            "title": "GBIF dataset",
            "doi": "10.15468/example",
            "recordCount": 321,
        },
        external_identifier="11111111-2222-3333-4444-555555555555",
    )

    facts = adapter.extract_structured_facts(record)
    related = adapter.find_related(record)

    assert any(f.fact_type_candidate == "recordCount" for f in facts)
    assert any(r.identifier_type == IdentifierType.GBIF_DATASET_KEY for r in related)
    assert any(r.identifier_type == IdentifierType.DATASET_DOI for r in related)
    adapter.close()


def test_gbif_extracts_dna_derived_occurrence_fields(retrieval_config):
    adapter = GbifAdapter(SourceConfig(name="gbif", enabled=True, base_url="https://api.gbif.org/v1"), retrieval_config)
    record = _record(
        "gbif",
        {
            "key": "11111111-2222-3333-4444-555555555555",
            "title": "GBIF dataset",
            "_occurrence_preview": {
                "results": [
                    {
                        "extensions": {
                            "http://rs.gbif.org/terms/1.0/DNADerivedData": [
                                {
                                    "pcr_primer_name_forward": "515F",
                                    "pcr_primer_name_reverse": "806R",
                                    "annealingTemp": "55 C",
                                    "ampliconSize": "291 bp",
                                }
                            ]
                        }
                    }
                ]
            },
        },
        external_identifier="11111111-2222-3333-4444-555555555555",
    )

    facts = adapter.extract_structured_facts(record)
    values = {f.fact_type_candidate: f.raw_value for f in facts}

    assert values["pcr_primer_name_forward"] == "515F"
    assert values["pcr_primer_name_reverse"] == "806R"
    assert values["annealingTemp"] == "55 C"
    assert values["ampliconSize"] == "291 bp"
    adapter.close()
