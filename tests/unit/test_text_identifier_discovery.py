import httpx

from fair_ocean_agent.database.enums import IdentifierType, RelationshipType, SupportType
from fair_ocean_agent.discovery.text_identifiers import (
    extract_repository_identifiers_from_text,
    resolve_sra_sample_accessions_to_studies,
    verify_deterministic_identifier,
    xml_to_text,
)
from fair_ocean_agent.sources.base import RelatedIdentifier, SourceConfig, SourceRecord, SourceRecordNotFoundError
from fair_ocean_agent.sources.ena import EnaAdapter


def test_extract_repository_identifiers_from_text_finds_accessions_and_dataset_dois():
    text = (
        "Raw reads are under NCBI BioProject PRJNA515494 and SRA study SRP040596. "
        "The environmental dataset is available as https://doi.org/10.1594/PANGAEA.923577 "
        "and 10.26008/1912/bco-dmo.765432.1."
    )

    related = extract_repository_identifiers_from_text(text, source_name="paper_scan")
    values = {(r.identifier_type, r.value) for r in related}

    assert (IdentifierType.BIOPROJECT_ACCESSION, "PRJNA515494") in values
    assert (IdentifierType.SRA_STUDY_ACCESSION, "SRP040596") in values
    assert (IdentifierType.DATASET_DOI, "10.1594/pangaea.923577") in values
    assert (IdentifierType.DATASET_DOI, "10.26008/1912/bco-dmo.765432.1") in values


def test_extract_repository_identifiers_from_text_ignores_generic_citation_dois():
    text = "Prior work is cited as 10.1038/s41598-023-48804-z, not as a dataset source."

    related = extract_repository_identifiers_from_text(text, source_name="paper_scan")

    assert related == []


def test_xml_to_text_collapses_article_xml():
    xml = "<article><sec><title>Data Availability</title><p>BioProject PRJNA996732.</p></sec></article>"

    assert "BioProject PRJNA996732" in xml_to_text(xml)


def test_extract_repository_identifiers_from_text_tags_tier_2_confidence():
    text = "Raw reads are under NCBI BioProject PRJNA515494."
    related = extract_repository_identifiers_from_text(text, source_name="paper_scan")
    assert related[0].confidence == SupportType.DETERMINISTICALLY_DERIVED


class _FakeAdapter:
    def __init__(self, found: bool):
        self._found = found

    def fetch_record(self, identifier):
        if not self._found:
            raise SourceRecordNotFoundError(f"no record for {identifier}")
        from datetime import datetime, timezone

        return SourceRecord(
            source_name="fake", external_identifier=identifier, raw={}, retrieved_at=datetime.now(timezone.utc),
            content_hash="deadbeef",
        )


def test_verify_deterministic_identifier_confirms_via_candidate_adapter():
    rel = RelatedIdentifier(
        identifier_type=IdentifierType.BIOPROJECT_ACCESSION, value="PRJNA515494",
        relationship_type=RelationshipType.IS_DATASET_FOR, source="paper_scan",
        confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    adapters = {"ncbi_bioproject": _FakeAdapter(found=True)}
    assert verify_deterministic_identifier(adapters, rel) is True


def test_verify_deterministic_identifier_drops_unresolvable_hit():
    rel = RelatedIdentifier(
        identifier_type=IdentifierType.BIOPROJECT_ACCESSION, value="PRJNA000000",
        relationship_type=RelationshipType.IS_DATASET_FOR, source="paper_scan",
        confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    adapters = {"ncbi_bioproject": _FakeAdapter(found=False), "ena": _FakeAdapter(found=False)}
    assert verify_deterministic_identifier(adapters, rel) is False


def test_verify_deterministic_identifier_returns_false_when_no_candidate_adapter_enabled():
    rel = RelatedIdentifier(
        identifier_type=IdentifierType.BIOPROJECT_ACCESSION, value="PRJNA515494",
        relationship_type=RelationshipType.IS_DATASET_FOR, source="paper_scan",
        confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    assert verify_deterministic_identifier({}, rel) is False


def test_verify_deterministic_identifier_skips_pangaea_adapter_for_a_bcodmo_doi():
    """A dataset DOI's own prefix says which repository owns it -- a BCO-DMO
    DOI must not be tried against the pangaea adapter first (which would
    incorrectly "confirm" it if the fake always succeeds)."""
    rel = RelatedIdentifier(
        identifier_type=IdentifierType.DATASET_DOI, value="10.26008/1912/bco-dmo.765432.1",
        relationship_type=RelationshipType.IS_DATASET_FOR, source="paper_scan",
        confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    adapters = {"pangaea": _FakeAdapter(found=True), "bcodmo": _FakeAdapter(found=True)}
    assert verify_deterministic_identifier(adapters, rel) is True
    # Confirm the pangaea adapter really was skipped, not just coincidentally successful too:
    adapters_pangaea_only_fails = {"pangaea": _FakeAdapter(found=True), "bcodmo": _FakeAdapter(found=False)}
    assert verify_deterministic_identifier(adapters_pangaea_only_fails, rel) is False


def _ena_adapter_resolving_all_samples_to(study_accession: str, *, retrieval_config) -> EnaAdapter:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"study_accession": study_accession}])

    return EnaAdapter(
        SourceConfig(name="ena", enabled=True, base_url="https://www.ebi.ac.uk/ena/portal/api", rate_limit_per_second=1000),
        retrieval_config,
        transport=httpx.MockTransport(handler),
    )


def test_resolve_sra_sample_accessions_to_studies_expands_range_and_routes_by_prefix(retrieval_config):
    """Real regression from 10.1073/pnas.2103275118: the paper's Data
    Availability statement cites only "SRS7105074 - SRS7105095" (an SRA
    Sample-level accession range) and never states its own study
    accession. ENA's own study_accession field for this NCBI/SRA-native
    submission resolves to a PRJNA... BioProject, not an ENA-native
    accession -- confirms guess_identifier_type routing, not a hardcoded
    ENA_STUDY_ACCESSION assumption."""
    text = "Sequencing data have been deposited in NCBI SRA: SRS7105074 - SRS7105095."
    adapter = _ena_adapter_resolving_all_samples_to("PRJNA649058", retrieval_config=retrieval_config)

    related = resolve_sra_sample_accessions_to_studies({"ena": adapter}, text, source_name="paper_scan")

    assert len(related) == 1  # all 22 samples in the range resolve to the same parent study -- deduped
    assert related[0].identifier_type == IdentifierType.BIOPROJECT_ACCESSION
    assert related[0].value == "PRJNA649058"
    adapter.close()


def test_resolve_sra_sample_accessions_to_studies_handles_single_accession_no_range(retrieval_config):
    text = "Raw reads are available under accession SRS7105074."
    adapter = _ena_adapter_resolving_all_samples_to("PRJNA649058", retrieval_config=retrieval_config)

    related = resolve_sra_sample_accessions_to_studies({"ena": adapter}, text, source_name="paper_scan")

    assert related[0].value == "PRJNA649058"
    adapter.close()


def test_resolve_sra_sample_accessions_to_studies_returns_empty_without_ena_adapter():
    text = "SRS7105074 - SRS7105095"
    assert resolve_sra_sample_accessions_to_studies({}, text, source_name="paper_scan") == []


def test_resolve_sra_sample_accessions_to_studies_returns_empty_when_no_accessions_present(retrieval_config):
    adapter = _ena_adapter_resolving_all_samples_to("PRJNA649058", retrieval_config=retrieval_config)
    assert resolve_sra_sample_accessions_to_studies({"ena": adapter}, "No accessions here.", source_name="x") == []
    adapter.close()
