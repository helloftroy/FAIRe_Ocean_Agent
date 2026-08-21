import httpx

from fair_ocean_agent.database.enums import IdentifierType, RelationshipType, SupportType
from fair_ocean_agent.discovery.text_identifiers import (
    extract_dataset_repository_identifiers_from_text,
    extract_repository_identifiers_from_text,
    resolve_sra_run_accessions_to_studies,
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


def test_resolve_sra_run_accessions_to_studies_finds_and_resolves_run_accessions(retrieval_config):
    """Real gap: only sample-level (SRS/ERS/DRS) accessions were
    recognized before this -- run-level (SRR/ERR/DRR) citations in a Data
    Availability statement were invisible."""
    text = "Raw reads are archived under SRR12335159."
    adapter = _ena_adapter_resolving_all_samples_to("PRJNA649058", retrieval_config=retrieval_config)

    related = resolve_sra_run_accessions_to_studies({"ena": adapter}, text, source_name="paper_scan")

    assert len(related) == 1
    assert related[0].identifier_type == IdentifierType.BIOPROJECT_ACCESSION
    assert related[0].value == "PRJNA649058"
    adapter.close()


def test_resolve_sra_run_accessions_to_studies_expands_run_range(retrieval_config):
    text = "Runs SRR12335159 - SRR12335161 are available."
    adapter = _ena_adapter_resolving_all_samples_to("PRJNA649058", retrieval_config=retrieval_config)

    related = resolve_sra_run_accessions_to_studies({"ena": adapter}, text, source_name="paper_scan")

    assert len(related) == 1  # all 3 runs resolve to the same parent study -- deduped
    adapter.close()


def test_extract_dataset_repository_identifiers_from_text_finds_all_four_pass2_repos():
    text = (
        "Sequences are deposited at Zenodo (10.5281/zenodo.10381280), "
        "Dryad (10.5061/dryad.xksn02vdx), Figshare (10.6084/m9.figshare.21653471), "
        "and OSF (10.17605/OSF.IO/EZCUJ)."
    )
    related = extract_dataset_repository_identifiers_from_text(text, source_name="paper_scan")
    values = {r.value for r in related}
    assert "10.5281/zenodo.10381280" in values
    assert "10.5061/dryad.xksn02vdx" in values
    assert "10.6084/m9.figshare.21653471" in values
    assert "10.17605/osf.io/ezcuj" in values  # normalize_doi lowercases
    assert all(r.identifier_type == IdentifierType.DATASET_DOI for r in related)


def test_extract_dataset_repository_identifiers_from_text_ignores_unrelated_dois():
    text = "This work builds on 10.1038/s41598-023-48804-z."
    assert extract_dataset_repository_identifiers_from_text(text, source_name="paper_scan") == []


def test_verify_deterministic_identifier_skips_zenodo_adapter_for_a_dryad_doi(retrieval_config):
    """Mirrors the existing pangaea/bcodmo skip-guard test -- a Dryad DOI
    must not be tried against the zenodo adapter first."""
    rel = RelatedIdentifier(
        identifier_type=IdentifierType.DATASET_DOI, value="10.5061/dryad.xksn02vdx",
        relationship_type=RelationshipType.IS_DATASET_FOR, source="paper_scan",
        confidence=SupportType.DETERMINISTICALLY_DERIVED,
    )
    adapters = {"zenodo": _FakeAdapter(found=True), "dryad": _FakeAdapter(found=True)}
    assert verify_deterministic_identifier(adapters, rel) is True
    adapters_zenodo_only_fails = {"zenodo": _FakeAdapter(found=True), "dryad": _FakeAdapter(found=False)}
    assert verify_deterministic_identifier(adapters_zenodo_only_fails, rel) is False
