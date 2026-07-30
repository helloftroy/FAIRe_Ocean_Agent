from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.discovery.text_identifiers import extract_repository_identifiers_from_text, xml_to_text


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
