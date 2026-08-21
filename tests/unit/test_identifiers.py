import pytest

from fair_ocean_agent.database.enums import IdentifierType
from fair_ocean_agent.identity.identifiers import (
    IdentifierError,
    guess_identifier_type,
    normalize_bioproject_accession,
    normalize_biosample_accession,
    normalize_doi,
    normalize_ena_study_accession,
    normalize_identifier,
    normalize_pmcid,
    normalize_pmid,
    normalize_sra_study_accession,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.1234/ABC.def", "10.1234/abc.def"),
        ("https://doi.org/10.1234/abc.def", "10.1234/abc.def"),
        ("http://dx.doi.org/10.1234/abc.def", "10.1234/abc.def"),
        ("doi:10.1234/abc.def", "10.1234/abc.def"),
        ("  10.1234/abc.def  ", "10.1234/abc.def"),
        ("10.1128/msystems.00184-16open_in_new", "10.1128/msystems.00184-16"),
        ("10.1128/msystems.00184-16 External Link", "10.1128/msystems.00184-16"),
    ],
)
def test_normalize_doi_valid(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize("raw", ["not-a-doi", "10.abc/xyz", "", "10.1234"])
def test_normalize_doi_invalid(raw):
    with pytest.raises(IdentifierError):
        normalize_doi(raw)


def test_normalize_bioproject_accession():
    assert normalize_bioproject_accession("prjna123456") == "PRJNA123456"
    assert normalize_bioproject_accession("PRJEB1") == "PRJEB1"
    with pytest.raises(IdentifierError):
        normalize_bioproject_accession("PRJXX123")


def test_normalize_biosample_accession():
    assert normalize_biosample_accession("samn12345678") == "SAMN12345678"
    with pytest.raises(IdentifierError):
        normalize_biosample_accession("XYZ12345")


def test_normalize_sra_study_accession():
    assert normalize_sra_study_accession("srp000001") == "SRP000001"
    with pytest.raises(IdentifierError):
        normalize_sra_study_accession("PRJNA123456")


def test_normalize_ena_study_accession():
    assert normalize_ena_study_accession("erp000001") == "ERP000001"
    assert normalize_ena_study_accession("prjeb1") == "PRJEB1"
    with pytest.raises(IdentifierError):
        normalize_ena_study_accession("SRP000001")


def test_normalize_pmid():
    assert normalize_pmid("PMID:12345") == "12345"
    assert normalize_pmid(" 12345 ") == "12345"
    with pytest.raises(IdentifierError):
        normalize_pmid("not-a-number")


def test_normalize_pmcid():
    assert normalize_pmcid("12345") == "PMC12345"
    assert normalize_pmcid("pmc12345") == "PMC12345"
    with pytest.raises(IdentifierError):
        normalize_pmcid("not-numeric")


def test_normalize_identifier_dispatches_by_type():
    assert (
        normalize_identifier(IdentifierType.DOI, "https://doi.org/10.1234/abc")
        == "10.1234/abc"
    )
    assert normalize_identifier(IdentifierType.CRUISE_ID, "  RV-2024-01  ") == "RV-2024-01"


@pytest.mark.parametrize(
    "raw,expected_type",
    [
        ("PRJNA123456", IdentifierType.BIOPROJECT_ACCESSION),
        ("SAMN12345678", IdentifierType.BIOSAMPLE_ACCESSION),
        ("DRA005630", IdentifierType.SRA_SUBMISSION_ACCESSION),
        ("ERP000001", IdentifierType.ENA_STUDY_ACCESSION),
        ("SRP000001", IdentifierType.SRA_STUDY_ACCESSION),
        ("DRS000001", IdentifierType.SRA_SAMPLE_ACCESSION),
        ("DRX000001", IdentifierType.SRA_EXPERIMENT_ACCESSION),
        ("DRR000001", IdentifierType.SRA_RUN_ACCESSION),
        ("DRZ000001", IdentifierType.SRA_ANALYSIS_ACCESSION),
        ("GCA_000001405.29", IdentifierType.ASSEMBLY_ACCESSION),
        ("PRJCA123456", IdentifierType.CNCB_PROJECT_ACCESSION),
        ("SAMC123456", IdentifierType.CNCB_BIOSAMPLE_ACCESSION),
        ("CRA123456", IdentifierType.CNCB_STUDY_ACCESSION),
        ("CRX123456", IdentifierType.CNCB_EXPERIMENT_ACCESSION),
        ("CRR123456", IdentifierType.CNCB_RUN_ACCESSION),
        ("PMC12345", IdentifierType.PMCID),
        ("10.1234/abc.def", IdentifierType.DOI),
        ("https://example.org/data", IdentifierType.URL),
        ("12345", IdentifierType.PMID),
    ],
)
def test_guess_identifier_type(raw, expected_type):
    assert guess_identifier_type(raw) == expected_type


def test_guess_identifier_type_unknown_returns_none():
    assert guess_identifier_type("some free-text label") is None
