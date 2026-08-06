"""Integration tests for fetch_record's real esearch/elink/efetch (NCBI) and
search (ENA) orchestration + XML/JSON parsing, via httpx.MockTransport --
no live network, but exercises the actual parsing code path fetch_record
uses (unlike test_ncbi_ena_parsing.py, which hand-builds `raw` dicts)."""
import httpx
import pytest

from fair_ocean_agent.sources.base import SourceConfig, SourceRecordNotFoundError
from fair_ocean_agent.sources.ena import EnaAdapter
from fair_ocean_agent.sources.ncbi import NcbiBioProjectAdapter, NcbiBioSampleAdapter

BIOPROJECT_XML = """<?xml version="1.0" ?>
<RecordSet><DocumentSummary uid="1425045">
    <Project>
        <ProjectID><ArchiveID accession="PRJNA1425045" archive="NCBI" id="1425045"/></ProjectID>
        <ProjectDescr>
            <Title>SF Bay 18S Metabarcoding Monitoring</Title>
            <Description>Results from a survey of filtered seawater samples.</Description>
        </ProjectDescr>
    </Project>
    <Submission submitted="2026-02-17"/>
</DocumentSummary></RecordSet>"""

BIOSAMPLE_XML = """<?xml version="1.0" ?>
<BioSampleSet><BioSample accession="SAMN1">
    <Description><Title>Sample one</Title></Description>
    <Owner><Name abbreviation="UKN">University of Konstanz, Corentin Fournier</Name></Owner>
    <Attributes>
        <Attribute attribute_name="collection_date">2023-12-06</Attribute>
        <Attribute attribute_name="depth">1</Attribute>
    </Attributes>
</BioSample><BioSample accession="SAMN2">
    <Description><Title>Sample two</Title></Description>
    <Attributes>
        <Attribute attribute_name="collection_date">2023-12-07</Attribute>
    </Attributes>
</BioSample></BioSampleSet>"""


def _ncbi_transport(esearch_bioproject_ids=("1425045",), elink_biosample_ids=("111", "222")):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        if path.endswith("esearch.fcgi") and params.get("db") == "bioproject":
            return httpx.Response(200, json={"esearchresult": {"idlist": list(esearch_bioproject_ids)}})
        if path.endswith("elink.fcgi"):
            return httpx.Response(
                200,
                json={
                    "linksets": [
                        {"linksetdbs": [{"linkname": "bioproject_biosample", "links": list(elink_biosample_ids)}]}
                    ]
                },
            )
        if path.endswith("efetch.fcgi") and params.get("db") == "bioproject":
            return httpx.Response(200, text=BIOPROJECT_XML)
        if path.endswith("efetch.fcgi") and params.get("db") == "biosample":
            return httpx.Response(200, text=BIOSAMPLE_XML)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def test_bioproject_fetch_record_parses_real_shaped_xml(retrieval_config):
    adapter = NcbiBioProjectAdapter(
        SourceConfig(name="ncbi_bioproject", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport(),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["accession"] == "PRJNA1425045"
    assert record.raw["title"] == "SF Bay 18S Metabarcoding Monitoring"
    assert record.raw["submitted"] == "2026-02-17"
    adapter.close()


def test_bioproject_fetch_record_not_found_when_esearch_empty(retrieval_config):
    adapter = NcbiBioProjectAdapter(
        SourceConfig(name="ncbi_bioproject", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport(esearch_bioproject_ids=()),
    )
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("PRJNA0000000")
    adapter.close()


def test_biosample_fetch_record_discovers_and_parses_linked_samples(retrieval_config):
    adapter = NcbiBioSampleAdapter(
        SourceConfig(name="ncbi_biosample", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport(),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["total_linked_samples"] == 2
    assert record.raw["truncated"] is False
    accessions = {s["accession"] for s in record.raw["samples"]}
    assert accessions == {"SAMN1", "SAMN2"}
    sample_one = next(s for s in record.raw["samples"] if s["accession"] == "SAMN1")
    assert sample_one["attributes"]["collection_date"] == "2023-12-06"
    assert sample_one["attributes"]["depth"] == "1"
    assert sample_one["owner"] == {
        "name": "University of Konstanz, Corentin Fournier",
        "abbreviation": "UKN",
    }
    adapter.close()


def test_biosample_fetch_record_not_found_when_no_linked_samples(retrieval_config):
    adapter = NcbiBioSampleAdapter(
        SourceConfig(name="ncbi_biosample", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport(elink_biosample_ids=()),
    )
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("PRJNA1425045")
    adapter.close()


def _ncbi_transport_discriminating(
    *,
    esearch_bioproject_ids=("1425045",),
    bioproject_esummary: dict | None = None,
    forward_biosample_ids=("111", "222"),
    reverse_bioproject_ids=("1425045",),
):
    """Unlike _ncbi_transport above, discriminates elink calls by
    dbfrom/db (needed now that fetch_record makes two distinct elink calls:
    bioproject->biosample for the sample list, biosample->bioproject for
    the reverse-verification signal) and serves esummary.fcgi for
    _esearch_verified_uid's multi-UID disambiguation path."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        if path.endswith("esearch.fcgi") and params.get("db") == "bioproject":
            return httpx.Response(200, json={"esearchresult": {"idlist": list(esearch_bioproject_ids)}})
        if path.endswith("esummary.fcgi") and params.get("db") == "bioproject":
            ids = params.get("id", "").split(",")
            result = {"uids": ids}
            for uid in ids:
                result[uid] = {"project_acc": (bioproject_esummary or {}).get(uid, "")}
            return httpx.Response(200, json={"result": result})
        if path.endswith("elink.fcgi") and params.get("dbfrom") == "bioproject":
            return httpx.Response(
                200,
                json={"linksets": [{"linksetdbs": [{"linkname": "bioproject_biosample", "links": list(forward_biosample_ids)}]}]},
            )
        if path.endswith("elink.fcgi") and params.get("dbfrom") == "biosample":
            return httpx.Response(
                200,
                json={"linksets": [{"linksetdbs": [{"linkname": "biosample_bioproject", "links": list(reverse_bioproject_ids)}]}]},
            )
        if path.endswith("efetch.fcgi") and params.get("db") == "bioproject":
            return httpx.Response(200, text=BIOPROJECT_XML)
        if path.endswith("efetch.fcgi") and params.get("db") == "biosample":
            return httpx.Response(200, text=BIOSAMPLE_XML)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def test_esearch_verified_uid_picks_matching_candidate_not_first(retrieval_config):
    """Regression test for the real PRJNA529480/PRJEB73262 bug: esearch
    returns two BioProject UIDs for one accession search; the first
    (esearch's own relevance ranking) is actually a DIFFERENT project that
    just mentions the accession in its own title/description. The fix must
    pick whichever candidate's own project_acc (via esummary) actually
    matches what was searched for, never just the first UID."""
    adapter = NcbiBioProjectAdapter(
        SourceConfig(name="ncbi_bioproject", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport_discriminating(
            esearch_bioproject_ids=("1356142", "1425045"),
            bioproject_esummary={"1356142": "PRJEB73262", "1425045": "PRJNA1425045"},
        ),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["uid"] == "1425045"
    assert record.raw["uid_resolution_ambiguous"] is False
    facts = adapter.extract_structured_facts(record)
    assert not any(f.fact_type_candidate == "ambiguous_uid_resolution" for f in facts)
    adapter.close()


def test_esearch_verified_uid_flags_review_when_no_candidate_matches(retrieval_config):
    """When esearch returns multiple UIDs and none of their own accessions
    (per esummary) match what was searched for, fall back to the first UID
    but emit a review-flagged PROJECT-level RawFact rather than silently
    trusting the guess."""
    adapter = NcbiBioProjectAdapter(
        SourceConfig(name="ncbi_bioproject", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport_discriminating(
            esearch_bioproject_ids=("1356142", "9999999"),
            bioproject_esummary={"1356142": "PRJEB73262", "9999999": "PRJEB99999"},
        ),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["uid"] == "1356142"
    assert record.raw["uid_resolution_ambiguous"] is True
    facts = adapter.extract_structured_facts(record)
    review_facts = [f for f in facts if f.fact_type_candidate == "ambiguous_uid_resolution"]
    assert len(review_facts) == 1
    assert review_facts[0].review_status == "needs_review"
    assert review_facts[0].confidence_metadata["esearch_uid_candidates"] == {
        "1356142": "PRJEB73262", "9999999": "PRJEB99999",
    }
    adapter.close()


def test_biosample_fetch_record_flags_review_on_reverse_elink_mismatch(retrieval_config):
    """Even when esearch itself is unambiguous, the independent
    biosample->bioproject reverse-elink signal disagreeing must still be
    surfaced as a review-flagged fact -- this is the check that would have
    caught PRJEB73262 directly, since a MAG-reanalysis BioSample's own
    reverse elink points back to ITS OWN BioProject, not the one whose
    accession search happened to surface it first."""
    adapter = NcbiBioSampleAdapter(
        SourceConfig(name="ncbi_biosample", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport_discriminating(
            esearch_bioproject_ids=("1425045",),
            forward_biosample_ids=("111", "222"),
            reverse_bioproject_ids=("999999",),  # does NOT include 1425045
        ),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["reverse_elink_verified"] is False
    facts = adapter.extract_structured_facts(record)
    review_facts = [f for f in facts if f.fact_type_candidate == "ambiguous_uid_resolution"]
    assert len(review_facts) == 1
    assert review_facts[0].confidence_metadata["reverse_elink_verified"] is False
    adapter.close()


def test_biosample_fetch_record_no_review_flag_when_signals_agree(retrieval_config):
    adapter = NcbiBioSampleAdapter(
        SourceConfig(name="ncbi_biosample", enabled=True, base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ncbi_transport_discriminating(
            esearch_bioproject_ids=("1425045",),
            forward_biosample_ids=("111", "222"),
            reverse_bioproject_ids=("1425045",),
        ),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["reverse_elink_verified"] is True
    facts = adapter.extract_structured_facts(record)
    assert not any(f.fact_type_candidate == "ambiguous_uid_resolution" for f in facts)
    adapter.close()


def _ena_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("result") == "study":
            return httpx.Response(
                200,
                json=[
                    {
                        "study_accession": "PRJNA1425045",
                        "secondary_study_accession": "SRP677779",
                        "study_title": "SF Bay 18S Metabarcoding Monitoring",
                        "study_description": "desc",
                        "center_name": "SFEI",
                        "first_public": "2026-02-19",
                    }
                ],
            )
        if params.get("result") == "read_run":
            return httpx.Response(
                200,
                json=[
                    {
                        "run_accession": "SRR1",
                        "sample_accession": "SAMN1",
                        "library_strategy": "AMPLICON",
                        "library_source": "METAGENOMIC",
                        "fastq_ftp": "ftp.sra.ebi.ac.uk/vol1/fastq/SRR001/SRR1.fastq.gz",
                        "fastq_bytes": "12345",
                    }
                ],
            )
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def test_ena_fetch_record_resolves_study_and_runs(retrieval_config):
    adapter = EnaAdapter(
        SourceConfig(name="ena", enabled=True, base_url="https://www.ebi.ac.uk/ena/portal/api", rate_limit_per_second=1000),
        retrieval_config,
        transport=_ena_transport(),
    )
    record = adapter.fetch_record("PRJNA1425045")

    assert record.raw["study"]["study_title"] == "SF Bay 18S Metabarcoding Monitoring"
    assert record.raw["study"]["secondary_study_accession"] == "SRP677779"
    assert len(record.raw["runs"]) == 1
    assert record.raw["runs"][0]["run_accession"] == "SRR1"
    adapter.close()


def test_ena_fetch_record_not_found_when_study_search_empty(retrieval_config):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    adapter = EnaAdapter(
        SourceConfig(name="ena", enabled=True, base_url="https://www.ebi.ac.uk/ena/portal/api", rate_limit_per_second=1000),
        retrieval_config,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SourceRecordNotFoundError):
        adapter.fetch_record("PRJNA0000000")
    adapter.close()
