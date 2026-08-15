"""Tests for the "node-adding" citation-expansion discovery pass:
handle_discover_citing_studies (workflow/handlers.py). Uses httpx.MockTransport
against a real NcbiBioProjectAdapter -- no live network -- exercising the
actual bioproject->pubmed elink + pubmed esummary orchestration code path,
plus the depth/fan-out safety valves (config.py's DiscoveryConfig)."""
import httpx

from fair_ocean_agent.config import AppConfig, DiscoveryConfig
from fair_ocean_agent.database.enums import IdentifierType, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Study, Task
from fair_ocean_agent.identity.identifiers import normalize_identifier
from fair_ocean_agent.sources.base import SourceConfig
from fair_ocean_agent.sources.ncbi import _elink_ids
from fair_ocean_agent.sources.ncbi import NcbiBioProjectAdapter
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task


def _bioproject_adapter(transport, retrieval_config):
    return NcbiBioProjectAdapter(
        SourceConfig(
            name="ncbi_bioproject",
            enabled=True,
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit_per_second=1000,
        ),
        retrieval_config,
        transport=transport,
    )


def _citation_transport(
    *,
    bioproject_ids=("529480",),
    biosample_ids=("11268033",),
    citing_pmids=("33288718",),
    pubmed_dois=None,
):
    pubmed_dois = pubmed_dois if pubmed_dois is not None else {"33288718": "10.1073/pnas.2005917117"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        if path.endswith("esearch.fcgi") and params.get("db") == "bioproject":
            return httpx.Response(200, json={"esearchresult": {"idlist": list(bioproject_ids)}})
        if path.endswith("esearch.fcgi") and params.get("db") == "biosample":
            return httpx.Response(200, json={"esearchresult": {"idlist": list(biosample_ids)}})
        if path.endswith("esummary.fcgi") and params.get("db") == "biosample":
            ids = params.get("id", "").split(",")
            result = {"uids": ids}
            for uid in ids:
                result[uid] = {"accession": "SAMN11268033"}
            return httpx.Response(200, json={"result": result})
        if params.get("dbfrom") in {"bioproject", "biosample"} and params.get("db") == "pubmed":
            return httpx.Response(
                200,
                json={
                    "linksets": [
                        {"linksetdbs": [{"linkname": f"{params.get('dbfrom')}_pubmed", "links": list(citing_pmids)}]}
                    ]
                },
            )
        if path.endswith("esummary.fcgi") and params.get("db") == "pubmed":
            ids = params.get("id", "").split(",")
            result = {"uids": ids}
            for pmid in ids:
                doi = pubmed_dois.get(pmid)
                result[pmid] = {"articleids": [{"idtype": "doi", "value": doi}]} if doi else {"articleids": []}
            return httpx.Response(200, json={"result": result})
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def _seed_parent_study(session, *, bioproject_accession="PRJNA529480", discovery_depth=0):
    study = Study(canonical_status="candidate", discovery_depth=discovery_depth)
    session.add(study)
    session.flush()
    session.add(
        ExternalIdentifier(
            study_id=study.study_id,
            identifier_type=IdentifierType.BIOPROJECT_ACCESSION.value,
            identifier_value=normalize_identifier(IdentifierType.BIOPROJECT_ACCESSION, bioproject_accession),
        )
    )
    session.flush()
    return study


def _citing_task(session, parent, bioproject_accession="PRJNA529480"):
    task = enqueue_task(
        session,
        TaskType.DISCOVER_CITING_STUDIES,
        study_id=parent.study_id,
        payload={"bioproject_accession": bioproject_accession},
        idempotency_key=f"test:{parent.study_id}",
    )
    session.commit()
    return task


def _biosample_citing_task(session, parent, biosample_accession="SAMN11268033"):
    task = enqueue_task(
        session,
        TaskType.DISCOVER_CITING_STUDIES,
        study_id=parent.study_id,
        payload={"biosample_accession": biosample_accession},
        idempotency_key=f"test:biosample:{parent.study_id}",
    )
    session.commit()
    return task


def test_elink_ids_collects_all_link_blocks_without_duplicates():
    class FakeHttp:
        def get_json(self, url, params):
            return {
                "linksets": [
                    {
                        "linksetdbs": [
                            {"linkname": "biosample_pubmed", "links": ["1", "2"]},
                            {"linkname": "biosample_pubmed_refs", "links": ["2", "3"]},
                        ]
                    }
                ]
            }, False

    assert _elink_ids(FakeHttp(), "https://example.test", "biosample", "pubmed", "123") == ["1", "2", "3"]


def test_creates_new_study_for_citing_pmid_with_doi(db_session, monkeypatch, retrieval_config):
    parent = _seed_parent_study(db_session)
    task = _citing_task(db_session, parent)

    adapter = _bioproject_adapter(_citation_transport(), retrieval_config)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_bioproject": adapter})

    handlers.handle_discover_citing_studies(db_session, task)
    db_session.commit()

    citing_studies = db_session.query(Study).filter(Study.study_id != parent.study_id).all()
    assert len(citing_studies) == 1
    citing = citing_studies[0]
    assert citing.discovery_depth == 1
    assert citing.discovery_parent_study_id == parent.study_id
    assert citing.discovery_root_study_id == parent.study_id
    assert citing.discovery_trigger == "bioproject_pubmed_citation"
    assert citing.canonical_status == "candidate"

    doi_ident = (
        db_session.query(ExternalIdentifier)
        .filter_by(study_id=citing.study_id, identifier_type=IdentifierType.DOI.value)
        .one()
    )
    assert doi_ident.identifier_value == normalize_identifier(IdentifierType.DOI, "10.1073/pnas.2005917117")
    assert doi_ident.verified is True

    new_task = (
        db_session.query(Task)
        .filter_by(study_id=citing.study_id, task_type=TaskType.DISCOVER_IDENTIFIERS.value)
        .one()
    )
    assert new_task is not None


def test_biosample_pubmed_links_create_new_study(db_session, monkeypatch, retrieval_config):
    parent = _seed_parent_study(db_session)
    db_session.add(
        ExternalIdentifier(
            study_id=parent.study_id,
            identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value,
            identifier_value="SAMN11268033",
        )
    )
    db_session.flush()
    task = _biosample_citing_task(db_session, parent)

    adapter = _bioproject_adapter(_citation_transport(), retrieval_config)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_biosample": adapter})

    handlers.handle_discover_citing_studies(db_session, task)
    db_session.commit()

    citing = db_session.query(Study).filter(Study.study_id != parent.study_id).one()
    assert citing.discovery_depth == 1
    assert citing.discovery_parent_study_id == parent.study_id
    assert citing.discovery_root_study_id == parent.study_id
    assert citing.discovery_trigger == "biosample_pubmed_citation"

    doi_ident = (
        db_session.query(ExternalIdentifier)
        .filter_by(study_id=citing.study_id, identifier_type=IdentifierType.DOI.value)
        .one()
    )
    assert doi_ident.identifier_value == normalize_identifier(IdentifierType.DOI, "10.1073/pnas.2005917117")
    assert doi_ident.source == "ncbi_biosample_pubmed_citation"


def test_already_known_doi_is_not_duplicated(db_session, monkeypatch, retrieval_config):
    parent = _seed_parent_study(db_session)
    already = Study(canonical_status="candidate")
    db_session.add(already)
    db_session.flush()
    db_session.add(
        ExternalIdentifier(
            study_id=already.study_id,
            identifier_type=IdentifierType.DOI.value,
            identifier_value=normalize_identifier(IdentifierType.DOI, "10.1073/pnas.2005917117"),
        )
    )
    db_session.flush()
    task = _citing_task(db_session, parent)

    adapter = _bioproject_adapter(_citation_transport(), retrieval_config)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_bioproject": adapter})

    handlers.handle_discover_citing_studies(db_session, task)
    db_session.commit()

    citing_studies = (
        db_session.query(Study).filter(Study.study_id.notin_([parent.study_id, already.study_id])).all()
    )
    assert citing_studies == []


def test_depth_cap_flags_review_instead_of_expanding(db_session, monkeypatch, retrieval_config):
    """A parent already AT the configured depth cap must not spawn further
    citing studies, and must record a review-flagged fact rather than
    silently doing nothing."""
    parent = _seed_parent_study(db_session, discovery_depth=1)
    task = _citing_task(db_session, parent)

    adapter = _bioproject_adapter(_citation_transport(), retrieval_config)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_bioproject": adapter})
    monkeypatch.setattr(
        handlers, "load_config", lambda: AppConfig(discovery=DiscoveryConfig(citation_expansion_max_depth=1))
    )

    handlers.handle_discover_citing_studies(db_session, task)
    db_session.commit()

    citing_studies = db_session.query(Study).filter(Study.study_id != parent.study_id).all()
    assert citing_studies == []
    capped_facts = (
        db_session.query(RawFact)
        .filter_by(study_id=parent.study_id, fact_type_candidate="citing_pmid_not_expanded")
        .all()
    )
    assert len(capped_facts) == 1
    assert capped_facts[0].review_status == "needs_review"


def test_max_citing_papers_cap_flags_excess(db_session, monkeypatch, retrieval_config):
    parent = _seed_parent_study(db_session)
    task = _citing_task(db_session, parent)

    pmids = [str(i) for i in range(5)]
    dois = {pmid: f"10.1000/citing.{pmid}" for pmid in pmids}
    adapter = _bioproject_adapter(
        _citation_transport(citing_pmids=tuple(pmids), pubmed_dois=dois), retrieval_config
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_bioproject": adapter})
    monkeypatch.setattr(
        handlers, "load_config", lambda: AppConfig(discovery=DiscoveryConfig(max_citing_papers_per_bioproject=2))
    )

    handlers.handle_discover_citing_studies(db_session, task)
    db_session.commit()

    citing_studies = db_session.query(Study).filter(Study.study_id != parent.study_id).all()
    assert len(citing_studies) == 2
    capped_facts = (
        db_session.query(RawFact)
        .filter_by(study_id=parent.study_id, fact_type_candidate="citing_pmid_not_expanded")
        .all()
    )
    assert len(capped_facts) == 1
    assert capped_facts[0].confidence_metadata["excess_pmid_count"] == 3


def test_no_bioproject_adapter_enabled_is_a_no_op(db_session, monkeypatch):
    parent = _seed_parent_study(db_session)
    task = _citing_task(db_session, parent)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})

    handlers.handle_discover_citing_studies(db_session, task)  # must not raise
    db_session.commit()

    assert db_session.query(Study).filter(Study.study_id != parent.study_id).count() == 0


def test_no_citing_pmids_is_a_no_op(db_session, monkeypatch, retrieval_config):
    parent = _seed_parent_study(db_session)
    task = _citing_task(db_session, parent)
    adapter = _bioproject_adapter(_citation_transport(citing_pmids=()), retrieval_config)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"ncbi_bioproject": adapter})

    handlers.handle_discover_citing_studies(db_session, task)
    db_session.commit()

    assert db_session.query(Study).filter(Study.study_id != parent.study_id).count() == 0
