"""Integration test for handle_discover_identifiers: fake, in-process
adapters (no network) standing in for Crossref/Europe PMC/OpenAlex, so the
orchestration logic (Source/RawFact creation with bibliographic fields,
related-identifier merging, Stage 2 dedup) is tested independent of any
live API.
"""
from datetime import datetime, timezone

import pytest

from fair_ocean_agent.database.enums import (
    CanonicalStatus,
    DataAvailabilityStatus,
    EntityLevel,
    IdentifierType,
    RelationshipType,
    SupportType,
)
from fair_ocean_agent.database.models import DataAsset, ExternalIdentifier, RawFact, Source, Study, StudySource
from fair_ocean_agent.identity.identifiers import normalize_doi
from fair_ocean_agent.sources.base import RawFactCandidate, RelatedIdentifier, SourceRecord, SourceRecordNotFoundError
from fair_ocean_agent.sources.datacite import DataCiteAdapter
from fair_ocean_agent.sources.europe_pmc import EuropePmcAdapter
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.database.enums import TaskType


class FakeAdapter:
    def __init__(self, name, record=None, facts=None, publication_fields=None, related=None, not_found=False):
        self.name = name
        self._record = record
        self._facts = facts or []
        self._publication_fields = publication_fields or {}
        self._related = related or []
        self._not_found = not_found
        self.closed = False

    def fetch_record(self, identifier):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no {self.name} record")
        return self._record

    def extract_structured_facts(self, record):
        return self._facts

    def parse_publication_fields(self, record):
        return self._publication_fields

    def find_related(self, record):
        return self._related

    def close(self):
        self.closed = True


class FakeNcbiBioSampleAdapter(FakeAdapter):
    """Adds fetch_record_by_accessions on top of FakeAdapter's plain
    fetch_record, for testing _fetch_ncbi_record_with_biosample_fallback's
    fallback path without touching the real NCBI adapter/network."""

    def __init__(self, *, fallback_record=None, fallback_not_found=False, **kwargs):
        super().__init__(**kwargs)
        self._fallback_record = fallback_record
        self._fallback_not_found = fallback_not_found
        self.fallback_calls: list[tuple[str, list[str]]] = []

    def fetch_record_by_accessions(self, bioproject_accession, accessions):
        self.fallback_calls.append((bioproject_accession, list(accessions)))
        if self._fallback_not_found:
            raise SourceRecordNotFoundError("no fallback record")
        return self._fallback_record


class FakeEuropePmcFullTextAdapter(EuropePmcAdapter):
    name = "europe_pmc"

    def __init__(self, fulltext_xml: str):
        self._fulltext_xml = fulltext_xml
        self._record = _make_record("europe_pmc")

    def fetch_record(self, identifier):
        return self._record

    def fetch_fulltext_xml(self, pmcid):
        return self._fulltext_xml

    def extract_structured_facts(self, record):
        return []

    def parse_publication_fields(self, record):
        return {"title": "DOI paper with repository IDs", "fulltext_available": True}

    def find_related(self, record):
        return [
            RelatedIdentifier(
                identifier_type=IdentifierType.PMCID,
                value="PMC5000000",
                relationship_type=RelationshipType.IS_PUBLICATION_OF,
                source="europe_pmc",
            )
        ]

    def close(self):
        pass


def _make_record(source_name: str) -> SourceRecord:
    return SourceRecord(
        source_name=source_name,
        external_identifier="10.1234/x",
        url=f"https://example.org/{source_name}",
        raw={"stub": True},
        retrieved_at=datetime.now(timezone.utc),
        content_hash="deadbeef",
    )


def _seeded_study_with_doi(session, doi="10.1234/x") -> Study:
    study = Study(title=None)
    session.add(study)
    session.flush()
    session.add(
        ExternalIdentifier(
            study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value=normalize_doi(doi)
        )
    )
    session.flush()
    return study


def _task_for(session, study) -> "handlers.Task":
    task = enqueue_task(session, TaskType.DISCOVER_IDENTIFIERS, study_id=study.study_id)
    session.commit()
    return task


def test_handler_creates_source_facts_and_publication(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    record = _make_record("crossref")
    fake = FakeAdapter(
        "crossref",
        record=record,
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="title",
                raw_field_name="title",
                raw_value="A Fake Study Title",
                source_locator="crossref.message.title",
                support_type=SupportType.STRUCTURED_SOURCE,
            )
        ],
        publication_fields={"doi": "10.1234/x", "title": "A Fake Study Title", "publication_year": 2020},
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    sources = db_session.query(Source).filter_by(study_id=study.study_id).all()
    assert len(sources) == 1 and sources[0].source_name == "crossref"
    assert sources[0].publication_year == 2020

    # New Source creation always routes through identity/source_linking.py's
    # create_source() choke point, so every Source gets a matching "home"
    # study_sources row.
    study_source = db_session.query(StudySource).filter_by(source_id=sources[0].source_id).one()
    assert study_source.study_id == study.study_id
    assert study_source.relationship_type == RelationshipType.IS_HOME_OF.value
    assert study_source.confidence == SupportType.STRUCTURED_SOURCE.value

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id).all()
    assert len(facts) == 1 and facts[0].raw_value == "A Fake Study Title"

    refreshed = db_session.get(Study, study.study_id)
    assert refreshed.title == "A Fake Study Title"


def test_handler_continues_when_one_adapter_has_no_record(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    missing = FakeAdapter("europe_pmc", not_found=True)
    present = FakeAdapter(
        "openalex",
        record=_make_record("openalex"),
        publication_fields={"openalex_id": "W1", "title": "Still Works"},
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": missing, "openalex": present})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    sources = db_session.query(Source).filter_by(study_id=study.study_id).all()
    assert [s.source_name for s in sources] == ["openalex"]


def test_handler_merges_study_on_related_identifier_collision(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session, doi="10.1234/new")
    other_study = Study(title="Pre-existing study found by PMID")
    db_session.add(other_study)
    db_session.flush()
    db_session.add(
        ExternalIdentifier(study_id=other_study.study_id, identifier_type=IdentifierType.PMID.value, identifier_value="999999")
    )
    db_session.commit()

    task = _task_for(db_session, study)
    fake = FakeAdapter(
        "europe_pmc",
        record=_make_record("europe_pmc"),
        related=[
            RelatedIdentifier(
                identifier_type=IdentifierType.PMID,
                value="999999",
                relationship_type=RelationshipType.IS_PUBLICATION_OF,
                source="europe_pmc",
            )
        ],
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": fake})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    refreshed_other = db_session.get(Study, other_study.study_id)
    assert refreshed_other.canonical_status == CanonicalStatus.MERGED.value

    # the DOI-bearing study absorbed the PMID that used to belong to other_study
    merged_ids = {
        (ei.identifier_type, ei.identifier_value)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (IdentifierType.PMID.value, "999999") in merged_ids


def test_handler_raises_not_implemented_without_doi(db_session):
    study = Study(title="No DOI here")
    db_session.add(study)
    db_session.flush()
    task = _task_for(db_session, study)

    with pytest.raises(NotImplementedError):
        handlers.handle_discover_identifiers(db_session, task)


def test_handler_is_idempotent_on_retry(db_session, monkeypatch):
    """Simulates a retried task: calling the handler twice for the same
    study/adapter must not duplicate Source/RawFact rows, and the Source's
    bibliographic fields must still end up fully populated."""
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    fake = FakeAdapter(
        "crossref",
        record=_make_record("crossref"),
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="title",
                raw_field_name="title",
                raw_value="Retried Study Title",
                source_locator="crossref.message.title",
                support_type=SupportType.STRUCTURED_SOURCE,
            )
        ],
        publication_fields={"doi": "10.1234/x", "title": "Retried Study Title", "publication_year": 2021},
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"crossref": fake})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()
    handlers.handle_discover_identifiers(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=study.study_id).count() == 1
    source = db_session.query(Source).filter_by(study_id=study.study_id).one()
    assert source.publication_year == 2021


def test_handler_mines_fulltext_identifiers_and_resolves_repository_sources(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title>"
        "<p>Raw reads are available under BioProject PRJNA515494.</p>"
        "</sec></article>"
    )
    biosample_adapter = FakeAdapter(
        "ncbi_biosample",
        record=_make_record("ncbi_biosample"),
        facts=[
            RawFactCandidate(
                entity_level=EntityLevel.SAMPLE,
                fact_type_candidate="collection_date",
                raw_field_name="collection_date",
                raw_value="2020-01",
                source_locator="ncbi_biosample.SAMN1.collection_date",
                entity_external_id="SAMN1",
                entity_label="SAMN1",
            )
        ],
    )
    # Tier-2 evidence (a regex-matched accession from prose) is only trusted
    # once verify_deterministic_identifier confirms it resolves against its
    # own source API -- an ncbi_bioproject adapter that can fetch_record()
    # this exact accession is what makes that verification succeed here.
    bioproject_adapter = FakeAdapter("ncbi_bioproject", record=_make_record("ncbi_bioproject"))
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {
            "europe_pmc": europe_pmc_adapter,
            "ncbi_biosample": biosample_adapter,
            "ncbi_bioproject": bioproject_adapter,
        },
    )

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    identifiers = {
        (ei.identifier_type, ei.identifier_value, ei.source)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (
        IdentifierType.BIOPROJECT_ACCESSION.value,
        "PRJNA515494",
        "europe_pmc_fulltext_identifier_scan",
    ) in identifiers
    assert db_session.query(Source).filter_by(study_id=study.study_id, source_name="ncbi_biosample").count() == 1
    assert db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_fulltext").count() == 0
    # Give-up tracking: a real per-sample entity got created via the
    # resolved BioProject/BioSample chain, so this counts as accessible.
    assert study.data_availability_status == DataAvailabilityStatus.ACCESSIBLE.value


def test_handler_marks_not_accessible_when_no_repository_search_pass_finds_anything(db_session, monkeypatch):
    """Give-up tracking, per an explicit user request: once the staged
    repository search (BioProject/SRA/ENA, then Zenodo/Dryad/Figshare/OSF,
    then DataCite) has fully run and found nothing accessible, the study
    is marked NOT_ACCESSIBLE so future rediscovery backfills stop
    re-searching it."""
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title>"
        "<p>Data are available from the corresponding author upon request.</p>"
        "</sec></article>"
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": europe_pmc_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    assert study.data_availability_status == DataAvailabilityStatus.NOT_ACCESSIBLE.value


def test_handler_never_marks_a_primer_reference_citation_study(db_session, monkeypatch):
    """A study reached only via primer-reference chasing has a different
    goal (a primer sequence, or the next reference in the citation chain),
    so the accessible-data question doesn't apply to it -- must stay
    UNKNOWN even when the repository search finds nothing."""
    study = _seeded_study_with_doi(db_session)
    study.discovery_trigger = "primer_reference_citation"
    db_session.flush()
    task = _task_for(db_session, study)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title><p>No data statement.</p></sec></article>"
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": europe_pmc_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    assert study.data_availability_status == DataAvailabilityStatus.UNKNOWN.value


def test_discover_identifiers_from_fulltext_uses_local_pdf_with_no_pmcid(db_session, monkeypatch, tmp_path):
    """Real gap, confirmed live: a closed-access paper with no PMCID at all
    (e.g. 10.1111/jeu.12975, 10.1002/edn3.570) previously had NO route to
    ever surface its own BioProject/BioSample accessions -- this function
    required a PMCID unconditionally before ever checking for a local PDF
    override, the same latent gap handle_extract_text_facts had."""
    study = Study(title="No PMCID, local PDF only")
    db_session.add(study)
    db_session.flush()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    monkeypatch.setenv(handlers.LOCAL_PDF_PATH_ENV, str(pdf_path))
    monkeypatch.setattr(
        handlers, "extract_pdf_text", lambda path: "Raw reads are available under BioProject PRJNA515494."
    )
    bioproject_adapter = FakeAdapter("ncbi_bioproject", record=_make_record("ncbi_bioproject"))

    handlers._discover_identifiers_from_fulltext(db_session, study, {"ncbi_bioproject": bioproject_adapter})
    db_session.commit()

    identifiers = {
        (ei.identifier_type, ei.identifier_value, ei.source)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (IdentifierType.BIOPROJECT_ACCESSION.value, "PRJNA515494", "local_pdf_identifier_scan") in identifiers


def _seeded_study_with_pmcid_for_fulltext_scan(session, pmcid="PMC9999999") -> Study:
    study = Study(title="A study")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.PMCID.value, identifier_value=pmcid))
    session.flush()
    return study


class FakeDataCiteAdapter(DataCiteAdapter):
    """DATASET_DOI isn't in verify_deterministic_identifier's
    STRUCTURED_SEQUENCE_IDENTIFIER_TYPES short-circuit set, so a Pass-3 hit
    still goes through a real fetch_record verification call -- override
    both, rather than only find_datasets_citing, to stand in for the real
    HTTP-backed adapter."""

    def __init__(self, citing_dois: list[str]):
        self._citing_dois = citing_dois

    def find_datasets_citing(self, article_doi: str) -> list[str]:
        return self._citing_dois

    def fetch_record(self, identifier: str) -> SourceRecord:
        return _make_record("datacite")


def test_discover_identifiers_from_fulltext_tries_pass3_datacite_when_pass1_and_pass2_find_nothing(db_session):
    """Regression test: handlers.py used RelationshipType.IS_DATASET_FOR to
    build the Pass 3 RelatedIdentifier without importing RelationshipType
    at all -- confirmed live on a real cluster run (NameError: name
    'RelationshipType' is not defined), never caught by
    test_datacite.py's own tests since those only exercise
    DataCiteAdapter.find_datasets_citing in isolation, not this
    integration point."""
    study = _seeded_study_with_pmcid_for_fulltext_scan(db_session)
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.DOI.value, identifier_value="10.1371/journal.pone.0109118"))
    db_session.flush()

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title><p>No accessions or dataset links mentioned here.</p></sec></article>"
    )
    datacite_adapter = FakeDataCiteAdapter(citing_dois=["10.5281/zenodo.99999999"])

    handlers._discover_identifiers_from_fulltext(
        db_session, study, {"europe_pmc": europe_pmc_adapter, "datacite": datacite_adapter}
    )
    db_session.commit()

    identifiers = {
        (ei.identifier_type, ei.identifier_value, ei.source)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (IdentifierType.DATASET_DOI.value, "10.5281/zenodo.99999999", "datacite_related_identifiers") in identifiers


def test_discover_identifiers_from_fulltext_skips_pass2_when_pass1_already_found_something(db_session):
    """Staged search, per an explicit user request: Zenodo/Dryad/Figshare/
    OSF (Pass 2) are only queried once BioProject/SRA/ENA accession mining
    (Pass 1) has come up empty. A Zenodo adapter present but never called
    is the real assertion here, not just "no Zenodo identifier saved"."""
    study = _seeded_study_with_pmcid_for_fulltext_scan(db_session)

    class _FailIfCalledAdapter(FakeAdapter):
        def fetch_record(self, identifier):
            raise AssertionError("Pass 2 adapter should never be called when Pass 1 already found something")

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title>"
        "<p>Raw reads are under NCBI BioProject PRJNA515494. "
        "Related data also at 10.5281/zenodo.10381280.</p></sec></article>"
    )
    bioproject_adapter = FakeAdapter("ncbi_bioproject", record=_make_record("ncbi_bioproject"))
    zenodo_adapter = _FailIfCalledAdapter("zenodo")

    handlers._discover_identifiers_from_fulltext(
        db_session, study, {"europe_pmc": europe_pmc_adapter, "ncbi_bioproject": bioproject_adapter, "zenodo": zenodo_adapter}
    )
    # Doesn't raise -> zenodo_adapter.fetch_record was correctly never called.


def test_discover_identifiers_from_fulltext_tries_pass2_when_pass1_finds_nothing(db_session):
    study = _seeded_study_with_pmcid_for_fulltext_scan(db_session)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        "<article><sec><title>Data Availability</title>"
        "<p>Related data at 10.5281/zenodo.10381280.</p></sec></article>"
    )
    zenodo_adapter = FakeAdapter("zenodo", record=_make_record("zenodo"))
    handlers._discover_identifiers_from_fulltext(
        db_session, study, {"europe_pmc": europe_pmc_adapter, "zenodo": zenodo_adapter}
    )
    db_session.commit()

    identifiers = {
        (ei.identifier_type, ei.identifier_value)
        for ei in db_session.query(ExternalIdentifier).filter_by(study_id=study.study_id).all()
    }
    assert (IdentifierType.DATASET_DOI.value, "10.5281/zenodo.10381280") in identifiers


def test_handler_discovers_supplements_inline_during_doi_driven_discovery(db_session, monkeypatch):
    """A DOI-driven DISCOVER_IDENTIFIERS pass must also surface
    supplementary-material references for free, not only via a separate
    manually-triggered DISCOVER_SUPPLEMENTS backfill -- this is the same
    open full text already being fetched for identifier mining."""
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        """<article><body><sec>
        <supplementary-material id="TS1"><media xmlns:xlink="http://www.w3.org/1999/xlink"
        xlink:href="Table_1.csv" mimetype="text" mime-subtype="csv"><?size 40?></media></supplementary-material>
        </sec></body></article>"""
    )
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": europe_pmc_adapter})

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    sources = db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_supplement").all()
    assert {s.external_identifier for s in sources} == {"Table_1.csv"}
    assets = db_session.query(DataAsset).filter_by(study_id=study.study_id).all()
    assert {a.file_name for a in assets} == {"Table_1.csv"}


def test_handler_extracts_publication_metadata_inline_during_doi_driven_discovery(db_session, monkeypatch):
    """A DOI-driven DISCOVER_IDENTIFIERS pass must also resolve
    license/accessRights/paper_authors_list/project_contact
    (from JATS <permissions>/<contrib-group>) and bibliographicCitation
    (from Crossref) for free -- no LLM, no extra network cost. rightsHolder
    is now extracted in the full-text LLM stage from rights text.
    Same shape
    as the inline supplement-discovery test above."""
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)

    europe_pmc_adapter = FakeEuropePmcFullTextAdapter(
        """<article><body><sec><permissions>
        <copyright-holder>Test Authors</copyright-holder>
        <license license-type="open-access" xmlns:xlink="http://www.w3.org/1999/xlink"
        xlink:href="http://creativecommons.org/licenses/by/4.0/"></license>
        </permissions>
        <contrib-group><contrib contrib-type="author" corresp="yes">
        <name><surname>Doe</surname><given-names>Jane</given-names></name>
        <email>jane@example.org</email></contrib></contrib-group>
        </sec></body></article>"""
    )
    crossref_adapter = FakeAdapter(
        "crossref",
        record=SourceRecord(
            source_name="crossref",
            external_identifier="10.1234/x",
            raw={
                "title": ["A test paper"],
                "author": [{"given": "Jane", "family": "Doe"}],
                "published": {"date-parts": [[2020]]},
                "container-title": ["Test Journal"],
                "DOI": "10.1234/x",
            },
            retrieved_at=datetime.now(timezone.utc),
            content_hash="deadbeef",
        ),
    )
    monkeypatch.setattr(
        handlers, "_build_enabled_adapters", lambda: {"europe_pmc": europe_pmc_adapter, "crossref": crossref_adapter}
    )

    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()

    facts = {
        f.fact_type_candidate: f.raw_value
        for f in db_session.query(RawFact).filter_by(
            study_id=study.study_id, extraction_method="deterministic_publication_metadata"
        )
    }
    assert facts["license"] == "http://creativecommons.org/licenses/by/4.0/"
    assert "rightsHolder" not in facts
    assert facts["accessRights"] == "open access"
    assert facts["paper_authors_list"] == "Jane Doe"
    assert facts["project_contact"] == "Jane Doe <jane@example.org>"
    assert "Doe J" in facts["bibliographicCitation"]

    fact_count = db_session.query(RawFact).filter_by(
        study_id=study.study_id, extraction_method="deterministic_publication_metadata"
    ).count()

    # Idempotent: re-running the same (simulated retry) task must not
    # duplicate facts.
    handlers.handle_discover_identifiers(db_session, task)
    db_session.commit()
    assert (
        db_session.query(RawFact)
        .filter_by(study_id=study.study_id, extraction_method="deterministic_publication_metadata")
        .count()
        == fact_count
    )


def test_handler_raises_runtime_error_when_no_adapters_enabled(db_session, monkeypatch):
    study = _seeded_study_with_doi(db_session)
    task = _task_for(db_session, study)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {})

    with pytest.raises(RuntimeError):
        handlers.handle_discover_identifiers(db_session, task)


def test_ncbi_biosample_fallback_not_used_when_elink_succeeds(db_session):
    """The common case: fetch_record itself succeeds, so the fallback
    (and its BIOSAMPLE_ACCESSION lookup) is never even attempted."""
    study = Study(title="Elink succeeds")
    db_session.add(study)
    db_session.flush()
    real_record = _make_record("ncbi_biosample")
    adapter = FakeNcbiBioSampleAdapter(name="ncbi_biosample", record=real_record)

    result = handlers._fetch_ncbi_record_with_biosample_fallback(db_session, study, "ncbi_biosample", adapter, "PRJNA1")

    assert result is real_record
    assert adapter.fallback_calls == []


def test_ncbi_biosample_fallback_used_when_elink_empty_and_accessions_known(db_session):
    """Regression test for a real live gap (confirmed 2026-08-31 against a
    real BioProject, PRJNA762627): NCBI's own bioproject<->biosample elink
    cross-reference can be empty even though real BioSamples exist and
    were already discovered independently (e.g. via ENA). When
    fetch_record raises SourceRecordNotFoundError, the fallback should
    fire using whatever BIOSAMPLE_ACCESSION identifiers are already
    attached to the study."""
    study = Study(title="Elink empty, accessions known")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value, identifier_value="SAMN1"))
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value, identifier_value="SAMN2"))
    db_session.commit()
    fallback_record = _make_record("ncbi_biosample")
    adapter = FakeNcbiBioSampleAdapter(name="ncbi_biosample", not_found=True, fallback_record=fallback_record)

    result = handlers._fetch_ncbi_record_with_biosample_fallback(db_session, study, "ncbi_biosample", adapter, "PRJNA1")

    assert result is fallback_record
    assert adapter.fallback_calls == [("PRJNA1", ["SAMN1", "SAMN2"])]


def test_ncbi_biosample_fallback_skipped_when_no_accessions_known(db_session):
    study = Study(title="Elink empty, nothing to fall back to")
    db_session.add(study)
    db_session.flush()
    adapter = FakeNcbiBioSampleAdapter(name="ncbi_biosample", not_found=True)

    result = handlers._fetch_ncbi_record_with_biosample_fallback(db_session, study, "ncbi_biosample", adapter, "PRJNA1")

    assert result is None
    assert adapter.fallback_calls == []


def test_ncbi_biosample_fallback_returns_none_when_fallback_also_fails(db_session):
    study = Study(title="Elink empty, fallback also empty")
    db_session.add(study)
    db_session.flush()
    db_session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.BIOSAMPLE_ACCESSION.value, identifier_value="SAMN1"))
    db_session.commit()
    adapter = FakeNcbiBioSampleAdapter(name="ncbi_biosample", not_found=True, fallback_not_found=True)

    result = handlers._fetch_ncbi_record_with_biosample_fallback(db_session, study, "ncbi_biosample", adapter, "PRJNA1")

    assert result is None
    assert adapter.fallback_calls == [("PRJNA1", ["SAMN1"])]


def test_ncbi_bioproject_never_uses_fallback_even_when_not_found(db_session):
    """The fallback is ncbi_biosample-specific -- ncbi_bioproject has no
    fetch_record_by_accessions method at all and shouldn't be expected to."""
    study = Study(title="ncbi_bioproject not found")
    db_session.add(study)
    db_session.flush()
    adapter = FakeAdapter(name="ncbi_bioproject", not_found=True)

    result = handlers._fetch_ncbi_record_with_biosample_fallback(db_session, study, "ncbi_bioproject", adapter, "PRJNA1")

    assert result is None
