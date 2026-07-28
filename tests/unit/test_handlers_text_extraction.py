"""Integration tests for handle_extract_text_facts: a fake europe_pmc
adapter (no network) standing in for the real one, and a MockLLMBackend
standing in for a real model server -- no live network, no real inference
endpoint required."""
import json

import pytest

from fair_ocean_agent.database.enums import IdentifierType, TaskType
from fair_ocean_agent.database.models import ExternalIdentifier, RawFact, Source, Study
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.disabled import DisabledLLMBackend
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.sources.base import SourceRecordNotFoundError
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task

FULLTEXT_XML = """<article><body>
<sec><title>Materials and Methods</title>
<sec><title>Sampling</title><p>Water samples were collected on 4 January 2022 at a depth of 5 meters.</p></sec>
</sec>
</body></article>"""

MULTI_SECTION_XML = """<article><body>
<sec><title>Methods</title>
<sec><title>Sampling</title><p>Water samples were collected on 4 January 2022 at a depth of 5 meters.</p></sec>
<sec><title>PCR</title><p>PCR used the primers mlCOIintF and jgHCO2198.</p></sec>
</sec>
</body></article>"""


class FakeEuropePmcAdapter:
    name = "europe_pmc"

    def __init__(self, fulltext_xml=FULLTEXT_XML, not_found=False):
        self._fulltext_xml = fulltext_xml
        self._not_found = not_found

    def fetch_fulltext_xml(self, pmcid):
        if self._not_found:
            raise SourceRecordNotFoundError(f"no fulltext for {pmcid}")
        return self._fulltext_xml

    def close(self):
        pass


class SecondCallFailsBackend(MockLLMBackend):
    def generate(self, *args, **kwargs):
        if len(self.calls) == 1:
            raise LLMBackendError("simulated section timeout")
        return super().generate(*args, **kwargs)


def _seeded_study_with_pmcid(session, pmcid="PMC1234567") -> Study:
    study = Study(title="A study")
    session.add(study)
    session.flush()
    session.add(ExternalIdentifier(study_id=study.study_id, identifier_type=IdentifierType.PMCID.value, identifier_value=pmcid))
    session.flush()
    return study


def _task_for(session, study):
    task = enqueue_task(session, TaskType.EXTRACT_TEXT_FACTS, study_id=study.study_id)
    session.commit()
    return task


@pytest.fixture(autouse=True)
def _reset_llm_cache():
    handlers.reset_llm_backend_cache()
    yield
    handlers.reset_llm_backend_cache()


def test_handler_extracts_and_persists_verified_facts(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_quote": "collected on 4 January 2022"}]
    )
    handlers._llm_backend_cache = MockLLMBackend(label="mock-model", responses=[response])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_fulltext").one()
    assert source.external_identifier == "PMC1234567"

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").all()
    assert len(facts) == 1
    assert facts[0].evidence_quote == "collected on 4 January 2022"
    assert facts[0].model_name == "mock-model"


def test_handler_drops_facts_with_fabricated_evidence(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    response = json.dumps([{"fact_type_candidate": "fake", "raw_value": "x", "evidence_quote": "not in the source at all"}])
    handlers._llm_backend_cache = MockLLMBackend(responses=[response])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").all()
    assert facts == []
    # the Source row is still created -- retrieval + section selection succeeded, just no facts survived
    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1


def test_handler_is_idempotent_on_retry(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_quote": "collected on 4 January 2022"}]
    )
    handlers._llm_backend_cache = MockLLMBackend(responses=[response, response])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()
    handlers.handle_extract_text_facts(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").count() == 1


def test_handler_preserves_successful_sections_when_later_section_times_out(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_quote": "collected on 4 January 2022"}]
    )
    handlers._llm_backend_cache = SecondCallFailsBackend(label="flaky-model", responses=[response])
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=MULTI_SECTION_XML)},
    )

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_fulltext").count() == 1
    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").all()
    assert len(facts) == 1
    assert facts[0].raw_value == "2022-01-04"


def test_handler_raises_not_implemented_without_pmcid(db_session):
    study = Study(title="No PMCID")
    db_session.add(study)
    db_session.flush()
    task = _task_for(db_session, study)

    with pytest.raises(NotImplementedError):
        handlers.handle_extract_text_facts(db_session, task)


def test_handler_no_ops_when_no_open_access_fulltext(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter(not_found=True)})

    handlers.handle_extract_text_facts(db_session, task)  # must not raise
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 0


def test_handler_no_ops_when_no_relevant_sections(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    irrelevant_xml = "<article><body><sec><title>Introduction</title><p>text</p></sec></body></article>"
    monkeypatch.setattr(
        handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=irrelevant_xml)}
    )

    handlers.handle_extract_text_facts(db_session, task)  # must not raise
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 0


def test_handler_raises_llm_backend_error_when_llm_disabled(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})
    # Explicitly inject a DisabledLLMBackend rather than relying on
    # _build_llm_backend_cached() resolving config/local.yaml's
    # llm.enabled: false -- a developer's own machine-specific overrides
    # (e.g. a real local model configured for manual testing) must never
    # change what this test is actually checking: that a disabled backend
    # raises loudly.
    handlers._llm_backend_cache = DisabledLLMBackend()

    with pytest.raises(LLMBackendError):
        handlers.handle_extract_text_facts(db_session, task)
