"""Integration tests for handle_extract_text_facts: a fake europe_pmc
adapter (no network) standing in for the real one, and a MockLLMBackend
standing in for a real model server -- no live network, no real inference
endpoint required."""
import json

import pytest

from fair_ocean_agent.database.enums import EntityLevel, IdentifierType, ReviewStatus, SupportType, TaskStatus, TaskType
from fair_ocean_agent.database.models import Entity, ExternalIdentifier, RawFact, Source, Study
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.disabled import DisabledLLMBackend
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.sources.base import RawFactCandidate
from fair_ocean_agent.sources.base import SourceRecordNotFoundError
from fair_ocean_agent.workflow import handlers
from fair_ocean_agent.workflow.task_queue import enqueue_task
from fair_ocean_agent.workflow.worker import run_worker

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

PCR_FLAG_XML = """<article><body>
<sec><title>Materials and Methods</title>
<p>A TaqMan qPCR assay used a FAM reporter dye and BHQ quencher.</p>
</sec>
</body></article>"""

ABSTRACT_SCOPE_XML = """<article><front><article-meta><abstract>
<p>This study investigates prokaryotic microorganisms, including bacteria and archaea,
across water-column and sediment samples from the deep sea.</p>
</abstract></article-meta></front><body>
<sec><title>Materials and Methods</title>
<sec><title>Sampling</title><p>Water samples were collected on 4 January 2022 at a depth of 5 meters.</p></sec>
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
    def __init__(self, *args, fail_after_calls=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._fail_after_calls = fail_after_calls

    def generate(self, *args, **kwargs):
        if len(self.calls) >= self._fail_after_calls:
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
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "SAMPLING.P001"}]
    )
    handlers._llm_backend_cache = MockLLMBackend(label="mock-model", responses=[response])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id, source_name="europe_pmc_fulltext").one()
    assert source.external_identifier == "PMC1234567"
    assert source.source_version == f"{handlers.PROMPT_VERSION}:mock-model"

    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").all()
    assert len(facts) == 1
    assert facts[0].evidence_quote == "Sampling Water samples were collected on 4 January 2022 at a depth of 5 meters."
    assert facts[0].confidence_metadata == {"evidence_ids": ["SAMPLING.P001"]}
    assert facts[0].model_name == "mock-model"


def test_handler_can_force_local_pdf_fulltext_source(db_session, monkeypatch, tmp_path):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    monkeypatch.setenv(handlers.LOCAL_PDF_PATH_ENV, str(pdf_path))
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: pytest.fail("Europe PMC should not be called when local PDF override is set"),
    )
    monkeypatch.setattr(
        handlers,
        "extract_pdf_sections",
        lambda path: [{"title": "Sampling", "text": "Water samples were collected on 4 January 2022."}],
    )
    monkeypatch.setattr(handlers, "extract_pdf_text", lambda path: "Abstract\n\nMethods")
    monkeypatch.setattr(handlers, "detect_and_correct_elev_depth_mislabeling", lambda *args, **kwargs: None)
    monkeypatch.setattr(handlers, "detect_text_search_flags", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_phix_percentage_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_controlled_search_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_section_categories_present", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_study_factor", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_study_target_taxonomic_scope", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_rights_holder", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_funding_source", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "extract_section_category_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_llm_judged_search_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        handlers,
        "extract_facts_from_section",
        lambda *args, **kwargs: (
            [
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate="collection_date",
                    raw_field_name="collection_date",
                    raw_value="2022-01-04",
                    source_locator="pdf:Sampling",
                    support_type=SupportType.EXPLICIT,
                    evidence_quote="Water samples were collected on 4 January 2022.",
                )
            ],
            None,
        ),
    )
    handlers._llm_backend_cache = MockLLMBackend(label="mock-model", responses=[])

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id, source_name="local_pdf_fulltext").one()
    assert source.external_identifier == str(pdf_path.resolve())
    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").all()
    assert len(facts) == 1
    assert facts[0].raw_value == "2022-01-04"


def test_handler_persists_deterministic_text_search_flags(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    handlers._llm_backend_cache = MockLLMBackend(label="mock-model", responses=["[]"])
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=PCR_FLAG_XML)},
    )

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    facts = {
        fact.fact_type_candidate: fact
        for fact in db_session.query(RawFact).filter_by(
            study_id=study.study_id,
            extraction_method="deterministic_text_search_flagging",
        )
    }
    assert {"probe_based_qPCR_ddPCR_assay_0_1", "pcr_0_1"} <= set(facts)
    assert facts["probe_based_qPCR_ddPCR_assay_0_1"].raw_value == "true"
    assert facts["probe_based_qPCR_ddPCR_assay_0_1"].evidence_quote == (
        "Materials and Methods A TaqMan qPCR assay used a FAM reporter dye and BHQ quencher."
    )
    assert facts["probe_based_qPCR_ddPCR_assay_0_1"].support_type == "deterministically_derived"
    assert facts["probe_based_qPCR_ddPCR_assay_0_1"].prompt_version == handlers.PROMPT_VERSION


def test_handler_persists_abstract_generated_target_taxonomic_scope(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    handlers._llm_backend_cache = MockLLMBackend(
        label="mock-model",
        responses=[
            json.dumps({"study_factor": "Water-column and sediment sample type in deep-sea habitats."}),
            json.dumps(
                {
                    "study_target_taxonomic_scope": (
                        "prokaryotic microorganisms | bacteria | archaea"
                    )
                }
            ),
            "[]",
        ],
    )
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=ABSTRACT_SCOPE_XML)},
    )

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    fact = (
        db_session.query(RawFact)
        .filter_by(
            study_id=study.study_id,
            fact_type_candidate="study_target_taxonomic_scope",
            extraction_method="llm_generated_study_target_taxonomic_scope",
        )
        .one()
    )
    assert fact.raw_value == "prokaryotic microorganisms | bacteria | archaea"
    assert fact.support_type == "inferred"
    assert "including bacteria and archaea" in fact.evidence_quote


def test_handler_materializes_a_real_entity_per_assay_tag(db_session, monkeypatch):
    """Two distinct assays' primer facts (assay_tag on the LLM response)
    must each get a real Entity + RawFact.entity_id -- not collapse into
    one untethered study-wide fact -- so mapping/faire.py can produce
    separate projectMetadata rows for each assay downstream.

    annealing_temperature is now gated on pcr_0_1 (extraction/faire_fields.py's
    required_any_flags) -- the fake fulltext must genuinely mention PCR so
    the real deterministic detector activates the flag and the mocked
    LLM response's gated facts aren't filtered out by allowed_fact_types."""
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    pcr_fulltext_xml = """<article><body>
<sec><title>Materials and Methods</title>
<sec><title>Sampling</title><p>PCR reactions used primers for 16S and 18S targets at varying annealing temperatures.</p></sec>
</sec>
</body></article>"""

    response = json.dumps(
        [
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "55C",
                "evidence_id": "SAMPLING.P001",
                "assay_tag": "16S-V3V4",
            },
            {
                "fact_type_candidate": "annealing_temperature",
                "raw_value": "60C",
                "evidence_id": "SAMPLING.P001",
                "assay_tag": "18S-V9",
            },
        ]
    )
    handlers._llm_backend_cache = MockLLMBackend(label="mock-model", responses=[response])
    monkeypatch.setattr(
        handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=pcr_fulltext_xml)}
    )

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    assay_entities = {
        entity.external_identifier: entity
        for entity in db_session.query(Entity).filter_by(study_id=study.study_id, entity_level=EntityLevel.ASSAY.value)
    }
    assert set(assay_entities) == {"16S-V3V4", "18S-V9"}

    facts = {
        fact.raw_value: fact
        for fact in db_session.query(RawFact).filter_by(
            study_id=study.study_id, extraction_method="llm_text_extraction"
        )
    }
    assert facts["55C"].entity_id == assay_entities["16S-V3V4"].entity_id
    assert facts["60C"].entity_id == assay_entities["18S-V9"].entity_id


def test_handler_drops_facts_with_fabricated_evidence(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    response = json.dumps([{"fact_type_candidate": "fake", "raw_value": "x", "evidence_id": "SAMPLING.P999"}])
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
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "SAMPLING.P001"}]
    )
    handlers._llm_backend_cache = MockLLMBackend(responses=[response, response])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()
    handlers.handle_extract_text_facts(db_session, task)  # simulated retry
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 1
    assert db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").count() == 1


def test_handler_reprocesses_fulltext_with_new_prompt_version_or_model(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    old_source = Source(
        study_id=study.study_id,
        source_type="article_fulltext",
        source_name="europe_pmc_fulltext",
        external_identifier="PMC1234567",
        source_version="text-extraction-v3:older-model",
    )
    db_session.add(old_source)
    db_session.flush()
    old_fact = RawFact(
        study_id=study.study_id,
        source_id=old_source.source_id,
        raw_field_name="target_gene",
        raw_value="16S rRNA | 16S",
        fact_type_candidate="target_gene",
        entity_level=EntityLevel.STUDY.value,
        support_type="explicit",
        extraction_method="deterministic_text_search_flagging",
        prompt_version="text-extraction-v3",
        review_status=ReviewStatus.ACCEPTED.value,
    )
    db_session.add(old_fact)
    db_session.commit()

    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "SAMPLING.P001"}]
    )
    handlers._llm_backend_cache = MockLLMBackend(label="new-model", responses=[response])
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    assert db_session.query(Source).filter_by(study_id=study.study_id).count() == 2
    assert (
        db_session.query(Source)
        .filter_by(
            study_id=study.study_id,
            source_name="europe_pmc_fulltext",
            source_version=f"{handlers.PROMPT_VERSION}:new-model",
        )
        .count()
        == 1
    )
    new_model_facts = db_session.query(RawFact).filter_by(study_id=study.study_id, model_name="new-model").all()
    assert {fact.fact_type_candidate for fact in new_model_facts} == {
        "collection_date",
        "assay_target_taxa",
        # neg_cont_0_1/pos_cont_0_1 now confidently default to "0" rather
        # than staying unresolved when no control/blank mention is found.
        "neg_cont_0_1",
        "pos_cont_0_1",
    }
    db_session.refresh(old_fact)
    assert old_fact.review_status == ReviewStatus.REJECTED.value


def test_handler_fails_atomically_when_later_section_times_out(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    response = json.dumps(
        [{"fact_type_candidate": "collection_date", "raw_value": "2022-01-04", "evidence_id": "SAMPLING.P001"}]
    )
    handlers._llm_backend_cache = SecondCallFailsBackend(label="flaky-model", responses=[response], fail_after_calls=1)
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=MULTI_SECTION_XML)},
    )

    with pytest.raises(LLMBackendError, match="simulated section timeout"):
        handlers.handle_extract_text_facts(db_session, task)
    # The real worker does this rollback before marking the task for retry.
    db_session.rollback()

    assert db_session.query(Source).filter_by(
        study_id=study.study_id,
        source_name="europe_pmc_fulltext",
    ).count() == 0
    assert db_session.query(RawFact).filter_by(
        study_id=study.study_id,
        extraction_method="llm_text_extraction",
    ).count() == 0


def test_handler_fails_atomically_after_exhausted_json_repairs(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    handlers._llm_backend_cache = MockLLMBackend(
        label="invalid-json-model",
        responses=["this is not json"],
    )
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter()},
    )

    with pytest.raises(LLMBackendError, match="invalid JSON after retries"):
        handlers.handle_extract_text_facts(db_session, task)
    db_session.rollback()

    assert db_session.query(Source).filter_by(
        study_id=study.study_id,
        source_name="europe_pmc_fulltext",
    ).count() == 0
    assert db_session.query(RawFact).filter_by(
        study_id=study.study_id,
        extraction_method="llm_text_extraction",
    ).count() == 0


def test_worker_marks_incomplete_paper_pass_for_retry_not_completed(db_session, monkeypatch):
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)
    handlers._llm_backend_cache = MockLLMBackend(
        label="invalid-json-model",
        responses=["this is not json"],
    )
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: {"europe_pmc": FakeEuropePmcAdapter()},
    )

    summary = run_worker(
        db_session,
        worker_id="paper-pass-test",
        max_tasks=1,
    )

    db_session.refresh(task)
    assert summary == {"processed": 1, "completed": 0, "failed": 1}
    assert task.status == TaskStatus.RETRY_PENDING.value
    assert "invalid JSON after retries" in task.last_error
    assert db_session.query(Source).filter_by(
        study_id=study.study_id,
        source_name="europe_pmc_fulltext",
    ).count() == 0
    assert db_session.query(RawFact).filter_by(
        study_id=study.study_id,
        extraction_method="llm_text_extraction",
    ).count() == 0


def test_handler_raises_not_implemented_without_pmcid(db_session):
    study = Study(title="No PMCID")
    db_session.add(study)
    db_session.flush()
    task = _task_for(db_session, study)

    with pytest.raises(NotImplementedError):
        handlers.handle_extract_text_facts(db_session, task)


def test_handler_processes_local_pdf_even_with_no_pmcid_at_all(db_session, monkeypatch, tmp_path):
    """Real gap, confirmed live: a closed-access paper never deposited in
    PMC (no PMCID at all -- e.g. 10.1111/jeu.12975, 10.1002/edn3.570) has
    no route to Europe PMC full text, but a locally-supplied PDF is a
    genuine, independent alternative source. The PMCID requirement below
    used to run unconditionally before ever checking for a local PDF
    override, so this exact case raised NotImplementedError regardless of
    whether a PDF was supplied."""
    study = Study(title="No PMCID, local PDF only")
    db_session.add(study)
    db_session.flush()
    task = _task_for(db_session, study)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    monkeypatch.setenv(handlers.LOCAL_PDF_PATH_ENV, str(pdf_path))
    monkeypatch.setattr(
        handlers,
        "_build_enabled_adapters",
        lambda: pytest.fail("Europe PMC should not be called when local PDF override is set"),
    )
    monkeypatch.setattr(
        handlers,
        "extract_pdf_sections",
        lambda path: [{"title": "Sampling", "text": "Water samples were collected on 4 January 2022."}],
    )
    monkeypatch.setattr(handlers, "extract_pdf_text", lambda path: "Abstract\n\nMethods")
    monkeypatch.setattr(handlers, "detect_and_correct_elev_depth_mislabeling", lambda *args, **kwargs: None)
    monkeypatch.setattr(handlers, "detect_text_search_flags", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_phix_percentage_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_controlled_search_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_section_categories_present", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_study_factor", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_study_target_taxonomic_scope", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_rights_holder", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "generate_funding_source", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "extract_section_category_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(handlers, "detect_llm_judged_search_facts", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        handlers,
        "extract_facts_from_section",
        lambda *args, **kwargs: (
            [
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate="collection_date",
                    raw_field_name="collection_date",
                    raw_value="2022-01-04",
                    source_locator="pdf:Sampling",
                    support_type=SupportType.EXPLICIT,
                    evidence_quote="Water samples were collected on 4 January 2022.",
                )
            ],
            None,
        ),
    )
    handlers._llm_backend_cache = MockLLMBackend(label="mock-model", responses=[])

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    source = db_session.query(Source).filter_by(study_id=study.study_id, source_name="local_pdf_fulltext").one()
    assert source.external_identifier == str(pdf_path.resolve())
    facts = db_session.query(RawFact).filter_by(study_id=study.study_id, extraction_method="llm_text_extraction").all()
    assert len(facts) == 1


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


def test_handler_keeps_text_probeable_structured_fields_in_paper_prompt(db_session, monkeypatch):
    """Structured/API values are evidence, not truth. If the same FAIRe
    field is also in the paper-text checklist, structured resolution must
    not remove it from the prompt; a paper-vs-API disagreement should
    become a review_required conflict after mapping."""
    study = _seeded_study_with_pmcid(db_session)
    sample = Entity(
        study_id=study.study_id,
        entity_level=EntityLevel.SAMPLE.value,
        external_identifier="SAMN1",
    )
    db_session.add(sample)
    db_session.flush()
    db_session.add(
        RawFact(
            study_id=study.study_id,
            entity_id=sample.entity_id,
            raw_field_name="depth",
            raw_value="5",
            fact_type_candidate="depth",
            entity_level=EntityLevel.SAMPLE.value,
            support_type="structured_source",
            extraction_method="adapter:ncbi_biosample",
        )
    )
    db_session.flush()
    task = _task_for(db_session, study)

    backend = MockLLMBackend(responses=["[]"])
    handlers._llm_backend_cache = backend
    monkeypatch.setattr(handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter()})

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    assert backend.calls, "expected at least one LLM call"
    # At least the checklist-style call must retain depth -- not every
    # call, since handle_extract_text_facts now also makes several other
    # LLM calls with entirely different prompt shapes (study_factor,
    # study_target_taxonomic_scope, the section-category pipeline's own
    # calls, ...).
    assert any("- depth:" in call["prompt"] for call in backend.calls)


def test_handler_asks_about_everything_when_nothing_resolved_yet(db_session, monkeypatch):
    """No StandardizedValue rows at all (MAP_FAIRE hasn't run) must leave
    the full checklist intact -- nothing gets excluded via structured-first
    when nothing has been resolved yet. The handler now passes
    focuses=EXTRACTION_FOCUSES (see extraction/text.py's v17 docstring),
    so "depth" and "dna_extraction_kit" land in separate topic-focused
    calls rather than one combined prompt -- across all calls is still the
    right check, same as the per-topic-focus split always required before
    v8 collapsed it into one pass. The fixture text must actually mention
    both topics: segments_for_focus's own keyword-cue skip means a focus
    whose topic never comes up in the text legitimately never fires,
    independent of (and not to be confused with) structured-first
    exclusion, which is what this test exists to check."""
    study = _seeded_study_with_pmcid(db_session)
    task = _task_for(db_session, study)

    xml = """<article><body>
<sec><title>Materials and Methods</title>
<sec><title>Sampling</title><p>Water samples were collected on 4 January 2022 at a depth of 5 meters.</p></sec>
<sec><title>DNA extraction</title><p>DNA was extracted using the DNeasy PowerSoil Kit.</p></sec>
</sec>
</body></article>"""
    backend = MockLLMBackend(responses=["[]"])
    handlers._llm_backend_cache = backend
    monkeypatch.setattr(
        handlers, "_build_enabled_adapters", lambda: {"europe_pmc": FakeEuropePmcAdapter(fulltext_xml=xml)}
    )

    handlers.handle_extract_text_facts(db_session, task)
    db_session.commit()

    assert backend.calls
    assert any("depth" in call["prompt"] for call in backend.calls)
    assert any("dna_extraction_kit" in call["prompt"] for call in backend.calls)
