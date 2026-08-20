"""Tests for extraction/study_factor.py -- the one deliberately
generative (not extractive) field in this pipeline, per an explicit user
instruction: "I want the LLM to read the abstract, then generate a
sentence about what this study is testing."
"""
import json

import pytest

from fair_ocean_agent.extraction.study_factor import (
    generate_study_factor,
    generate_study_target_taxonomic_scope,
)
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.mock import MockLLMBackend

_ABSTRACT_XML = """<article><front><article-meta><abstract>
<p>We compared microbial community composition across three habitat types (reef, sand, seagrass)
at two time points to test whether habitat type or season better predicts community structure.</p>
</abstract></article-meta></front></article>"""


def test_generate_study_factor_returns_generated_sentence_from_abstract():
    summary = "Habitat type and season as predictors of microbial community structure."
    response = json.dumps({"study_factor": summary})
    backend = MockLLMBackend(responses=[response])

    facts = generate_study_factor(backend, _ABSTRACT_XML, locator_prefix="test")

    assert len(facts) == 1
    fact = facts[0]
    assert fact.fact_type_candidate == "study_factor"
    assert fact.raw_value == summary
    assert fact.support_type.value == "inferred"
    assert "habitat types" in fact.evidence_quote


def test_generate_study_factor_accepts_pdf_plain_text_abstract():
    summary = "Habitat type and season as predictors of microbial community structure."
    backend = MockLLMBackend(responses=[json.dumps({"study_factor": summary})])
    pdf_text = """Title

Abstract
We compared microbial community composition across three habitat types (reef, sand, seagrass)
at two time points to test whether habitat type or season better predicts community structure.

Introduction
This part should not be included in the abstract evidence.
"""

    facts = generate_study_factor(backend, pdf_text, locator_prefix="pdf")

    assert len(facts) == 1
    assert facts[0].raw_value == summary
    assert "habitat types" in facts[0].evidence_quote
    assert "This part should not be included" not in facts[0].evidence_quote


def test_generate_study_target_taxonomic_scope_returns_pipe_values_from_abstract():
    response = json.dumps(
        {"study_target_taxonomic_scope": "prokaryotic microorganisms | bacteria | archaea"}
    )
    backend = MockLLMBackend(responses=[response])

    facts = generate_study_target_taxonomic_scope(backend, _ABSTRACT_XML, locator_prefix="test")

    assert len(facts) == 1
    fact = facts[0]
    assert fact.fact_type_candidate == "study_target_taxonomic_scope"
    assert fact.raw_value == "prokaryotic microorganisms | bacteria | archaea"
    assert fact.support_type.value == "inferred"
    assert "microbial community composition" in fact.evidence_quote
    assert "Extract the organisms or broad biological/taxonomic group" in backend.calls[0]["prompt"]


def test_generate_study_target_taxonomic_scope_accepts_pdf_plain_text_abstract():
    backend = MockLLMBackend(
        responses=[json.dumps({"study_target_taxonomic_scope": "corals | microbial communities"})]
    )
    pdf_text = """Abstract
Settlement cues in reef-building corals were compared across oceans, with microbial communities
characterized from cue samples.

Materials and Methods
DNA extraction details follow.
"""

    facts = generate_study_target_taxonomic_scope(backend, pdf_text, locator_prefix="pdf")

    assert len(facts) == 1
    assert facts[0].raw_value == "corals | microbial communities"
    assert "Settlement cues" in facts[0].evidence_quote
    assert "DNA extraction" not in facts[0].evidence_quote


def test_generate_study_factor_no_abstract_makes_no_llm_call():
    backend = MockLLMBackend(responses=["{}"])
    facts = generate_study_factor(backend, "<article><body/></article>", locator_prefix="test")
    assert facts == []
    assert backend.calls == []


def test_generate_study_target_taxonomic_scope_no_abstract_makes_no_llm_call():
    backend = MockLLMBackend(responses=["{}"])
    facts = generate_study_target_taxonomic_scope(backend, "<article><body/></article>", locator_prefix="test")
    assert facts == []
    assert backend.calls == []


def test_generate_study_factor_no_fulltext_at_all():
    backend = MockLLMBackend(responses=["{}"])
    facts = generate_study_factor(backend, None, locator_prefix="test")
    assert facts == []
    assert backend.calls == []


def test_generate_study_factor_empty_sentence_returns_no_facts():
    response = json.dumps({"study_factor": ""})
    backend = MockLLMBackend(responses=[response])
    facts = generate_study_factor(backend, _ABSTRACT_XML, locator_prefix="test")
    assert facts == []


def test_generate_study_target_taxonomic_scope_empty_value_returns_no_facts():
    response = json.dumps({"study_target_taxonomic_scope": ""})
    backend = MockLLMBackend(responses=[response])
    facts = generate_study_target_taxonomic_scope(backend, _ABSTRACT_XML, locator_prefix="test")
    assert facts == []


def test_generate_study_factor_raises_on_invalid_json_after_retries():
    backend = MockLLMBackend(responses=["not json"])
    with pytest.raises(LLMBackendError):
        generate_study_factor(backend, _ABSTRACT_XML, locator_prefix="test")


def test_generate_study_target_taxonomic_scope_raises_on_invalid_json_after_retries():
    backend = MockLLMBackend(responses=["not json"])
    with pytest.raises(LLMBackendError):
        generate_study_target_taxonomic_scope(backend, _ABSTRACT_XML, locator_prefix="test")


def test_generate_study_factor_malformed_xml_returns_no_facts():
    backend = MockLLMBackend(responses=["{}"])
    facts = generate_study_factor(backend, "<not valid xml", locator_prefix="test")
    assert facts == []
    assert backend.calls == []


def test_generate_study_target_taxonomic_scope_malformed_xml_returns_no_facts():
    backend = MockLLMBackend(responses=["{}"])
    facts = generate_study_target_taxonomic_scope(backend, "<not valid xml", locator_prefix="test")
    assert facts == []
    assert backend.calls == []

