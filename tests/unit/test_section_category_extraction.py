"""Tests for extraction/section_category_extraction.py (Stage 2 LLM
sentence-categorization and Stage 3 category-scoped term extraction)."""
import json

import pytest

from fair_ocean_agent.extraction.section_categories import SECTION_CATEGORIES
from fair_ocean_agent.extraction.section_category_extraction import (
    categorize_paragraphs,
    extract_category_terms,
    extract_section_category_facts,
)
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.mock import MockLLMBackend

_PCR1_CATEGORY = next(c for c in SECTION_CATEGORIES if c.name == "pcr1_primary_amplification")


def test_categorize_paragraphs_empty_input_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])
    assert categorize_paragraphs(backend, []) == {}
    assert backend.calls == []


def test_categorize_paragraphs_tags_sentences_from_llm_response():
    response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["pcr1_primary_amplification"]},
            {"sentence_id": "S0.1", "categories": ["library_prep_sequencing", "assay_definition"]},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    paragraph = "PCR was performed. Libraries were sequenced on Illumina."
    result = categorize_paragraphs(
        backend, [(paragraph, frozenset({"pcr1_primary_amplification", "library_prep_sequencing", "assay_definition"}))]
    )
    assert result[0][0] == ("PCR was performed.", frozenset({"pcr1_primary_amplification"}))
    assert result[0][1] == (
        "Libraries were sequenced on Illumina.",
        frozenset({"library_prep_sequencing", "assay_definition"}),
    )


def test_categorize_paragraphs_omitted_sentence_gets_empty_category_set():
    response = json.dumps([{"sentence_id": "S0.0", "categories": ["pcr1_primary_amplification"]}])
    backend = MockLLMBackend(responses=[response])
    paragraph = "PCR was performed. An unrelated aside sentence."
    result = categorize_paragraphs(backend, [(paragraph, frozenset({"pcr1_primary_amplification"}))])
    assert result[0][1] == ("An unrelated aside sentence.", frozenset())


def test_categorize_paragraphs_ignores_unknown_category_names():
    response = json.dumps([{"sentence_id": "S0.0", "categories": ["pcr1_primary_amplification", "not_a_real_category"]}])
    backend = MockLLMBackend(responses=[response])
    result = categorize_paragraphs(backend, [("PCR was performed.", frozenset({"pcr1_primary_amplification"}))])
    assert result[0][0][1] == frozenset({"pcr1_primary_amplification"})


def test_categorize_paragraphs_raises_on_invalid_json_after_retries():
    backend = MockLLMBackend(responses=["not json"])
    with pytest.raises(LLMBackendError):
        categorize_paragraphs(backend, [("PCR was performed.", frozenset({"pcr1_primary_amplification"}))])


def test_extract_category_terms_no_candidates_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, "Nothing relevant here at all.", locator_prefix="test")
    assert facts == []
    assert backend.calls == []


def test_extract_category_terms_extracts_verbatim_values_and_pipe_joins_conflicts():
    run_text = (
        "PCR amplification was performed in a total reaction volume of 25 uL. "
        "The annealing temperature was 55C for 30 seconds. "
        "PCR cycles: 35 cycles were performed."
    )
    response = json.dumps(
        [
            {"field": "amplificationReactionVolume", "raw_value": "25 uL", "quote_id": "Q001"},
            {"field": "annealingTemp", "raw_value": "55C", "quote_id": "Q002"},
            {"field": "pcr_cycles", "raw_value": "35 cycles", "quote_id": "Q003"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    by_field = {f.fact_type_candidate: f for f in facts}
    assert by_field["amplificationReactionVolume"].raw_value == "25 uL"
    assert by_field["annealingTemp"].raw_value == "55C"
    assert by_field["pcr_cycles"].raw_value == "35 cycles"
    assert all(f.support_type.value == "explicit" for f in facts)


def test_extract_category_terms_rejects_hallucinated_field_names():
    run_text = "PCR amplification was performed in a total reaction volume of 25 uL."
    response = json.dumps([{"field": "not_a_real_pcr1_field", "raw_value": "25 uL", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_rejects_values_not_present_in_the_cited_quote():
    """The 'take word for word from text and not generate' instruction is
    enforced programmatically, not just via the prompt -- a value that
    doesn't literally appear in its own cited quote is discarded even if
    the model returns it."""
    run_text = "PCR amplification was performed in a total reaction volume of 25 uL."
    response = json.dumps(
        [{"field": "amplificationReactionVolume", "raw_value": "50 uL", "quote_id": "Q001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_dedups_identical_values_across_quotes():
    run_text = "PCR cycles: 35 cycles were performed. Later, 35 cycles were confirmed again."
    response = json.dumps(
        [
            {"field": "pcr_cycles", "raw_value": "35 cycles", "quote_id": "Q001"},
            {"field": "pcr_cycles", "raw_value": "35 cycles", "quote_id": "Q002"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    assert len(facts) == 1
    assert facts[0].raw_value == "35 cycles"


def test_extract_category_terms_raises_on_invalid_json_after_retries():
    run_text = "PCR amplification was performed in a total reaction volume of 25 uL."
    backend = MockLLMBackend(responses=["not json"])
    with pytest.raises(LLMBackendError):
        extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")


def test_extract_category_terms_surfaces_rrna_f_r_primer_name_quote():
    run_text = (
        "The V3-V4 region of the 16S rRNA gene was amplified using universal primers "
        "16S rRNA F and 16S rRNA R."
    )
    response = json.dumps(
        [
            {"field": "pcr_primer_name_forward", "raw_value": "16S rRNA F", "quote_id": "Q001"},
            {"field": "pcr_primer_name_reverse", "raw_value": "16S rRNA R", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])

    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")

    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_field["pcr_primer_name_forward"] == "16S rRNA F"
    assert by_field["pcr_primer_name_reverse"] == "16S rRNA R"


def test_extract_section_category_facts_full_pipeline_on_real_paper_text():
    """Grounded in the real PNAS 10.1073/pnas.2005917117 supplementary
    methods text this module's Stage 1 gate was validated against
    (real dense, multi-category paragraph shape), exercising Stage 1
    (paragraph gate) -> Stage 2 (LLM categorization, mocked) -> Stage 2.5
    (run-grouping) -> Stage 3 (LLM term extraction, mocked) end to end.
    Uses phrasing that genuinely matches a real CategoryTerm's own
    distinctive search cues (not just the looser category-level
    keywords), since Stage 3's term-level gate is deliberately as strict
    as the category-level one is loose."""
    text = (
        "PCR amplification was performed. The forward primer name was 515F and the reverse "
        "primer name was 806R. The sequencing instrument was an Ion Torrent Personal Genome "
        "Machine."
    )

    categorization_response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["pcr1_primary_amplification"]},
            {"sentence_id": "S0.1", "categories": ["pcr1_primary_amplification"]},
            {"sentence_id": "S0.2", "categories": ["library_prep_sequencing"]},
        ]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        if 'extracting FAIRe "PCR1 / primary amplification"' in prompt:
            return json.dumps(
                [
                    {"field": "pcr_primer_name_forward", "raw_value": "515F", "quote_id": "Q001"},
                    {"field": "pcr_primer_name_reverse", "raw_value": "806R", "quote_id": "Q001"},
                ]
            )
        if 'extracting FAIRe "Library preparation' in prompt:
            return json.dumps(
                [{"field": "instrument", "raw_value": "Ion Torrent Personal Genome Machine", "quote_id": "Q001"}]
            )
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Supplementary Methods", text)], locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["pcr_primer_name_forward"] == "515F"
    assert by_field["pcr_primer_name_reverse"] == "806R"
    assert by_field["instrument"] == "Ion Torrent Personal Genome Machine"


def test_extract_section_category_facts_sample_prep_category_on_real_paper_text():
    """Grounded in the real fmicb paper (10.3389/fmicb.2024.1295149) text
    that motivated this whole category and its samp_vol_we_dna_ext
    sample-type-routing fix: a real paragraph mixing filtration, lysis
    buffer, a named extraction kit, and both a water-sample aside and the
    real sediment-specific extraction amount."""
    text = (
        "Post filtration, the Sterivex filter cartridges were cracked open and the filter paper was "
        "chipped into small pieces and resuspended in the lysis buffer of the E.Z.N.A. Soil DNA kit. "
        "DNA extraction was performed as per the manufacturer's instructions. For sediment samples, "
        "500 mg of dried sediment samples were used for DNA extraction."
    )
    categorization_response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["sample_prep"]},
            {"sentence_id": "S0.1", "categories": ["sample_prep"]},
            {"sentence_id": "S0.2", "categories": ["sample_prep"]},
        ]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        if 'extracting FAIRe "Sample preparation' in prompt:
            return json.dumps(
                [
                    {"field": "nucl_acid_ext_kit", "raw_value": "Soil DNA kit", "quote_id": "Q002"},
                    {"field": "samp_vol_we_dna_ext", "raw_value": "500 mg", "quote_id": "Q004"},
                ]
            )
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Methods", text)], locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["nucl_acid_ext_kit"] == "Soil DNA kit"
    assert by_field["samp_vol_we_dna_ext"] == "500 mg"


def test_extract_section_category_facts_no_gated_paragraphs_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])
    facts = extract_section_category_facts(
        backend, [("Intro", "The sky was blue and the ocean was calm that day.")], locator_prefix="test"
    )
    assert facts == []
    assert backend.calls == []


def test_extract_section_category_facts_captures_bioinfo_method_additional_verbatim():
    """Per an explicit user request, once the bioinformatics-pipeline
    categories (raw read preprocessing, OTU/ASV generation, taxonomic
    assignment -- deliberately excluding wet-lab categories like PCR1/
    library prep) are classified, their pipeline sentences are captured
    verbatim as bioinfo_method_additional, with downstream stats excluded."""
    text = (
        "Reads were quality filtered and trimmed using Cutadapt. "
        "Chimeric sequences were removed using UCHIME. "
        "Taxonomy was assigned using the SILVA reference database."
    )
    categorization_response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["raw_read_preprocessing"]},
            {"sentence_id": "S0.1", "categories": ["otu_asv_generation_filtering"]},
            {"sentence_id": "S0.2", "categories": ["taxonomic_assignment"]},
        ]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Methods", text)], locator_prefix="test")
    by_field = {f.fact_type_candidate: f for f in facts}
    fact = by_field["bioinfo_method_additional"]
    assert "Reads were quality filtered and trimmed using Cutadapt." in fact.raw_value
    assert "Chimeric sequences were removed using UCHIME." in fact.raw_value
    assert "Taxonomy was assigned using the SILVA reference database." in fact.raw_value
    assert fact.support_type.value == "deterministically_derived"
    assert fact.confidence_metadata["categories"] == [
        "raw_read_preprocessing",
        "otu_asv_generation_filtering",
        "taxonomic_assignment",
    ]


def test_extract_section_category_facts_bioinfo_method_additional_excludes_downstream_stats():
    text = (
        "Clean high-quality reads were further analyzed in QIIME2 2022.2 pipeline using DADA2 Package "
        "to obtain Amplicon Sequence Variants (ASVs). "
        "Clustering of the obtained ASVs into their respective groups was done using SILVA 138 SSU reference database. "
        "Correlation plots were constructed in PAST 4.07b to assess the correlation between microbial groups and "
        "the physicochemical parameters. "
        "Similarity percentage analysis (SIMPER—R vegan function) was performed to assess the contribution of "
        "the taxonomic group contributing maximum towards the differences in the abundances between the samples."
    )
    categorization_response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["otu_asv_generation_filtering"]},
            {"sentence_id": "S0.1", "categories": ["taxonomic_assignment"]},
            {"sentence_id": "S0.2", "categories": ["taxonomic_assignment"]},
            {"sentence_id": "S0.3", "categories": ["taxonomic_assignment"]},
        ]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        return "[]"

    facts = extract_section_category_facts(
        MockLLMBackend(responses=fake_backend),
        [("Methods", text)],
        locator_prefix="test",
    )

    by_field = {f.fact_type_candidate: f for f in facts}
    value = by_field["bioinfo_method_additional"].raw_value
    assert "QIIME2 2022.2 pipeline" in value
    assert "SILVA 138 SSU reference database" in value
    assert "Correlation plots" not in value
    assert "SIMPER" not in value


def test_extract_section_category_facts_bioinfo_method_additional_excludes_wet_lab_categories():
    text = "PCR amplification was performed in a total reaction volume of 25 uL."
    categorization_response = json.dumps(
        [{"sentence_id": "S0.0", "categories": ["pcr1_primary_amplification"]}]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Methods", text)], locator_prefix="test")
    assert "bioinfo_method_additional" not in {f.fact_type_candidate for f in facts}
