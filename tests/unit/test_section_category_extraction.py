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
_SAMPLE_PREP_CATEGORY = next(c for c in SECTION_CATEGORIES if c.name == "sample_prep")
_OTU_ASV_CATEGORY = next(c for c in SECTION_CATEGORIES if c.name == "otu_asv_generation_filtering")


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
        "PCR used a commercial master mix (Qiagen HotStarTaq). "
        "The annealing temperature was 55C for 30 seconds. "
        "PCR cycles: 35 cycles were performed."
    )
    response = json.dumps(
        [
            {"field": "commercial_mm", "raw_value": "Qiagen HotStarTaq", "quote_id": "Q001"},
            {"field": "annealingTemp", "raw_value": "55C", "quote_id": "Q002"},
            {"field": "pcr_cycles", "raw_value": "35 cycles", "quote_id": "Q003"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    by_field = {f.fact_type_candidate: f for f in facts}
    assert by_field["commercial_mm"].raw_value == "Qiagen HotStarTaq"
    assert by_field["annealingTemp"].raw_value == "55C"
    assert by_field["pcr_cycles"].raw_value == "35 cycles"
    assert all(f.support_type.value == "explicit" for f in facts)


def test_extract_category_terms_rejects_hallucinated_field_names():
    run_text = "PCR amplification was performed in a total reaction volume of 25 uL."
    response = json.dumps([{"field": "not_a_real_pcr1_field", "raw_value": "25 uL", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_rejects_method_description_as_a_concentration_value():
    """Regression guard for a real bug found live (10.1002/ece3.6071):
    "concentration" is meant to hold a short numeric measurement, but the
    model returned a whole sentence describing the kit/instrument used to
    measure it (which literally IS verbatim-present in the quote, so the
    verbatim guard alone wouldn't catch this) instead of leaving the field
    empty when no actual number was ever stated."""
    run_text = (
        "DNA concentration was measured with the Quant-iT dsDNA HS assay kit (Thermo Scientific) "
        "using a Qubit 2.0 Fluorometer (Life Technologies)."
    )
    response = json.dumps(
        [
            {
                "field": "concentration",
                "raw_value": (
                    "DNA concentration was measured with the Quant-iT dsDNA HS assay kit "
                    "(Thermo Scientific) using a Qubit 2.0 Fluorometer (Life Technologies)"
                ),
                "quote_id": "Q001",
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_boolean_field_accepts_1_with_no_digit_in_its_own_quote():
    """Real bug found live (10.1371/journal.pone.0303937): the generic
    verbatim-substring guard is meaningless (and actively harmful) for a
    "_0_1" judged classification field -- "1"/"0" are never literally
    "copied" from the quote, they're inferred. The one quote that
    unambiguously supports active filtration ("...connected to compressed
    air and an overpressure of 2 bar was applied.") has no literal "1"
    character anywhere in it, so the old blanket verbatim check silently
    discarded the correct answer."""
    run_text = (
        "A pressure barrel containing the water sample was connected to compressed air and an "
        "overpressure of two bar was applied."
    )
    response = json.dumps(
        [{"field": "filter_passive_active_0_1", "raw_value": "1", "quote_id": "Q001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert len(facts) == 1
    assert facts[0].raw_value == "1"


def test_extract_category_terms_boolean_field_rejects_non_boolean_values():
    run_text = "A pressure barrel containing the water sample was connected to compressed air."
    response = json.dumps(
        [{"field": "filter_passive_active_0_1", "raw_value": "active", "quote_id": "Q001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_boolean_field_prefers_1_over_conflicting_0_and_narrows_evidence():
    """Real bug found live: three candidate quotes for the same boolean
    field disagreed ("0", "1", "0") -- the middle one ("...connected to
    compressed air and an overpressure of 2 bar was applied.") is the
    only one that actually, unambiguously answers the question; the other
    two just mention an unrelated flowmeter/device with nothing to do
    with active vs. passive filtration. "1" must win, and the final
    evidence_quote must cite only the quote that actually supports it."""
    run_text = (
        "Filtration used a self-constructed monitoring device (flowmeter). A pressure barrel "
        "containing the water sample was connected to compressed air and an overpressure of two "
        "bar was applied. A mini-flowmeter (model FCH-0) was connected at the outlet."
    )
    response = json.dumps(
        [
            {"field": "filter_passive_active_0_1", "raw_value": "0", "quote_id": "Q001"},
            {"field": "filter_passive_active_0_1", "raw_value": "1", "quote_id": "Q002"},
            {"field": "filter_passive_active_0_1", "raw_value": "0", "quote_id": "Q003"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert len(facts) == 1
    assert facts[0].raw_value == "1"
    assert "compressed air" in facts[0].evidence_quote
    assert "flowmeter" not in facts[0].evidence_quote.lower()


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


# --- primer reference/traceability (explicit user request) ------------------


def test_extract_category_terms_primer_reference_extracted_when_no_sequence_found():
    """When only the primer's name is given (no sequence), a trailing bare
    citation is a valid fallback source -- per an explicit user request:
    'if a primer sequence isn't listed we just take the name... I want to
    be able to also take the reference for primers'."""
    run_text = "We amplified the V4 region using the universal primers 515F/806R (Caporaso et al., 2011)."
    response = json.dumps(
        [
            {"field": "pcr_primer_name_forward", "raw_value": "515F", "quote_id": "Q001"},
            {"field": "pcr_primer_name_reverse", "raw_value": "806R", "quote_id": "Q001"},
            {"field": "pcr_primer_reference_forward", "raw_value": "(Caporaso et al., 2011)", "quote_id": "Q001"},
            {"field": "pcr_primer_reference_reverse", "raw_value": "(Caporaso et al., 2011)", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    by_type = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_type["pcr_primer_reference_forward"] == "(Caporaso et al., 2011)"
    assert by_type["pcr_primer_reference_reverse"] == "(Caporaso et al., 2011)"
    assert "primer_forward_source_unresolved" not in by_type
    assert "primer_reverse_source_unresolved" not in by_type


def test_extract_category_terms_primer_reference_dropped_when_sequence_also_present():
    """Per an explicit user request: reference extraction is only a
    fallback for when the sequence isn't reported -- if the sequence IS
    extracted, the reference must not also be populated alongside it."""
    run_text = (
        "The forward primer sequence was 5'-GTGYCAGCMGCCGCGGTAA-3'. "
        "We used the universal primers 515F/806R (Caporaso et al., 2011)."
    )
    response = json.dumps(
        [
            {"field": "pcr_primer_forward", "raw_value": "5'-GTGYCAGCMGCCGCGGTAA-3'", "quote_id": "Q001"},
            {"field": "pcr_primer_reference_forward", "raw_value": "(Caporaso et al., 2011)", "quote_id": "Q002"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    by_type = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_type["pcr_primer_forward"] == "5'-GTGYCAGCMGCCGCGGTAA-3'"
    assert "pcr_primer_reference_forward" not in by_type
    assert "primer_forward_source_unresolved" not in by_type


def test_extract_category_terms_flags_primer_when_neither_sequence_nor_reference_found():
    """Per an explicit user request: 'if no reference is given and no
    sequence given, flag it' -- a candidate for a future targeted
    supplement crawl, surfaced as an internal-only diagnostic fact."""
    run_text = "We amplified the V4 region using the universal primers 515F/806R for all samples."
    response = json.dumps(
        [
            {"field": "pcr_primer_name_forward", "raw_value": "515F", "quote_id": "Q001"},
            {"field": "pcr_primer_name_reverse", "raw_value": "806R", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    by_type = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_type["primer_forward_source_unresolved"] == "1"
    assert by_type["primer_reverse_source_unresolved"] == "1"
    assert "pcr_primer_reference_forward" not in by_type


def test_extract_category_terms_rejects_primer_reference_without_a_real_citation_shape():
    """A hallucinated 'reference' with no citation/DOI shape in its own
    quote is discarded rather than trusted -- the broadened cues alone
    would otherwise let a plain primer-name sentence become a reference
    candidate with nothing real to cite."""
    run_text = "We amplified the V4 region using the universal primers 515F/806R for all samples."
    response = json.dumps(
        [
            {"field": "pcr_primer_name_forward", "raw_value": "515F", "quote_id": "Q001"},
            {"field": "pcr_primer_reference_forward", "raw_value": "515F/806R", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")
    by_type = {f.fact_type_candidate: f.raw_value for f in facts}
    assert "pcr_primer_reference_forward" not in by_type
    assert by_type["primer_forward_source_unresolved"] == "1"


def test_extract_category_terms_rejects_value_for_field_the_quote_was_never_offered_for():
    """Regression guard for a real bug found live (10.1371/journal.pone.0303937):
    the verbatim guard only checks that a value's TEXT appears in its cited
    quote, not that the returned field name is one of the field(s) that
    quote was actually candidate-tagged for. A quote tagged only
    [screen_other] ("...spurious sequences...the lowest number of reads in
    a sample was 18,696...") also contains a read count, and the model
    attached it to min_reads_cutoff instead of the screen_other field the
    quote was offered for -- accepted before this fix since "18,696" is
    genuinely verbatim in that same quote's text."""
    run_text = (
        "After removal of the low quality, chimera and spurious sequences (false positive sequences) "
        "during the bioinformatic treatment and the filtration of the lowest abundant reads, the lowest "
        "number of reads in a sample was 18,696 for the first experiment."
    )
    response = json.dumps([{"field": "min_reads_cutoff", "raw_value": "18,696", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _OTU_ASV_CATEGORY, run_text, locator_prefix="test")
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
    run_text = "PCR used a commercial master mix (Qiagen HotStarTaq)."
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


def test_extract_category_terms_surfaces_bare_primer_pair_naming():
    """Regression guard for a real gap found live (10.1038/s42003-024-06136-2):
    a paper simply naming its primer pair directly ("the universal primers
    of X/Y") -- the overwhelmingly common real-world phrasing -- matched
    none of pcr_primer_name_forward/reverse's original cues (all meta-
    descriptive: "primer designated", "primer ID", "primer abbreviation"),
    so this sentence never even became a candidate quote and the field
    silently went from populated in an earlier run to empty. Captures the
    real prompt to confirm the sentence is actually OFFERED as a candidate
    for both fields, not just that a canned mock response round-trips."""
    run_text = (
        "Amplicon of the 16 S rRNA gene was prepared using the two-round PCR amplification "
        "strategy with the universal primers of Uni519F/806r, as described in Zhao et al."
    )
    captured_prompt = {}

    def fake_backend(prompt: str) -> str:
        captured_prompt["value"] = prompt
        return json.dumps(
            [
                {"field": "pcr_primer_name_forward", "raw_value": "Uni519F", "quote_id": "Q001"},
                {"field": "pcr_primer_name_reverse", "raw_value": "806r", "quote_id": "Q001"},
            ]
        )

    backend = MockLLMBackend(responses=fake_backend)

    facts = extract_category_terms(backend, _PCR1_CATEGORY, run_text, locator_prefix="test")

    prompt = captured_prompt["value"]
    quote_line = next(line for line in prompt.splitlines() if "Uni519F/806r" in line)
    # The bracketed term list prefixing this specific candidate quote line
    # (e.g. "Q001 [pcr_primer_name_forward, pcr_primer_name_reverse]: ...")
    # is what actually proves the sentence was offered as a candidate for
    # these fields -- both names also appear, unconditionally, in the
    # prompt's own field-definition list further up, so checking the whole
    # prompt wouldn't catch a regression back to the old, narrower cues.
    assert "pcr_primer_name_forward" in quote_line
    assert "pcr_primer_name_reverse" in quote_line

    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_field["pcr_primer_name_forward"] == "Uni519F"
    assert by_field["pcr_primer_name_reverse"] == "806r"


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
    real sediment-specific extraction amount. Also regression-covers a real
    bug this same sentence exposed live: "E.Z.N.A." 's internal periods used
    to be misread as sentence boundaries, shredding this one real sentence
    into fragments and truncating both nucl_acid_ext_kit (lost its own
    "E.Z.N.A." brand prefix) and nucl_acid_ext_lysis (cut off before the kit
    name) -- now merged back into one whole candidate quote (Q001) by
    _split_into_sentences' dotted-abbreviation handling."""
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
                    {"field": "nucl_acid_ext_kit", "raw_value": "E.Z.N.A. Soil DNA kit", "quote_id": "Q001"},
                    {
                        "field": "nucl_acid_ext_lysis",
                        "raw_value": "resuspended in the lysis buffer of the E.Z.N.A. Soil DNA kit",
                        "quote_id": "Q001",
                    },
                    {"field": "samp_vol_we_dna_ext", "raw_value": "500 mg", "quote_id": "Q003"},
                ]
            )
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Methods", text)], locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["nucl_acid_ext_kit"] == "E.Z.N.A. Soil DNA kit"
    assert by_field["nucl_acid_ext_lysis"] == "resuspended in the lysis buffer of the E.Z.N.A. Soil DNA kit"
    assert by_field["samp_vol_we_dna_ext"] == "500 mg"


def test_extract_category_terms_samp_store_sol_from_real_paper_text():
    """Grounded in the real 10.1371/journal.pone.0303937 text: the storage
    solution ("lysis buffer") and the later storage event ("stored ...
    at -20C") are in separate sentences, connected only by context.
    Regression guard for a real gap: samp_store_sol's original cues
    ("stored in lysis buffer") never matched real phrasing ("immersed
    EACH with 3 ml of lysis buffer") at all. Per an explicit, repeated
    user request, this fix (and this field's own leftover-immersion
    coverage) lives on samp_store_sol only now -- the near-duplicate
    prepped_samp_store_sol was consolidated away entirely, never both."""
    run_text = (
        "Then, the biomass was immersed each with 3 ml of lysis buffer (50 mM Tris-HCl buffer pH 8.0, "
        "50 mM EDTA, 50 mM EGTA). The tubes were stored on dry ice for the rest of the ship cruise and "
        "later in the lab at -20 degrees C in a freezer."
    )
    response = json.dumps(
        [
            {"field": "samp_store_sol", "raw_value": "lysis buffer", "quote_id": "Q001"},
            {"field": "samp_store_temp", "raw_value": "-20 degrees C", "quote_id": "Q002"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_field["samp_store_sol"] == "lysis buffer"


def test_extract_category_terms_lysis_and_sep_cues_cover_a_real_manual_protocol():
    """Grounded in the real 10.1371/journal.pone.0303937 CTAB/phenol-
    chloroform manual extraction protocol: neither nucl_acid_ext_lysis
    (bead-beating/sonication language) nor nucl_acid_ext_sep (phenol-
    chloroform/precipitation language) originally had cues broad enough
    to catch this real, common manual-protocol phrasing at all -- per an
    explicit, user-supplied expanded cue list."""
    run_text = (
        "Then, 200 ul of zirconium beads were added to each tube and the tubes were treated "
        "in an ultrasonic water bath for 1 min. The DNA was purified by phenol-chloroform "
        "extraction. The precipitated DNA was collected by centrifugation."
    )
    response = json.dumps(
        [
            {
                "field": "nucl_acid_ext_lysis",
                "raw_value": "treated in an ultrasonic water bath",
                "quote_id": "Q001",
            },
            {"field": "nucl_acid_ext_sep", "raw_value": "purified by phenol-chloroform extraction", "quote_id": "Q002"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_field["nucl_acid_ext_lysis"] == "treated in an ultrasonic water bath"
    assert by_field["nucl_acid_ext_sep"] == "purified by phenol-chloroform extraction"


def test_extract_category_terms_derives_dna_extraction_amount_from_unit_only_response():
    run_text = "Briefly, DNA was extracted from ~7 g of sediment from each selected depth."
    response = json.dumps(
        [
            {"field": "samp_size_unit", "raw_value": "g", "quote_id": "Q001"},
            {"field": "samp_vol_we_dna_ext_unit", "raw_value": "g", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])

    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert "samp_size_unit" not in by_field
    assert "samp_size" not in by_field
    assert by_field["samp_vol_we_dna_ext_unit"] == "g"
    assert by_field["samp_vol_we_dna_ext"] == "~7 g"


def test_extract_category_terms_derives_collected_sample_size_from_unit_only_response():
    run_text = "At each station, 10 L of water were collected and immediately filtered."
    response = json.dumps([{"field": "samp_size_unit", "raw_value": "L", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])

    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}
    assert by_field["samp_size_unit"] == "L"
    assert by_field["samp_size"] == "10 L"


def test_split_into_sentences_keeps_dotted_abbreviation_intact():
    """Regression guard for the exact real bug above, isolated at the
    sentence-splitter level: a naive [.!?]-then-whitespace split treats
    every internal period in "E.Z.N.A." as its own sentence end, since none
    of those periods happen to be followed by whitespace except the last
    one -- which IS followed by whitespace before the next real word,
    making it indistinguishable from a genuine sentence boundary without
    this dotted-abbreviation check."""
    from fair_ocean_agent.extraction.section_category_extraction import _split_into_sentences

    text = "Resuspended in the lysis buffer of the E.Z.N.A. Soil DNA kit. Extraction followed the kit manual."
    assert _split_into_sentences(text) == [
        "Resuspended in the lysis buffer of the E.Z.N.A. Soil DNA kit.",
        "Extraction followed the kit manual.",
    ]


def test_split_into_sentences_keeps_et_al_citation_intact():
    """Real bug found live while building primer-reference extraction: 'et
    al.' is mistaken for a sentence boundary just like 'vol.', permanently
    separating a primer's name from the citation meant to identify it
    ("...primer 515F, described by Caporaso et al. (2011), and...")."""
    from fair_ocean_agent.extraction.section_category_extraction import _split_into_sentences

    text = (
        "The forward primer 515F was described by Caporaso et al. (2011). "
        "The reverse primer 806R was described by Apprill et al. (2015)."
    )
    assert _split_into_sentences(text) == [
        "The forward primer 515F was described by Caporaso et al. (2011).",
        "The reverse primer 806R was described by Apprill et al. (2015).",
    ]


def test_split_into_sentences_splits_semicolon_joined_enumerated_steps():
    """Real bug found live (10.1093/ismejo/wrae013): a methods sentence
    describing three DIFFERENT tools for three DIFFERENT purposes as one
    semicolon-joined enumerated list has only ONE terminal period (at the
    very end), so the old period-only splitter offered the WHOLE blob as
    a single candidate -- corrupting every field pulled from it (the
    model attached "SeqPrep" -- really the adapter-trimming tool named in
    clause (i) -- to error_rate_tool, whose real answer, "Trimmomatic",
    only appears in clause (iii))."""
    from fair_ocean_agent.extraction.section_category_extraction import _split_into_sentences

    text = (
        "Quality trimming was conducted by: (i) removing Illumina adapters using SeqPrep 1.2 with "
        "default settings targeting the adapter sequences [64]; (ii) remove any leftover PhiX "
        "control sequences by mapping the reads to the PhiX genome (NCBI Reference Sequence: "
        "NC_001422.1) using bowtie2 2.3.5.1 [65], and (iii) remove low quality and short reads "
        "using Trimmomatic 0.39 with settings: LEADING:20, TRAILING:20, and MINLEN:80."
    )
    assert _split_into_sentences(text) == [
        "Quality trimming was conducted by: (i) removing Illumina adapters using SeqPrep 1.2 with "
        "default settings targeting the adapter sequences [64];",
        "(ii) remove any leftover PhiX control sequences by mapping the reads to the PhiX genome "
        "(NCBI Reference Sequence: NC_001422.1) using bowtie2 2.3.5.1 [65];",
        "(iii) remove low quality and short reads using Trimmomatic 0.39 with settings: LEADING:20, "
        "TRAILING:20, and MINLEN:80.",
    ]


def test_split_into_sentences_never_splits_a_semicolon_inside_a_citation_list():
    from fair_ocean_agent.extraction.section_category_extraction import _split_into_sentences

    text = "This has been shown before (Smith 2020; Jones 2021). It matters for the analysis."
    assert _split_into_sentences(text) == [
        "This has been shown before (Smith 2020; Jones 2021).",
        "It matters for the analysis.",
    ]


def test_extract_section_category_facts_routes_leftover_sentences_by_scope():
    """Regression guard for a real bug found live (10.1371/journal.pone.0303937):
    prep_method_additional's own leftover-capture (built to fix an earlier
    gap) swept in nucleic-acid-extraction-specific procedural detail too,
    when the user wanted that scoped to its own dedicated field,
    nucl_acid_ext_method_additional, instead -- per an explicit user
    request. A leftover sentence about reagent volumes (extraction-shaped)
    must land in nucl_acid_ext_method_additional, not prep_method_
    additional; an unrelated leftover sentence about the filtration
    apparatus stays in prep_method_additional. (zirconium beads/
    ultrasonic-water-bath, this test's original example, now match
    nucl_acid_ext_lysis directly -- its own cues were broadened per a
    later, separate user-supplied cue list -- so they no longer exercise
    the leftover path at all; reagent volumes still do.)"""
    text = (
        "Filtration used a self-built pressure filtration rig on the ship. "
        "The reagent volumes were adjusted slightly from the standard protocol."
    )
    categorization_response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["sample_prep"]},
            {"sentence_id": "S0.1", "categories": ["sample_prep"]},
        ]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Methods", text)], locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}

    assert "self-built pressure filtration rig" in by_field["prep_method_additional"]
    assert "reagent volumes" not in by_field["prep_method_additional"]
    assert "reagent volumes" in by_field["nucl_acid_ext_method_additional"]
    assert "pressure filtration rig" not in by_field["nucl_acid_ext_method_additional"]


def test_extract_section_category_facts_sample_collection_terms():
    """Sample-collection terms deliberately fold into the same sample_prep
    category rather than a new classifier, per an explicit user
    instruction -- confirms Stage 1/2/2.5 catch this text and Stage 3
    extracts each field's own quote correctly, including samp_category
    (captured as raw evidence only -- it has no MappingRule, see
    mapping/rules.py's own comment for why)."""
    text = (
        "Water samples were collected using a Niskin bottle attached to a CTD rosette. "
        "At each station, 10 L of water were collected and immediately filtered. "
        "Samples were collected during the Malaspina 2010 expedition. "
        "Triplicate filters from each station were pooled to form one composite sample. "
        "A field blank was included as a negative control at each sampling event."
    )
    categorization_response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["sample_prep"]},
            {"sentence_id": "S0.1", "categories": ["sample_prep"]},
            {"sentence_id": "S0.2", "categories": ["sample_prep"]},
            {"sentence_id": "S0.3", "categories": ["sample_prep"]},
            {"sentence_id": "S0.4", "categories": ["sample_prep"]},
        ]
    )

    def fake_backend(prompt: str) -> str:
        if "categorizing sentences" in prompt:
            return categorization_response
        if 'extracting FAIRe "Sample preparation' in prompt:
            return json.dumps(
                [
                    {"field": "samp_collect_device", "raw_value": "Niskin bottle", "quote_id": "Q001"},
                    {"field": "samp_size", "raw_value": "10 L", "quote_id": "Q002"},
                    {"field": "samp_size_unit", "raw_value": "L", "quote_id": "Q002"},
                    {"field": "internal_expedition_id", "raw_value": "Malaspina 2010", "quote_id": "Q003"},
                    {
                        "field": "sample_composed_of",
                        "raw_value": "Triplicate filters from each station were pooled to form one composite sample",
                        "quote_id": "Q004",
                    },
                    {"field": "samp_category", "raw_value": "negative control", "quote_id": "Q005"},
                ]
            )
        return "[]"

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_section_category_facts(backend, [("Methods", text)], locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["samp_collect_device"] == "Niskin bottle"
    assert by_field["samp_size"] == "10 L"
    assert by_field["samp_size_unit"] == "L"
    assert by_field["internal_expedition_id"] == "Malaspina 2010"
    assert by_field["sample_composed_of"] == (
        "Triplicate filters from each station were pooled to form one composite sample"
    )
    assert by_field["samp_category"] == "negative control"


def test_extract_section_category_facts_no_gated_paragraphs_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])
    facts = extract_section_category_facts(
        backend, [("Intro", "The sky was blue and the ocean was calm that day.")], locator_prefix="test"
    )
    assert facts == []
    assert backend.calls == []

