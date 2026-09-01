"""Tests for extraction/section_category_extraction.py (Stage 2 LLM
sentence-categorization and Stage 3 category-scoped term extraction)."""
import json
import re

import pytest

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.extraction.section_categories import SECTION_CATEGORIES
from fair_ocean_agent.extraction.section_category_extraction import (
    _normalization_parts,
    categorize_paragraphs,
    extract_category_terms,
    extract_pulled_env_var_facts,
    extract_section_category_facts,
    normalize_controlled_sample_prep_facts,
)
from fair_ocean_agent.llm.base import LLMBackendError
from fair_ocean_agent.llm.mock import MockLLMBackend
from fair_ocean_agent.sources.base import RawFactCandidate

_SAMPLE_PREP_CATEGORY = next(c for c in SECTION_CATEGORIES if c.name == "sample_prep")


def test_normalization_parts_drops_leaked_prompt_placeholder():
    """Real gap found live (PMC10988111): several fields' own
    normalization prompts instruct "Use other:<short literal technique>"
    as a format template for the model to fill in -- a weak/confused
    model echoed that placeholder back verbatim instead of substituting a
    real value, producing a stored value of literally
    "other:<short literal technique>". A real extracted term never
    contains literal angle brackets, so this is dropped outright rather
    than kept as if it were real content."""
    assert _normalization_parts("other:<short literal technique>") == []
    assert _normalization_parts("other:<short literal technique> | bead beating") == ["bead beating"]
    assert _normalization_parts(["other:<short literal device>", "Niskin bottle"]) == ["Niskin bottle"]
    # legitimate values with no placeholder syntax are unaffected
    assert _normalization_parts("bead beating | sonication") == ["bead beating", "sonication"]


def test_categorize_paragraphs_empty_input_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])
    assert categorize_paragraphs(backend, []) == {}
    assert backend.calls == []


def test_categorize_paragraphs_tags_sentences_from_llm_response():
    response = json.dumps(
        [
            {"sentence_id": "S0.0", "categories": ["sample_prep"]},
            {"sentence_id": "S0.1", "categories": []},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    paragraph = "Samples were stored at -80C. An unrelated aside sentence."
    result = categorize_paragraphs(backend, [(paragraph, frozenset({"sample_prep"}))])
    assert result[0][0] == ("Samples were stored at -80C.", frozenset({"sample_prep"}))
    assert result[0][1] == ("An unrelated aside sentence.", frozenset())


def test_categorize_paragraphs_omitted_sentence_gets_empty_category_set():
    response = json.dumps([{"sentence_id": "S0.0", "categories": ["sample_prep"]}])
    backend = MockLLMBackend(responses=[response])
    paragraph = "Samples were stored at -80C. An unrelated aside sentence."
    result = categorize_paragraphs(backend, [(paragraph, frozenset({"sample_prep"}))])
    assert result[0][1] == ("An unrelated aside sentence.", frozenset())


def test_categorize_paragraphs_ignores_unknown_category_names():
    response = json.dumps([{"sentence_id": "S0.0", "categories": ["sample_prep", "not_a_real_category"]}])
    backend = MockLLMBackend(responses=[response])
    result = categorize_paragraphs(backend, [("Samples were stored at -80C.", frozenset({"sample_prep"}))])
    assert result[0][0][1] == frozenset({"sample_prep"})


def test_categorize_paragraphs_raises_on_invalid_json_after_retries():
    backend = MockLLMBackend(responses=["not json"])
    with pytest.raises(LLMBackendError):
        categorize_paragraphs(backend, [("Samples were stored at -80C.", frozenset({"sample_prep"}))])


def test_categorize_paragraphs_chunks_a_large_paper_instead_of_one_giant_call():
    """Real live audit (10.1371/journal.pone.0303937, see this function's
    own comment): one call for a whole paper's worth of candidate
    sentences got silently truncated by the model's own output-token
    limit, dropping everything past the cutoff. 130 one-sentence
    paragraphs (well past _MAX_SENTENCES_PER_CATEGORIZATION_CALL=60) must
    produce multiple smaller calls, not one call asking the model to
    tag 130 sentences at once -- and every single sentence's tag must
    still come back correctly regardless of which batch it landed in."""
    paragraphs = [(f"Sentence number {i} about sample prep.", frozenset({"sample_prep"})) for i in range(130)]

    def respond(prompt: str) -> str:
        ids_in_this_call = re.findall(r"S(\d+)\.0", prompt)
        return json.dumps([{"sentence_id": f"S{i}.0", "categories": ["sample_prep"]} for i in ids_in_this_call])

    backend = MockLLMBackend(responses=respond)
    result = categorize_paragraphs(backend, paragraphs)

    assert len(backend.calls) == 3  # 60 + 60 + 10
    sentences_per_call = [len(re.findall(r"\bS\d+\.0:", call["prompt"])) for call in backend.calls]
    assert sentences_per_call == [60, 60, 10]
    for i in range(130):
        assert result[i][0][1] == frozenset({"sample_prep"}), f"sentence {i} lost its tag"


def test_extract_category_terms_no_candidates_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, "Nothing relevant here at all.", locator_prefix="test")
    assert facts == []
    assert backend.calls == []


def test_extract_category_terms_chunks_a_dense_category_instead_of_one_giant_call():
    """Same class of risk as categorize_paragraphs' own confirmed
    truncation incident (see _MAX_CANDIDATES_PER_TERM_EXTRACTION_CALL's
    comment), one layer down in Stage 3: a category with many matched
    candidate quotes must go out as multiple smaller calls, and the
    cross-batch accumulation into `grouped` (this function's own
    post-processing runs once, after every batch) must still merge every
    batch's real values into one final pipe-joined fact -- not just the
    last batch's."""
    run_text = " ".join(f"Sample {i} was collected using a sampler labeled SN-{i:03d}." for i in range(1, 46))

    def respond(prompt: str) -> str:
        # raw_value must be the literal SN-NNN substring so it survives
        # extract_category_terms' own verbatim-in-quote guard -- the
        # quote_id -> SN number mapping is recovered from the prompt's own
        # candidate-quote lines, same as a real model reading them.
        return json.dumps(
            [
                {"field": "samp_collect_device", "raw_value": f"SN-{sn}", "quote_id": qid}
                for qid, sn in re.findall(r"(Q\d+) \[samp_collect_device\]: Sample \d+ .*?SN-(\d+)", prompt)
            ]
        )

    backend = MockLLMBackend(responses=respond)
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    assert len(backend.calls) == 2  # 40 + 5
    by_field = {f.fact_type_candidate: f for f in facts}
    values = set(by_field["samp_collect_device"].raw_value.split(" | "))
    assert values == {f"SN-{i:03d}" for i in range(1, 46)}


def test_extract_category_terms_extracts_verbatim_values_and_pipe_joins_conflicts():
    """nucl_acid_ext_kit/samp_store_temp/sterilise_method each live in a
    different workflow phase (extraction/storage/collection -- see
    TERM_PHASES_BY_CATEGORY), so this exercises three separate phase
    calls, not one. The mock responds per-call from the prompt's own
    field list rather than a hardcoded quote_id, since each phase's own
    candidate numbering restarts at Q001."""
    run_text = (
        "DNA was extracted using the DNeasy PowerSoil kit. "
        "Samples were stored at -80C prior to extraction. "
        "Sterile 10-mL cutoff syringes were used for subsampling."
    )
    answers = {
        "nucl_acid_ext_kit": "DNeasy PowerSoil kit",
        "samp_store_temp": "-80C",
        "sterilise_method": "Sterile 10-mL cutoff syringes",
    }

    def respond(prompt: str) -> str:
        items = []
        for qid, bracketed in re.findall(r"(Q\d+) \[([^\]]+)\]:", prompt):
            for field in (name.strip() for name in bracketed.split(",")):
                if field in answers:
                    items.append({"field": field, "raw_value": answers[field], "quote_id": qid})
        return json.dumps(items)

    backend = MockLLMBackend(responses=respond)
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {f.fact_type_candidate: f for f in facts}
    assert by_field["nucl_acid_ext_kit"].raw_value == "DNeasy PowerSoil kit"
    assert by_field["samp_store_temp"].raw_value == "-80C"
    assert by_field["sterilise_method"].raw_value == "Sterile 10-mL cutoff syringes"
    assert all(f.support_type.value == "explicit" for f in facts)


def test_extract_category_terms_rejects_hallucinated_field_names():
    run_text = "DNA was extracted using the DNeasy PowerSoil kit."
    response = json.dumps([{"field": "not_a_real_sample_prep_field", "raw_value": "DNeasy PowerSoil kit", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
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


def test_extract_category_terms_pool_dna_num_keeps_pooling_sentences():
    run_text = (
        "DNA extracts from three replicate samples were pooled before PCR. "
        "RNA samples were pooled in equal volumes before library preparation."
    )
    response = json.dumps(
        [
            {
                "field": "pool_dna_num",
                "raw_value": "DNA extracts from three replicate samples were pooled before PCR",
                "quote_id": "Q001",
            },
            {
                "field": "pool_dna_num",
                "raw_value": "RNA samples were pooled in equal volumes before library preparation",
                "quote_id": "Q002",
            },
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "pool_dna_num"
    assert facts[0].raw_value == (
        "DNA extracts from three replicate samples were pooled before PCR | "
        "RNA samples were pooled in equal volumes before library preparation"
    )


def test_extract_category_terms_derives_dna_cleanup_0_1_from_resolved_method():
    """Real bug caught live (10.7717/peerj.9857): dna_cleanup_0_1 was left
    blank while dna_cleanup_method resolved to a real value from the exact
    same quote ("Amplicons were cleaned using PCR clean-up kit
    (Fermentas)") -- the boolean's own narrower cue list never matched it.
    A resolved method is definitionally "yes, a cleanup happened"."""
    run_text = "Amplicons were cleaned using PCR clean-up kit (Fermentas) prior to the second PCR."
    response = json.dumps(
        [{"field": "dna_cleanup_method", "raw_value": "PCR clean-up kit (Fermentas)", "quote_id": "Q001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["dna_cleanup_method"] == "PCR clean-up kit (Fermentas)"
    assert by_field["dna_cleanup_0_1"] == "1"


def test_extract_category_terms_does_not_derive_dna_cleanup_0_1_when_already_answered():
    run_text = "No cleanup was performed on the extracted DNA."
    response = json.dumps([{"field": "dna_cleanup_0_1", "raw_value": "0", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["dna_cleanup_0_1"] == "0"


def test_extract_category_terms_pool_dna_num_rejects_library_pooling_without_sample_context():
    run_text = "Sequencing libraries were pooled in equimolar amounts before loading on the MiSeq."
    response = json.dumps(
        [
            {
                "field": "pool_dna_num",
                "raw_value": "Sequencing libraries were pooled in equimolar amounts before loading on the MiSeq",
                "quote_id": "Q001",
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    assert facts == []


def test_extract_category_terms_pool_dna_num_accepts_replicates_as_sample_context():
    """Real gap found live (10.3390/microorganisms10030558): "Three
    replicates were pooled and purified using the QIAquick Gel Extraction
    Kits" names what got pooled as "replicates" -- a real, common way of
    describing multiple parallel PCR reactions/aliquots being combined
    before cleanup, missed entirely since _POOLING_SAMPLE_CONTEXT_RE's
    original word list didn't include it."""
    run_text = "Three replicates were pooled and purified using the QIAquick Gel Extraction Kits (QIAGEN GmbH)."
    response = json.dumps(
        [
            {
                "field": "pool_dna_num",
                "raw_value": "Three replicates were pooled and purified using the QIAquick Gel Extraction Kits (QIAGEN GmbH)",
                "quote_id": "Q001",
            }
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "pool_dna_num"


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


def test_extract_category_terms_derives_active_filter_for_sterivex_without_active_language():
    """Generic filtration evidence should still fill filter_passive_active_0_1.

    Sterivex cartridge filters are active-by-design, even when papers do
    not state explicit active/pump/pressure wording.
    """
    run_text = "Water samples were filtered through a 0.22 um Sterivex cartridge filter."
    response = json.dumps(
        [
            {"field": "filter_name", "raw_value": "Sterivex", "quote_id": "Q001"},
            {"field": "size_frac", "raw_value": "0.22 um", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["filter_name"].raw_value == "Sterivex"
    assert by_type["size_frac"].raw_value == "0.22 um"
    assert by_type["filter_passive_active_0_1"].raw_value == "1"
    assert "Sterivex" in by_type["filter_passive_active_0_1"].evidence_quote


def test_extract_category_terms_derives_passive_filter_when_generic_filter_present_without_active_mechanism():
    run_text = "Water samples were filtered through a 0.22 um cartridge filter."
    response = json.dumps(
        [
            {"field": "filter_name", "raw_value": "cartridge filter", "quote_id": "Q001"},
            {"field": "size_frac", "raw_value": "0.22 um", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["filter_passive_active_0_1"].raw_value == "0"


def test_extract_category_terms_sterivex_upgrades_llm_zero_to_active():
    run_text = "Water samples were filtered through a 0.22 um Sterivex cartridge filter."
    response = json.dumps(
        [
            {"field": "filter_name", "raw_value": "Sterivex", "quote_id": "Q001"},
            {"field": "filter_passive_active_0_1", "raw_value": "0", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["filter_passive_active_0_1"].raw_value == "1"


def test_extract_category_terms_adds_all_size_fracs_from_filter_cascade_quote():
    """Real regression from the PLOS plankton-filtration paper:
    one filtration sentence contains 180-um, 5.0-um, and 0.2-um filters.
    The model may return only the first value, but all three are simple
    pore-size facts once the quote is already a size_frac candidate."""
    run_text = (
        "The water was filtered through a 180-μm filter, followed by a "
        "5.0-μm polycarbonate membrane and a 0.2-μm polycarbonate membrane."
    )
    response = json.dumps(
        [{"field": "size_frac", "raw_value": "180-μm", "quote_id": "Q001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")

    by_type = {fact.fact_type_candidate: fact for fact in facts}
    assert by_type["size_frac"].raw_value == "180-μm | 5.0-μm | 0.2-μm"


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
    run_text = "Samples were stored at -80C prior to extraction."
    response = json.dumps(
        [{"field": "samp_store_temp", "raw_value": "-20C", "quote_id": "Q001"}]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_rejects_value_for_field_the_quote_was_never_offered_for():
    """Regression guard for a real bug found live (10.1371/journal.pone.0303937):
    the verbatim guard only checks that a value's TEXT appears in its cited
    quote, not that the returned field name is one of the field(s) that
    quote was actually candidate-tagged for. A quote tagged only
    [samp_mat_process] ("...chopped into small pieces...the total dried
    weight was 7 g.") also contains a weight, and a model could attach it
    to samp_size instead of the samp_mat_process field the quote was
    offered for -- accepted before this fix since "7 g" is genuinely
    verbatim in that same quote's text."""
    run_text = "The subsampled sediment was chopped into small pieces and the total dried weight was 7 g."
    response = json.dumps([{"field": "samp_size", "raw_value": "7 g", "quote_id": "Q001"}])
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_dedups_identical_values_across_quotes():
    run_text = "Samples were stored at -80C. Later, storage at -80C was confirmed again."
    response = json.dumps(
        [
            {"field": "samp_store_temp", "raw_value": "-80C", "quote_id": "Q001"},
            {"field": "samp_store_temp", "raw_value": "-80C", "quote_id": "Q002"},
        ]
    )
    backend = MockLLMBackend(responses=[response])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert len(facts) == 1
    assert facts[0].raw_value == "-80C"


def test_extract_category_terms_skips_a_phase_with_invalid_json_without_raising():
    """Real regression found live: with Stage 3 split into one call per
    workflow phase, a single phase's own call returning invalid JSON used
    to raise LLMBackendError and abort the WHOLE category -- including
    every OTHER phase's already-succeeded facts -- since splitting one
    call into several independent ones otherwise multiplies the odds
    that *something* fails while keeping the same "lose everything" blast
    radius. A failing phase is now logged and skipped; other phases'
    facts still come through."""
    run_text = (
        "DNA was extracted using the DNeasy PowerSoil kit. "
        "Samples were stored at -80C prior to extraction."
    )
    backend = MockLLMBackend(responses=["not json"])
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    assert facts == []


def test_extract_category_terms_one_failing_phase_does_not_lose_another_phases_facts():
    run_text = (
        "Sterile 10-mL cutoff syringes were used for subsampling. "
        "DNA was extracted using the DNeasy PowerSoil kit."
    )

    def respond(prompt: str) -> str:
        if "nucl_acid_ext_kit" in prompt:
            return "not json"
        return json.dumps(
            [{"field": "sterilise_method", "raw_value": "Sterile 10-mL cutoff syringes", "quote_id": "Q001"}]
        )

    backend = MockLLMBackend(responses=respond)
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {f.fact_type_candidate: f.raw_value for f in facts}
    assert by_field["sterilise_method"] == "Sterile 10-mL cutoff syringes"
    assert "nucl_acid_ext_kit" not in by_field


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


def test_extract_category_terms_storage_duration_location_and_method_candidates():
    run_text = (
        "Each sediment sub-section was put into sterile bags and stored onboard at -20C. "
        "These samples were then transported to the laboratory under frozen conditions. "
        "The dried samples were transferred to sterile tubes until further use."
    )
    captured_prompt = {}

    def fake_backend(prompt: str) -> str:
        captured_prompt["value"] = prompt
        return json.dumps(
            [
                {"field": "samp_store_loc", "raw_value": "onboard", "quote_id": "Q001"},
                {"field": "samp_store_temp", "raw_value": "-20C", "quote_id": "Q001"},
                {
                    "field": "samp_store_method_additional",
                    "raw_value": "transported to the laboratory under frozen conditions",
                    "quote_id": "Q002",
                },
                {"field": "samp_store_dur", "raw_value": "until further use", "quote_id": "Q003"},
            ]
        )

    backend = MockLLMBackend(responses=fake_backend)
    facts = extract_category_terms(backend, _SAMPLE_PREP_CATEGORY, run_text, locator_prefix="test")
    by_field = {fact.fact_type_candidate: fact.raw_value for fact in facts}

    assert by_field["samp_store_loc"] == "onboard"
    assert by_field["samp_store_temp"] == "-20C"
    assert by_field["samp_store_method_additional"] == "transported to the laboratory under frozen conditions"
    assert by_field["samp_store_dur"] == "until further use"

    prompt = captured_prompt["value"]
    onboard_quote = next(line for line in prompt.splitlines() if "stored onboard" in line)
    transported_quote = next(line for line in prompt.splitlines() if "transported to the laboratory" in line)
    until_quote = next(line for line in prompt.splitlines() if "until further use" in line)
    assert "samp_store_loc" in onboard_quote
    assert "samp_store_method_additional" in transported_quote
    assert "samp_store_dur" in until_quote


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


def test_normalize_controlled_sample_prep_facts_standardizes_then_appends_quotes():
    facts = [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="nucl_acid_ext_lysis",
            raw_field_name="nucl_acid_ext_lysis",
            raw_value="treated in an ultrasonic water bath | proteinase K digestion",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="treated in an ultrasonic water bath | proteinase K digestion",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="nucl_acid_ext_sep",
            raw_field_name="nucl_acid_ext_sep",
            raw_value="DNA was purified by phenol-chloroform extraction",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="DNA was purified by phenol-chloroform extraction",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="prep_method_additional",
            raw_field_name="prep_method_additional",
            raw_value="Sediment was freeze-dried and ground with a mortar and pestle",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="Sediment was freeze-dried and ground with a mortar and pestle",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="nucl_acid_ext_method_additional",
            raw_field_name="nucl_acid_ext_method_additional",
            raw_value="RNA was removed by RNase treatment",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="RNA was removed by RNase treatment",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="samp_collect_device",
            raw_field_name="samp_collect_device",
            raw_value="Water samples were collected using Niskin bottles",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="Water samples were collected using Niskin bottles",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="samp_collect_method",
            raw_field_name="samp_collect_method",
            raw_value="Integrated water samples were collected from the upper 50 m",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="Integrated water samples were collected from the upper 50 m",
        ),
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="samp_mat_process",
            raw_field_name="samp_mat_process",
            raw_value="Samples were filtered, freeze-dried, and ground before DNA extraction",
            source_locator="test",
            support_type=SupportType.EXPLICIT,
            evidence_quote="Samples were filtered, freeze-dried, and ground before DNA extraction",
        )
    ]
    backend = MockLLMBackend(
        responses=[
            json.dumps({"nucl_acid_ext_lysis": "sonication | proteinase K"}),
            json.dumps({"nucl_acid_ext_sep": "phenol-chloroform"}),
            json.dumps({"prep_method_additional": "freeze-drying | grinding"}),
            json.dumps({"nucl_acid_ext_method_additional": "RNase treatment"}),
            json.dumps({"samp_collect_device": "Niskin bottle"}),
            json.dumps({"samp_collect_method": "integrated-depth sampling"}),
            json.dumps({"samp_mat_process": "filtration | freeze-drying | grinding"}),
        ]
    )

    normalized = normalize_controlled_sample_prep_facts(backend, facts, locator_prefix="test")

    by_field = {fact.fact_type_candidate: fact for fact in normalized}
    assert by_field["nucl_acid_ext_lysis_normalized"].raw_value == (
        "sonication | proteinase K | treated in an ultrasonic water bath | proteinase K digestion"
    )
    assert by_field["nucl_acid_ext_sep_normalized"].raw_value == (
        "phenol-chloroform | DNA was purified by phenol-chloroform extraction"
    )
    assert by_field["prep_method_additional_normalized"].raw_value == (
        "freeze-drying | grinding | Sediment was freeze-dried and ground with a mortar and pestle"
    )
    assert by_field["nucl_acid_ext_method_additional_normalized"].raw_value == (
        "RNase treatment | RNA was removed by RNase treatment"
    )
    assert by_field["samp_collect_device_normalized"].raw_value == (
        "Niskin bottle | Water samples were collected using Niskin bottles"
    )
    assert by_field["samp_collect_method_normalized"].raw_value == (
        "integrated-depth sampling | Integrated water samples were collected from the upper 50 m"
    )
    assert by_field["samp_mat_process_normalized"].raw_value == (
        "filtration | freeze-drying | grinding | Samples were filtered, freeze-dried, and ground before DNA extraction"
    )
    assert "treated in an ultrasonic water bath" in backend.calls[0]["prompt"]
    assert "DNA was purified by phenol-chloroform extraction" in backend.calls[1]["prompt"]
    assert "Sediment was freeze-dried and ground" in backend.calls[2]["prompt"]
    assert "RNA was removed by RNase treatment" in backend.calls[3]["prompt"]
    assert "Water samples were collected using Niskin bottles" in backend.calls[4]["prompt"]
    assert "Integrated water samples were collected" in backend.calls[5]["prompt"]
    assert "Samples were filtered, freeze-dried" in backend.calls[6]["prompt"]


def test_normalize_controlled_sample_prep_facts_makes_no_call_without_source_quotes():
    backend = MockLLMBackend(responses=[json.dumps({"nucl_acid_ext_lysis": "sonication"})])

    normalized = normalize_controlled_sample_prep_facts(backend, [], locator_prefix="test")

    assert normalized == []
    assert backend.calls == []


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


def _env_var_block_fact(quote: str) -> RawFactCandidate:
    return RawFactCandidate(
        entity_level=EntityLevel.STUDY,
        fact_type_candidate="x_env_var_block",
        raw_field_name="x_env_var_block",
        raw_value=quote,
        source_locator="test",
        support_type=SupportType.EXPLICIT,
        evidence_quote=quote,
    )


def test_extract_pulled_env_var_facts_keeps_only_verbatim_grounded_pairs():
    """Real evidence (10.1093/ismejo/wrae013, STUDY-01c947869c9d): a
    genuine in-situ reading and a later climate-room/acclimation reading
    both use the same variable names and both state real numbers -- the
    prompt is responsible for excluding the latter (untestable without a
    live LLM), but the verbatim guard here must still reject any value the
    model claims that doesn't literally appear in its own cited quote."""
    quote = (
        "In situ bottom water temperature (6.5°C), dissolved O2 (11.9 mg L-1), and salinity (6.4 PSU) were "
        "measured with a ProODO probe."
    )
    response = json.dumps(
        [
            {"variable": "temperature", "value": "6.5°C", "quote_id": "Q001"},
            {"variable": "salinity", "value": "6.4 PSU", "quote_id": "Q001"},
            # Hallucinated -- never appears in the quote -- must be dropped.
            {"variable": "pH", "value": "8.1", "quote_id": "Q001"},
        ]
    )
    backend = MockLLMBackend(responses=[response])

    facts = extract_pulled_env_var_facts(backend, [_env_var_block_fact(quote)], locator_prefix="test")

    assert len(facts) == 1
    assert facts[0].fact_type_candidate == "x_pulled_env_var"
    assert facts[0].raw_value == "temperature = 6.5°C | salinity = 6.4 PSU"
    assert facts[0].evidence_quote == quote


def test_extract_pulled_env_var_facts_empty_llm_response_produces_no_fact():
    """Real evidence (STUDY-1e007d7a6809): a quote that only names sampling
    depths and a qualitative "chlorophyll maxima" location, with no actual
    measured chlorophyll number, should yield nothing once the model
    correctly reports no genuine pairs."""
    quote = (
        "Water samples were collected from depths of 30 m, chlorophyll maxima (Cmax), 200 m, 600 m, 2,000 m, "
        "3,500 m, and near the bottom with the help of Niskin bottles."
    )
    backend = MockLLMBackend(responses=["[]"])

    facts = extract_pulled_env_var_facts(backend, [_env_var_block_fact(quote)], locator_prefix="test")

    assert facts == []


def test_extract_pulled_env_var_facts_no_source_fact_makes_no_llm_call():
    backend = MockLLMBackend(responses=["[]"])

    facts = extract_pulled_env_var_facts(backend, [], locator_prefix="test")

    assert facts == []
    assert backend.calls == []
