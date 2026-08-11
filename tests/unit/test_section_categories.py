"""Tests for extraction/section_categories.py.

The reference-list/figure-caption regression cases below use real
sentence fragments from a real paper's cached supplementary methods
(PNAS 10.1073/pnas.2005917117, retrieved from this project's own audit
database) -- not synthetic text -- since these are the exact false
positives that shaped this module's design.
"""
from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.extraction.section_categories import (
    SECTION_CATEGORIES,
    candidate_categories_for_paragraph,
    derive_pcr_0_1_from_category_detection,
    detect_section_categories_present,
    group_sentences_into_category_runs,
    split_into_paragraphs,
)
from fair_ocean_agent.sources.base import RawFactCandidate

# Locks in the exact per-category term counts from the user's own term
# tables, so a future edit that accidentally drops or duplicates an entry
# fails loudly here instead of silently shrinking Stage 3's eventual
# field coverage.
_EXPECTED_TERM_COUNTS = {
    # New category, per an explicit user request: sample preparation /
    # storage / nucleic acid extraction (pre-PCR). 36 -> 42 after a second
    # batch added concentration/concentration_method/concentration_unit/
    # date_ext/ratioOfAbsorbance260_280/stationed_sample_dur.
    "sample_prep": 42,
    # assay_validation and nucl_acid_amp were removed entirely per an
    # explicit user request (8 -> 6).
    "assay_definition": 6,
    "pcr1_primary_amplification": 26,
    "targeted_qpcr_ddpcr_detection": 27,
    "pcr2_indexing": 11,
    "library_prep_sequencing": 9,
    "raw_read_preprocessing": 11,
    # otu_raw_description is no longer a CategoryTerm here -- it's
    # generated, not extracted (13 -> 12).
    "otu_asv_generation_filtering": 12,
    # tax_assign_cat is no longer a CategoryTerm here -- it's a controlled
    # enum classification (search_flags.py's own LLMJudgedSearchField,
    # allowed_values-constrained), not a verbatim quote (8 -> 7). tax_class_
    # other is also no longer a CategoryTerm -- it's generated, not
    # extracted (7 -> 6).
    "taxonomic_assignment": 6,
}

# A real DNA-extraction paragraph flowing directly into amplicon PCR,
# sequencing, quality trimming, chimera removal, OTU clustering, and
# taxonomic classification with no blank-line break between any of them --
# confirmed against the real PNAS supplementary methods text. This is the
# exact evidence that motivated sentence-level (not whole-paragraph)
# category tagging.
_REAL_DENSE_METHODS_PARAGRAPH = (
    "DNA extraction. DNA for amplicon sequencing and qPCR was extracted from ~0.5 gram of "
    "sediment per sample using the PowerLyze DNA extraction kits. "
    "Amplicon sequencing and sequence analysis. 16S rRNA genes were amplified using the primer "
    "pair 515F/806R in a two-round amplicon preparation, with an optimal PCR cycle number in the "
    "first round to minimize over-amplification. Amplicon libraries were sequenced on an Ion "
    "Torrent Personal Genome Machine. Sequencing reads were quality filtered and trimmed to 220 bp "
    "using the USEARCH pipeline and chimera were detected and removed using UCHIME. Trimmed reads "
    "were clustered into operational taxonomy units (OTUs) at >97% nucleotide sequence identity "
    "using UPARSE. The taxonomic classification of OTUs was performed using the lowest common "
    "ancestor algorithm implemented in the CREST package."
)

_REAL_QPCR_PARAGRAPH = (
    "Quantification of total microbial community and anammox bacteria. Abundances of anammox "
    "bacteria was quantified using qPCR by targeting the hzo gene using the primer pair hzoF1/hzoR1. "
    "All gene abundances were determined in triplicate by qPCR, and standard deviations are "
    "presented using horizontal error bars."
)

_REAL_UNRELATED_GEOCHEM_PARAGRAPH = (
    "Instead, we calculated the root mean square error (RMSE) of the modeled and measured profiles "
    "of O2, Mn(II), NO3-, NH4+, and DIC in the four AMOR cores. Gene abundances below detection "
    "limit were arbitrarily shown as 100 copies g-1."
)

_REAL_REFERENCE_ENTRY = (
    "27. P. Engstrom, C. R. Penton, A. H. Devol, Anaerobic ammonium oxidation in deep-sea "
    "sediments off the Washington margin. Limnology and Oceanography 54, 1643-1652 (2009)."
)

_REAL_FIGURE_CAPTION = (
    "Fig. S 5. Cell-specific rates of anammox bacteria in NATZ. Cell-specific rates of anammox "
    "were calculated by dividing the modeled bulk anammox reaction rate by the anammox cell number "
    "quantified by qPCR targeting the hzo gene."
)


def test_all_nine_categories_are_defined_with_nonempty_keyword_lists():
    assert {c.name for c in SECTION_CATEGORIES} == {
        "sample_prep",
        "assay_definition",
        "pcr1_primary_amplification",
        "targeted_qpcr_ddpcr_detection",
        "pcr2_indexing",
        "library_prep_sequencing",
        "raw_read_preprocessing",
        "otu_asv_generation_filtering",
        "taxonomic_assignment",
    }
    for category in SECTION_CATEGORIES:
        assert category.keywords


def test_split_into_paragraphs_drops_the_reference_section_entirely():
    text = f"{_REAL_QPCR_PARAGRAPH}\n\n26. Some earlier reference.\n\n{_REAL_REFERENCE_ENTRY}"
    paragraphs = split_into_paragraphs(text)
    assert paragraphs == [_REAL_QPCR_PARAGRAPH]


def test_split_into_paragraphs_drops_figure_captions_anywhere_not_just_at_the_end():
    text = f"{_REAL_FIGURE_CAPTION}\n\n{_REAL_QPCR_PARAGRAPH}"
    paragraphs = split_into_paragraphs(text)
    assert paragraphs == [_REAL_QPCR_PARAGRAPH]


def test_candidate_categories_for_paragraph_finds_multiple_real_categories_in_dense_text():
    """Real evidence: one continuous supplementary-methods paragraph
    legitimately spans PCR1, library prep/sequencing, raw read
    preprocessing, OTU/ASV generation, and taxonomic assignment -- the
    paragraph-level gate is deliberately over-inclusive; sentence-level
    tagging (LLM, not built in this module) is what narrows it down."""
    categories = candidate_categories_for_paragraph(_REAL_DENSE_METHODS_PARAGRAPH)
    assert "pcr1_primary_amplification" in categories
    assert "library_prep_sequencing" in categories
    assert "raw_read_preprocessing" in categories
    assert "otu_asv_generation_filtering" in categories
    assert "taxonomic_assignment" in categories


def test_candidate_categories_for_paragraph_finds_qpcr_cleanly():
    categories = candidate_categories_for_paragraph(_REAL_QPCR_PARAGRAPH)
    assert categories == {"targeted_qpcr_ddpcr_detection", "assay_definition", "pcr1_primary_amplification"}


def test_candidate_categories_for_paragraph_no_match_on_unrelated_text():
    assert candidate_categories_for_paragraph("The sky was blue and the ocean was calm that day.") == frozenset()


def test_detect_section_categories_present_end_to_end_on_real_text():
    text = (
        f"{_REAL_DENSE_METHODS_PARAGRAPH}\n\n"
        f"{_REAL_QPCR_PARAGRAPH}\n\n"
        f"{_REAL_UNRELATED_GEOCHEM_PARAGRAPH}\n\n"
        f"{_REAL_FIGURE_CAPTION}\n\n"
        f"{_REAL_REFERENCE_ENTRY}"
    )
    facts = detect_section_categories_present([("Supplementary Methods", text)], locator_prefix="test")
    detected = {fact.fact_type_candidate for fact in facts}
    assert detected == {
        # The dense paragraph's own opening sentence ("DNA for amplicon
        # sequencing and qPCR was extracted from ~0.5 gram of sediment per
        # sample using the PowerLyze DNA extraction kits") is a genuine,
        # real sample_prep-category match ("DNA... was extracted",
        # "extraction kits").
        "sample_prep_0_1",
        "assay_definition_0_1",
        "pcr1_primary_amplification_0_1",
        "targeted_qpcr_ddpcr_detection_0_1",
        "library_prep_sequencing_0_1",
        "raw_read_preprocessing_0_1",
        "otu_asv_generation_filtering_0_1",
        "taxonomic_assignment_0_1",
    }
    # pcr2_indexing never appears anywhere in the supplied text -- a real
    # true negative (this paper's own two-round amplicon prep never
    # describes an explicit indexing/barcoding step), must not be flagged.
    assert "pcr2_indexing_0_1" not in detected
    for fact in facts:
        assert fact.raw_value == "1"


def test_group_sentences_into_category_runs_basic_single_category():
    tagged = [
        ("Sentence one.", frozenset({"pcr1_primary_amplification"})),
        ("Sentence two, untagged connective.", frozenset()),
        ("Sentence three.", frozenset({"pcr1_primary_amplification"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs == {
        "pcr1_primary_amplification": "Sentence one. Sentence two, untagged connective. Sentence three.",
    }


def test_group_sentences_into_category_runs_cat1_cat2_cat1_grouped_with_cat2_excluded():
    """Exact user specification: 'if a paragraph goes from cat 1 to cat 2
    back to cat 1, the two cat 1 will be grouped consecutively with cat 2
    removed.'"""
    tagged = [
        ("First PCR sentence.", frozenset({"pcr1_primary_amplification"})),
        ("Indexing PCR sentence.", frozenset({"pcr2_indexing"})),
        ("Back to first PCR.", frozenset({"pcr1_primary_amplification"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs["pcr1_primary_amplification"] == "First PCR sentence. Back to first PCR."
    assert runs["pcr2_indexing"] == "Indexing PCR sentence."


def test_group_sentences_into_category_runs_multi_tagged_sentence_continues_both_runs():
    tagged = [
        ("Assay and PCR1 sentence.", frozenset({"assay_definition", "pcr1_primary_amplification"})),
        ("Only PCR1 continues.", frozenset({"pcr1_primary_amplification"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs["assay_definition"] == "Assay and PCR1 sentence."
    assert runs["pcr1_primary_amplification"] == "Assay and PCR1 sentence. Only PCR1 continues."


def test_category_term_counts_match_the_user_supplied_term_tables():
    for category in SECTION_CATEGORIES:
        assert len(category.terms) == _EXPECTED_TERM_COUNTS[category.name], category.name


def test_category_terms_have_unique_native_names_within_each_category():
    for category in SECTION_CATEGORIES:
        names = [term.native_name for term in category.terms]
        assert len(names) == len(set(names)), category.name


def test_fallback_only_terms_have_no_search_cues_and_are_not_independently_searched():
    fallback_terms = {
        term.native_name: term
        for category in SECTION_CATEGORIES
        for term in category.terms
        if term.fallback_only
    }
    assert set(fallback_terms) == {
        "prep_method_additional",
        "pcr_method_additional",
        "targeted_detection_method_additional",
        "pcr2_method_additional",
        "seq_method_additional",
    }
    for term in fallback_terms.values():
        assert term.search_cues == ()


def test_multi_sentence_terms_match_faire_description_style_fields():
    multi_sentence_terms = {
        term.native_name
        for category in SECTION_CATEGORIES
        for term in category.terms
        if term.allows_multi_sentence
    }
    assert multi_sentence_terms == {
        "otu_final_description",
        "samp_mat_process",
        "samp_store_method_additional",
        "nucl_acid_ext_method_additional",
    }


def test_target_subfragment_and_assay_name_cues_are_deliberately_distinct():
    """User's own explicit distinction: 'V4 region' cues target_subfragment,
    but 'V4 assay'/'V4 primer set' cues assay_name -- both terms must not
    share the bare substring in a way that collapses the distinction."""
    assay_definition = next(c for c in SECTION_CATEGORIES if c.name == "assay_definition")
    terms_by_name = {t.native_name: t for t in assay_definition.terms}
    assert "V4 region" in terms_by_name["target_subfragment"].search_cues
    assert not any(cue == "V4 region" for cue in terms_by_name["assay_name"].search_cues)


def test_group_sentences_into_category_runs_untagged_sentence_after_a_break_is_dropped():
    tagged = [
        ("PCR1 sentence.", frozenset({"pcr1_primary_amplification"})),
        ("Other category sentence.", frozenset({"pcr2_indexing"})),
        ("Untagged, ambiguous which side it belongs to.", frozenset()),
        ("PCR1 resumes.", frozenset({"pcr1_primary_amplification"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs["pcr1_primary_amplification"] == "PCR1 sentence. PCR1 resumes."


def _detection_fact(fact_type: str) -> RawFactCandidate:
    return RawFactCandidate(
        entity_level=EntityLevel.STUDY,
        fact_type_candidate=fact_type,
        raw_field_name=fact_type,
        raw_value="1",
        source_locator="test:locator",
        support_type=SupportType.DETERMINISTICALLY_DERIVED,
        evidence_quote="some evidence",
    )


def test_derive_pcr_0_1_from_pcr1_detection():
    facts = [_detection_fact("pcr1_primary_amplification_0_1")]
    derived = derive_pcr_0_1_from_category_detection(facts)
    assert derived is not None
    assert derived.fact_type_candidate == "pcr_0_1"
    assert derived.raw_value == "1"


def test_derive_pcr_0_1_from_qpcr_detection_when_pcr1_absent():
    """A qPCR/ddPCR-only paper never uses the bare word 'PCR'/'amplified'
    at all (word-boundary matching never matches 'PCR' inside 'qPCR') --
    pcr_0_1 must still derive from targeted_qpcr_ddpcr_detection_0_1
    alone, matching the old regex-based flag's own scope (it explicitly
    included qPCR/ddPCR patterns)."""
    facts = [_detection_fact("targeted_qpcr_ddpcr_detection_0_1")]
    derived = derive_pcr_0_1_from_category_detection(facts)
    assert derived is not None
    assert derived.fact_type_candidate == "pcr_0_1"


def test_derive_pcr_0_1_is_none_when_neither_source_category_detected():
    facts = [_detection_fact("otu_asv_generation_filtering_0_1")]
    assert derive_pcr_0_1_from_category_detection(facts) is None
    assert derive_pcr_0_1_from_category_detection([]) is None
