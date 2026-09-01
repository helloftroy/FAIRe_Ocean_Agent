"""Tests for extraction/section_categories.py.

The reference-list/figure-caption regression cases below use real
sentence fragments from a real paper's cached supplementary methods
(PNAS 10.1073/pnas.2005917117, retrieved from this project's own audit
database) -- not synthetic text -- since these are the exact false
positives that shaped this module's design.
"""
from fair_ocean_agent.extraction.section_categories import (
    SECTION_CATEGORIES,
    _term_pattern,
    candidate_categories_for_paragraph,
    derive_pcr_0_1_from_category_detection,
    detect_section_categories_present,
    group_sentences_into_category_runs,
    split_into_paragraphs,
)

_SAMPLE_PREP_CATEGORY = next(c for c in SECTION_CATEGORIES if c.name == "sample_prep")


def _term_cues_match(native_name: str, sentence: str) -> bool:
    term = next(t for t in _SAMPLE_PREP_CATEGORY.terms if t.native_name == native_name)
    return any(_term_pattern(cue).search(sentence) for cue in term.search_cues)

# Locks in the exact per-category term count from the user's own term
# tables, so a future edit that accidentally drops or duplicates an entry
# fails loudly here instead of silently shrinking Stage 3's eventual
# field coverage. sample_prep is the only category left: the other 8
# (assay_definition, pcr1_primary_amplification, targeted_qpcr_ddpcr_
# detection, pcr2_indexing, library_prep_sequencing, raw_read_
# preprocessing, otu_asv_generation_filtering, taxonomic_assignment) were
# removed entirely, not merely disabled -- per an explicit user request,
# after a live 6-study audit found sample_prep accounted for ~72% of
# everything the whole category-pipeline ever produced while several
# others produced zero real facts, and Stage 2 (categorize_paragraphs)
# was the single most expensive step in the entire extraction pipeline.
# 35 -> 36: x_env_var_block added, bundling ~18 low-yield physicochemical
# sampleMetadata fields (temp, salinity, ph, nitrate, chlorophyll, ...)
# into one term per an explicit user request.
_EXPECTED_TERM_COUNTS = {
    "sample_prep": 36,
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


def test_only_sample_prep_category_is_defined_with_nonempty_keyword_list():
    assert {c.name for c in SECTION_CATEGORIES} == {"sample_prep"}
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


def test_split_into_paragraphs_strips_table_body_glued_to_sample_prep_sentence():
    table_glued_to_method = (
        "Cue Site Region # of quality- filtered reads # of OTUs # of reads uniquely mapping to OTUs "
        "Mapping efficiency A1 Orpheus Island (GBR) Pacific 2760 6 2714 0.983 A2 Orpheus Island "
        "(GBR) Pacific 4906 10 3566 0.727 P Pohnpei Pacific NA NA NA NA "
        "DNA was isolated from ground-up cue samples as described in ."
    )

    paragraphs = split_into_paragraphs(table_glued_to_method)

    assert paragraphs == ["DNA was isolated from ground-up cue samples as described in ."]


def test_split_into_paragraphs_removes_glued_inline_subsection_heading():
    """Real evidence (10.7717/peerj.9857, the same coral-spawning paper as
    the table-glued-sentence test above): a short subsection heading
    ("Caribbean spawn I") survived PDF text extraction glued directly onto
    the sentence that follows it, with no period or blank line separating
    them. Left in place, this fragment was previously extracted verbatim
    as if it were a real internal_expedition_id value. It must be removed
    entirely, and the sentence before/after must still read as coherent,
    unbroken prose (not split into a separate paragraph, which would
    fragment the paragraph-level keyword gate and risk losing real content
    immediately after the heading)."""
    glued = (
        "Samples were stored in seawater at 80 °C. Caribbean spawn I On the evening of "
        "August 31, 2010, gamete bundles were collected with mesh nets directly from three "
        "distinct colonies. Pacific spawn I In November 2010, at Orpheus Island Research "
        "Station, the same type of experiments were conducted."
    )

    paragraphs = split_into_paragraphs(glued)

    assert len(paragraphs) == 1
    assert "Caribbean spawn I" not in paragraphs[0]
    assert "Pacific spawn I" not in paragraphs[0]
    assert "80 °C. On the evening of August 31, 2010" in paragraphs[0]
    assert "distinct colonies. In November 2010, at Orpheus Island" in paragraphs[0]


def test_split_into_paragraphs_leaves_real_sentences_ending_in_a_number_alone():
    """Negative controls for the glued-heading regex above: real,
    un-glued prose that happens to end a sentence with a station number,
    kit name, or citation year must not be split or altered."""
    real_sentences = [
        "A total of 34 samples were collected from 6 stations. Station 1 was located near the river mouth.",
        "Libraries were prepared using the Nextera XT kit (Illumina). Illumina adapters were then ligated.",
        "This study builds on prior work (Smith et al. 2019). Field sampling took place in 2020.",
    ]
    for text in real_sentences:
        assert split_into_paragraphs(text) == [text]


def test_samp_store_temp_cues_match_stored_in_medium_at_temperature():
    """Real bug caught live (10.7717/peerj.9857): "Samples were stored in
    seawater at 80 °C." never matched samp_store_temp's "stored at" cue,
    since "in seawater" sits between "stored" and "at" -- "stored at" is a
    literal, gap-free 2-word phrase cue."""
    assert _term_cues_match("samp_store_temp", "Samples were stored in seawater at 80 °C.")
    # the original gap-free phrasing must keep matching too
    assert _term_cues_match("samp_store_temp", "Samples were stored at -80°C until extraction.")


def test_samp_size_cues_match_liters_of_seawater_was_filtered():
    """Real gap found live (10.3390/microorganisms10030558): "Two liters
    of seawater was filtered onto 0.22 µm membrane filter..." never
    matched any samp_size cue -- the original cues only covered "were
    collected" phrasing, but filtration IS the collection/concentration
    step for a water sample, so "was/were filtered" is at least as common
    a way to report sample volume for marine/aquatic eDNA papers."""
    assert _term_cues_match("samp_size", "Two liters of seawater was filtered onto 0.22 µm membrane filter for DNA extraction.")
    assert _term_cues_match("samp_size", "Three liters of seawater were filtered through a 0.2 µm filter.")
    # the original "were collected" phrasing must keep matching too
    assert _term_cues_match("samp_size", "10 L of water were collected at each station.")


def test_concentration_cues_match_spelled_out_template_amount():
    """Real gap found live (10.3390/microorganisms10030558): "Fifty
    nanograms of DNA was used as a template for PCR amplification"
    reports the DNA input amount as a spelled-out number with a bare
    mass unit (no "/uL"), which none of the original ng/uL-style cues
    matched -- confirmed live via the user this value was never
    hallucinated, just never reached by any existing cue."""
    assert _term_cues_match("concentration", "Fifty nanograms of DNA was used as a template for PCR amplification.")
    # the original ng/uL phrasing must keep matching too
    assert _term_cues_match("concentration", "Final DNA concentration was 12.4 ng/uL.")


def test_candidate_categories_for_paragraph_finds_sample_prep_in_dense_text():
    """Real evidence: this dense supplementary-methods paragraph's own
    opening sentence ("DNA for amplicon sequencing and qPCR was extracted
    from ~0.5 gram of sediment per sample using the PowerLyze DNA
    extraction kits") is a genuine sample_prep match ("DNA... was
    extracted", "extraction kits")."""
    categories = candidate_categories_for_paragraph(_REAL_DENSE_METHODS_PARAGRAPH)
    assert "sample_prep" in categories


def test_candidate_categories_for_paragraph_finds_sample_prep_from_storage_text():
    paragraph = (
        "Sediment sub-sections were stored onboard at -20C and transported to the laboratory "
        "under frozen conditions until further use."
    )
    categories = candidate_categories_for_paragraph(paragraph)
    assert "sample_prep" in categories


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
    # The dense paragraph's own opening sentence ("DNA for amplicon
    # sequencing and qPCR was extracted from ~0.5 gram of sediment per
    # sample using the PowerLyze DNA extraction kits") is a genuine,
    # real sample_prep-category match ("DNA... was extracted",
    # "extraction kits") -- the only category left to detect.
    assert detected == {"sample_prep_0_1"}
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
        "nucl_acid_ext_method_additional",
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
        "samp_mat_process",
        "samp_store_method_additional",
        "x_env_var_block",
    }


def test_group_sentences_into_category_runs_untagged_sentence_after_a_break_is_dropped():
    tagged = [
        ("PCR1 sentence.", frozenset({"pcr1_primary_amplification"})),
        ("Other category sentence.", frozenset({"pcr2_indexing"})),
        ("Untagged, ambiguous which side it belongs to.", frozenset()),
        ("PCR1 resumes.", frozenset({"pcr1_primary_amplification"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs["pcr1_primary_amplification"] == "PCR1 sentence. PCR1 resumes."


def test_group_sentences_into_category_runs_bridges_a_short_untagged_gap():
    """Up to _MAX_BRIDGING_SENTENCES consecutive untagged (genuinely
    connective) sentences still bridge a run -- this is the existing,
    intended behavior the cap below must not break for short gaps."""
    tagged = [
        ("Samples were stored in seawater.", frozenset({"sample_prep"})),
        ("This was done to preserve them.", frozenset()),
        ("Temperature was kept constant.", frozenset()),
        ("Samples were later processed.", frozenset({"sample_prep"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs["sample_prep"] == (
        "Samples were stored in seawater. This was done to preserve them. "
        "Temperature was kept constant. Samples were later processed."
    )


def test_group_sentences_into_category_runs_stops_absorbing_after_too_many_untagged_sentences():
    """Real bug found live (10.7717/peerj.333): a run started by one
    genuine sample_prep sentence ("Samples were stored in seawater at 80
    C.") silently absorbed an entire table caption and several paragraphs
    of coral-settlement-assay narrative via unlimited bridging, since none
    of that unrelated text was ever tagged with any category. More than
    _MAX_BRIDGING_SENTENCES untagged sentences in a row must end the run
    without ever including the unrelated bridge sentences, even if a
    same-category sentence appears later."""
    tagged = [
        ("Samples were stored in seawater at 80 C.", frozenset({"sample_prep"})),
        ("Table 1 Settlement cue panel and metabarcoding statistics.", frozenset()),
        ("CCA cue information including the name of the cue.", frozenset()),
        ("Gamete bundles were collected with mesh nets.", frozenset()),
        ("Bundles were cross-fertilized for one hour.", frozenset()),
        ("A completely unrelated sample_prep sentence far later.", frozenset({"sample_prep"})),
    ]
    runs = group_sentences_into_category_runs(tagged)
    assert runs["sample_prep"] == (
        "Samples were stored in seawater at 80 C. A completely unrelated sample_prep sentence far later."
    )
    assert "Table 1" not in runs["sample_prep"]
    assert "Gamete bundles" not in runs["sample_prep"]


def test_derive_pcr_0_1_from_bare_pcr_mention():
    facts = [derive_pcr_0_1_from_category_detection([("Methods", "PCR was performed using the primer pair 515F/806R.")])]
    derived = facts[0]
    assert derived is not None
    assert derived.fact_type_candidate == "pcr_0_1"
    assert derived.raw_value == "1"


def test_derive_pcr_0_1_from_qpcr_mention_with_no_bare_pcr_word():
    """A qPCR/ddPCR-only paper never uses the bare word 'PCR'/'amplified'
    at all (word-boundary matching never matches 'PCR' inside 'qPCR') --
    pcr_0_1 must still derive from the qPCR/ddPCR vocabulary alone,
    reverted back to this module's own original independent regex scan
    after pcr1_primary_amplification/targeted_qpcr_ddpcr_detection (its
    intermediate replacement) were removed entirely."""
    derived = derive_pcr_0_1_from_category_detection(
        [("Methods", "A TaqMan qPCR assay used a FAM reporter dye and BHQ quencher.")]
    )
    assert derived is not None
    assert derived.fact_type_candidate == "pcr_0_1"


def test_derive_pcr_0_1_is_none_when_no_pcr_mention_anywhere():
    assert derive_pcr_0_1_from_category_detection([("Methods", "The sky was blue and the ocean was calm.")]) is None
    assert derive_pcr_0_1_from_category_detection([]) is None
