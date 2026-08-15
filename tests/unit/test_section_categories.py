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
    # date_ext/ratioOfAbsorbance260_280. 42 -> 48
    # after a "sample collection" batch added samp_collect_device/
    # samp_collect_method/samp_size/samp_size_unit/sample_composed_of/
    # sample_derived_from/samp_category (folded into this same category
    # rather than a new classifier, per an explicit user instruction).
    # 48 -> 39: per an explicit, repeated user request, removed
    # prepped_samp_store_temp/prepped_samp_store_dur/prepped_samp_store_sol
    # (consolidated into samp_store_sol instead -- the user didn't want two
    # near-duplicate storage-solution fields), dna_store_loc, nucl_acid_ext,
    # nucl_acid_ext_modify, date_ext, ratioOfAbsorbance260_280, size_frac_low.
    # 39 -> 40: internal_expedition_id added back as a sampleMetadata
    # cruise/expedition/campaign/station identifier, still folded into
    # sample_prep because it appears in the initial sampling narrative.
    # 40 -> 36: precip_chem_prep/precip_force_prep/precip_temp_prep/
    # precip_time_prep removed entirely per an explicit user request
    # ("negligible... don't want to waste compute on them").
    # 36 -> 37: sterilise_method moved here from a deterministic
    # ControlledSearchField, per an explicit user request (sterilization
    # of sampling equipment is a sample-collection concept).
    # 37 -> 35: concentration_method/concentration_unit removed per an
    # explicit user request; concentration now preserves value+unit together.
    # 35 -> 36: x_env_var_block added, bundling ~18 low-yield physicochemical
    # sampleMetadata fields (temp, salinity, ph, nitrate, chlorophyll, ...)
    # into one term per an explicit user request ("group all the above
    # together, just to make it easier for Qwen").
    "sample_prep": 36,
    # assay_validation and nucl_acid_amp were removed entirely per an
    # explicit user request (8 -> 6).
    "assay_definition": 6,
    # 26 -> 24 after removing pcr_primer_vol_forward/reverse entirely per
    # user request. 24 -> 18: amplificationReactionVolume, pcr_cond,
    # pcr_dna_vol, pcr_primer_conc_forward, pcr_primer_conc_reverse, and
    # thermocycler removed entirely per an explicit user request
    # ("negligible... don't want to waste compute on them") -- pcr_cycles
    # and annealingTemp deliberately kept.
    "pcr1_primary_amplification": 18,
    "targeted_qpcr_ddpcr_detection": 27,
    # 11 -> 4: pcr2_amplificationReactionVolume, pcr2_commercial_mm,
    # pcr2_cond, pcr2_custom_mm, pcr2_dna_vol, pcr2_plate_id, and
    # pcr2_thermocycler removed entirely per an explicit user request --
    # pcr2_cycles and pcr2_annealingTemp deliberately kept.
    "pcr2_indexing": 4,
    # 8 -> 7: lib_screen removed entirely per an explicit user request.
    "library_prep_sequencing": 7,
    # error_rate_cutoff was removed from category-term extraction; error
    # evidence is now handled by the dedicated judged search fields.
    # 10 -> 11: screen_nontarget_method duplicated here too (its primary
    # home is otu_asv_generation_filtering) since a real paper's own PhiX
    # spike-in removal happens at the raw-read stage, not after OTU/ASV
    # generation -- see extraction/section_categories.py's own comment.
    # 11 -> 1: demux_tool/demux_max_mismatch, error_rate_tool/
    # error_rate_type, merge_tool/merge_min_overlap, and min_len_cutoff/
    # min_len_tool removed entirely per an explicit user request.
    # 1 -> 0: screen_nontarget_method removed entirely too, per a later
    # explicit user request -- this duplicate (its primary home was
    # otu_asv_generation_filtering below) goes with it.
    "raw_read_preprocessing": 0,
    # otu_raw_description is no longer a CategoryTerm here -- it's
    # generated, not extracted (13 -> 12). 12 -> 5: chimera_check_method/
    # chimera_check_param, min_reads_cutoff/min_reads_tool/
    # min_reads_cutoff_unit, otu_clust_cutoff, and otu_final_description
    # removed entirely per an explicit user request; otu_raw_description
    # (already generated, not a CategoryTerm) removed too.
    # 5 -> 2: screen_geograph_method/screen_nontarget_method/screen_other
    # removed entirely per an explicit user request; screen_contam_method
    # stays (real decontamination step) but got a new prompt/cues.
    "otu_asv_generation_filtering": 2,
    # tax_assign_cat is no longer a CategoryTerm here -- it's a controlled
    # enum classification (search_flags.py's own LLMJudgedSearchField,
    # allowed_values-constrained), not a verbatim quote (8 -> 7). tax_class_
    # other is also no longer a CategoryTerm -- it's generated, not
    # extracted (7 -> 6). otu_db_custom was merged into otu_db (6 -> 5).
    # 5 -> 2: tax_class_collapse/tax_class_id_cutoff/tax_class_query_cutoff
    # removed entirely per an explicit user request.
    "taxonomic_assignment": 2,
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


def test_split_into_paragraphs_strips_table_body_glued_to_sample_prep_sentence():
    table_glued_to_method = (
        "Cue Site Region # of quality- filtered reads # of OTUs # of reads uniquely mapping to OTUs "
        "Mapping efficiency A1 Orpheus Island (GBR) Pacific 2760 6 2714 0.983 A2 Orpheus Island "
        "(GBR) Pacific 4906 10 3566 0.727 P Pohnpei Pacific NA NA NA NA "
        "DNA was isolated from ground-up cue samples as described in ."
    )

    paragraphs = split_into_paragraphs(table_glued_to_method)

    assert paragraphs == ["DNA was isolated from ground-up cue samples as described in ."]


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


def test_candidate_categories_for_paragraph_routes_amplification_program_to_primary_pcr():
    paragraph = (
        "Amplification of the V3-V5 hypervariable regions of the 16S rRNA gene was performed "
        "with Phusion High Fidelity DNA polymerase. The amplification program consisted of "
        "initial denaturation at 98C, followed by 30 cycles of denaturation, annealing at "
        "62.4C, and extension."
    )

    categories = candidate_categories_for_paragraph(paragraph)

    assert "pcr1_primary_amplification" in categories


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
        "nucl_acid_ext_method_additional",
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
        "samp_mat_process",
        "samp_store_method_additional",
        "x_env_var_block",
    }


def test_target_subfragment_and_assay_name_cues_are_deliberately_distinct():
    """User's own explicit distinction: 'V4 region' cues target_subfragment,
    but 'V4 assay'/'V4 primer set' cues assay_name -- both terms must not
    share the bare substring in a way that collapses the distinction."""
    assay_definition = next(c for c in SECTION_CATEGORIES if c.name == "assay_definition")
    terms_by_name = {t.native_name: t for t in assay_definition.terms}
    assert "V4 region" in terms_by_name["target_subfragment"].search_cues
    assert not any(cue == "V4 region" for cue in terms_by_name["assay_name"].search_cues)


def test_assay_definition_routes_5prime_ssu_portion_and_amplicon_band_size():
    assay_definition = next(c for c in SECTION_CATEGORIES if c.name == "assay_definition")
    terms_by_name = {t.native_name: t for t in assay_definition.terms}

    assert "conserved 5′ portion" in terms_by_name["target_subfragment"].search_cues
    assert "bp bands" in terms_by_name["ampliconSize"].search_cues

    paragraph = (
        "The conserved 5′ portion of the eukaryotic small-subunit ribosomal RNA gene "
        "(18S SSU) was amplified. Amplicons (~550 bp bands) were successfully obtained "
        "from 6 out of 7 samples."
    )

    assert "assay_definition" in candidate_categories_for_paragraph(paragraph)


def test_raw_read_and_taxonomy_terms_include_pear_and_kraken_tools():
    taxonomy = next(c for c in SECTION_CATEGORIES if c.name == "taxonomic_assignment")
    taxonomy_terms = {t.native_name: t for t in taxonomy.terms}
    assert "Kraken 2" in taxonomy_terms["otu_seq_comp_appr"].search_cues

    assert "raw_read_preprocessing" in candidate_categories_for_paragraph(
        "Paired-end reads were merged using PEAR 0.9.10."
    )
    assert "taxonomic_assignment" in candidate_categories_for_paragraph(
        "Representative sequences were classified taxonomically using Kraken 2.0.9."
    )


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
