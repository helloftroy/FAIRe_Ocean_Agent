from fair_ocean_agent.extraction.faire_fields import (
    FALLBACK_NARRATIVE_FIELDS,
    FIELD_GROUPS,
    LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS,
    LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS,
    all_faire_hints,
    all_field_names,
    assay_scoped_field_names,
    field_names_for_reference,
    native_name_to_faire_hint,
    render_field_reference,
)
from fair_ocean_agent.exports.faire import class_columns
from fair_ocean_agent.standards.faire_registry import build_faire_registry


def test_all_field_names_covers_every_group_and_fallback():
    names = all_field_names()
    for fields in FIELD_GROUPS.values():
        for f in fields:
            assert f.native_name in names
    for f in FALLBACK_NARRATIVE_FIELDS:
        assert f.native_name in names


def test_no_duplicate_field_names_across_groups():
    """Every native_name must be unique across groups and the fallback list
    -- a duplicate would mean the same concept was accidentally defined
    twice, and the prompt would show it twice too."""
    all_names = []
    for fields in FIELD_GROUPS.values():
        all_names.extend(f.native_name for f in fields)
    all_names.extend(f.native_name for f in FALLBACK_NARRATIVE_FIELDS)
    assert len(all_names) == len(set(all_names))


def test_native_names_never_equal_a_faire_hint_of_another_field():
    """Guards the actual bug this taxonomy was redesigned to prevent: a raw
    fact's identity (native_name) must never collide with a FAIRe field
    spelling used elsewhere as a hint -- otherwise a downstream consumer
    couldn't tell whether a given fact_type_candidate string was a native
    concept name or a standard's own vocabulary."""
    native_names = {f.native_name for fields in FIELD_GROUPS.values() for f in fields}
    hints = all_faire_hints()
    # A field is allowed to coincidentally share its own native_name and
    # faire_hint (e.g. "assay_name"/"assay_name") -- that's a real overlap
    # in spelling, not a collision between two *different* fields' identity
    # and hint.
    for fields in FIELD_GROUPS.values():
        for f in fields:
            other_hints = hints - ({f.faire_hint} if f.faire_hint else set())
            assert f.native_name not in other_hints, (
                f"{f.native_name!r} is used as another field's FAIRe hint -- ambiguous"
            )


_ALL_FLAGS = frozenset({"pcr_0_1", "probe_based_qPCR_ddPCR_assay_0_1"})


def test_render_field_reference_includes_every_field_and_group_header():
    rendered = render_field_reference(active_flags=_ALL_FLAGS)
    for group_name, fields in FIELD_GROUPS.items():
        assert f"{group_name}:" in rendered
        for f in fields:
            if f.faire_hint in LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS:
                continue
            assert f.native_name in rendered
            assert f.hint in rendered


def test_render_field_reference_includes_faire_hints():
    rendered = render_field_reference(active_flags=_ALL_FLAGS)
    for hint in all_faire_hints() - LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS:
        assert hint in rendered


def test_primer_volume_and_concentration_hints_cover_the_aggregate_each_primer_phrasing():
    """Regression guard: a real paper (ISME J 10.1093/ismejo/wrae013) states
    "...and 1 uL of each primer" -- one aggregate volume covering both
    primers, never naming forward/reverse separately. Confirmed live that
    the model reliably fills both forward_primer_volume/
    reverse_primer_volume from this phrasing once explicitly told to reuse
    an aggregate 'each primer'/'both primers' value for both fields."""
    rendered = render_field_reference(active_flags=frozenset({"pcr_0_1"}))
    for native_name in (
        "forward_primer_volume",
        "reverse_primer_volume",
        "forward_primer_concentration",
        "reverse_primer_concentration",
    ):
        line = next(line for line in rendered.splitlines() if line.startswith(f"- {native_name}:"))
        assert "each primer" in line


def test_render_field_reference_includes_fallback_section():
    rendered = render_field_reference()
    assert "General narrative fallback" in rendered
    for f in FALLBACK_NARRATIVE_FIELDS:
        if f.native_name in LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS:
            continue
        assert f.native_name in rendered


def test_fallback_fields_have_no_faire_hint():
    """Fallback fields are deliberately coarse narrative catch-alls, not a
    specific FAIRe field's atomic content -- they must never carry a hint
    that implies otherwise."""
    for f in FALLBACK_NARRATIVE_FIELDS:
        assert f.faire_hint is None


def test_field_groups_cover_concepts_named_in_the_milestone_8_request():
    """Regression guard for the concrete gap the user named: PCR volumes,
    primer concentrations, assay names, controls, replicate structure,
    thresholds, standard curves, taxonomy outputs. Checked via native_name
    (a raw fact's real identity), not FAIRe's own spelling."""
    names = all_field_names()
    requested = {
        "pcr_reaction_volume",  # PCR volumes
        "forward_primer_concentration",  # primer concentrations
        "assay_name",  # assay names
        "negative_control_type",  # controls
        "biological_replicate_count",  # replicate structure
        "quantification_cycle_threshold",  # thresholds
        "standard_curve_slope",  # standard curves
        "scientific_name",  # taxonomy outputs
    }
    missing = requested - names
    assert not missing, f"Concepts requested but missing from the taxonomy: {missing}"


def test_native_name_to_faire_hint_round_trips_a_known_field():
    mapping = native_name_to_faire_hint()
    assert mapping["annealing_temperature"] == "annealingTemp"
    assert mapping["standard_curve_r_squared"] == "r2"


# biological_rep is the one deliberate exception to "exclude or gate":
# search_flags.CONTROLLED_SEARCH_FIELDS's own "biological_rep" entry is
# itself unconditional (matches the real FAIRe schema, which has no
# conditional requirement for it -- a general sample-design concept, not a
# PCR-specific one), and gold case controls-replicates-001 proves the LLM
# checklist side must stay unconditionally active too (the benchmark only
# exercises the LLM path, and that gold case expects
# "biological_replicate_count" regardless of PCR content). Excluding the
# LLM version would silently break that passing gold case; gating it (as a
# prior round did) would wrongly hide it from non-PCR papers (e.g. shotgun
# metagenomics) that still report replicate counts. Unlike the
# adapter_forward/adapter_reverse duplicate this same guard caught (where
# one mechanism could rewrite/mangle a sequence the other reports
# verbatim), a plain replicate count has low real risk of the two
# mechanisms disagreeing harmfully -- this is accepted, deliberate
# redundancy for coverage, not an unreconciled gap.
#
# trim_method/min_len_tool/min_len_cutoff (LLM native names
# adapter_trimming_method/length_filtering_tool/minimum_read_length):
# CONTROLLED_SEARCH_FIELDS' own entries here are narrow, tool-name-specific
# regex matches (only "Trimmomatic"/"MINLEN"), not general free-text
# tool-name extraction. Confirmed on two real papers: excluding the LLM
# version would have lost adapter_trimming_method entirely for a paper
# using a custom Perl script (PeerJ 10.7717/peerj.333) or SeqPrep (ISME J
# 10.1093/ismejo/wrae013) -- neither Trimmomatic, so the deterministic
# detector alone would never catch them. On a real Trimmomatic paper (PLOS
# ONE 10.1371/journal.pone.0303937) both mechanisms already fire today and
# agree ("trimmomatic"/"trimmomatic", "500 bp"/"500 bp"), confirming this
# is the same low-conflict-risk pattern as biological_rep, not a genuine
# unreconciled gap.
_ACCEPTED_UNCONDITIONAL_OVERLAPS = frozenset(
    {"biological_rep", "trim_method", "min_len_tool", "min_len_cutoff"}
)


def test_controlled_search_field_overlaps_are_excluded_or_flag_gated():
    """Standing guard against the exact failure mode this task fixed:
    search_flags.CONTROLLED_SEARCH_FIELDS and this taxonomy can both cover
    the same real-world concept under two different mechanisms (a
    deterministic curated-term matcher and the LLM checklist). Every
    CONTROLLED_SEARCH_FIELDS term_name that also appears here (as a
    native_name or a faire_hint) must be handled deliberately -- either
    fully excluded (LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS, when there's no
    natural flag to gate on, e.g. seq_kit), flag-gated
    (required_any_flags, when the LLM should stay a richer complement,
    e.g. target_gene/thermocycler/commercial_master_mix/assay_type), or an
    explicitly documented accepted overlap
    (_ACCEPTED_UNCONDITIONAL_OVERLAPS, e.g. biological_rep) -- never left
    silently duplicated with no record of the decision. Catches the next
    new deterministic detector landing without its LLM-side twin being
    reconciled, the way assay_type/biological_rep both did mid-development
    of this very test."""
    from fair_ocean_agent.extraction.search_flags import CONTROLLED_SEARCH_FIELDS

    controlled_names = {f.term_name for f in CONTROLLED_SEARCH_FIELDS}
    for fields in FIELD_GROUPS.values():
        for f in fields:
            if f.native_name not in controlled_names and f.faire_hint not in controlled_names:
                continue
            handled = (
                f.faire_hint in LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS
                or bool(f.required_any_flags)
                or f.faire_hint in _ACCEPTED_UNCONDITIONAL_OVERLAPS
                or f.native_name in _ACCEPTED_UNCONDITIONAL_OVERLAPS
            )
            assert handled, (
                f"{f.native_name!r} (hint {f.faire_hint!r}) duplicates a "
                "search_flags.CONTROLLED_SEARCH_FIELDS entry but is neither "
                "excluded, flag-gated, nor a documented accepted overlap -- "
                "both mechanisms would fire for the same real fact with no "
                "record of that being intentional"
            )


def test_llm_judged_search_field_overlaps_are_excluded_or_flag_gated():
    """Same guard as test_controlled_search_field_overlaps_are_excluded_or_flag_gated,
    but for search_flags.LLM_JUDGED_SEARCH_FIELDS -- a second, separate
    deterministic-ish mechanism (a narrow, quote-anchored LLM pass over
    curated search_terms) that this taxonomy's general checklist can
    equally duplicate. Real example this guard would have caught up front:
    forward_sequencing_adapter/reverse_sequencing_adapter (general
    checklist) duplicated LLM_JUDGED_SEARCH_FIELDS's own
    adapter_forward/adapter_reverse entries -- two independent LLM calls
    each able to produce a fact for the same real adapter sequence."""
    from fair_ocean_agent.extraction.search_flags import LLM_JUDGED_SEARCH_FIELDS

    judged_names = {f.term_name for f in LLM_JUDGED_SEARCH_FIELDS}
    for fields in FIELD_GROUPS.values():
        for f in fields:
            if f.native_name not in judged_names and f.faire_hint not in judged_names:
                continue
            handled = (
                f.faire_hint in LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS
                or bool(f.required_any_flags)
                or f.faire_hint in _ACCEPTED_UNCONDITIONAL_OVERLAPS
                or f.native_name in _ACCEPTED_UNCONDITIONAL_OVERLAPS
            )
            assert handled, (
                f"{f.native_name!r} (hint {f.faire_hint!r}) duplicates a "
                "search_flags.LLM_JUDGED_SEARCH_FIELDS entry but is neither "
                "excluded, flag-gated, nor a documented accepted overlap -- "
                "two independent LLM calls would fire for the same real fact"
            )


def test_assay_scoped_field_names_is_subset_of_all_field_names():
    assert assay_scoped_field_names() <= all_field_names()


def test_assay_scoped_field_names_excludes_sample_and_control_type_fields():
    """biological_replicate_count is about a sample/treatment, not an
    assay. negative_control_type/positive_control_type map to FAIRe's
    sampleMetadata, not projectMetadata, so tagging them with an assay_tag
    would have no effect downstream -- excluded to keep the prompt
    simpler for zero lost benefit."""
    names = assay_scoped_field_names()
    assert "biological_replicate_count" not in names
    assert "negative_control_type" not in names
    assert "positive_control_type" not in names


def test_assay_scoped_field_names_includes_pcr_and_qpcr_fields():
    names = assay_scoped_field_names()
    assert "assay_name" in names
    assert "annealing_temperature" in names
    assert "pcr_replicate_count" in names
    assert "standard_curve_r_squared" in names


# --- Structured-first extraction: excluding already-resolved FAIRe fields ---
# (structured sources like NCBI/ENA/PANGAEA get a chance to resolve a field
# before the LLM is ever asked about it -- see extraction/text.py's
# resolved_faire_fields_for_study and workflow/handlers.py's wiring)


def test_render_field_reference_omits_excluded_faire_hints():
    rendered = render_field_reference(exclude_faire_hints=frozenset({"annealingTemp"}), active_flags=_ALL_FLAGS)
    # Substring containment isn't precise enough here: "second_pcr_annealing_temperature"
    # (a different field, different hint) legitimately contains
    # "annealing_temperature" as a substring of its own name.
    assert "- annealing_temperature:" not in rendered
    assert "[FAIRe hint: annealingTemp]" not in rendered
    # An unrelated concept in the same group must still be present.
    assert "target_gene" in rendered


def test_render_field_reference_omits_group_left_empty_by_exclusion():
    """DNA extraction's every faire_hint, if all excluded, must drop the
    whole group heading too -- an empty "DNA extraction:" heading with no
    bullets under it would confuse the model, not help it."""
    dna_hints = frozenset(f.faire_hint for f in FIELD_GROUPS["DNA extraction"])
    rendered = render_field_reference(exclude_faire_hints=dna_hints, active_flags=_ALL_FLAGS)
    assert "DNA extraction:" not in rendered
    # A different, non-excluded group must still render.
    assert "PCR / assay setup:" in rendered


def test_render_field_reference_keeps_nonexcluded_fallback_fields():
    """Dynamic hint exclusion must not remove unrelated fallbacks; the one
    fallback covered by the static low-value policy stays absent."""
    rendered = render_field_reference(exclude_faire_hints=all_faire_hints())
    assert "General narrative fallback" in rendered
    for f in FALLBACK_NARRATIVE_FIELDS:
        if f.native_name in LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS:
            assert f.native_name not in rendered
        else:
            assert f.native_name in rendered


def test_render_field_reference_with_no_exclusions_matches_default():
    assert render_field_reference(exclude_faire_hints=frozenset()) == render_field_reference()


def test_low_value_optional_fields_are_excluded_from_llm_only():
    expected_fields = frozenset(
        {
            "informationWithheld",
            "dataGeneralizations",
            "pcr_analysis_software",
            "pcr_method_additional",
            "pcr2_analysis_software",
            "pcr2_method_additional",
            "seq_method_additional",
            "woce_sect",
            "sequencing_location",
            "block_seq",
            "block_ref",
            "block_taxa",
            "inhibition_check_0_1",
            "inhibition_check",
            "samp_collect_method",
            "samp_store_method_additional",
            "assay_name",
            "lib_conc",
            "lib_conc_unit",
            "lib_conc_meth",
            "platform",
            "instrument",
            "lib_layout",
            "seq_kit",
            "adapter_forward",
            "adapter_reverse",
        }
    )
    rendered = render_field_reference()
    allowed_names = field_names_for_reference()

    assert LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS == expected_fields
    for field in expected_fields:
        if field in ("instrument", "platform"):
            # Too generic a substring: "instrument" appears inside an
            # unrelated hint ("...instrument/method used to measure DNA
            # concentration"), and "platform" is still legitimately present
            # via the still-active sequencing_platform *fallback* narrative
            # field (a different, unexcluded concept -- only the atomic
            # sequencing_platform_general native name was excluded).
            # test_platform_instrument_lib_layout_excluded_from_llm_checklist
            # below checks these precisely via field name, not substring.
            continue
        assert field not in rendered
    assert "sequencing_location" not in rendered
    assert "sequencing_location" not in allowed_names
    assert "PCR_amplification_conditions" not in rendered
    assert "PCR_amplification_conditions" not in allowed_names
    assert LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS == frozenset(
        {"PCR_amplification_conditions", "collection_method", "storage_conditions", "environmental_context"}
    )


def test_llm_exclusions_do_not_remove_faire_registry_or_export_fields():
    # woce_sect is guarded by the LLM policy for forward compatibility, but
    # is not a field in the authoritative FAIRe v1.0.2 schema/workbook.
    upstream_fields = LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS - {"woce_sect"}
    registry_fields = {term["upstream_field_name"] for term in build_faire_registry()}
    # This exclusion policy now spans projectMetadata, sampleMetadata, and
    # experimentRunMetadata fields (not just projectMetadata) -- check each
    # excluded field is still a real, exported column in whichever of
    # those three tables it actually belongs to.
    export_fields = (
        set(class_columns("projectMetadata"))
        | set(class_columns("sampleMetadata"))
        | set(class_columns("experimentRunMetadata"))
    )

    assert upstream_fields <= registry_fields
    assert upstream_fields <= export_fields
    assert "woce_sect" not in registry_fields
    assert "woce_sect" not in export_fields


def test_sample_metadata_llm_checklist_is_narrowed_to_location_date_depth():
    """Regression guard for an explicit user review of which sample-level
    fields are realistically findable in prose: only coordinates
    (decimalLatitude/decimalLongitude), collection_date (eventDate), and
    depth (minimumDepthInMeters/maximumDepthInMeters) remain. Everything
    else about a sample -- including samp_category (sample vs. control),
    which isn't even in this taxonomy at all -- is structured-source-only."""
    names = field_names_for_reference()
    assert {"collection_date", "depth", "coordinates"} <= names
    for excluded in (
        "sample_collection_method",
        "sample_storage_conditions",
        "collection_method",
        "storage_conditions",
        "environmental_context",
    ):
        assert excluded not in names
    assert "samp_category" not in all_field_names()


def test_platform_instrument_lib_layout_excluded_from_llm_checklist():
    """Regression guard for a follow-up NOAA checklist review: these are
    real projectMetadata fields (not experimentRunMetadata, contrary to an
    earlier decision in this module's history), already 100% covered by
    ENA's own structured instrument_platform/instrument_model/
    library_layout facts wherever a study has one -- excluded from the LLM
    checklist, kept in the taxonomy/registry/exports for structured
    adapters."""
    names = field_names_for_reference()
    for excluded in ("sequencing_platform_general", "sequencing_instrument", "library_layout"):
        assert excluded not in names
    assert {"sequencing_platform_general", "sequencing_instrument", "library_layout"} <= all_field_names()
