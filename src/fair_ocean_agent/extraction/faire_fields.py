"""The extraction concept taxonomy: the set of source-native concepts
`extraction/text.py`'s prompt asks the model to look for, grouped the way
FAIRe groups them (PCR/assay, controls, qPCR/standard curve, sequencing,
bioinformatics, taxonomic assignment) for organizational convenience only.

**A raw fact's identity is never a standard's vocabulary.** Each entry's
`native_name` -- what `fact_type_candidate` actually gets set to -- is a
plain, standard-agnostic description of the concept (`annealing_temperature`,
`r_squared`, `reference_database`), not FAIRe's own slot spelling
(`annealingTemp`, `r2`, `otu_db`). `faire_hint` records which FAIRe field
this concept *might* correspond to, but that hint is carried as a
separate, optional suggestion (`candidate_standard_fields` on the
extracted fact, stored in `RawFact.confidence_metadata` -- see
extraction/text.py) -- never folded into the fact's own name. A raw fact
produced under this taxonomy is exactly as standard-independent as one
produced by any repository adapter; standardization/mapping onto FAIRe
(or any other standard) stays mapping/faire.py's job, a separate
downstream step over raw_facts, same as always.

An earlier version of this module used FAIRe's own slot names directly as
`fact_type_candidate` (`annealingTemp` as the fact's own identity, not a
hint about it) -- that coupled a raw fact's identity to one specific
standard, which is exactly what raw_facts elsewhere in this pipeline
deliberately avoids (a repository-adapter fact's `fact_type_candidate` is
never phrased in Darwin Core or MIxS's own vocabulary spelling either).
Fixed by splitting `native_name` (the fact's real identity) from
`faire_hint` (a suggestion about it) once a user caught the coupling
before any real data was built on it.

This is still the single source of truth the prompt is rendered from
(`render_field_reference`) and tests validate against
(`all_field_names`/`all_faire_hints`) -- editing the taxonomy here is
editing what the model is shown and what a gold case is checked against,
in one place.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaireExtractionField:
    native_name: str
    hint: str
    faire_hint: str | None = None
    example: str | None = None


FIELD_GROUPS: dict[str, tuple[FaireExtractionField, ...]] = {
    "DNA extraction": (
        FaireExtractionField("dna_extraction_kit", "name of the extraction kit used", "nucl_acid_ext_kit", "DNeasy PowerWater Kit"),
        FaireExtractionField("dna_lysis_method", "lysis method (e.g. physical, chemical, enzymatic)", "nucl_acid_ext_lysis", "bead-beating"),
        FaireExtractionField("dna_separation_method", "how DNA was separated/purified (e.g. spin column, magnetic beads)", "nucl_acid_ext_sep"),
        FaireExtractionField("sample_volume_for_extraction", "volume or mass of sample processed for extraction", "samp_vol_we_dna_ext", "500 mL"),
        FaireExtractionField("sample_volume_for_extraction_unit", "unit for sample_volume_for_extraction", "samp_vol_we_dna_ext_unit", "mL"),
        FaireExtractionField("dna_concentration", "DNA concentration after extraction", "concentration", "12.4 ng/uL"),
        FaireExtractionField("dna_concentration_method", "instrument/method used to measure DNA concentration", "concentration_method", "Qubit fluorometer"),
        FaireExtractionField("absorbance_260_280_ratio", "A260/A280 absorbance ratio (DNA purity)", "ratioOfAbsorbance260_280", "1.85"),
        FaireExtractionField("dna_cleanup_method", "DNA clean-up/purification method or kit name", "dna_cleanup_method"),
    ),
    "PCR / assay setup": (
        FaireExtractionField("assay_name", "a short name/identifier the paper gives its assay", "assay_name", "18S-V4-eukaryote"),
        FaireExtractionField("assay_type", "targeted, metabarcoding, or other detection approach", "assay_type"),
        FaireExtractionField("target_gene", "targeted gene or locus", "target_gene", "16S rRNA"),
        FaireExtractionField("target_subfragment", "targeted hypervariable subregion", "target_subfragment", "V4"),
        FaireExtractionField("forward_primer_sequence", "forward primer sequence, 5' to 3'", "pcr_primer_forward"),
        FaireExtractionField("reverse_primer_sequence", "reverse primer sequence, 5' to 3'", "pcr_primer_reverse"),
        FaireExtractionField("forward_primer_name", "forward primer's name", "pcr_primer_name_forward", "515F"),
        FaireExtractionField("reverse_primer_name", "reverse primer's name", "pcr_primer_name_reverse", "926R"),
        FaireExtractionField("forward_primer_concentration", "forward primer's stock concentration", "pcr_primer_conc_forward", "10 uM"),
        FaireExtractionField("reverse_primer_concentration", "reverse primer's stock concentration", "pcr_primer_conc_reverse", "10 uM"),
        FaireExtractionField("forward_primer_volume", "forward primer volume per reaction", "pcr_primer_vol_forward", "1 uL"),
        FaireExtractionField("reverse_primer_volume", "reverse primer volume per reaction", "pcr_primer_vol_reverse", "1 uL"),
        FaireExtractionField("amplicon_size", "expected amplicon length in base pairs, excluding primers/adapters", "ampliconSize", "411 bp"),
        FaireExtractionField("pcr_reaction_volume", "total PCR reaction volume", "amplificationReactionVolume", "25 uL"),
        FaireExtractionField("template_dna_volume", "template DNA volume added per PCR reaction", "pcr_dna_vol", "2 uL"),
        FaireExtractionField("thermocycler", "thermocycler manufacturer and model", "thermocycler"),
        FaireExtractionField("annealing_temperature", "PCR annealing temperature", "annealingTemp", "55C"),
        FaireExtractionField("pcr_cycle_count", "number of PCR cycles", "pcr_cycles", "35"),
        FaireExtractionField("pcr_conditions", "full description of PCR reaction conditions/thermal profile", "pcr_cond"),
        FaireExtractionField("commercial_master_mix", "commercial master mix name/brand, if one was used", "commercial_mm"),
        FaireExtractionField("custom_master_mix", "custom master mix composition, if a commercial one was not used", "custom_mm"),
    ),
    "Controls & replicates": (
        FaireExtractionField("negative_control_type", "type of negative control used", "neg_cont_type", "extraction blank"),
        FaireExtractionField("positive_control_type", "type of positive control used", "pos_cont_type", "synthetic DNA standard"),
        FaireExtractionField("biological_replicate_count", "number of biological replicates collected per sample/treatment", "biological_rep", "3"),
        FaireExtractionField("pcr_replicate_count", "number of PCR technical replicates per sample", "pcr_rep", "3"),
    ),
    "qPCR / standard curve": (
        FaireExtractionField("quantification_cycle_threshold", "the fluorescence threshold value used for Cq/Ct", "thresholdQuantificationCycle"),
        FaireExtractionField("quantification_cycle", "a reported quantification cycle (Cq/Ct) value", "quantificationCycle"),
        FaireExtractionField("qpcr_standard_type", "type of qPCR standard used (e.g. gBlock, plasmid, synthetic gene fragment)", "std_type"),
        FaireExtractionField("qpcr_standard_concentration", "input quantity of the qPCR standard", "std_conc"),
        FaireExtractionField("qpcr_standard_concentration_unit", "unit for qpcr_standard_concentration", "std_conc_unit", "copies/uL"),
        FaireExtractionField("qpcr_standard_source", "source/supplier of the qPCR standard", "std_source"),
        FaireExtractionField("standard_curve_slope", "slope of the qPCR standard curve", "slope"),
        FaireExtractionField("standard_curve_intercept", "intercept of the qPCR standard curve", "intercept"),
        FaireExtractionField("standard_curve_r_squared", "R-squared value of the qPCR standard curve", "r2"),
        FaireExtractionField("qpcr_amplification_efficiency", "qPCR amplification efficiency (%)", "efficiency", "98%"),
        FaireExtractionField("estimated_copy_number", "estimated concentration of target molecules/copies", "estimatedNumberOfCopies"),
        FaireExtractionField("estimated_copy_number_unit", "unit for estimated_copy_number", "estimatedNumberOfCopies_unit", "copies/reaction"),
        FaireExtractionField("estimated_copy_number_method", "method used to estimate target copy number", "estimatedNumberOfCopies_method"),
        FaireExtractionField("assay_limit_of_detection", "assay's limit of detection (LOD)", "pcr_assay_lod"),
        FaireExtractionField("assay_limit_of_detection_unit", "unit for assay_limit_of_detection", "pcr_assay_lod_unit"),
        FaireExtractionField("assay_limit_of_quantification", "assay's limit of quantification (LOQ)", "pcr_assay_loq"),
        FaireExtractionField("assay_limit_of_quantification_unit", "unit for assay_limit_of_quantification", "pcr_assay_loq_unit"),
    ),
    "Sequencing / library prep": (
        FaireExtractionField("sequencing_platform_general", "general sequencing platform (e.g. Illumina, PacBio, Oxford Nanopore)", "platform"),
        FaireExtractionField("sequencing_instrument", "specific sequencer manufacturer and model", "instrument", "Illumina MiSeq"),
        FaireExtractionField("sequencing_kit", "sequencing kit name", "seq_kit", "MiSeq Reagent Kit v3"),
        FaireExtractionField("library_layout", "single, paired, or other read layout", "lib_layout"),
        FaireExtractionField("forward_sequencing_adapter", "forward sequencing adapter sequence", "adapter_forward"),
        FaireExtractionField("reverse_sequencing_adapter", "reverse sequencing adapter sequence", "adapter_reverse"),
        FaireExtractionField("library_concentration", "concentration of the prepared sequencing library", "lib_conc"),
        FaireExtractionField("library_concentration_method", "method used to estimate library concentration", "lib_conc_meth"),
        FaireExtractionField("library_concentration_unit", "unit for library_concentration", "lib_conc_unit"),
        FaireExtractionField("phix_percentage", "% PhiX spiked into the sequencing run", "phix_perc", "10%"),
        FaireExtractionField("sequencing_location", "facility/lab where sequencing was performed", "sequencing_location"),
    ),
    "Bioinformatics workflow": (
        FaireExtractionField("adapter_trimming_method", "primer/adapter trimming method, including software and version", "trim_method"),
        FaireExtractionField("adapter_trimming_parameters", "trimming parameters/cutoffs used, if non-default", "trim_param"),
        FaireExtractionField("demultiplexing_tool", "software (with version) used to demultiplex reads", "demux_tool"),
        FaireExtractionField("read_merging_tool", "software (with version) used to merge paired-end reads", "merge_tool"),
        FaireExtractionField("read_merge_minimum_overlap", "minimum overlap required to merge paired-end reads", "merge_min_overlap", "12 bp"),
        FaireExtractionField("denoising_tool", "software used for denoising/error-correction (e.g. DADA2)", "error_rate_tool"),
        FaireExtractionField("minimum_read_length", "minimum read length threshold used for filtering", "min_len_cutoff"),
        FaireExtractionField("length_filtering_tool", "software used to filter reads by length", "min_len_tool"),
        FaireExtractionField("chimera_detection_method", "chimera detection approach, including software/version", "chimera_check_method"),
        FaireExtractionField("clustering_tool", "software (with version) used for OTU/ASV clustering", "otu_clust_tool"),
        FaireExtractionField("clustering_similarity_threshold", "percent similarity threshold used for OTU/ASV clustering", "otu_clust_cutoff", "97%"),
        FaireExtractionField("reference_database", "reference database(s) used for taxonomic assignment, with version", "otu_db", "SILVA 138"),
        FaireExtractionField("taxonomic_assignment_method", "taxonomic assignment approach (e.g. BLAST, naive Bayesian classifier)", "tax_assign_cat"),
        FaireExtractionField("bioinformatics_sop_reference", "reference/link/DOI to the bioinformatics standard operating procedure", "sop_bioinformatics"),
    ),
    "Taxonomic assignment output": (
        FaireExtractionField("scientific_name", "a taxon name the paper reports as assigned/detected", "scientificName"),
        FaireExtractionField("taxon_rank", "the taxonomic rank of an assigned name (e.g. species, genus)", "taxonRank"),
        FaireExtractionField("taxon_kingdom", "kingdom-level name for a reported taxon", "kingdom"),
        FaireExtractionField("taxon_phylum", "phylum-level name for a reported taxon", "phylum"),
        FaireExtractionField("taxon_class", "class-level name for a reported taxon", "class"),
        FaireExtractionField("taxon_order", "order-level name for a reported taxon", "order"),
        FaireExtractionField("taxon_family", "family-level name for a reported taxon", "family"),
        FaireExtractionField("taxon_genus", "genus-level name for a reported taxon", "genus"),
        FaireExtractionField("species_epithet", "the species epithet of a reported scientific name", "specificEpithet"),
        FaireExtractionField("percent_sequence_identity", "% sequence identity to a reference used for taxonomic assignment", "percent_match"),
        FaireExtractionField("percent_query_coverage", "% query coverage against a reference sequence", "percent_query_cover"),
        FaireExtractionField("taxonomic_assignment_confidence", "confidence/bootstrap score for a taxonomic assignment", "confidence_score"),
    ),
}


# Coarse, narrative fallback fact types -- kept for when a paper describes
# a concept in prose without stating any of the atomic fields above (e.g.
# "PCR was performed as previously described [ref]" gives no annealing
# temperature or cycle count to extract atomically, but is still a real,
# evidence-backed fact worth recording under one of these). Pre-dates this
# module (Milestone 4); mapping/rules.py already has rules treating these
# as free-text fallbacks onto FAIRe's own "*_method_additional" fields.
# No faire_hint: these are deliberately coarse narrative catch-alls, not a
# specific FAIRe field's atomic content.
FALLBACK_NARRATIVE_FIELDS: tuple[FaireExtractionField, ...] = (
    FaireExtractionField("DNA_extraction_method", "general DNA extraction narrative, if no atomic kit/method field applies"),
    FaireExtractionField("PCR_amplification_conditions", "general PCR narrative, if no atomic PCR field applies"),
    FaireExtractionField("storage_conditions", "how samples were stored (temperature, preservative, duration)"),
    FaireExtractionField("sequencing_platform", "sequencing platform mentioned only in a general narrative sentence"),
    FaireExtractionField("collection_method", "how samples were physically collected (device, gear, procedure)"),
    FaireExtractionField("environmental_context", "general environmental setting narrative not cleanly one of env_broad_scale/env_local_scale/env_medium"),
)


def all_field_names() -> frozenset[str]:
    """Every valid `fact_type_candidate` this taxonomy defines -- always a
    native_name, never a faire_hint (a hint is never a fact's own
    identity)."""
    names = {f.native_name for fields in FIELD_GROUPS.values() for f in fields}
    names |= {f.native_name for f in FALLBACK_NARRATIVE_FIELDS}
    return frozenset(names)


def all_faire_hints() -> frozenset[str]:
    """Every FAIRe field name this taxonomy can suggest as a
    `candidate_standard_fields` hint -- for validating that a hint the
    model returns is one this taxonomy actually knows about."""
    return frozenset(f.faire_hint for fields in FIELD_GROUPS.values() for f in fields if f.faire_hint)


def native_name_to_faire_hint() -> dict[str, str]:
    """Maps a taxonomy native_name to its FAIRe hint, where one exists --
    lets a caller reconstruct the intended hint if a model omits
    `candidate_standard_fields` for a field this taxonomy knows a hint
    for."""
    return {f.native_name: f.faire_hint for fields in FIELD_GROUPS.values() for f in fields if f.faire_hint}


def render_field_reference(exclude_faire_hints: frozenset[str] = frozenset()) -> str:
    """Renders FIELD_GROUPS + FALLBACK_NARRATIVE_FIELDS as the itemized,
    group-headed checklist `extraction/text.py`'s prompt embeds -- one
    source of truth, so the taxonomy a maintainer edits here is exactly
    what the model is shown. Each line shows the native_name (what
    fact_type_candidate must be set to) and, in parentheses, which FAIRe
    field it hints at (what candidate_standard_fields may suggest) --
    clearly two different things, never merged into one name.

    `exclude_faire_hints` drops any FIELD_GROUPS entry whose faire_hint is
    already in that set -- used to skip asking the model about concepts
    already resolved from structured sources (NCBI/ENA/PANGAEA/...) for a
    given study, before ever calling the LLM (see
    extraction/text.py's resolved_faire_fields_for_study). An entry with no
    faire_hint can never match and always stays; FALLBACK_NARRATIVE_FIELDS
    (no faire_hint at all, by design) are never filtered by this -- there's
    no structured-field correspondence to check them against. A group left
    with zero remaining entries after filtering is omitted entirely rather
    than rendered as an empty heading."""
    lines: list[str] = []
    for group_name, fields in FIELD_GROUPS.items():
        remaining = [f for f in fields if f.faire_hint not in exclude_faire_hints]
        if not remaining:
            continue
        lines.append(f"{group_name}:")
        for f in remaining:
            example = f" (e.g. \"{f.example}\")" if f.example else ""
            hint_note = f" [FAIRe hint: {f.faire_hint}]" if f.faire_hint else ""
            lines.append(f"- {f.native_name}: {f.hint}{example}{hint_note}")
    lines.append("General narrative fallback (use only if no concept above applies; no FAIRe hint):")
    for f in FALLBACK_NARRATIVE_FIELDS:
        lines.append(f"- {f.native_name}: {f.hint}")
    return "\n".join(lines)
