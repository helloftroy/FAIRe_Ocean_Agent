"""The FAIRe-aware raw fact taxonomy: the exact set of atomic fields
`extraction/text.py`'s prompt asks the model to look for, grouped the way
FAIRe itself groups them (PCR/assay, controls, qPCR/standard curve,
sequencing, bioinformatics, taxonomic assignment).

This is the single source of truth for that taxonomy -- the prompt is
rendered from it (`render_field_reference`), so the list of fields a
maintainer edits here is exactly the list a reader sees in the prompt and
the list `all_field_names()` exposes for tests. Before this module
existed, the extraction prompt asked the model to find whatever it
thought relevant under an open-ended `fact_type_candidate` of its own
choosing (e.g. "PCR_amplification_conditions" as one blob) -- useful for
never missing something explicitly stated, but useless for populating
FAIRe's atomic fields (PCR volumes, primer concentrations, annealing
temperature, standard-curve slope/intercept, ...), since nothing forced
the model's vocabulary to line up with FAIRe's own field names. Naming
each field here exactly the way FAIRe's schema.yaml does (see
schemas/faire/README.md) means a fact this extracts can map onto FAIRe
with an exact-label rule instead of a fuzzy one -- see mapping/rules.py's
own docstring for how that mapping layer treats a fact's name.

Field `hint` text is a short, prompt-facing paraphrase of FAIRe's own
slot description (schemas/faire/schema.yaml), not a copy of the full
official wording -- kept brief deliberately, since ~70 fields' worth of
full FAIRe prose would bloat the prompt for no benefit; the authoritative
long-form definition is always schemas/faire/schema.yaml itself.

Not every FAIRe field belongs here. Left out on purpose:
- Fields that are inherently per-sample/per-replicate bookkeeping rather
  than a single paper-level narrative fact (`technical_rep_id`,
  `biological_rep_relation`, `samp_name`, `pcr_plate_id`) -- a paper's
  Methods text doesn't usually state "this specific replicate is
  numbered 3," and forcing the model to invent one would risk
  fabrication, not extraction.
- Fields already well-covered by structured NCBI/ENA repository data
  (`collection_date`, `lat_lon`, `depth`, `env_broad_scale`, ...) --
  extending atomic coverage here is about what repository metadata
  *doesn't* give: assay/PCR/sequencing/bioinformatics/taxonomy detail
  that only ever appears in a paper's own Methods section.
- Enum-valued instrument-set fields unlikely to appear as free prose
  (`automaticThresholdQuantificationCycle`, `automaticBaselineValue` --
  "was the threshold set automatically or manually" is qPCR-software
  metadata, not something a paper's Methods section narrates).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaireExtractionField:
    name: str
    hint: str
    example: str | None = None


FIELD_GROUPS: dict[str, tuple[FaireExtractionField, ...]] = {
    "DNA extraction": (
        FaireExtractionField("nucl_acid_ext_kit", "name of the extraction kit used", "DNeasy PowerWater Kit"),
        FaireExtractionField("nucl_acid_ext_lysis", "lysis method (e.g. physical, chemical, enzymatic)", "bead-beating"),
        FaireExtractionField("nucl_acid_ext_sep", "how DNA was separated/purified (e.g. spin column, magnetic beads)"),
        FaireExtractionField("samp_vol_we_dna_ext", "volume or mass of sample processed for extraction", "500 mL"),
        FaireExtractionField("samp_vol_we_dna_ext_unit", "unit for samp_vol_we_dna_ext", "mL"),
        FaireExtractionField("concentration", "DNA concentration after extraction", "12.4 ng/uL"),
        FaireExtractionField("concentration_method", "instrument/method used to measure DNA concentration", "Qubit fluorometer"),
        FaireExtractionField("ratioOfAbsorbance260_280", "A260/A280 absorbance ratio (DNA purity)", "1.85"),
        FaireExtractionField("dna_cleanup_method", "DNA clean-up/purification method or kit name"),
    ),
    "PCR / assay setup": (
        FaireExtractionField("assay_name", "a short name/identifier the paper gives its assay", "18S-V4-eukaryote"),
        FaireExtractionField("assay_type", "targeted, metabarcoding, or other detection approach"),
        FaireExtractionField("target_gene", "targeted gene or locus", "16S rRNA"),
        FaireExtractionField("target_subfragment", "targeted hypervariable subregion", "V4"),
        FaireExtractionField("pcr_primer_forward", "forward primer sequence, 5' to 3'"),
        FaireExtractionField("pcr_primer_reverse", "reverse primer sequence, 5' to 3'"),
        FaireExtractionField("pcr_primer_name_forward", "forward primer's name", "515F"),
        FaireExtractionField("pcr_primer_name_reverse", "reverse primer's name", "926R"),
        FaireExtractionField("pcr_primer_conc_forward", "forward primer's stock concentration", "10 uM"),
        FaireExtractionField("pcr_primer_conc_reverse", "reverse primer's stock concentration", "10 uM"),
        FaireExtractionField("pcr_primer_vol_forward", "forward primer volume per reaction", "1 uL"),
        FaireExtractionField("pcr_primer_vol_reverse", "reverse primer volume per reaction", "1 uL"),
        FaireExtractionField("ampliconSize", "expected amplicon length in base pairs, excluding primers/adapters", "411 bp"),
        FaireExtractionField("amplificationReactionVolume", "total PCR reaction volume", "25 uL"),
        FaireExtractionField("pcr_dna_vol", "template DNA volume added per PCR reaction", "2 uL"),
        FaireExtractionField("thermocycler", "thermocycler manufacturer and model"),
        FaireExtractionField("annealingTemp", "PCR annealing temperature", "55C"),
        FaireExtractionField("pcr_cycles", "number of PCR cycles", "35"),
        FaireExtractionField("pcr_cond", "full description of PCR reaction conditions/thermal profile"),
        FaireExtractionField("commercial_mm", "commercial master mix name/brand, if one was used"),
        FaireExtractionField("custom_mm", "custom master mix composition, if a commercial one was not used"),
    ),
    "Controls & replicates": (
        FaireExtractionField("neg_cont_type", "type of negative control used", "extraction blank"),
        FaireExtractionField("pos_cont_type", "type of positive control used", "synthetic DNA standard"),
        FaireExtractionField("biological_rep", "number of biological replicates collected per sample/treatment", "3"),
        FaireExtractionField("pcr_rep", "number of PCR technical replicates per sample", "3"),
    ),
    "qPCR / standard curve": (
        FaireExtractionField("thresholdQuantificationCycle", "the fluorescence threshold value used for Cq/Ct"),
        FaireExtractionField("quantificationCycle", "a reported quantification cycle (Cq/Ct) value"),
        FaireExtractionField("std_type", "type of qPCR standard used (e.g. gBlock, plasmid, synthetic gene fragment)"),
        FaireExtractionField("std_conc", "input quantity of the qPCR standard"),
        FaireExtractionField("std_conc_unit", "unit for std_conc", "copies/uL"),
        FaireExtractionField("std_source", "source/supplier of the qPCR standard"),
        FaireExtractionField("slope", "slope of the qPCR standard curve"),
        FaireExtractionField("intercept", "intercept of the qPCR standard curve"),
        FaireExtractionField("r2", "R-squared value of the qPCR standard curve"),
        FaireExtractionField("efficiency", "qPCR amplification efficiency (%)", "98%"),
        FaireExtractionField("estimatedNumberOfCopies", "estimated concentration of target molecules/copies"),
        FaireExtractionField("estimatedNumberOfCopies_unit", "unit for estimatedNumberOfCopies", "copies/reaction"),
        FaireExtractionField("estimatedNumberOfCopies_method", "method used to estimate target copy number"),
        FaireExtractionField("pcr_assay_lod", "assay's limit of detection (LOD)"),
        FaireExtractionField("pcr_assay_lod_unit", "unit for pcr_assay_lod"),
        FaireExtractionField("pcr_assay_loq", "assay's limit of quantification (LOQ)"),
        FaireExtractionField("pcr_assay_loq_unit", "unit for pcr_assay_loq"),
    ),
    "Sequencing / library prep": (
        FaireExtractionField("platform", "general sequencing platform (e.g. Illumina, PacBio, Oxford Nanopore)"),
        FaireExtractionField("instrument", "specific sequencer manufacturer and model", "Illumina MiSeq"),
        FaireExtractionField("seq_kit", "sequencing kit name", "MiSeq Reagent Kit v3"),
        FaireExtractionField("lib_layout", "single, paired, or other read layout"),
        FaireExtractionField("adapter_forward", "forward sequencing adapter sequence"),
        FaireExtractionField("adapter_reverse", "reverse sequencing adapter sequence"),
        FaireExtractionField("lib_conc", "concentration of the prepared sequencing library"),
        FaireExtractionField("lib_conc_meth", "method used to estimate library concentration"),
        FaireExtractionField("lib_conc_unit", "unit for lib_conc"),
        FaireExtractionField("phix_perc", "% PhiX spiked into the sequencing run", "10%"),
        FaireExtractionField("sequencing_location", "facility/lab where sequencing was performed"),
    ),
    "Bioinformatics workflow": (
        FaireExtractionField("trim_method", "primer/adapter trimming method, including software and version"),
        FaireExtractionField("trim_param", "trimming parameters/cutoffs used, if non-default"),
        FaireExtractionField("demux_tool", "software (with version) used to demultiplex reads"),
        FaireExtractionField("merge_tool", "software (with version) used to merge paired-end reads"),
        FaireExtractionField("merge_min_overlap", "minimum overlap required to merge paired-end reads", "12 bp"),
        FaireExtractionField("error_rate_tool", "software used for denoising/error-correction (e.g. DADA2)"),
        FaireExtractionField("min_len_cutoff", "minimum read length threshold used for filtering"),
        FaireExtractionField("min_len_tool", "software used to filter reads by length"),
        FaireExtractionField("chimera_check_method", "chimera detection approach, including software/version"),
        FaireExtractionField("otu_clust_tool", "software (with version) used for OTU/ASV clustering"),
        FaireExtractionField("otu_clust_cutoff", "percent similarity threshold used for OTU/ASV clustering", "97%"),
        FaireExtractionField("otu_db", "reference database(s) used for taxonomic assignment, with version", "SILVA 138"),
        FaireExtractionField("tax_assign_cat", "taxonomic assignment approach (e.g. BLAST, naive Bayesian classifier)"),
        FaireExtractionField("sop_bioinformatics", "reference/link/DOI to the bioinformatics standard operating procedure"),
    ),
    "Taxonomic assignment output": (
        FaireExtractionField("scientificName", "a taxon name the paper reports as assigned/detected"),
        FaireExtractionField("taxonRank", "the taxonomic rank of an assigned name (e.g. species, genus)"),
        FaireExtractionField("kingdom", "kingdom-level name for a reported taxon"),
        FaireExtractionField("phylum", "phylum-level name for a reported taxon"),
        FaireExtractionField("class", "class-level name for a reported taxon"),
        FaireExtractionField("order", "order-level name for a reported taxon"),
        FaireExtractionField("family", "family-level name for a reported taxon"),
        FaireExtractionField("genus", "genus-level name for a reported taxon"),
        FaireExtractionField("specificEpithet", "the species epithet of a reported scientificName"),
        FaireExtractionField("percent_match", "% sequence identity to a reference used for taxonomic assignment"),
        FaireExtractionField("percent_query_cover", "% query coverage against a reference sequence"),
        FaireExtractionField("confidence_score", "confidence/bootstrap score for a taxonomic assignment"),
    ),
}


# Coarse, narrative fallback fact types -- kept for when a paper describes
# a concept in prose without stating any of the atomic fields above (e.g.
# "PCR was performed as previously described [ref]" gives no annealing
# temperature or cycle count to extract atomically, but is still a real,
# evidence-backed fact worth recording under one of these). Pre-dates this
# module (Milestone 4); mapping/rules.py already has rules treating these
# as free-text fallbacks onto FAIRe's own "*_method_additional" fields.
FALLBACK_NARRATIVE_FIELDS: tuple[FaireExtractionField, ...] = (
    FaireExtractionField("DNA_extraction_method", "general DNA extraction narrative, if no atomic kit/method field applies"),
    FaireExtractionField("PCR_amplification_conditions", "general PCR narrative, if no atomic PCR field applies"),
    FaireExtractionField("storage_conditions", "how samples were stored (temperature, preservative, duration)"),
    FaireExtractionField("sequencing_platform", "sequencing platform mentioned only in a general narrative sentence"),
    FaireExtractionField("collection_method", "how samples were physically collected (device, gear, procedure)"),
    FaireExtractionField("environmental_context", "general environmental setting narrative not cleanly one of env_broad_scale/env_local_scale/env_medium"),
)


def all_field_names() -> frozenset[str]:
    names = {f.name for fields in FIELD_GROUPS.values() for f in fields}
    names |= {f.name for f in FALLBACK_NARRATIVE_FIELDS}
    return frozenset(names)


def render_field_reference() -> str:
    """Renders FIELD_GROUPS + FALLBACK_NARRATIVE_FIELDS as the itemized,
    group-headed checklist `extraction/text.py`'s prompt embeds -- one
    source of truth, so the taxonomy a maintainer edits here is exactly
    what the model is shown."""
    lines: list[str] = []
    for group_name, fields in FIELD_GROUPS.items():
        lines.append(f"{group_name}:")
        for f in fields:
            example = f" (e.g. \"{f.example}\")" if f.example else ""
            lines.append(f"- {f.name}: {f.hint}{example}")
    lines.append("General narrative fallback (use only if no atomic field above applies):")
    for f in FALLBACK_NARRATIVE_FIELDS:
        lines.append(f"- {f.name}: {f.hint}")
    return "\n".join(lines)
