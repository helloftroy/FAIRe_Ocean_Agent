"""PCR/library-prep/bioinformatics methods-text categorization: the
keyword data and deterministic (no-LLM) building blocks for a two-stage
categorize-then-extract pipeline, per an explicit user design (see
README's "Section-categorization pipeline for PCR-family fields" once
written up).

Three deterministic pieces live here, all directly testable without an
LLM:

1. `SECTION_CATEGORIES` -- the keyword taxonomy itself, one entry per
   methods-text category (PCR1, targeted qPCR/ddPCR, PCR2/indexing,
   assay definition, library prep + sequencing, raw read preprocessing,
   OTU/ASV generation + filtering, taxonomic assignment). More categories
   (data availability, copyright, ...) get appended here as they're
   defined -- nothing else in this module needs to change shape when
   that happens.
2. `split_into_paragraphs` -- paragraph-level splitting with a structural
   pre-filter that drops reference-list and figure/table-caption
   paragraphs *before* any keyword matching runs. Confirmed necessary
   against real data: a real PNAS supplementary-methods bibliography
   entry ("27. P. Engstrom... Anaerobic ammonium oxidation...") matched
   the PCR1 category's own "amplicon"/"primer" keywords purely because a
   cited paper's *title* mentioned them -- citation text is never this
   paper's own methods, regardless of what it happens to mention.
3. `group_sentences_into_category_runs` -- the run-grouping algorithm for
   Stage 2.5 of the pipeline: once each sentence in a flagged paragraph
   has been tagged (by the LLM, Stage 2 -- not built here) with zero or
   more category names, this groups each category's own sentences into
   text runs. A run starts at a sentence tagged with that category and
   extends through any untagged (bridging/connective) sentences, ending
   the instant a sentence tagged with a *different* category appears.
   Multiple runs of the same category within one paragraph (e.g.
   cat1 -> cat2 -> cat1) are concatenated together, with the interrupting
   category's own text excluded throughout -- an explicit user
   specification, not a guess.

Also here: `detect_section_categories_present`, a lightweight, paragraph-
gated (same reference/caption filtering as above) presence check emitting
one `<category>_0_1` RawFactCandidate per category found anywhere in the
supplied texts. This is deliberately a *diagnostic* signal for tuning the
keyword lists against real papers ("if sections are being detected even
though they aren't there I'll have to constrain the list a bit" -- the
user's own framing), not itself part of the eventual field-extraction
path (that's Stage 1's job, wired separately once Stage 2/3 exist). It
reuses the SAME paragraph/reference-filtering gate as the real pipeline
so a false "detected" here means the real Stage 1 would have produced the
same false positive -- a flag that only ever saw sentence-level context
(no reference-list filtering) would be a worse, less faithful preview of
that.

`<category>_0_1` is deliberately NOT registered as a real FAIRe checklist
term (not added to schemas/faire/classes.yaml) -- these are pipeline-
internal audit columns, same INTERNAL_*-column precedent as
exports/faire.py's own INTERNAL_STUDY_ID_FIELD/
INTERNAL_ALIAS_SAMPLE_IDS_FIELD, not something submittable to
NOAA/GBIF/OBIS as part of the real FAIRe checklist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.sources.base import RawFactCandidate


@dataclass(frozen=True)
class CategoryTerm:
    """One FAIRe field within a SectionCategory, with its own distinctive
    search cues -- deliberately narrower/more specific than the category-
    level `keywords` above (which only decide whether a paragraph deserves
    a closer look at all). These feed Stage 3 (category-scoped quote-
    judged extraction, not yet built): within a category's own assembled
    run-text (Stage 2.5's output), each term's own cues gate which
    sentences are even shown to the LLM as candidates for that term --
    the same quote-candidate-then-judge mechanism
    extraction/search_flags.py's LLMJudgedSearchField already uses, just
    scoped to one category's own fields and own text instead of the full
    taxonomy and the whole paper."""

    native_name: str
    search_cues: tuple[str, ...] = ()
    # "Fallback only" terms (e.g. pcr_method_additional) capture whatever
    # in-category text didn't match any other term's own cues, rather
    # than being independently keyword-searched -- an explicit user
    # instruction, not a guess: "Fallback only: unmatched but clearly
    # PCR1-specific method text; do not independently keyword-search."
    fallback_only: bool = False
    # otu_final_description: FAIRe defines this as a full (possibly
    # multi-sentence) description, not a single short value -- Stage 3's
    # eventual extraction prompt must accept multiple sentences for this,
    # not truncate to one phrase.
    allows_multi_sentence: bool = False
    # Verbatim LLM prompt definition (user-supplied), used as this term's
    # "meaning" line in Stage 3's judgment prompt -- whether the expected
    # value reads as a number, a software/package name, or a free-text
    # phrase is inferred from this wording at prompt-construction time
    # rather than hardcoded per term, since the shared prompt template
    # already carries one global "take word for word from the text, never
    # generate" instruction that applies uniformly regardless of value
    # shape (an explicit user instruction: "In all cases, have the
    # program take word for word from text and not generate").
    definition: str = ""


@dataclass(frozen=True)
class SectionCategory:
    name: str  # slug; the detection fact_type_candidate is f"{name}_0_1"
    label: str  # human-readable, for future LLM categorization prompts
    keywords: tuple[str, ...]
    terms: tuple[CategoryTerm, ...] = ()


SECTION_CATEGORIES: tuple[SectionCategory, ...] = (
    SectionCategory(
        name="sample_prep",
        label="Sample preparation / storage / nucleic acid extraction",
        keywords=(
            # General sample preparation / handling -- physical/chemical
            # handling after collection and before PCR, per an explicit
            # user category definition.
            "sample preparation", "sample processing", "processed prior to extraction",
            "prepared for DNA extraction", "sample handling", "subsampled", "subsampling",
            "aliquoted", "homogenized", "homogenised", "ground", "crushed", "milled",
            "cut into pieces", "chopped", "dissected", "mixed", "vortexed", "centrifuged",
            "pelleted", "supernatant", "resuspended", "washed", "rinsed", "filtered",
            "filtration", "filter", "filter cartridge", "filter membrane", "membrane filter",
            "cartridge filter", "Sterivex", "pore size", "mesh size",
            # Sample storage / preservation -- deliberately in the same
            # bucket: commonly occurs right between sampling and extraction.
            "stored", "storage", "preserved", "preservation", "frozen", "freeze",
            "-20°C", "-80°C", "4°C", "on ice", "dry ice", "liquid nitrogen",
            "RNAlater", "ethanol", "lysis buffer", "preservation buffer", "transported frozen",
            "transported on ice", "shipped frozen",
            # Drying / concentration / precipitation
            "freeze-dried", "lyophilized", "lyophilised", "air-dried", "dried",
            "vacuum dried", "vacuum concentrator", "concentrated", "concentration",
            "precipitated", "precipitation",
            # Nucleic-acid extraction -- the strongest portion of this bucket.
            "DNA extraction", "DNA was extracted", "RNA extraction", "RNA was extracted",
            "nucleic acid extraction", "nucleic acids were extracted", "DNA isolation",
            "RNA isolation", "DNA purification", "extraction protocol", "extraction procedure",
            "extracted according to", "manufacturer's instructions", "manufacturer's protocol",
            "extraction kit", "DNA kit", "RNA kit",
            # Lysis
            "lysis", "lysed", "cell lysis", "bead beating", "bead-beating", "bead mill",
            "homogenizer", "sonication", "sonicated", "freeze-thaw", "proteinase K",
            # DNA quantification/purity and deployment duration -- added
            # alongside a second batch of user-supplied term definitions.
            "DNA concentration", "Qubit", "NanoDrop", "260/280", "A260/280", "absorbance ratio",
            "deployed", "deployment period", "extracted on",
        ),
        terms=(
            # General sample-handling narrative -- FAIRe's own samp_mat_process
            # is explicitly defined as "any processing applied to the sample
            # during or after retrieving the sample from environment (e.g.
            # sieving, storage)" -- a natural home for subsampling/
            # homogenizing/grinding/centrifuging/washing language that has no
            # more specific dedicated field of its own.
            CategoryTerm("samp_mat_process", (
                "sample preparation", "sample processing", "prepared for DNA extraction",
                "sample handling", "processed prior to extraction", "subsampled", "subsampling",
                "aliquoted", "homogenized", "homogenised", "ground", "crushed", "milled",
                "centrifuged", "pelleted", "resuspended", "washed", "rinsed",
            ), allows_multi_sentence=True, definition='Physical or chemical processing applied to a sample during or after collection, such as filtration, sieving, homogenization, precipitation, or subsampling.'),
            CategoryTerm("prep_method_additional", fallback_only=True, definition='Additional information about sample preparation, preservation, or pre-extraction storage that does not fit another specific field.'),

            # Filtration
            CategoryTerm("filter_material", (
                "filter material", "cellulose filter", "cellulose ester filter", "nylon filter",
                "glass fiber filter", "polyethersulfone filter", "PES filter", "membrane filter",
                "filter membrane",
            ), definition='Material from which the sample filter membrane is made.'),
            CategoryTerm("filter_name", (
                "Sterivex", "filter cartridge", "cartridge filter", "commercial filter",
                "filter brand", "filter product",
            ), definition='Commercial name or brand/model of the filter used.'),
            CategoryTerm("filter_diameter", (
                "filter diameter", "diameter of the filter", "mm filter", "mm diameter filter",
            ), definition='Physical diameter of a circular filter, usually in mm. Do not confuse with pore size.'),
            CategoryTerm("filter_surface_area", (
                "filter surface area", "filter area", "surface area of the filter",
            ), definition='Total surface area of the filter membrane, usually in mm².'),
            CategoryTerm("size_frac", (
                "pore size", "filtering pore size", "filter pore size", "µm filter", "um filter",
                "0.22 µm", "0.2 µm", "0.45 µm",
            ), definition='Pore size of the main filter used to collect/sample material, in µm.'),
            CategoryTerm("size_frac_low", (
                "pre-filter", "prefilter pore size", "mesh size", "pre-filtered through",
                "pre-sorted through",
            ), definition='Mesh or pore size of a pre-filter used to exclude material larger than that threshold, in µm.'),
            CategoryTerm("prefilter_material", (
                "pre-filter material", "prefilter material", "pre-sort material",
            ), definition='Material used for a pre-filter or pre-sort step before the main sample filtration.'),
            CategoryTerm("filter_passive_active_0_1", (
                "active filtration", "passive filtration", "pumped through the filter",
                "submerged filter", "passive sampler", "actively filtered", "passively filtered",
            ), definition='Whether filtration/collection was active using a pump or fan (1) or passive by environmental exposure/submersion (0).'),
            CategoryTerm("pump_flow_rate", (
                "pump flow rate", "flow rate of", "pumped at a rate", "L/min", "flow rate was",
            ), definition='Flow rate of the pump used during filtration.'),
            CategoryTerm("pump_flow_rate_unit", (
                "L/min", "L/h", "L/s", "m3/min", "m3/h", "m3/s",
            ), definition='Unit associated with the filtration pump flow rate.'),
            CategoryTerm("stationed_sample_dur", (
                "deployed for", "deployment period", "filter was deployed", "sampler was deployed",
                "in situ for", "deployed in the environment",
            ), definition='Length of time a filter or other stationed sampler was deployed directly in the environment.'),

            # Sample storage / preservation (prior to DNA extraction)
            CategoryTerm("samp_store_temp", (
                "stored at", "storage temperature", "-20°C", "-80°C", "4°C", "on ice",
                "dry ice", "liquid nitrogen", "ambient temperature", "stored frozen", "stored cold",
            ), definition='Temperature at which the original environmental sample was stored.'),
            CategoryTerm("samp_store_dur", (
                "stored for", "storage duration", "prior to extraction for", "stored until",
            ), definition='Duration the original environmental sample was stored before processing or DNA extraction.'),
            CategoryTerm("samp_store_loc", (
                "stored in a freezer", "stored at the", "storage location", "stored in the lab",
            ), definition='Physical location where the original sample was stored, such as a particular freezer or laboratory room.'),
            CategoryTerm("samp_store_sol", (
                "stored in RNAlater", "stored in ethanol", "storage solution", "preservation buffer",
                "stored in lysis buffer", "Longmire's buffer", "preserved in ethanol",
            ), definition='Solution in which the original environmental sample was stored or preserved.'),
            CategoryTerm("samp_store_method_additional", (
                "transported frozen", "transported on ice", "shipped frozen", "storage conditions",
                "shipped on dry ice", "shipped on ice",
            ), allows_multi_sentence=True, definition='Additional useful information about how the original environmental sample was stored or preserved.'),
            CategoryTerm("prepped_samp_store_temp", (
                "prepared sample was stored at", "processed sample was stored at",
                "extract was stored at prior to", "subsample was stored at",
            ), definition='Temperature at which the processed/prepared sample was stored before DNA extraction.'),
            CategoryTerm("prepped_samp_store_dur", (
                "prepared sample was stored for", "processed sample was stored for",
            ), definition='Duration a processed sample, such as a filter or homogenized subsample, was stored before DNA extraction.'),
            CategoryTerm("prepped_samp_store_sol", (
                "prepared sample was stored in", "processed sample was stored in",
            ), definition='Solution in which the processed/prepared sample was stored before DNA extraction.'),
            CategoryTerm("dna_store_loc", (
                "extracted DNA was stored", "DNA samples were stored at", "extracts were stored in",
            ), definition='Physical location where the extracted DNA was stored, such as a freezer or laboratory room.'),

            # Drying / concentration / precipitation
            CategoryTerm("precip_chem_prep", (
                "precipitated with", "ethanol precipitation", "isopropanol precipitation",
                "sodium chloride precipitation", "precipitation buffer",
            ), definition='Chemical(s) used to precipitate/concentrate sample material before DNA extraction.'),
            CategoryTerm("precip_force_prep", (
                "centrifuged at", "centrifuge force", "x g for", "rpm for", "relative centrifugal force",
            ), definition='Centrifugation force used during pre-extraction precipitation.'),
            CategoryTerm("precip_temp_prep", (
                "precipitated at", "precipitation temperature",
            ), definition='Temperature used during pre-extraction precipitation.'),
            CategoryTerm("precip_time_prep", (
                "precipitated for", "precipitation for", "precipitation time",
            ), definition='Duration of the pre-extraction precipitation step.'),

            # Nucleic-acid extraction
            CategoryTerm("nucl_acid_ext", (
                "extraction protocol", "extraction procedure", "extracted according to",
                "following the protocol of", "extraction SOP", "protocol available at",
            ), definition='Citation, URL, SOP, or protocol reference describing the nucleic acid extraction procedure.'),
            CategoryTerm("nucl_acid_ext_kit", (
                "extraction kit", "DNA kit", "RNA kit", "using the kit", "manufacturer's instructions",
                "manufacturer's protocol", "Soil DNA kit", "DNeasy", "PowerSoil",
            ), definition='Name of the commercial kit used to extract DNA/RNA from the sample.'),
            CategoryTerm("nucl_acid_ext_modify", (
                "modified the manufacturer's protocol", "protocol was modified",
                "with the following modification", "deviating from the manufacturer's instructions",
            ), definition='Any explicitly stated modification made to a commercial or established extraction protocol.'),
            CategoryTerm("nucl_acid_ext_method_additional", (
                "DNA extraction was performed", "DNA was extracted", "RNA extraction",
                "RNA was extracted", "nucleic acid extraction", "nucleic acids were extracted",
                "DNA isolation", "RNA isolation", "extraction procedure",
            ), allows_multi_sentence=True, definition='Additional useful nucleic-acid extraction details not captured by another extraction field.'),
            CategoryTerm("date_ext", (
                "extraction was performed on", "extracted on", "DNA was extracted on",
                "extraction date",
            ), definition='Date on which nucleic acid extraction was performed.'),
            CategoryTerm("dna_cleanup_0_1", (
                "DNA was cleaned", "DNA was purified", "DNA purification", "cleanup was performed",
                "no additional cleanup", "purification step",
            ), definition='Whether extracted DNA was subsequently cleaned or purified: yes = 1, no = 0.'),
            CategoryTerm("dna_cleanup_method", (
                "cleaned using", "purified using", "cleanup kit", "purification kit",
            ), definition='Method or commercial kit used to clean/purify extracted DNA.'),
            CategoryTerm("pool_dna_num", (
                "extracts were pooled", "pooled extracts", "number of extracts pooled",
                "DNA extracts were combined",
            ), definition='Number of separate DNA extracts pooled together into one sample before PCR.'),
            CategoryTerm("concentration", (
                "DNA concentration was", "concentration of the extracted DNA", "concentration of DNA",
                "concentration was measured", "final DNA concentration",
            ), definition='Concentration of total DNA after extraction.'),
            CategoryTerm("concentration_method", (
                "measured using a Qubit", "measured using a NanoDrop", "concentration was measured using",
                "quantified using", "Qubit fluorometer", "NanoDrop spectrophotometer",
            ), definition='Method or instrument used to measure extracted DNA concentration.'),
            CategoryTerm("concentration_unit", (
                "ng/µL", "ng/mL", "µg/mL", "ng per µL",
            ), definition='Unit used for extracted DNA concentration, such as ng/µL.'),
            CategoryTerm("ratioOfAbsorbance260_280", (
                "260/280 ratio", "A260/280", "absorbance ratio", "260/280 nm ratio", "A260/A280",
            ), definition='Reported absorbance-ratio measurement used to assess extracted DNA purity.'),
            CategoryTerm("samp_vol_we_dna_ext", (
                "used for DNA extraction", "used for extraction", "processed for DNA extraction",
                "used for RNA extraction", "used for nucleic acid extraction",
            ), definition='Amount of sample or subsample actually processed for DNA extraction; this is not necessarily the total amount originally collected.'),
            CategoryTerm("samp_vol_we_dna_ext_unit", (
                "mg of dried", "mg of sediment", "g of sediment", "mL of", "L of water",
            ), definition='Unit for the amount of sample processed for DNA extraction, such as mg, g, mL, L, or cm².'),

            # Lysis
            CategoryTerm("nucl_acid_ext_lysis", (
                "lysis", "lysed", "cell lysis", "bead beating", "bead-beating", "bead mill",
                "homogenizer", "sonication", "sonicated", "freeze-thaw", "proteinase K",
                "chemical lysis", "enzymatic lysis", "thermal lysis",
            ), definition='General approach used to lyse DNA-containing material, such as physical, thermal, chemical, enzymatic, or osmotic lysis.'),
            CategoryTerm("nucl_acid_ext_sep", (
                "separated using", "column-based", "spin column", "magnetic beads",
                "phenol-chloroform", "phenol chloroform", "silica column",
            ), definition='Approach used to separate/purify DNA from the sample mixture, such as column-based separation, magnetic beads, centrifugation, precipitation, or phenol-chloroform.'),
        ),
    ),
    SectionCategory(
        name="assay_definition",
        label="Assay definition / target",
        keywords=(
            # What assay/marker is used and what it targets -- not the PCR
            # recipe itself.
            "assay", "assay name", "target assay", "targeted assay", "metabarcoding assay",
            "primer set", "primer pair", "marker", "genetic marker", "molecular marker",
            "locus", "target gene", "target region", "target locus", "amplicon", "amplicon size",
            "amplicon length", "12S", "16S", "18S", "23S", "28S", "COI", "CO1", "cox1", "CytB",
            "cytochrome b", "rbcL", "ITS", "ITS1", "ITS2",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
            "V3-V4", "V4-V5", "D1-D2", "D2-D3",
            "MiFish", "Teleo", "Leray", "TAReuk", "universal primers", "species-specific primers",
            "taxon-specific primers", "designed to amplify", "designed to detect", "designed to target",
            "primers targeting", "assay targeting", "specific for", "probe targeting", "target taxa",
            "target taxon", "target species", "taxonomic target", "metabarcoding", "targeted detection",
        ),
        terms=(
            CategoryTerm("assay_name", (
                "assay name", "assay designated", "assay referred to as", "primer set called",
                "primer pair called", "marker assay", "assay identifier", "primer set designated",
                "hereafter referred to as", "assay developed by",
            ), definition='The short name or identifier used for a particular assay, primer set, marker assay, or amplified region. Extract an explicitly stated assay name when possible; do not invent one.'),
            CategoryTerm("assay_type", (
                "metabarcoding assay", "DNA metabarcoding", "community metabarcoding", "targeted assay",
                "species-specific assay", "taxon-specific assay", "targeted detection",
                "single-species detection", "broad community profiling", "multi-taxon detection",
            ), definition='Whether the assay is designed for targeted detection of one/few taxa or metabarcoding of a broader biological community.'),
            # assay_validation removed entirely per an explicit user
            # request ("remove assay_validation from FAIRe list, i don't
            # want to see it on the table print outs") -- suppressed from
            # export in exports/faire.py's PROJECT_METADATA_SUPPRESSED_
            # FIELDS and no longer extracted anywhere.
            CategoryTerm("targetTaxonomicAssay", (
                "primers targeting", "primers designed to amplify", "assay designed to detect",
                "assay targeting", "probe targeting", "specific for", "species-specific primers for",
                "taxon-specific primers for", "universal primers for", "designed for detection of",
            ), definition='The organism or taxonomic group that the assay, primers, or probe were designed to detect or amplify. This is assay capability, not necessarily the narrower taxonomic focus of the study.'),
            # "plus remaining controlled gene names" per the user -- this
            # list is deliberately non-exhaustive; a controlled gene-name
            # vocabulary lives elsewhere (not duplicated here).
            CategoryTerm("target_gene", (
                "target gene", "marker gene", "target locus", "marker locus", "12S rRNA", "16S rRNA",
                "18S rRNA", "COI", "cytochrome b", "rbcL",
            ), definition='The gene or genetic marker targeted for amplification, such as 12S, 16S, 18S, COI, CytB, ITS, or rbcL.'),
            # Deliberate distinction from assay_name: "V4 region" here,
            # but "V4 assay"/"V4 primer set" cues assay_name instead.
            CategoryTerm("target_subfragment", (
                "target subfragment", "hypervariable region", "variable region", "V4 region",
                "V9 region", "V3-V4 region", "ITS1 region", "ITS2 region", "D1-D2 region", "P6 loop",
            ), definition='A smaller named region within the target gene/locus, such as V4, V9, V3-V4, ITS1, ITS2, D1-D2, or the trnL P6 loop.'),
            CategoryTerm("ampliconSize", (
                "amplicon size", "amplicon length", "expected amplicon", "expected product size",
                "PCR product size", "amplicon of approximately", "bp amplicon", "base-pair amplicon",
                "target fragment length", "product length",
            ), definition='The expected length of the amplified target DNA fragment in base pairs, excluding primers, adapters, and indexes.'),
            # nucl_acid_amp removed entirely per an explicit user request
            # ("remove nucl_acid_amp from the search, and from the list --
            # i don't ever want to see it again") -- suppressed from
            # export in exports/faire.py's PROJECT_METADATA_SUPPRESSED_
            # FIELDS and no longer extracted anywhere.
        ),
    ),
    SectionCategory(
        name="pcr1_primary_amplification",
        label="PCR1 / primary amplification",
        keywords=(
            "PCR", "PCR amplification", "amplification", "amplified", "amplicon",
            "polymerase chain reaction", "PCR reaction", "PCR reactions", "reaction mixture",
            "PCR mixture", "master mix", "mastermix", "template DNA", "DNA template", "primer",
            "primers", "forward primer", "reverse primer", "primer concentration", "primer volume",
            "reaction volume", "thermal cycling", "cycling conditions", "PCR conditions",
            "initial denaturation", "denaturation", "annealing", "annealing temperature", "extension",
            "elongation", "final extension", "final elongation", "PCR cycles", "cycles of amplification",
            "thermocycler", "thermal cycler", "technical replicates", "PCR replicates", "replicate PCRs",
            "blocking primer", "blocking oligonucleotide", "PNA clamp", "LNA clamp", "PCR inhibition",
            "inhibition test", "inhibition check", "internal amplification control",
        ),
        terms=(
            CategoryTerm("amplificationReactionVolume", (
                "total reaction volume", "final reaction volume", "PCR reaction volume",
                "reaction volume of", "performed in a volume of", "uL reaction", "amplification volume",
                "total PCR volume", "final volume per reaction", "reaction mixture volume",
            ), definition='The total final volume of one PCR reaction, not the amount of DNA, primer, or master mix added.'),
            CategoryTerm("annealingTemp", (
                "annealing temperature", "annealed at", "annealing at", "annealing step",
                "annealing phase", "primer annealing", "C annealing", "touchdown annealing",
                "annealing gradient", "anneal for",
            ), definition='The temperature used specifically during the primer-annealing stage of PCR. Do not return denaturation or extension temperatures.'),
            CategoryTerm("commercial_mm", (
                "commercial master mix", "PCR master mix", "premixed master mix", "pre-made master mix",
                "2x master mix", "ready-to-use mix", "manufacturer's master mix", "commercial PCR mix",
                "master mix supplied by", "PCR reaction mix kit",
            ), definition='The name, manufacturer, and ideally version/product information of a commercially prepared PCR master mix.'),
            CategoryTerm("custom_mm", (
                "reaction mixture consisted of", "reaction contained", "custom master mix",
                "master mix was prepared", "buffer and dNTPs", "MgCl2 and dNTPs",
                "polymerase and buffer", "BSA was added", "reaction components were",
                "components per reaction",
            ), definition='The explicitly reported composition of a PCR reaction mixture assembled from individual components rather than a commercial pre-made master mix.'),
            CategoryTerm("inhibition_check_0_1", (
                "tested for PCR inhibition", "inhibition was checked", "inhibition testing",
                "PCR inhibition assessment", "inhibition control included",
                "internal amplification control", "internal positive control",
                "spike-in inhibition test", "serial dilution inhibition test", "evaluated for inhibitors",
            ), definition='Whether the study explicitly tested DNA extracts or samples for PCR inhibition.'),
            CategoryTerm("inhibition_check", (
                "inhibition assessed by", "inhibition evaluated using", "tested for inhibition using",
                "internal amplification control", "spiked with control DNA",
                "serial dilution to assess inhibition", "Cq shift", "Ct shift", "inhibition assay",
                "inhibition mitigation",
            ), definition='How PCR inhibition was tested, assessed, or mitigated, including dilution tests, internal controls, spike-ins, purification, or other procedures.'),
            CategoryTerm("pcr_analysis_software", (
                "PCR data analyzed using", "PCR data analysed using", "amplification data analyzed using",
                "amplification curves analyzed with", "Cq values calculated using",
                "Ct values calculated using", "PCR run analyzed with", "fluorescence data analyzed using",
                "PCR analysis software", "instrument data analysis software",
            ), definition='Software used to analyze PCR/qPCR amplification-run data, such as amplification curves, Ct/Cq values, or instrument output. Do not return general sequence-analysis software.'),
            CategoryTerm("pcr_cond", (
                "PCR conditions", "cycling conditions", "thermal cycling profile",
                "thermocycling conditions", "PCR program", "thermal profile", "amplification conditions",
                "cycling protocol", "denaturation annealing extension",
                "initial denaturation followed by",
            ), definition='The overall PCR thermal-cycling conditions, including stages such as initial denaturation, denaturation, annealing, extension, and final extension.'),
            CategoryTerm("pcr_cycles", (
                "PCR cycles", "amplification cycles", "cycles of amplification", "cycled for",
                "number of cycles", "35 cycles", "40 cycles", "x 35 cycles", "35-cycle amplification",
                "repeated for 35 cycles",
            ), definition='The number of amplification cycles performed during the primary PCR.'),
            CategoryTerm("pcr_dna_vol", (
                "template DNA volume", "DNA volume per reaction", "uL template DNA",
                "DNA template added", "template added per reaction", "DNA input volume",
                "PCR template volume", "volume of DNA extract", "extract volume per reaction",
                "uL of extract",
            ), definition='The volume of DNA extract or DNA template added to each primary PCR reaction.'),
            CategoryTerm("pcr_method_additional", fallback_only=True, definition='Other explicitly reported information about the primary PCR method that is useful but does not belong in another specific PCR field.'),
            CategoryTerm("pcr_primer_conc_forward", (
                "forward primer concentration", "final forward primer concentration",
                "forward primer at", "nM forward primer", "uM forward primer",
                "concentration of forward primer", "forward primer final concentration",
                "forward oligo concentration", "F primer concentration",
                "forward primer stock concentration",
            ), definition='The concentration of the forward PCR primer used in the reaction. Preserve whether the reported value is stock or final concentration.'),
            CategoryTerm("pcr_primer_conc_reverse", (
                "reverse primer concentration", "final reverse primer concentration",
                "reverse primer at", "nM reverse primer", "uM reverse primer",
                "concentration of reverse primer", "reverse primer final concentration",
                "reverse oligo concentration", "R primer concentration",
                "reverse primer stock concentration",
            ), definition='The concentration of the reverse PCR primer used in the reaction. Preserve whether the reported value is stock or final concentration.'),
            CategoryTerm("pcr_primer_forward", (
                "forward primer sequence", "forward oligonucleotide sequence", "forward primer 5'-3'",
                "F primer sequence", "forward sequence", "forward primer:", "sense primer sequence",
                "Fwd primer sequence", "forward oligo 5'",
            ), definition="The nucleotide sequence of the forward PCR primer, reported in the 5'->3' direction."),
            CategoryTerm("pcr_primer_reverse", (
                "reverse primer sequence", "reverse oligonucleotide sequence", "reverse primer 5'-3'",
                "R primer sequence", "reverse sequence", "reverse primer:", "antisense primer sequence",
                "Rev primer sequence", "reverse oligo 5'",
            ), definition="The nucleotide sequence of the reverse PCR primer, reported in the 5'->3' direction."),
            CategoryTerm("pcr_primer_name_forward", (
                "forward primer name", "forward primer designated", "forward primer called",
                "forward primer identifier", "forward primer ID", "F primer name",
                "forward oligo name", "forward primer label", "Fwd primer name",
                "forward primer abbreviation", "primer pair", "primer names",
                "primer set", "16S rRNA F",
            ), definition='The published or study-specific name/identifier of the forward primer, not its nucleotide sequence.'),
            CategoryTerm("pcr_primer_name_reverse", (
                "reverse primer name", "reverse primer designated", "reverse primer called",
                "reverse primer identifier", "reverse primer ID", "R primer name",
                "reverse oligo name", "reverse primer label", "Rev primer name",
                "reverse primer abbreviation", "primer pair", "primer names",
                "primer set", "16S rRNA R",
            ), definition='The published or study-specific name/identifier of the reverse primer, not its nucleotide sequence.'),
            CategoryTerm("pcr_primer_reference_forward", (
                "forward primer described by", "forward primer from", "forward primer reference",
                "forward primer published in", "forward primer developed by",
                "forward primer according to", "forward primer adapted from", "forward primer source",
                "forward primer citation", "forward primer DOI",
            ), definition='The citation, DOI, publication, or source describing the forward primer.'),
            CategoryTerm("pcr_primer_reference_reverse", (
                "reverse primer described by", "reverse primer from", "reverse primer reference",
                "reverse primer published in", "reverse primer developed by",
                "reverse primer according to", "reverse primer adapted from", "reverse primer source",
                "reverse primer citation", "reverse primer DOI",
            ), definition='The citation, DOI, publication, or source describing the reverse primer.'),
            CategoryTerm("pcr_primer_vol_forward", (
                "forward primer volume", "volume of forward primer", "uL forward primer",
                "forward primer added per reaction", "forward primer volume per reaction",
                "F primer volume", "aliquot of forward primer", "forward oligo volume",
                "uL of forward primer",
            ), definition='The physical volume of forward-primer solution added to each PCR reaction. Do not confuse with concentration.'),
            CategoryTerm("pcr_primer_vol_reverse", (
                "reverse primer volume", "volume of reverse primer", "uL reverse primer",
                "reverse primer added per reaction", "reverse primer volume per reaction",
                "R primer volume", "aliquot of reverse primer", "reverse oligo volume",
                "uL of reverse primer",
            ), definition='The physical volume of reverse-primer solution added to each PCR reaction. Do not confuse with concentration.'),
            CategoryTerm("pcr_rep", (
                "PCR technical replicates", "technical PCR replicates", "replicate PCR reactions",
                "replicate PCRs", "PCRs performed in duplicate", "PCRs performed in triplicate",
                "duplicate PCR reactions", "triplicate PCR reactions", "independent PCR reactions",
                "PCR replicates per sample",
            ), definition='The number of technical PCR replicate reactions performed per biological sample. Do not count biological, field, or extraction replicates.'),
            CategoryTerm("thermocycler", (
                "thermocycler", "thermal cycler", "PCR instrument", "PCR machine", "PCR performed on",
                "PCR performed using", "amplification performed on", "cycled in a",
                "thermal cycling system", "manufacturer and model",
            ), definition='The manufacturer and model of the thermal cycler used to perform the primary PCR. Do not return the sequencing instrument.'),
            CategoryTerm("block_ref", (
                "blocking primer described by", "blocking primer reference",
                "blocking oligonucleotide reference", "blocker developed by", "blocker published in",
                "blocking primer adapted from", "blocking primer according to",
                "blocking primer citation", "blocking primer DOI", "blocker source",
            ), definition='The citation, DOI, publication, or other source describing the blocking primer/oligonucleotide.'),
            CategoryTerm("block_seq", (
                "blocking primer sequence", "blocking oligonucleotide sequence", "blocker sequence",
                "PNA sequence", "LNA sequence", "blocking oligo 5'", "host-blocking sequence",
                "blocking sequence 5'-3'", "block sequence", "blocking primer nucleotide sequence",
            ), definition='The nucleotide sequence of the blocking primer, blocking oligonucleotide, PNA/LNA clamp, or equivalent blocker.'),
            CategoryTerm("block_taxa", (
                "blocking primer targeting", "blocker designed against", "blocked taxon",
                "host DNA blocked", "suppress amplification of", "prevent amplification of",
                "blocking primer specific for", "host-blocking primer for", "blocker targeting",
                "exclude amplification of",
            ), definition='The taxon whose amplification the blocking primer or blocker was intended to suppress.'),
        ),
    ),
    SectionCategory(
        name="targeted_qpcr_ddpcr_detection",
        label="Targeted qPCR / ddPCR detection",
        keywords=(
            "qPCR", "quantitative PCR", "quantitative real-time PCR", "real-time PCR", "real time PCR",
            "RT-qPCR", "ddPCR", "droplet digital PCR", "digital PCR", "dPCR", "targeted assay",
            "species-specific assay", "species-specific qPCR", "taxon-specific assay", "TaqMan",
            "TaqMan assay", "hydrolysis probe", "probe-based assay", "probe-based qPCR", "probe",
            "reporter", "quencher", "FAM", "HEX", "VIC", "BHQ", "ZEN", "MGB", "Cq", "Ct",
            "quantification cycle", "threshold cycle", "fluorescence threshold", "baseline",
            "automatic threshold", "standard curve", "calibration curve", "DNA standard",
            "synthetic standard", "gBlock", "plasmid standard", "copy number", "copies per reaction",
            "copies/reaction", "limit of detection", "LOD", "limit of quantification", "LOQ",
            "detection limit", "quantification limit", "detection criteria", "positive detection",
            "considered positive", "PCR efficiency", "amplification efficiency", "droplets",
            "positive droplets", "negative droplets", "QuantaSoft",
        ),
        terms=(
            CategoryTerm("amp_vis_method", (
                "amplicon visualisation", "amplicon visualization", "target detected by",
                "detection by qPCR", "detection by digital PCR", "detection by gel electrophoresis",
                "capillary electrophoresis detection", "real-time fluorescence detection",
                "amplicons visualized on", "presence determined by",
            ), definition='The experimental method used to determine or visualize whether the target amplicon was detected, such as qPCR fluorescence, ddPCR, gel electrophoresis, or another detection approach.'),
            CategoryTerm("automaticBaselineValue", (
                "baseline set automatically", "automatic baseline", "baseline determined automatically",
                "instrument-defined baseline", "software-defined baseline", "default baseline setting",
                "baseline manually set", "manual baseline", "baseline adjusted manually",
                "automatic baseline setting disabled",
            ), definition='Whether/how the qPCR fluorescence baseline was automatically determined rather than manually specified.'),
            CategoryTerm("automaticThresholdQuantificationCycle", (
                "threshold set automatically", "automatic fluorescence threshold",
                "automatic threshold setting", "instrument-defined threshold",
                "software-defined threshold", "default threshold setting", "threshold manually set",
                "manual threshold", "threshold adjusted manually", "automatic threshold disabled",
            ), definition='Whether/how the fluorescence threshold used for Cq/Ct determination was automatically set rather than manually specified.'),
            CategoryTerm("baselineValue", (
                "baseline cycles", "baseline range", "baseline interval", "baseline window",
                "baseline from cycle", "baseline through cycle", "baseline start cycle",
                "baseline end cycle", "cycles used for baseline", "background fluorescence cycles",
            ), definition='The explicitly reported baseline setting or baseline cycle range used for qPCR fluorescence analysis.'),
            CategoryTerm("detection_criteria", (
                "criteria for positive detection", "considered positive when",
                "positive detection required", "sample considered detected",
                "positive amplifications per sample", "at least two positive replicates",
                "detected in two of three", "detection criterion", "confirmed by Sanger sequencing",
                "positive call required",
            ), definition='The rule used to decide whether a sample was considered positive/detected, such as a required number of positive technical replicates or confirmatory analysis.'),
            CategoryTerm("lod_method", (
                "LOD determined by", "LOD calculated using", "LOD estimated using",
                "method for determining LOD", "limit of detection method", "LOD methodology",
                "LOD calculated according to", "95% detection probability", "probit analysis for LOD",
                "LOD model",
            ), definition="The statistical or experimental procedure used to calculate or determine the assay's limit of detection. Do not return the LOD value itself."),
            CategoryTerm("loq_method", (
                "LOQ determined by", "LOQ calculated using", "LOQ estimated using",
                "method for determining LOQ", "limit of quantification method", "LOQ methodology",
                "LOQ calculated according to", "quantification precision criterion",
                "CV threshold for LOQ", "LOQ model",
            ), definition="The statistical or experimental procedure used to calculate or determine the assay's limit of quantification. Do not return the LOQ value itself."),
            CategoryTerm("pcr_assay_lod", (
                "limit of detection", "LOD was", "LOD =", "detection limit", "95% LOD",
                "minimum detectable concentration", "minimum detectable copies",
                "lowest detectable concentration", "analytical detection limit",
                "copies detectable per reaction",
            ), definition='The reported limit of detection: the lowest target amount/concentration that the assay can reliably detect.'),
            CategoryTerm("pcr_assay_lod_LL", (
                "LOD lower 95% confidence limit", "LOD lower confidence bound", "LOD lower CI",
                "lower confidence limit for LOD", "lower bound of LOD", "95% CI lower limit",
                "LOD confidence interval lower", "LOD LL", "lower 95% CI for detection limit",
                "LOD lower endpoint",
            ), definition='The lower confidence bound or lower confidence limit associated with the reported LOD.'),
            CategoryTerm("pcr_assay_lod_UL", (
                "LOD upper 95% confidence limit", "LOD upper confidence bound", "LOD upper CI",
                "upper confidence limit for LOD", "upper bound of LOD", "95% CI upper limit",
                "LOD confidence interval upper", "LOD UL", "upper 95% CI for detection limit",
                "LOD upper endpoint",
            ), definition='The upper confidence bound or upper confidence limit associated with the reported LOD.'),
            CategoryTerm("pcr_assay_lod_techreps", (
                "LOD technical replicates", "replicates used for LOD", "LOD based on replicates",
                "replicate reactions at LOD", "number of replicates for LOD",
                "technical replicates per concentration", "replicates per dilution for LOD",
                "LOD replicate number", "LOD applied to three replicates", "replicate wells for LOD",
            ), definition='The number of technical replicate reactions used when establishing or applying the LOD.'),
            CategoryTerm("pcr_assay_lod_unit", (
                "LOD unit", "copies/reaction", "copies per reaction", "copies/uL", "copies per uL",
                "ng/uL", "ng per reaction", "fg per reaction", "genome equivalents",
                "molecules per reaction",
            ), definition='The measurement unit associated with the LOD, such as copies/reaction or copies/uL.'),
            CategoryTerm("pcr_assay_loq", (
                "limit of quantification", "LOQ was", "LOQ =", "quantification limit",
                "lowest quantifiable concentration", "minimum quantifiable copies",
                "analytical quantification limit", "lowest concentration quantified",
                "quantifiable copies per reaction", "lower limit of quantification",
            ), definition='The reported limit of quantification: the lowest target amount/concentration that can be quantitatively measured with acceptable performance.'),
            CategoryTerm("pcr_assay_loq_LL", (
                "LOQ lower 95% confidence limit", "LOQ lower confidence bound", "LOQ lower CI",
                "lower confidence limit for LOQ", "lower bound of LOQ",
                "95% CI lower limit for LOQ", "LOQ confidence interval lower", "LOQ LL",
                "lower 95% CI for quantification limit", "LOQ lower endpoint",
            ), definition='The lower confidence bound or lower confidence limit associated with the LOQ.'),
            CategoryTerm("pcr_assay_loq_UL", (
                "LOQ upper 95% confidence limit", "LOQ upper confidence bound", "LOQ upper CI",
                "upper confidence limit for LOQ", "upper bound of LOQ",
                "95% CI upper limit for LOQ", "LOQ confidence interval upper", "LOQ UL",
                "upper 95% CI for quantification limit", "LOQ upper endpoint",
            ), definition='The upper confidence bound or upper confidence limit associated with the LOQ.'),
            CategoryTerm("pcr_assay_loq_techreps", (
                "LOQ technical replicates", "replicates used for LOQ", "LOQ based on replicates",
                "replicate reactions at LOQ", "number of replicates for LOQ",
                "technical replicates per concentration", "replicates per dilution for LOQ",
                "LOQ replicate number", "LOQ applied to three replicates", "replicate wells for LOQ",
            ), definition='The number of technical replicate reactions used when establishing or applying the LOQ.'),
            CategoryTerm("pcr_assay_loq_unit", (
                "LOQ unit", "copies/reaction", "copies per reaction", "copies/uL", "copies per uL",
                "ng/uL", "ng per reaction", "fg per reaction", "genome equivalents",
                "molecules per reaction",
            ), definition='The measurement unit associated with the LOQ.'),
            CategoryTerm("probeQuencher", (
                "probe quencher", "quencher dye", "3' quencher", "dark quencher",
                "double-quenched probe", "quenched with", "quencher at 3' end", "internal quencher",
                "3' quenching group",
            ), definition='The quencher molecule attached to a fluorescent probe, such as BHQ, ZEN, Iowa Black, TAMRA, or MGB-associated quenching chemistry.'),
            CategoryTerm("probeReporter", (
                "reporter dye", "reporter fluorophore", "fluorescent reporter", "5' reporter",
                "probe labeled with", "probe labelled with", "fluorescent label", "5' fluorophore",
                "reporter at 5' end",
            ), definition='The fluorescent reporter/fluorophore attached to the probe, such as FAM, HEX, VIC, or Cy5.'),
            CategoryTerm("probe_conc", (
                "probe concentration", "final probe concentration", "probe at", "nM probe", "uM probe",
                "probe stock concentration", "hydrolysis probe concentration",
                "probe concentration per reaction", "final concentration of probe", "probe diluted to",
            ), definition='The reported concentration of the assay probe, preserving whether it is stock or final reaction concentration when stated.'),
            CategoryTerm("probe_ref", (
                "probe described by", "probe reference", "probe developed by", "probe published in",
                "probe adapted from", "probe according to", "previously described probe",
                "probe citation", "probe DOI", "probe source",
            ), definition='The publication, citation, DOI, or other source describing the probe.'),
            CategoryTerm("probe_seq", (
                "probe sequence", "hydrolysis probe sequence", "probe 5'-3'",
                "oligonucleotide probe sequence", "internal probe sequence",
                "probe nucleotide sequence", "probe oligo", "probe sequence was", "probe:",
            ), definition="The nucleotide sequence of the probe in the 5'->3' direction."),
            CategoryTerm("std_seq", (
                "standard DNA sequence", "standard sequence", "sequence of the standard",
                "synthetic standard sequence", "plasmid insert sequence", "standard template sequence",
                "control DNA sequence", "calibration standard sequence", "DNA standard sequence",
                "standard oligonucleotide sequence",
            ), definition='The nucleotide sequence of the DNA standard or calibration template used for targeted quantification.'),
            CategoryTerm("std_source", (
                "standard purchased from", "standard obtained from", "standard supplied by",
                "source of standard", "standard DNA provided by", "synthetic standard purchased",
                "commercial standard obtained", "standard manufactured by", "standard synthesized by",
                "standard provider",
            ), definition='Where the DNA/calibration standard came from, such as a commercial supplier, synthesis provider, organism, or laboratory source.'),
            CategoryTerm("std_type", (
                "type of standard", "plasmid standard", "genomic DNA standard", "gDNA standard",
                "amplicon standard", "synthetic double-stranded DNA", "synthetic DNA standard",
                "PCR product standard", "standard template type", "calibration DNA type",
            ), definition='The physical/type category of standard used, such as plasmid DNA, genomic DNA, synthetic double-stranded DNA, or PCR amplicon.'),
            CategoryTerm("targeted_detection_method_additional", fallback_only=True, definition='Other useful information describing targeted qPCR/ddPCR detection that does not fit one of the more specific detection fields.'),
            CategoryTerm("thresholdQuantificationCycle", (
                "fluorescence threshold", "threshold value", "threshold line",
                "fluorescence signal threshold", "delta Rn threshold", "RFU threshold",
                "threshold set to", "amplification threshold", "fluorescence cutoff", "threshold level",
            ), definition='The explicit fluorescence threshold level used to determine the qPCR quantification cycle. This is the signal threshold, not the resulting Ct/Cq cycle number.'),
        ),
    ),
    SectionCategory(
        name="pcr2_indexing",
        label="PCR2 / indexing PCR",
        keywords=(
            "second PCR", "second-round PCR", "second round PCR", "second-step PCR", "second step PCR",
            "second amplification", "second amplification step", "PCR2", "PCR 2",
            "second PCR amplification", "indexing PCR", "index PCR", "indexing reaction",
            "index amplification", "barcode PCR", "barcoding PCR", "barcode amplification", "adapter PCR",
            "adapter amplification", "adapter-tagging PCR", "index adapter PCR", "dual-index PCR",
            "dual indexing PCR", "Nextera index PCR", "Nextera indexing PCR", "limited-cycle PCR",
            "limited cycle PCR", "additional PCR", "subsequent PCR", "subsequent amplification",
            "followed by a second PCR", "followed by indexing PCR", "followed by index amplification",
            "indices were added by PCR", "barcodes were added by PCR", "adapters were added by PCR",
        ),
        # Every PCR2 term is deliberately scoped to text already gated
        # into this category (Stage 1 keywords above already require
        # second/indexing-PCR context) so none of these need to
        # independently re-require that context themselves -- an
        # explicit user instruction: "I'd make every PCR2 matcher require
        # second/indexing-PCR context, because otherwise it will steal
        # PCR1 values." Category-scoping (Stage 3 will only ever search
        # within this category's own run-text) satisfies this
        # structurally, without duplicating the requirement per term.
        terms=(
            CategoryTerm("pcr2_amplificationReactionVolume", (
                "second PCR reaction volume", "PCR2 reaction volume", "indexing PCR volume",
                "index PCR volume", "second amplification volume", "second PCR in a volume of",
                "PCR2 total volume", "indexing reaction volume", "uL second PCR",
                "uL indexing PCR",
            ), definition='The total final volume of one second/indexing-PCR reaction.'),
            CategoryTerm("pcr2_analysis_software", (
                "second PCR data analyzed using", "PCR2 data analyzed using",
                "indexing PCR analysis software", "second amplification analyzed with",
                "PCR2 run analyzed", "index PCR data analysis", "PCR2 amplification curves",
                "second PCR instrument software", "index PCR software analysis",
                "PCR2 data processing software",
            ), definition='Software used to analyze the second/indexing-PCR run or amplification data.'),
            CategoryTerm("pcr2_annealingTemp", (
                "second PCR annealing", "PCR2 annealing", "indexing PCR annealing",
                "index PCR annealing", "second amplification annealed at", "PCR2 annealed at",
                "annealing temperature for second PCR", "indexing annealing temperature",
                "second-step annealing", "PCR2 annealing temperature",
            ), definition='The primer-annealing temperature used specifically during the second/indexing PCR.'),
            CategoryTerm("pcr2_commercial_mm", (
                "second PCR master mix", "PCR2 master mix", "indexing PCR master mix",
                "index PCR master mix", "commercial mix for second PCR",
                "second-step commercial master mix", "indexing reaction premix",
                "2x mix for indexing PCR", "PCR2 premixed master mix",
                "master mix used in second PCR",
            ), definition='The commercial pre-made master mix used specifically for PCR2/indexing PCR.'),
            CategoryTerm("pcr2_custom_mm", (
                "second PCR reaction contained", "PCR2 reaction mixture", "indexing PCR contained",
                "second PCR components", "PCR2 custom mix",
                "second amplification mixture consisted of", "index PCR reaction contained",
                "buffer in second PCR", "PCR2 dNTP concentration", "PCR2 polymerase concentration",
            ), definition='The explicitly reported composition of a custom reaction mixture used specifically for PCR2/indexing PCR.'),
            CategoryTerm("pcr2_cond", (
                "second PCR conditions", "PCR2 conditions", "indexing PCR conditions",
                "index PCR cycling conditions", "second amplification conditions",
                "second PCR thermal profile", "PCR2 cycling protocol", "indexing thermal cycling",
                "second-step PCR program", "PCR2 thermocycling conditions",
            ), definition='The full thermal-cycling conditions used for PCR2/indexing PCR.'),
            CategoryTerm("pcr2_cycles", (
                "second PCR cycles", "PCR2 cycles", "indexing PCR cycles", "index PCR cycles",
                "second amplification cycles", "limited-cycle PCR", "8 indexing cycles",
                "cycled for eight cycles", "number of PCR2 cycles", "second-step cycle number",
            ), definition='The number of amplification cycles performed in PCR2/indexing PCR.'),
            CategoryTerm("pcr2_dna_vol", (
                "DNA volume in second PCR", "PCR2 template volume", "template added to second PCR",
                "first PCR product volume", "amplicon volume for indexing PCR",
                "uL PCR1 product", "PCR product added to PCR2", "template volume for index PCR",
                "DNA input to PCR2", "indexing PCR template volume",
            ), definition='The volume of template entering PCR2, often PCR1 product/amplicon rather than original extracted DNA.'),
            CategoryTerm("pcr2_method_additional", fallback_only=True, definition='Additional useful PCR2/indexing-PCR methodological information not captured in another PCR2 field.'),
            CategoryTerm("pcr2_plate_id", (
                "PCR2 plate ID", "second PCR plate", "indexing plate", "index PCR plate",
                "plate identifier", "plate ID", "plate barcode", "indexing PCR plate ID",
                "second-step plate", "PCR2 plate identifier",
            ), definition='The identifier/name assigned to the plate used for PCR2/indexing PCR.'),
            CategoryTerm("pcr2_thermocycler", (
                "second PCR performed on", "PCR2 performed on", "indexing PCR performed on",
                "index PCR performed using", "second amplification performed on",
                "PCR2 thermocycler", "indexing thermocycler", "second PCR thermal cycler",
                "PCR2 instrument", "second-step PCR system",
            ), definition='The manufacturer and model of the thermocycler used specifically for PCR2/indexing PCR.'),
        ),
    ),
    SectionCategory(
        name="library_prep_sequencing",
        label="Library preparation + sequencing",
        keywords=(
            "library preparation", "library prep", "library construction", "sequencing library",
            "libraries were prepared", "libraries were constructed", "library preparation kit",
            "one-step PCR", "two-step PCR", "ligation-based", "adapter ligation", "adapter", "adapters",
            "sequencing adapter", "adapter sequence", "P5", "P7", "Nextera", "index", "indices",
            "indexing", "barcode", "barcodes", "barcoding", "MID", "dual index", "dual indexing", "i5",
            "i7", "fusion primer", "tailed primer", "library purification", "library cleanup",
            "size selection", "size-selected", "AMPure", "AMPure XP", "SPRI", "SPRIselect",
            "Pippin Prep", "BluePippin", "gel purification", "library QC", "library quality control",
            "library quantified", "library quantification", "Qubit", "Bioanalyzer", "TapeStation",
            "Fragment Analyzer", "library concentration", "library molarity", "equimolar", "pooled",
            "pooling", "library pool", "normalized libraries", "sequenced", "sequencing", "sequenced on",
            "sequencing platform", "sequencing instrument", "sequencer", "Illumina", "MiSeq", "MiniSeq",
            "NextSeq", "HiSeq", "NovaSeq", "Ion Torrent", "Ion PGM", "DNBSEQ", "BGISEQ", "MGISEQ",
            "PacBio", "Sequel", "Revio", "Oxford Nanopore", "MinION", "GridION", "PromethION", "454",
            "GS FLX", "sequencing kit", "reagent kit", "sequencing chemistry", "MiSeq Reagent Kit",
            "flow cell", "paired-end", "paired end", "single-end", "single end", "2 x 250", "2x250",
            "2 x 150", "PE250", "PE150", "read length", "sequencing facility", "sequencing center",
            "sequencing centre", "sequenced at",
        ),
        terms=(
            CategoryTerm("adapter_forward", (
                "forward sequencing adapter", "forward adapter sequence", "Read 1 adapter",
                "R1 adapter sequence", "5' sequencing adapter", "forward adapter 5'-3'",
                "P5-side adapter", "forward overhang sequence", "adapter on forward primer",
                "forward library adapter", "forward overhang", "Illumina overhang",
                "primer overhang", "primer tail", "5′ tail", "sequencing tail",
                "fusion primer sequence", "adapter-tailed primer", "adapter-tagged primer",
                "adapter-linked primer",
            ), definition='The nucleotide sequence of the forward/Read-1 sequencing adapter or adapter overhang. Do not return the locus-specific PCR primer unless it is explicitly part of the adapter sequence requested.'),
            CategoryTerm("adapter_reverse", (
                "reverse sequencing adapter", "reverse adapter sequence", "Read 2 adapter",
                "R2 adapter sequence", "3' sequencing adapter", "reverse adapter 5'-3'",
                "P7-side adapter", "reverse overhang sequence", "adapter on reverse primer",
                "reverse library adapter", "reverse overhang", "Illumina overhang",
                "primer overhang", "primer tail", "5′ tail", "sequencing tail",
                "fusion primer sequence", "adapter-tailed primer", "adapter-tagged primer",
                "adapter-linked primer",
            ), definition='The nucleotide sequence of the reverse/Read-2 sequencing adapter or adapter overhang.'),
            CategoryTerm("barcoding_pcr_appr", (
                "one-step PCR", "single-step PCR", "two-step PCR", "two-stage PCR",
                "second PCR indexing", "fusion-primer approach", "tailed-primer approach",
                "indices incorporated during PCR", "adapter ligation", "ligation-based library",
            ), definition='How adapters/indexes/barcodes were incorporated into metabarcoding libraries: one-step PCR, two-step PCR, or ligation-based preparation.'),
            CategoryTerm("lib_screen", (
                "library screening", "library enrichment", "library size selection",
                "library purification", "library cleanup", "library quality control", "library QC",
                "library quantified", "library normalization", "equimolar pooling",
            ), definition='A description of library screening, purification, size selection, enrichment, quantification, QC, normalization, or related preparation before sequencing.'),
            CategoryTerm("instrument", (
                "sequencing instrument", "sequencer model", "instrument model",
                "sequenced using a", "sequenced on a", "sequencing performed on", "sequencer used",
                "manufacturer and model", "sequencing system model", "instrument used for sequencing",
            ), definition='The specific manufacturer/model of the sequencing instrument used.'),
            CategoryTerm("lib_layout", (
                "paired-end sequencing", "paired end reads", "single-end sequencing",
                "single end reads", "2 x 150 bp", "2 x 250 bp", "2x150", "2x250", "PE reads",
                "SE reads",
            ), definition='Whether sequencing reads were generated in paired-end or single-end configuration.'),
            CategoryTerm("platform", (
                "sequencing platform", "sequencing technology", "platform used for sequencing",
                "sequencing platform was", "high-throughput sequencing platform",
                "next-generation sequencing platform", "NGS platform", "platform manufacturer",
                "sequencing technology used", "sequenced using the platform",
            ), definition='The broad sequencing technology/platform family, such as Illumina, Ion Torrent, PacBio SMRT, Oxford Nanopore, or DNBSEQ.'),
            CategoryTerm("seq_kit", (
                "sequencing kit", "sequencing reagent kit", "reagent kit", "sequencing chemistry",
                "flow-cell kit", "sequencing cartridge", "cycle kit", "sequencing reagents",
                "chemistry version", "sequencing kit version",
            ), definition='The name/version of the sequencing reagent kit, sequencing chemistry, flow-cell kit, or cartridge used to generate sequence reads.'),
            CategoryTerm("seq_method_additional", fallback_only=True, definition='Other useful library-preparation or sequencing methodological information not represented by a more specific field.'),
        ),
    ),
    SectionCategory(
        name="raw_read_preprocessing",
        label="Raw read preprocessing",
        keywords=(
            # General
            "raw reads", "raw sequences", "FASTQ", "read preprocessing", "sequence preprocessing",
            "pre-processing", "processed reads", "quality control", "quality filtering",
            "quality filtered", "read filtering", "filtered reads",
            # Demultiplexing
            "demultiplexed", "demultiplexing", "demux", "assigned to samples", "barcode splitting",
            "index matching", "barcode mismatch", "index mismatch", "bcl2fastq", "BCL Convert",
            "qiime demux", "ngsfilter",
            # Primer / adapter trimming
            "primer removal", "primer trimming", "primers removed", "adapter removal",
            "adapter trimming", "adapters removed", "trimmed", "trimming", "Cutadapt", "q2-cutadapt",
            "Trimmomatic", "fastp", "BBDuk", "AdapterRemoval", "Trim Galore",
            # Quality/error filtering
            "Phred", "quality score", "Q score", "Q20", "Q30", "expected error", "expected errors",
            "maxEE", "quality threshold", "quality cutoff", "low-quality reads", "filterAndTrim",
            "fastq_maxee", "truncQ",
            # Length filtering
            "minimum length", "maximum length", "read length cutoff", "length filtering",
            "short reads removed", "reads shorter than", "reads longer than", "MINLEN", "min_len",
            # Paired-end merging
            "paired reads merged", "paired-end merging", "merge pairs", "merged reads", "joined reads",
            "forward and reverse reads", "minimum overlap", "overlap", "mergePairs",
            "fastq_mergepairs", "FLASH", "PEAR",
        ),
        # min_len_tool/min_reads_tool are deliberately kept apart from
        # each other: read-length filtering here vs. feature/ASV-abundance
        # filtering under otu_asv_generation_filtering below -- an
        # explicit user note ("This makes min_len_tool and min_reads_tool
        # much harder to confuse: read length versus feature abundance.").
        terms=(
            CategoryTerm("demux_max_mismatch", (
                "maximum barcode mismatches", "allowed barcode mismatches",
                "barcode mismatch tolerance", "maximum index mismatches", "allowed index mismatches",
                "index mismatch tolerance", "one mismatch allowed", "zero barcode mismatches",
                "mismatches in MID", "demultiplexing mismatch threshold",
            ), definition='The maximum number/rate of barcode or index mismatches permitted while assigning reads to samples during demultiplexing.'),
            CategoryTerm("demux_tool", (
                "demultiplexed using", "demultiplexing performed with", "reads demultiplexed with",
                "barcodes separated using", "reads assigned to samples using",
                "index-based demultiplexing with", "barcode splitting performed with",
                "demultiplexing software", "sample assignment performed using",
                "sequence reads were demultiplexed",
            ), definition='The software/tool used specifically to demultiplex reads--that is, assign multiplexed reads to samples using their barcode/index sequences.'),
            CategoryTerm("error_rate_cutoff", (
                "maximum expected errors", "expected error cutoff", "error threshold",
                "quality cutoff", "quality threshold", "minimum Phred score",
                "mean quality threshold", "reads below Q", "maxEE =", "maximum error rate",
            ), definition='The numerical threshold used to remove or trim reads based on sequencing quality/error, such as maxEE=2 or Q20.'),
            CategoryTerm("error_rate_tool", (
                "quality filtered using", "quality filtering performed with",
                "low-quality reads removed using", "reads filtered by quality using",
                "expected-error filtering using", "Phred filtering performed with",
                "quality trimming using", "read quality filtering software",
                "reads exceeding error threshold removed with", "quality-based filtering performed using",
            ), definition='The software/function used specifically to remove or trim reads according to a sequencing-quality/error threshold.'),
            CategoryTerm("error_rate_type", (
                "expected error", "expected errors", "expected error rate", "Phred score",
                "Phred quality", "quality filter", "quality filtered", "quality filtering",
                "quality trimmed", "quality trimming", "Q score", "quality score",
                "mean read quality", "average quality score", "predicted read accuracy",
            ), definition='The kind of error/quality metric used for filtering, principally Phred quality score, expected error rate, or another explicitly stated metric.'),
            CategoryTerm("merge_min_overlap", (
                "minimum overlap", "minimum overlap length", "minimum paired-end overlap",
                "at least 20 bp overlap", "overlap of at least", "required overlap",
                "minimum bases overlapping", "merge overlap threshold",
                "minimum forward-reverse overlap", "overlap requirement",
            ), definition='The minimum number of overlapping bases required to merge forward and reverse paired-end reads.'),
            CategoryTerm("merge_tool", (
                "paired reads merged using", "paired-end reads merged with",
                "forward and reverse reads merged using", "read pairs joined using",
                "pair merging performed with", "merged sequences using",
                "paired reads assembled using", "merging software", "forward/reverse assembly using",
                "read merging performed by",
            ), definition='The software/tool used specifically to merge paired-end forward and reverse reads.'),
            CategoryTerm("min_len_cutoff", (
                "minimum read length", "minimum sequence length", "length cutoff",
                "reads shorter than", "sequences shorter than", "minimum length threshold",
                "short reads removed below", "minimum retained length", "read length >=",
                "bp minimum length", "trimmed to",
            ), definition='The minimum sequence/read length allowed after preprocessing; shorter reads are removed.'),
            CategoryTerm("min_len_tool", (
                "length filtering performed using", "short reads removed using",
                "reads filtered by length using", "sequence length filtering with",
                "minimum-length filtering using", "reads below length removed with",
                "length-based filtering software", "read length cutoff applied using",
                "short sequence filtering using", "length filter implemented with",
            ), definition='The software/tool used specifically to remove reads because they are shorter than an allowed minimum length.'),
            CategoryTerm("trim_method", (
                "primer trimming performed with", "primer removal using", "adapter trimming using",
                "adapter removal using", "primers were removed with", "adapters were removed with",
                "technical sequences removed using", "primer sequences trimmed with",
                "adapter sequences trimmed with", "trimming software",
            ), definition='The software/method used specifically to remove PCR primers, sequencing adapters, adapter tails, or other technical sequences from reads.'),
            CategoryTerm("trim_param", (
                "minimum adapter overlap", "allowed primer mismatches", "maximum primer mismatches",
                "adapter error rate", "discard untrimmed reads", "anchored primer",
                "anchored adapter", "allow indels", "no indels", "primer trimming parameters",
            ), definition='Parameters controlling primer/adapter trimming, such as mismatch allowance, adapter error rate, minimum overlap, anchored matching, or discard-untrimmed behavior.'),
        ),
    ),
    SectionCategory(
        name="otu_asv_generation_filtering",
        label="OTU/ASV generation + filtering & curation",
        keywords=(
            # General
            "ASV", "ASVs", "amplicon sequence variant", "amplicon sequence variants", "sequence variant",
            "exact sequence variant", "ESV", "OTU", "OTUs", "operational taxonomic unit",
            "operational taxonomic units", "feature", "features", "feature table", "ASV table",
            "OTU table", "abundance table", "sequence table",
            # Denoising / ASV generation
            "denoised", "denoising", "ASVs were inferred", "sequence variants were inferred", "DADA2",
            "Deblur", "UNOISE", "UNOISE2", "UNOISE3", "learnErrors", "error model", "dereplication",
            "dereplicated", "unique sequences",
            # OTU clustering
            "OTU clustering", "clustered into OTUs", "sequences were clustered", "clustering threshold",
            "similarity threshold", "97% similarity", "99% similarity", "sequence identity", "UPARSE",
            "UCLUST", "USEARCH", "VSEARCH", "SWARM", "SUMACLUST", "CD-HIT",
            # Chimera removal
            "chimera", "chimeras", "chimeric", "bimera", "bimeras", "chimera removal",
            "chimera detection", "removeBimeraDenovo", "UCHIME", "uchime_denovo", "uchime_ref",
            # Low-abundance filtering
            "low abundance", "low-abundance", "rare ASVs", "rare OTUs", "minimum reads",
            "minimum read count", "read-count threshold", "relative abundance threshold", "singletons",
            "doubletons", "minsize", "fewer than reads",
            # Contaminant / control filtering
            "contaminant", "contamination filtering", "decontam", "microDecon",
            "negative control filtering", "blank filtering", "blank threshold", "background reads",
            "control threshold",
            # Final feature curation
            "non-target taxa removed", "non-target sequences removed", "host sequences removed",
            "chloroplast removed", "mitochondrial sequences removed", "unclassified removed",
            "unassigned removed", "manual curation", "filtered ASV table", "filtered OTU table",
            "feature table was filtered", "LULU", "replicates combined", "PCR replicates pooled",
        ),
        terms=(
            CategoryTerm("chimera_check_method", (
                "chimeras removed using", "chimera detection performed with",
                "chimeric sequences identified using", "bimeras removed using",
                "chimera checking using", "chimera removal software", "de novo chimera detection",
                "reference-based chimera detection", "chimeric reads filtered using",
                "chimera screening performed with",
            ), definition='The software/method used specifically to detect and/or remove chimeric PCR sequences during sequence processing.'),
            CategoryTerm("chimera_check_param", (
                "chimera parameters", "chimera filtering parameters", "chimera detection threshold",
                "minimum divergence", "abundance skew", "chimera score cutoff",
                "non-default chimera parameters", "chimera reference database parameter",
                "chimera detection settings", "chimera filtering cutoff",
            ), definition='Non-default parameters, thresholds, reference settings, or other settings used during chimera detection/removal.'),
            CategoryTerm("otu_clust_cutoff", (
                "clustered at 97% similarity", "clustered at 99% similarity",
                "OTU similarity threshold", "clustering similarity cutoff",
                "sequence identity threshold for clustering", "percent similarity for OTUs",
                "clustering cutoff", "0.03 distance cutoff", "3% sequence divergence",
                "100% similarity ASVs",
            ), definition='The sequence-similarity/identity threshold used to form OTUs; for exact sequence-variant workflows, preserve the explicitly stated ASV/100%-identity treatment rather than assuming one when absent.'),
            CategoryTerm("otu_clust_tool", (
                "OTUs clustered using", "OTU clustering performed with", "sequences clustered using",
                "ASVs inferred using", "ASV inference performed with",
                "sequence variants inferred using", "denoising performed with",
                "feature inference using", "OTU generation software", "ASV generation software",
            ), definition='The software/method used to generate OTUs or infer ASVs/sequence variants from cleaned reads.'),
            # otu_raw_description is deliberately NOT a CategoryTerm here --
            # per an explicit user request, it's generated by the LLM in its
            # own words from this category's combined run-text
            # (extraction/section_category_extraction.py's
            # _generate_otu_raw_description_fact), not extracted verbatim,
            # since a real paper's own raw-table sentence is often an
            # unhelpful cross-reference ("we employed the same pipeline as
            # [citation]") rather than a real description.
            CategoryTerm("min_reads_cutoff", (
                "minimum read count", "minimum reads", "read-count threshold", "fewer than 10 reads",
                "less than 10 reads", "low-abundance threshold", "relative abundance threshold",
                "minimum feature abundance", "singleton removal threshold", "minimum sequence count",
            ), definition='The minimum abundance/read-count threshold required for an ASV, OTU, sequence, or detection to be retained.'),
            CategoryTerm("min_reads_cutoff_unit", (
                "reads", "read count", "number of reads", "percent reads", "% relative abundance",
                "relative read abundance", "fraction of reads", "reads per sample", "reads per ASV",
                "reads per OTU",
            ), definition='The unit in which the minimum-abundance threshold is expressed, such as number of reads, percent relative abundance, or fraction of reads.'),
            CategoryTerm("min_reads_tool", (
                "low-abundance features removed using", "minimum-read filtering performed with",
                "rare ASVs removed using", "rare OTUs removed using",
                "feature abundance filtering using", "singletons removed using",
                "read-count filtering software", "relative-abundance filtering using",
                "low-count features filtered with", "abundance threshold applied using",
            ), definition='The software/method used specifically to remove low-abundance ASVs, OTUs, sequences, or detections based on a minimum read-count/relative-abundance rule.'),
            # FAIRe defines otu_final_description as a description of
            # processing applied to the FINAL (post-curation) OTU/ASV
            # table -- also multi-sentence.
            CategoryTerm("otu_final_description", (
                "final OTU table", "final ASV table", "final feature table", "curated OTU table",
                "curated ASV table", "final table curation", "post-curation table",
                "final dataset after filtering", "final community table", "final abundance table",
            ), allows_multi_sentence=True, definition='A source-faithful description of processing and curation applied to produce the final OTU/ASV/feature table after filtering. This may span multiple sentences.'),
            CategoryTerm("screen_contam_method", (
                "contaminants screened using", "contaminant removal", "contamination filtering",
                "negative-control filtering", "blank-based filtering",
                "control-based contaminant removal", "features detected in blanks",
                "background contamination threshold", "contaminant sequences removed",
                "contamination screening method",
            ), definition='The method used to identify/remove likely contamination, often using negative controls, blanks, prevalence/frequency rules, or contaminant-detection algorithms.'),
            CategoryTerm("screen_geograph_method", (
                "geographic screening", "geographical screening", "species distribution screened",
                "known geographic range", "known distribution", "outside known range",
                "geographically implausible", "regional occurrence records",
                "distribution information applied", "geographic plausibility",
            ), definition='The method used to remove or flag taxonomic detections based on geographic plausibility or known species distributions.'),
            CategoryTerm("screen_nontarget_method", (
                "non-target taxa removed", "non-target species removed",
                "non-target sequences filtered", "target taxa retained", "off-target taxa removed",
                "excluded non-target taxa", "taxonomic exclusion", "sequences outside target group",
                "non-target screening", "target-group filtering",
            ), definition='The method used to remove organisms/sequences outside the intended taxonomic target of the study.'),
            CategoryTerm("screen_other", (
                "additional screening criteria", "other screening criteria", "manual screening",
                "manual curation criteria", "ecological plausibility filtering",
                "additional sequence screening", "other filtering approach",
                "custom screening criterion", "further curation", "additional biological filtering",
            ), definition='Any other explicitly described biological/ecological screening or curation procedure that is not contamination, geography, or non-target filtering.'),
        ),
    ),
    SectionCategory(
        name="taxonomic_assignment",
        label="Taxonomic assignment",
        keywords=(
            # General
            "taxonomic assignment", "taxonomy assignment", "taxonomy was assigned", "assigned taxonomy",
            "taxonomically assigned", "taxonomic classification", "taxonomic identification",
            "taxonomic annotation", "taxonomy annotation", "classified taxonomically",
            "species identification", "taxonomic identity", "assigned to taxa", "assigned to species",
            "assigned to genus",
            # Assignment software / approaches
            "BLAST", "BLASTn", "MegaBLAST", "megablast", "QIIME 2 feature-classifier",
            "q2-feature-classifier", "classify-sklearn", "naive Bayes", "Naive Bayesian",
            "RDP Classifier", "SINTAX", "IDTAXA", "Kraken", "Kraken2", "PROTAX", "SEPP", "EPA-ng",
            "pplacer", "lowest common ancestor", "LCA", "best hit", "top hit", "sequence similarity",
            # Reference databases
            "reference database", "reference library", "reference sequences", "SILVA", "PR2",
            "GenBank", "NCBI nt", "NCBI nucleotide", "BOLD", "BOLD Systems", "UNITE", "RDP",
            "Greengenes", "MIDORI", "MIDORI2", "MitoFish", "MetaZooGene", "Diat.barcode", "PhytoREF",
            "custom database", "custom reference database", "in-house database",
            # Assignment criteria
            "percent identity", "percentage identity", "% identity", "sequence identity",
            "similarity cutoff", "identity cutoff", "query coverage", "coverage threshold",
            "confidence threshold", "confidence cutoff", "bootstrap", "E-value", "e-value", "best match",
            "best hit", "top hit", "lowest common ancestor", "consensus taxonomy",
            "species-level assignment", "genus-level assignment", "family-level assignment",
            "ambiguous assignment", "multiple matches",
        ),
        terms=(
            CategoryTerm("otu_db", (
                "reference database", "taxonomic reference database", "taxonomy database",
                "reference sequence database", "reference library", "matched against the database",
                "searched against the database", "taxonomy assigned against",
                "reference database version", "reference sequences from",
            ), definition='The reference sequence/taxonomy database used to assign taxonomic identities to OTUs/ASVs/sequences, including version/release when reported.'),
            CategoryTerm("otu_db_custom", (
                "custom reference database", "custom database", "in-house database",
                "locally curated database", "custom reference library", "database constructed from",
                "reference database was built", "database curated for this study",
                "bespoke database", "study-specific reference database",
            ), definition='A description of a custom, locally built, curated, or study-specific reference database used for taxonomic assignment.'),
            CategoryTerm("otu_seq_comp_appr", (
                "sequences aligned using", "sequence comparison performed using",
                "alignment performed with", "queries aligned against",
                "reference sequences searched using", "sequence similarity search performed with",
                "alignment software", "query sequences compared using",
                "reference matching performed using", "taxonomic alignment performed with",
                # User-supplied cues (verbatim), grounded in a real paper
                # (CREST4) that names its classification tool via
                # "taxonomic classification ... was performed using <tool>"
                # rather than describing an explicit "alignment" step.
                "taxonomic classification", "taxonomic assignment", "taxonomy assigned",
                "assigned taxonomy", "taxonomic identification", "classified taxonomically",
                "sequences classified", "OTUs classified", "ASVs classified",
                "representative sequences classified", "compared against reference",
                "compared with reference sequences", "searched against", "aligned against",
                "matched against", "sequence similarity search", "best hit", "reference database",
                "reference sequences", "BLAST", "BLASTn", "MegaBLAST", "USEARCH", "VSEARCH",
                "CREST", "CREST4",
            ), definition='Extract the name and version, if available, of the software, algorithm, or sequence-comparison tool used to compare OTU/ASV/feature sequences against reference sequences for taxonomic assignment. This includes alignment, similarity-search, or taxonomic sequence-comparison tools. Extract the tool only when the text shows that it was used to assign or identify taxonomy. Do not extract software used only for OTU/ASV generation, read trimming, assembly, or unrelated sequence comparisons.'),
            # tax_assign_cat is deliberately NOT a CategoryTerm here -- the
            # real FAIRe field is a controlled enum (sequence similarity /
            # sequence composition / phylogeny / probabilistic / other),
            # and per an explicit user confirmation ("this isn't in the
            # text, so needs to be inferred unfortunately") the correct
            # category is essentially never stated verbatim (a paper says
            # "CREST4"/"LCA", never the literal words "sequence
            # similarity"). Stage 3's own verbatim guard would reject any
            # such classification outright, so this field is handled
            # instead by extraction/search_flags.py's tax_assign_cat
            # LLMJudgedSearchField, which has no such guard and is
            # explicitly allowed to classify/infer into the fixed enum.
            CategoryTerm("tax_class_collapse", (
                "lowest common ancestor", "LCA", "collapsed to higher taxonomic level",
                "assigned to lowest common taxon", "lowest common taxonomic rank",
                "collapsed to genus", "collapsed to family", "consensus taxonomic level",
                "ambiguous hits assigned to", "taxonomic rank reduced to",
                # User-supplied cues (verbatim).
                "lowest common taxon", "lowest shared taxon", "lowest shared taxonomic rank",
                "common ancestor", "consensus taxonomy", "consensus classification",
                "consensus assignment", "assigned to genus level", "assigned to family level",
                "assigned to higher taxonomic level", "higher taxonomic rank", "resolved to genus",
                "retained at genus level", "reported at genus level", "assigned at the lowest rank",
                "lowest confidently assigned rank", "multiple hits", "ambiguous hits",
                "equally matching sequences", "best hits", "shared taxonomy", "common taxonomy",
            ), definition='Extract the rule used to resolve uncertain, ambiguous, or multiple taxonomic matches by assigning a sequence to a broader taxonomic rank. Examples include lowest common ancestor (LCA), consensus taxonomy, assignment to the lowest shared taxonomic rank, or falling back from species to genus/family when confidence or identity is insufficient. Do not require the paper to use the word "collapse."'),
            CategoryTerm("tax_class_id_cutoff", (
                "minimum percent identity", "minimum sequence identity", "identity cutoff",
                "identity threshold", "% identity required", "at least 97% identity",
                "sequence identity >=", "percent identity threshold", "minimum match identity",
                "taxonomic identity cutoff",
            ), definition='The minimum sequence-identity/percent-identity criterion required to accept a taxonomic assignment.'),
            CategoryTerm("tax_class_query_cutoff", (
                "minimum query coverage", "query coverage cutoff", "query coverage threshold",
                "% query coverage", "at least 90% query coverage", "alignment coverage threshold",
                "minimum query alignment", "query coverage >=", "fraction of query aligned",
                "minimum sequence coverage",
            ), definition='The minimum query/alignment coverage required to accept a taxonomic assignment.'),
            # tax_class_other is deliberately NOT a CategoryTerm here --
            # per an explicit user request ("tax_class_other can be all
            # classified 'TAXONOMIC ASSIGNMENT'. can ask the LLM to
            # summarize based on the section classified 'TAXONOMIC
            # ASSIGNMENT'."), it's generated by the LLM in its own words
            # from this category's combined run-text
            # (extraction/section_category_extraction.py's
            # _generate_tax_class_other_fact), not extracted verbatim.
        ),
    ),
)


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    category.name: tuple(_term_pattern(term) for term in category.keywords)
    for category in SECTION_CATEGORIES
}

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_REFERENCE_SECTION_HEADER_RE = re.compile(r"^\s*(references|bibliography|literature\s+cited)\s*$", re.IGNORECASE)
# A numbered reference-list entry, e.g. "27. P. Engstrom, C. R. Penton, ...
# Anaerobic ammonium oxidation in deep-sea sediments..." -- confirmed against
# a real PNAS supplementary-methods reference list.
_NUMBERED_REFERENCE_ENTRY_RE = re.compile(r"^\s*\d{1,3}\.\s+[A-Z]")
# Tolerant of PDF-extraction whitespace artifacts (a real supplementary
# PDF rendered "Fig. S 5." with a stray space between "S" and "5") --
# without \s* between the optional "S" and the digit, this silently missed
# real captions.
_FIGURE_TABLE_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table)\s*s?\s*\d", re.IGNORECASE)


def _drop_reference_section(paragraphs: list[str]) -> list[str]:
    """References sections are a contiguous block, almost always at the
    end of a methods/supplement document -- once the first reference-list
    marker is seen, every remaining paragraph is dropped rather than
    filtered one at a time, since a differently-formatted later entry
    (missing the leading number, e.g. a continuation line) could
    otherwise slip through a per-paragraph-only filter."""
    for index, paragraph in enumerate(paragraphs):
        stripped = paragraph.strip()
        if _REFERENCE_SECTION_HEADER_RE.match(stripped) or _NUMBERED_REFERENCE_ENTRY_RE.match(stripped):
            return paragraphs[:index]
    return paragraphs


def split_into_paragraphs(text: str) -> list[str]:
    """Blank-line-delimited paragraphs, references section dropped
    (see `_drop_reference_section`) and bare figure/table captions
    excluded one at a time (these can appear anywhere, not just at the
    end) -- both confirmed as real false-positive sources against a real
    PNAS supplementary-methods document (a cited paper's own title
    matching "amplicon"/"primer"; bare "Fig. S4. ..." caption lines
    matching "amplicon")."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    paragraphs = _drop_reference_section(paragraphs)
    return [p for p in paragraphs if not _FIGURE_TABLE_CAPTION_RE.match(p)]


def candidate_categories_for_paragraph(paragraph: str) -> frozenset[str]:
    """Stage 1: the cheap, deterministic keyword gate. A paragraph that
    matches zero categories here never reaches the LLM at all (Stage 2,
    not built in this module)."""
    return frozenset(
        category_name
        for category_name, patterns in _CATEGORY_PATTERNS.items()
        if any(pattern.search(paragraph) for pattern in patterns)
    )


# Most categories are correctly present after a single real keyword hit
# anywhere in the paper. targeted_qpcr_ddpcr_detection is the deliberate
# exception, per a real live-paper finding: a paper can mention "qPCR"
# exactly once, for a purpose that has nothing to do with a targeted qPCR/
# ddPCR assay (e.g. "total cell numbers were taken as the sum of the
# archaeal and bacterial 16S rRNA genes as determined by qPCR" -- a
# quantification aside, not the paper's own assay). A genuine qPCR/ddPCR
# paper reliably mentions this vocabulary more than once (assay setup,
# standard curve, LOD/LOQ, controls, ...), so this category requires at
# least 2 whole-document keyword mentions before being considered present
# at all -- every other category keeps the original single-mention gate.
_CATEGORY_MIN_DOCUMENT_MENTIONS: dict[str, int] = {"targeted_qpcr_ddpcr_detection": 2}


def _category_mention_count(category_name: str, texts: list[tuple[str, str]]) -> int:
    patterns = _CATEGORY_PATTERNS[category_name]
    return sum(
        len(pattern.findall(text)) for _title, text in texts for pattern in patterns
    )


def low_confidence_categories(texts: list[tuple[str, str]]) -> frozenset[str]:
    """Categories in `_CATEGORY_MIN_DOCUMENT_MENTIONS` whose whole-document
    keyword-mention count falls short of their configured minimum --
    callers subtract this from any per-paragraph candidate set so a single
    incidental mention never gates a whole category's worth of Stage 3
    extraction (or the INTERNAL_SECTION_DETECTION_FIELDS diagnostic
    column) on its own."""
    return frozenset(
        category_name
        for category_name, minimum in _CATEGORY_MIN_DOCUMENT_MENTIONS.items()
        if _category_mention_count(category_name, texts) < minimum
    )


def group_sentences_into_category_runs(
    tagged_sentences: list[tuple[str, frozenset[str]]],
) -> dict[str, str]:
    """Stage 2.5: given a paragraph's sentences, each already tagged
    (Stage 2, LLM) with zero or more category names, groups each present
    category's own text. A run for category X starts at a sentence
    tagged X and extends through following untagged (bridging) sentences,
    ending the instant a sentence tagged with a DIFFERENT category (not
    including X, even if X is also present on that same sentence)
    appears. Separate runs of the same category within one paragraph
    (category flips away and back) are concatenated together, in order,
    with the interrupting category's own sentences excluded throughout --
    an explicit user specification: "if a paragraph goes from cat 1 to
    cat 2 back to cat 1, the two cat 1 [runs] will be grouped
    consecutively with cat 2 removed."

    Computed independently per category (not as a single linear pass)
    so a sentence tagged with more than one category correctly continues
    every one of its own categories' runs while still breaking any OTHER
    category's currently-open run."""
    present_categories = {category for _, categories in tagged_sentences for category in categories}
    return {
        category: " ".join(_runs_for_category(tagged_sentences, category))
        for category in present_categories
    }


def _runs_for_category(tagged_sentences: list[tuple[str, frozenset[str]]], category: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    in_run = False
    for sentence, categories in tagged_sentences:
        if category in categories:
            current.append(sentence)
            in_run = True
        elif not categories:
            if in_run:
                current.append(sentence)
        else:
            if current:
                runs.append(" ".join(current))
            current = []
            in_run = False
    if current:
        runs.append(" ".join(current))
    return runs


_PCR_0_1_SOURCE_CATEGORIES = ("pcr1_primary_amplification_0_1", "targeted_qpcr_ddpcr_detection_0_1")


def derive_pcr_0_1_from_category_detection(
    section_category_facts: list[RawFactCandidate],
) -> RawFactCandidate | None:
    """`pcr_0_1` gates a broad swath of the existing PCR/assay checklist
    (extraction/faire_fields.py's "PCR / assay setup" group, several
    LLMJudgedSearchField entries) and used to be its own independent
    regex scan (extraction/search_flags.py's former standalone `pcr_0_1`
    TextSearchFlag entry, now removed) that explicitly matched
    `qPCR`/`ddPCR` as well as plain `PCR` -- an explicit user instruction
    ("this is essentially if category PCR1=True... worth rewiring so we
    don't duplicate its process") asked to derive it from category
    detection instead, so the two can never disagree.

    Derived from PCR1 *or* targeted qPCR/ddPCR detection, not PCR1 alone:
    a qPCR/ddPCR-only paper never uses the bare word "PCR"/"amplified" at
    all (confirmed by a real, pre-existing test fixture -- "A TaqMan qPCR
    assay used a FAM reporter dye and BHQ quencher" triggers
    `targeted_qpcr_ddpcr_detection_0_1` but not
    `pcr1_primary_amplification_0_1`, since word-boundary matching on
    bare "PCR" never matches inside "qPCR"), and the old regex-based flag
    already covered this case explicitly via its own separate `qPCR`/
    `ddPCR` patterns. Deriving from PCR1 alone would silently narrow
    `pcr_0_1`'s trigger conditions and stop unlocking the PCR checklist
    group for exactly this kind of real paper."""
    facts_by_type = {fact.fact_type_candidate: fact for fact in section_category_facts}
    for category_fact_type in _PCR_0_1_SOURCE_CATEGORIES:
        fact = facts_by_type.get(category_fact_type)
        if fact is not None:
            return RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="pcr_0_1",
                raw_field_name="pcr_0_1",
                raw_value="1",
                source_locator=fact.source_locator,
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=fact.evidence_quote,
                confidence_metadata={"detector": f"derived_from_{category_fact_type}"},
            )
    return None


def detect_section_categories_present(
    texts: list[tuple[str, str]], *, locator_prefix: str
) -> list[RawFactCandidate]:
    """One `<category>_0_1` fact per category with at least one
    keyword-matching, non-reference/non-caption paragraph anywhere in
    `texts` -- a diagnostic coverage signal for tuning the keyword lists
    above against real papers, not itself the extraction path. Uses the
    SAME paragraph-level gate `split_into_paragraphs`/
    `candidate_categories_for_paragraph` the real pipeline will use, so a
    "detected" here is a faithful preview of what Stage 1 would actually
    forward -- not a looser, sentence-only check that could disagree with
    it."""
    found: dict[str, str] = {}
    for title, text in texts:
        for paragraph in split_into_paragraphs(text):
            for category_name in candidate_categories_for_paragraph(paragraph):
                found.setdefault(category_name, paragraph)
    suppressed = low_confidence_categories(texts)
    facts: list[RawFactCandidate] = []
    for category in SECTION_CATEGORIES:
        if category.name in suppressed:
            continue
        evidence = found.get(category.name)
        if evidence is None:
            continue
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=f"{category.name}_0_1",
                raw_field_name=f"{category.name}_0_1",
                raw_value="1",
                source_locator=f"{locator_prefix}:section_category_detection:{category.name}",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=evidence[:500],
                confidence_metadata={"detector": "section_category_keyword_gate"},
            )
        )
    return facts
