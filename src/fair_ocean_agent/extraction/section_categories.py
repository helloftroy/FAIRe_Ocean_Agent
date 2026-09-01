"""PCR/library-prep/bioinformatics methods-text categorization: the
keyword data and deterministic (no-LLM) building blocks for a two-stage
categorize-then-extract pipeline, per an explicit user design (see
README's "Section-categorization pipeline for PCR-family fields" once
written up).

Three deterministic pieces live here, all directly testable without an
LLM:

1. `SECTION_CATEGORIES` -- the keyword taxonomy itself. Scoped down to
   just `sample_prep` per an explicit user request: a live 6-study audit
   found this one category accounted for ~72% of everything the whole
   pipeline ever produced, while several others (targeted qPCR/ddPCR,
   PCR2/indexing) produced zero real facts, and Stage 2 (categorize_
   paragraphs) was the single most expensive step in the entire
   extraction pipeline. The other 8 categories (PCR1, targeted qPCR/
   ddPCR, PCR2/indexing, assay definition, library prep + sequencing, raw
   read preprocessing, OTU/ASV generation + filtering, taxonomic
   assignment) were removed entirely, not merely disabled -- "for now",
   per the user's own framing; every term that had a real fallback
   elsewhere (broad-checklist, quote-judged, deterministic-regex,
   structured-API) keeps working through that mechanism, and the two
   that didn't (otu_seq_comp_appr, screen_contam_method) got a new
   quote-judged companion in extraction/search_flags.py so they weren't
   lost. More categories get appended here as they're defined -- nothing
   else in this module needs to change shape when that happens.
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
            # Sample collection -- deliberately folded into this same
            # category rather than a new classifier, per an explicit user
            # instruction ("i don't think this needs its own classifier...
            # if its included in the paper, its going to be right before
            # the sample"): collection language sits immediately adjacent
            # to preparation language in a paper's methods narrative.
            "Niskin bottle", "grab sampler", "van Veen grab", "gravity corer", "box corer",
            "plankton net", "collected using a", "collected with a", "collection device",
            "sampling device", "samples were collected", "collected by", "hand-collected",
            "collection method", "sampling method", "obtained by", "L of water were collected",
            "g of sediment were collected", "total volume collected", "pooled from",
            "composite sample", "composed of", "combined into one sample", "subsampled from",
            "subsample of", "parent sample", "negative control", "positive control",
            "PCR standard", "no-template control", "blank sample", "field blank",
            "cruise", "expedition", "voyage", "campaign", "sampling mission", "sampling campaign",
            "aboard R/V", "aboard the R/V", "research vessel", "station", "sampling station",
            # In-situ physicochemical/environmental measurements taken at the
            # sampling event -- bundled into one x_env_var_block term rather
            # than one CategoryTerm per variable, per an explicit user
            # request ("group all the above together, just to make it
            # easier for Qwen").
            "temperature", "water temperature", "in situ temperature", "in-situ temperature", "Temp",
            "salinity", "Sal", "dissolved oxygen", "pH", "chlorophyll", "chlorophyll a", "Chl a", "Chl-a",
            "chloro",
            "suspended particulate matter", "SPM", "total suspended solids", "TSS",
            "suspended solids", "organic matter", "particulate organic matter",
            "particulate organic carbon", "POC", "particulate organic nitrogen", "PON",
            "dissolved organic carbon", "DOC", "dissolved organic nitrogen", "DON",
            "dissolved inorganic carbon", "DIC", "dissolved inorganic nitrogen", "DIN",
            "total organic carbon", "TOC", "total nitrogen", "TN", "total dissolved nitrogen", "TDN",
            "total carbon", "total particulate carbon", "TPC", "nitrate", "NO3", "nitrite", "NO2",
            "physicochemical", "physico-chemical", "water quality parameters",
            "environmental parameters", "environmental variables", "in situ measurements",
            "in-situ measurements",
        ),
        terms=(
            # Sample collection -- device/method/size describe how the
            # ORIGINAL environmental sample was obtained, distinct from
            # samp_mat_process (post-collection handling) below.
            CategoryTerm("samp_collect_device", (
                "Niskin bottle", "corer", "gravity corer", "box corer", "van Veen grab",
                "grab sampler", "plankton net", "collected using a", "collected with a",
                "sampler", "syringe", "swab", "pump",
            ), definition='The physical instrument, container, sampler, or equipment used to collect the environmental sample, such as a Niskin bottle, corer, net, grab sampler, syringe, swab, or pump.'),
            CategoryTerm("samp_collect_method", (
                "collected by", "samples were collected using", "collection method",
                "sampling method", "obtained by", "hand-collected", "collected via",
                "samples were obtained by",
                # Real papers commonly say "taken", not "collected" --
                # confirmed live, a real gap (10.1371/journal.pone.0303937's
                # own "were taken by ship using an integrating water
                # sampler" matched none of the original cues).
                "were taken by", "were taken using", "taken by ship", "depth integration",
                "integrated samples", "integrating water sampler",
            ), definition='The procedure used to obtain the environmental or biological sample from its source. Describe how the sample was collected, including relevant collection technique, depth integration, coring, pumping, netting, swabbing, grabbing, or similar actions. This is about collection from the environment, not later filtration, storage, DNA extraction, PCR, or sequencing.'),
            # Moved here from a deterministic ControlledSearchField per an
            # explicit user request: sterilization/decontamination of
            # sampling equipment is itself a sample-collection concept, not
            # a project-level free-text search.
            CategoryTerm("sterilise_method", (
                "sterile", "sterilized", "sterilised", "decontaminated", "autoclaved", "bleach",
                "sodium hypochlorite", "ethanol", "UV", "UV-sterilized", "flamed", "DNA-away",
                "RNase-free", "DNase-free",
                "flame sterilised", "decontamination", "decontaminate", "clean room", "DNAZap",
                "single-use equipment",
                # Real gap found live: "The water sampler was sanitised and
                # rinsed in the water body between samples to avoid
                # contamination from previous sites" -- a real, common
                # British-spelling wording ("sanitised") this cue list
                # never covered at all, so the sentence fell through
                # entirely to screen_contam_method's own bare "contamination"
                # match instead (a separate bug, see search_flags.py's
                # _SCREEN_CONTAM_METHOD_CONTEXT_RE).
                "sanitised", "sanitized", "sanitise", "sanitize", "rinsed between samples",
                "rinsed in the water body between samples",
            ), definition=(
                'Extract the method used to sterilize or decontaminate sampling, laboratory, or '
                'processing equipment/materials to prevent cross-sample contamination before or between '
                'samples. Include the sterilizing agent or procedure and, when stated, relevant '
                'conditions such as concentration, exposure time, temperature, UV treatment, autoclaving, '
                'flaming, or bleach treatment.'
            )),
            CategoryTerm("samp_size", (
                "L of water were collected", "liters of water were collected",
                "g of sediment were collected", "total volume collected",
                "mass of sediment collected", "volume of water sampled", "kg of sediment",
                # Real gap found live (10.3390/microorganisms10030558): water
                # samples are routinely reported as "X liters of seawater was/
                # were filtered" -- filtration IS the collection/concentration
                # step for a water sample, not a separate later step, so this
                # phrasing is at least as common as "were collected" for
                # marine/aquatic eDNA papers specifically.
                "liters of seawater was filtered", "liters of seawater were filtered",
                "L of seawater was filtered", "L of seawater were filtered",
                "liters of water was filtered", "liters of water were filtered",
                "L of water was filtered", "L of water were filtered",
            ), definition='The total amount of environmental material originally collected for that sample, such as 10 L of water or 500 g of sediment. Do not confuse with the smaller amount later used for DNA extraction.'),
            CategoryTerm("samp_size_unit", (
                "L of water", "mL of water", "g of sediment", "mg of sediment", "kg of sediment",
                "cm2 of surface", "cm² of surface",
                "L of seawater", "mL of seawater",
            ), definition='The unit associated with `samp_size`, such as L, mL, g, mg, cm², or another explicitly reported unit.'),
            CategoryTerm("sample_composed_of", (
                "pooled from", "combined from", "composite sample", "composed of",
                "pooled samples", "combined into one sample", "samples were pooled",
            ), definition='The material or component(s) that make up the sample, especially when a sample contains multiple identifiable constituents or pooled components. Preserve the source description rather than inferring composition.'),
            CategoryTerm("sample_derived_from", (
                "subsampled from", "derived from sample", "subsample of", "parent sample",
                "originated from sample", "a subsample of sample",
            ), definition='The original sample, material, or parent specimen from which the current sample was produced by subsampling or processing. Use this to express parent→derived-sample relationships.'),
            CategoryTerm("internal_expedition_id", (
                "cruise", "cruise ID", "cruise_id", "expedition", "expedition ID",
                "voyage", "campaign", "sampling mission", "sampling campaign", "aboard R/V",
                "aboard the R/V", "research vessel", "station", "station ID", "station_id",
            ), definition='Extract the explicit name or identifier of the research expedition, campaign, ship cruise/voyage, broader sampling mission, or named sampling station/station series. Examples: Tara Oceans, MOSAiC, Malaspina 2010. Look for phrases such as cruise, expedition, voyage, station, campaign, or aboard R/V ... during .... Preserve the source identifier exactly.'),
            # Bundles real but individually low-yield physicochemical
            # FAIRe sampleMetadata fields (diss_inorg_carb, diss_inorg_nitro,
            # diss_org_carb, diss_org_nitro, nitrate, nitrite, org_matter,
            # part_org_carb, part_org_nitro, ph, suspend_part_matter,
            # tot_carb, tot_diss_nitro, tot_nitro, tot_org_carb,
            # tot_part_carb, chlorophyll, temp, salinity, dissolved oxygen,
            # etc.) into ONE term, per explicit user requests. The raw
            # pipe-joined value is exported as its own single column
            # (mapping/rules.py's own MappingRule, exports/faire.py's
            # CUSTOM_ENV_VAR_BLOCK_FIELD) and the individual fields it
            # replaces are suppressed from export entirely (exports/
            # faire.py's SAMPLE_METADATA_SUPPRESSED_FIELDS) -- deliberately
            # NOT decomposed back into those real columns.
            CategoryTerm("x_env_var_block", (
                "temperature", "water temperature", "Temp", "salinity", "Sal", "dissolved oxygen", "pH",
                "chlorophyll", "chlorophyll a", "Chl a", "Chl-a", "chloro", "suspended particulate matter",
                "SPM", "total suspended solids", "TSS", "suspended solids", "organic matter",
                "particulate organic matter", "particulate organic carbon", "POC",
                "particulate organic nitrogen", "PON", "dissolved organic carbon", "DOC",
                "dissolved organic nitrogen", "DON", "dissolved inorganic carbon", "DIC",
                "dissolved inorganic nitrogen", "DIN", "total organic carbon", "TOC",
                "total nitrogen", "TN", "total dissolved nitrogen", "TDN", "total carbon",
                "total particulate carbon", "TPC", "nitrate", "NO3", "nitrite", "NO2",
                "physicochemical", "physico-chemical", "water quality parameters",
                "environmental parameters", "environmental variables", "in situ measurements",
                "in-situ measurements",
            ), allows_multi_sentence=True, definition=(
                'Environmental measurements: Extract measured environmental variables associated '
                'with sampling, including temperature, salinity, dissolved oxygen, carbon-containing '
                'environmental variable, nitrogen-containing environmental variable, pH, chlorophyll, '
                'suspended particulate matter, and similar physicochemical measurements. Preserve the '
                'variable name, value, unit, depth/location/context, and measurement method when '
                'stated. Format each variable as "name/formula: value unit" (using whichever name or '
                'chemical formula is given in the text); if the quote reports more than one variable, '
                'return one object per variable so they can be combined.'
            )),
            # samp_category: deliberately NOT wired into a MappingRule (see
            # mapping/rules.py) -- capturing the raw evidence is useful,
            # but broadcasting a single extracted "negative control"/"PCR
            # standard" quote onto every authoritative sample in the study
            # would mislabel the real environmental samples too, since this
            # field identifies *which specific* sample is a control, unlike
            # every other sample_prep field's "same process applied to all
            # samples" semantics. Per an explicit user instruction, real
            # per-sample resolution should come from API/structured data
            # (already covered by the existing SAMPLE-level MappingRules);
            # this term exists only so the raw quote is visible for review.
            CategoryTerm("samp_category", (
                "negative control", "positive control", "PCR standard", "no-template control",
                "blank sample", "field blank", "served as a control", "included as a control",
            ), definition='Whether this text explicitly identifies a described sample as a negative control, positive control, PCR standard, or blank, as opposed to a regular environmental sample.'),

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
                # Real papers describe the physical handling steps, not the
                # word "processing" itself -- confirmed live, a real gap
                "filter membrane was", "membrane was curled", "curled up", "transferred into",
                "conical tube", "immersed", "cut into pieces", "chopped up",
            ), allows_multi_sentence=True, definition='Physical or chemical processing applied to the collected sample before nucleic-acid extraction. This can include filtration, pre-filtration, sieving, subsampling, homogenization, grinding, cutting, centrifugation, precipitation, freeze-drying, drying, washing, concentrating, or other preparation of the sample material. Do not include PCR, library preparation, sequencing, or bioinformatics.'),
            CategoryTerm("prep_method_additional", fallback_only=True, definition='Additional useful details about sample preparation that do not fit another specific sample-preparation field. Use this for important procedural details such as pressure used during filtration, order of multiple preparation steps, unusual apparatus, stopping criteria, special handling, or other preparation conditions. Do not duplicate information that is already fully captured by specific fields such as filter pore size, filter material, storage temperature, or extraction kit.'),

            # Filtration
            CategoryTerm("filter_material", (
                "filter material", "cellulose filter", "cellulose ester filter", "nylon filter",
                "glass fiber filter", "polyethersulfone filter", "PES filter", "membrane filter",
                "filter membrane",
            ), definition='Material from which the sample filter membrane is made.'),
            CategoryTerm("filter_name", (
                "Sterivex", "filter cartridge", "cartridge filter", "commercial filter",
                "filter brand", "filter product",
            ), definition='Commercial name or brand/model of the filter used. Do not return cross-reference placeholders such as "see below" or "see above".'),
            CategoryTerm("filter_diameter", (
                "filter diameter", "diameter of the filter", "mm filter", "mm diameter filter",
            ), definition='Physical diameter of a circular filter, usually in mm. Do not confuse with pore size.'),
            CategoryTerm("filter_surface_area", (
                "filter surface area", "filter area", "surface area of the filter",
            ), definition='Total surface area of the filter membrane, usually in mm².'),
            CategoryTerm("size_frac", (
                "pore size", "filtering pore size", "filter pore size", "µm filter", "um filter",
                "μm filter", "µm membrane", "μm membrane", "um membrane",
                "180-µm", "180-μm", "180-um", "5.0-µm", "5.0-μm", "5.0-um",
                "0.22 µm", "0.22 μm", "0.22 um", "0.22-µm", "0.22-μm", "0.22-um",
                "0.2 µm", "0.2 μm", "0.2 um", "0.2-µm", "0.2-μm", "0.2-um",
                "0.45 µm", "0.45 μm", "0.45 um", "0.45-µm", "0.45-μm", "0.45-um",
            ), definition='Pore size of the main filter used to collect/sample material, in µm.'),
            CategoryTerm("prefilter_material", (
                "pre-filter material", "prefilter material", "pre-sort material",
            ), definition='Material used for a pre-filter or pre-sort step before the main sample filtration.'),
            # Real papers overwhelmingly describe the mechanism (compressed
            # air, overpressure, a named pump) rather than ever using the
            # bare phrase "active filtration" -- confirmed live, a real gap
            # (10.1371/journal.pone.0303937's own compressed-air/overpressure
            # pressure-barrel filtration rig matched none of the original
            # cues, so this term was never even offered as a candidate).
            CategoryTerm("filter_passive_active_0_1", (
                "active filtration", "passive filtration", "pumped through the filter",
                "submerged filter", "passive sampler", "actively filtered", "passively filtered",
                "compressed air", "overpressure", "pressure vessel", "pressure barrel",
                "vacuum pump", "vacuum filtration", "peristaltic pump", "syringe pressure",
                "under pressure", "pressurized", "forced through the filter", "flowmeter",
                "flow rate", "flowrate", "filtered through", "filtered onto", "filtered on",
                "filtration through", "filtration onto", "filter membrane", "membrane filter",
                "filter cartridge", "cartridge filter", "Sterivex", "pore size", "µm filter",
                "μm filter", "um filter", "µm membrane", "μm membrane", "um membrane",
                "0.22 µm", "0.22 μm", "0.22 um", "0.22-µm", "0.22-μm", "0.22-um",
                "0.2 µm", "0.2 μm", "0.2 um", "0.2-µm", "0.2-μm", "0.2-um",
                "0.45 µm", "0.45 μm", "0.45 um", "0.45-µm", "0.45-μm", "0.45-um",
            ), definition='Whether the filtration/collection of water or air was active or passive. Return exactly 1 for active filtration and 0 for passive filtration. Active = 1: water or air is explicitly forced or moved through a filter using a pump, compressed air, vacuum, fan, syringe pressure, peristaltic pump, pressure vessel, or another mechanical driving force. Passive = 0: filtration is described but no active forcing mechanism is stated, or the filter/material is simply exposed, submerged, suspended, or stationed in the environment and material accumulates without mechanically forcing water or air through it. Do not interpret 0 as "no filtration" or 1 as "filtration present" -- this field distinguishes active versus passive filtration only.'),
            CategoryTerm("pump_flow_rate", (
                "pump flow rate", "flow rate of", "pumped at a rate", "L/min", "flow rate was",
            ), definition='Flow rate of the pump used during filtration.'),
            CategoryTerm("pump_flow_rate_unit", (
                "L/min", "L/h", "L/s", "m3/min", "m3/h", "m3/s",
            ), definition='Unit associated with the filtration pump flow rate.'),
            # Sample storage / preservation (prior to DNA extraction)
            CategoryTerm("samp_store_temp", (
                "stored at", "storage temperature", "-20°C", "-80°C", "4°C", "on ice",
                "dry ice", "liquid nitrogen", "ambient temperature", "stored frozen", "stored cold",
                "-20C", "-20 C", "-20 degrees C", "-80C", "-80 C", "-80 degrees C",
                "4C", "4 C", "4 degrees C",
                # "stored at" is a literal, gap-free 2-word cue -- a real gap
                # confirmed live (10.7717/peerj.9857): "Samples were stored
                # in seawater at 80 °C" never matched it, since "in seawater"
                # sits between "stored" and "at". "stored in" catches this
                # common "stored in <medium> at <temp>" construction too.
                "stored in",
            ), definition='Temperature at which the original environmental sample was stored.'),
            CategoryTerm("samp_store_dur", (
                "stored for", "storage duration", "prior to extraction for", "stored until",
                "kept until", "preserved until", "until extraction", "until DNA extraction",
                "until RNA extraction", "until processing", "until further processing",
                "until further use", "before extraction", "prior to extraction",
                "overnight", "for 24 h", "for 24 hours", "for 48 h", "for 48 hours",
                "for several days", "for several weeks", "for several months",
                "for the rest of the ship cruise",
            ), definition='Duration the original environmental sample was stored before processing or DNA extraction.'),
            CategoryTerm("samp_store_loc", (
                "stored in a freezer", "stored at the", "storage location", "stored in the lab",
                "stored onboard", "stored on board", "stored aboard", "stored in the laboratory",
                "stored at the laboratory", "stored in the field", "stored in the freezer",
                "stored in a refrigerator", "stored in a cold room", "onboard", "on board",
                "aboard", "in the laboratory", "to the laboratory", "transported to the laboratory",
                "transported to the lab",
            ), definition='Physical location where the original sample was stored, such as a particular freezer or laboratory room.'),
            CategoryTerm("samp_store_sol", (
                "stored in RNAlater", "stored in ethanol", "storage solution", "preservation buffer",
                "stored in lysis buffer", "Longmire's buffer", "preserved in ethanol",
                # Bare, single-word cues deliberately, not 2-word phrases --
                # confirmed live that real phrasing inserts words between
                # them ("immersed EACH with 3 ml of lysis buffer"), which a
                # fixed multi-word phrase cue silently never matches
                # (10.1371/journal.pone.0303937).
                "immersed", "resuspended", "submerged",
            ), definition='The solution in which the sample (original or, after processing, e.g. a filter membrane) was stored, preserved, immersed, or resuspended, such as RNAlater, ethanol, or a lysis buffer -- even if the immersion and a later storage event (e.g. "stored at -20C") are described in separate nearby sentences. Return ONLY the short solution name/phrase itself (e.g. for "the biomass was immersed with 3 ml of lysis buffer", return "lysis buffer"), never the surrounding sentence.'),
            CategoryTerm("samp_store_method_additional", (
                "transported frozen", "transported on ice", "shipped frozen", "storage conditions",
                "shipped on dry ice", "shipped on ice",
                "stored", "storage", "preserved", "kept", "frozen", "freeze", "transported",
                "transported under frozen conditions", "under frozen conditions", "stored onboard",
                "stored on board", "stored aboard", "stored until", "kept until",
                "preserved until", "until further use", "until extraction", "until processing",
                "stored at", "stored in", "stored on", "placed on ice", "kept on ice",
                "flash frozen", "snap frozen", "liquid nitrogen", "dry ice",
            ), allows_multi_sentence=True, definition='Additional useful information about how the original environmental sample was stored or preserved.'),

            # precip_chem_prep/precip_force_prep/precip_temp_prep/
            # precip_time_prep removed entirely per an explicit user
            # request ("negligible... don't want to waste compute on
            # them") -- suppressed from export in exports/faire.py's
            # SAMPLE_METADATA_SUPPRESSED_FIELDS and no longer extracted.

            # Nucleic-acid extraction
            # Broadened per an explicit user-supplied cue list.
            CategoryTerm("nucl_acid_ext_kit", (
                "extraction kit", "DNA kit", "RNA kit", "using the kit", "manufacturer's instructions",
                "manufacturer's protocol", "Soil DNA kit", "DNeasy", "PowerSoil",
                "DNA extraction kit", "RNA extraction kit", "nucleic acid extraction kit",
                "DNA isolation kit", "RNA isolation kit", "DNA purification kit", "genomic DNA kit",
                "commercial extraction kit", "extracted using", "isolated using",
                "DNA was extracted with", "DNA was isolated with",
                "according to the manufacturer's instructions", "according to the manufacturer's protocol",
                "following the kit protocol",
                "PowerWater", "PowerLyzer", "PowerLyze", "NucleoSpin", "E.Z.N.A.", "MagAttract",
                "Quick-DNA", "AllPrep", "PureLink", "QIAamp",
            ), definition='Name of the commercial kit used to extract DNA/RNA from the sample.'),
            # fallback_only per an explicit user request: a dedicated
            # leftover-capture pass (see _nucl_acid_ext_method_additional_
            # fact) scoped specifically to nucleic-acid-extraction-shaped
            # leftover sentences within the sample_prep run, not the whole
            # category's leftovers (that's prep_method_additional's job).
            CategoryTerm("nucl_acid_ext_method_additional", fallback_only=True, definition='Additional useful details about the nucleic-acid extraction workflow that do not fit another specific extraction field, such as reagent amounts or concentrations, bead-beating or sonication conditions, incubation times and temperatures, centrifugation conditions, wash steps, precipitation steps, elution conditions, repeated treatment steps, or other detailed extraction procedures.'),
            # User-supplied cues (verbatim), grounded in two real papers:
            # "DNA contamination was removed with the TURBO DNA-free kit
            # (Invitrogen)" (10.1093/ismejo/wrae013) and an isopropanol/
            # ethanol precipitation cleanup (10.1371/journal.pone.0303937)
            # -- the original cue lists only had "cleaned"/"purified"
            # phrasing, missing both "contamination was removed" and
            # precipitation-based cleanup language entirely.
            CategoryTerm("dna_cleanup_0_1", (
                "DNA was cleaned", "DNA was purified", "DNA purification", "cleanup was performed",
                "no additional cleanup", "purification step",
                "contamination was removed", "DNA-free kit", "DNase treatment", "DNase treated",
                "treated with DNase", "removed with the", "isopropanol precipitation",
                "ethanol precipitation", "DNA pellet was washed", "precipitated DNA",
            ), definition='Whether extracted DNA was subsequently cleaned or purified after the initial extraction: yes = 1, no = 0. This includes a commercial cleanup/purification kit, a DNA-free/DNase treatment to remove contaminants, AND a precipitation-based cleanup (e.g. isopropanol or ethanol precipitation followed by removing the supernatant/washing the pellet) -- precipitation-and-wash steps described narratively (without ever using the word "cleanup" or "purification") still count as yes = 1.'),
            CategoryTerm("dna_cleanup_method", (
                "cleaned using", "purified using", "cleanup kit", "purification kit",
                "contamination was removed with", "removed with the", "DNA-free kit",
                "DNase treatment", "treated with DNase", "isopropanol precipitation",
                "ethanol precipitation", "precipitated with isopropanol", "precipitated with ethanol",
            ), definition='Method or commercial kit used to clean/purify extracted DNA.'),
            CategoryTerm("pool_dna_num", (
                "pool", "pooled", "pooling", "samples were pooled", "sample was pooled",
                "pooled samples", "pooled sample", "DNA was pooled", "RNA was pooled",
                "DNA samples were pooled", "RNA samples were pooled", "DNA extracts were pooled",
                "RNA extracts were pooled", "extracts were pooled", "pooled extracts",
                "number of extracts pooled", "DNA extracts were combined", "RNA extracts were combined",
                "samples were combined", "combined samples",
            ), definition='Sentence(s) describing how DNA, RNA, nucleic-acid extracts, or samples were pooled or combined before downstream analysis. Preserve the source sentence rather than reducing it to only a number.'),
            CategoryTerm("concentration", (
                "DNA concentration was", "concentration of the extracted DNA", "concentration of DNA",
                "concentration was measured", "final DNA concentration", "ng/µL", "ng/mL", "µg/mL",
                "ng per µL", "ng/uL", "ng per uL",
                # Real gap found live (10.3390/microorganisms10030558):
                # "Fifty nanograms of DNA was used as a template for PCR
                # amplification" reports the DNA input amount as a
                # spelled-out number with a bare mass unit (no "/µL"),
                # not a numeral-with-per-volume-unit phrasing any cue
                # above matches at all -- confirmed live the value was
                # never missing/hallucinated, just never reached by any
                # existing cue. Deliberately NOT asking for "Fifty" ->
                # "50" numeral normalization here: extract_category_terms'
                # own master prompt already tells the model to copy values
                # WORD FOR WORD, and its verbatim-quote guard (section_
                # category_extraction.py) discards any value that doesn't
                # literally appear in its own quote -- a normalized "50
                # ng" would never match "Fifty nanograms..." and get
                # silently dropped, worse than just keeping "Fifty".
                "nanograms of DNA", "ng of DNA", "used as a template",
            ), definition='Concentration of total DNA after extraction, preserving both the numeric value and unit in the same value when reported, e.g. "12.4 ng/uL". Also accept the amount of DNA reported as PCR template input (e.g. "Fifty nanograms of DNA was used as a template") when no separate extract concentration is given, keeping the number exactly as the source states it (spelled out or numeral). Do not return only the unit.'),
            CategoryTerm("samp_vol_we_dna_ext", (
                "used for DNA extraction", "used for extraction", "processed for DNA extraction",
                "used for RNA extraction", "used for nucleic acid extraction",
                "DNA was extracted from", "DNA extracted from", "extracted from",
            ), definition='Amount of sample or subsample actually processed for DNA extraction; this is not necessarily the total amount originally collected.'),
            CategoryTerm("samp_vol_we_dna_ext_unit", (
                "mg of dried", "mg of sediment", "g of sediment", "mL of", "L of water",
            ), definition='Unit for the amount of sample processed for DNA extraction, such as mg, g, mL, L, or cm².'),

            # Lysis
            # Both broadened per an explicit user-supplied cue list.
            CategoryTerm("nucl_acid_ext_lysis", (
                "lysis", "lysed", "cell lysis", "bead beating", "bead-beating", "bead mill",
                "homogenizer", "sonication", "sonicated", "freeze-thaw", "proteinase K",
                "chemical lysis", "enzymatic lysis", "thermal lysis",
                "lysis buffer", "disruption", "cell disruption", "mechanical disruption",
                "zirconium beads", "glass beads", "ultrasonic", "heat lysis", "lysozyme",
                "SDS", "CTAB", "detergent", "osmotic lysis", "homogenized for lysis",
                "vortexed with beads", "cells were lysed by", "samples were disrupted using",
                "lysis was performed by", "bead-beaten for", "treated with lysozyme",
                "incubated with proteinase K", "SDS was added for lysis",
            ), definition='General approach used to lyse DNA-containing material, such as physical, thermal, chemical, enzymatic, or osmotic lysis.'),
            CategoryTerm("nucl_acid_ext_sep", (
                "separated using", "column-based", "spin column", "magnetic beads",
                "phenol-chloroform", "phenol chloroform", "silica column",
                "DNA was purified", "RNA was purified", "nucleic acid purification",
                "phase separation", "organic extraction", "phenol/chloroform", "chloroform extraction",
                "phenol chloroform isoamyl alcohol", "PCI extraction", "column purification",
                "silica membrane", "binding column", "magnetic bead purification",
                "bead-based purification", "solid-phase extraction", "precipitated DNA",
                "DNA precipitation", "isopropanol precipitation", "ethanol precipitation",
                "alcohol precipitation", "centrifuged for phase separation", "aqueous phase",
                "supernatant transferred", "DNA pellet", "eluted", "elution buffer",
                "DNA was purified by", "nucleic acids were separated using",
                "the aqueous phase was transferred", "DNA was precipitated with",
                "DNA bound to the column", "DNA was eluted from", "magnetic beads were used to purify",
            ), definition='Approach used to separate/purify DNA from the sample mixture, such as column-based separation, magnetic beads, centrifugation, precipitation, or phenol-chloroform.'),
        ),
    ),
)


# Splits sample_prep's own 34 searchable terms into smaller, ordered,
# workflow-phase groups for Stage 3 (section_category_extraction.py's
# extract_category_terms) -- per an explicit user request: a single call
# listing all 34 field definitions regardless of which ones that batch's
# quotes actually matched is unnecessary noise for a small model, and the
# real fix for one specific miss (pool_dna_num missing "Three replicates
# were pooled...", see section_category_extraction.py's own
# _POOLING_SAMPLE_CONTEXT_RE comment) already came from elsewhere, but a
# more focused per-phase prompt is still a real accuracy improvement:
# each phase's own call only ever lists ITS OWN terms as candidate
# fields, in the order these facts naturally appear in a paper's own
# Methods narrative (collection, then filtration, then storage, then
# nucleic-acid extraction/pooling -- "pool_dna_num... after sampling and
# before pcr", per that request). A category with no entry here (there
# are none today) falls back to one single phase covering every
# searchable term, i.e. today's existing behavior, unchanged.
SAMPLE_PREP_TERM_PHASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "collection",
        (
            "samp_collect_device", "samp_collect_method", "sterilise_method", "samp_size",
            "samp_size_unit", "sample_composed_of", "sample_derived_from", "internal_expedition_id",
            "x_env_var_block", "samp_category", "samp_mat_process",
        ),
    ),
    (
        "filtration",
        (
            "filter_material", "filter_name", "filter_diameter", "filter_surface_area", "size_frac",
            "prefilter_material", "filter_passive_active_0_1", "pump_flow_rate", "pump_flow_rate_unit",
        ),
    ),
    (
        "storage",
        ("samp_store_temp", "samp_store_dur", "samp_store_loc", "samp_store_sol", "samp_store_method_additional"),
    ),
    (
        "extraction",
        (
            "nucl_acid_ext_kit", "dna_cleanup_0_1", "dna_cleanup_method", "pool_dna_num", "concentration",
            "samp_vol_we_dna_ext", "samp_vol_we_dna_ext_unit", "nucl_acid_ext_lysis", "nucl_acid_ext_sep",
        ),
    ),
)
TERM_PHASES_BY_CATEGORY: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "sample_prep": SAMPLE_PREP_TERM_PHASES,
}


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
_TABLE_BODY_SEQUENCE_METRIC_RE = re.compile(
    r"\b(?:"
    r"#\s*of\s+(?:quality[-\s]*filtered\s+)?reads|"
    r"#\s*of\s+OTUs|"
    r"uniquely\s+mapping\s+to\s+OTUs|"
    r"mapping\s+efficiency"
    r")\b",
    re.IGNORECASE,
)
_METHOD_SENTENCE_START_RE = re.compile(
    r"\b(?:"
    r"DNA\s+was\s+(?:isolated|extracted)|"
    r"RNA\s+was\s+(?:isolated|extracted)|"
    r"nucleic\s+acids?\s+were\s+(?:isolated|extracted)"
    r")\b",
    re.IGNORECASE,
)


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


def _strip_leading_table_body(paragraph: str) -> str:
    """Trim PDF-extracted table bodies accidentally glued to methods prose.

    Some PDFs emit a bare table body without a leading "Table N" caption,
    followed immediately by the next real methods sentence. If the table
    contains sequencing metrics, and the paragraph later resumes with a
    nucleic-acid extraction sentence, keep the real method sentence and drop
    the table rows before it.
    """
    if not _TABLE_BODY_SEQUENCE_METRIC_RE.search(paragraph):
        return paragraph
    method_start = _METHOD_SENTENCE_START_RE.search(paragraph)
    if method_start is None:
        return paragraph
    return paragraph[method_start.start() :].strip()


# A short subsection heading occasionally survives PDF/DOCX text
# extraction glued directly onto the sentence that follows it, with no
# blank line -- or even a period of its own -- separating them (confirmed
# live, a real paper: "...stored in seawater at 80 C. Caribbean spawn I On
# the evening of August 31, 2010..." -- "Caribbean spawn I" is the
# heading; "Pacific spawn I In November 2010, at Orpheus Island Research
# Station..." is another instance of the same paper's own pattern). Left
# unsplit, the heading fragment either rides along as part of the
# following sentence's own extracted content (a real bug: "Pacific spawn
# I" was extracted verbatim as an internal_expedition_id value, since it
# superficially looks like a short campaign identifier), or an entire
# later, unrelated sample-collection event's narrative gets silently
# absorbed into the same run as the first (a real bug: prep_method_
# additional pulling in a second, later spawning event's full collection
# narrative). Requires a trailing roman numeral or number (e.g. "spawn
# I", "Station 2", "Site 3") -- the confirmed real-world shape of this
# artifact -- so a real, unglued sentence that merely happens to end a
# clause with a number doesn't accidentally get split (checked against 15
# realistic negative-control sentences: kit/brand names, station
# numbering, citation years, table references -- zero false positives).
# Removed outright (not split into its own paragraph) -- an earlier
# version split the paragraph at this point instead, but that fragmented
# Stage 1's own paragraph-level keyword gate: a real, un-glued sentence
# immediately after the heading (e.g. "gamete bundles were collected from
# FGBNMS") can easily fail to independently contain any category keyword
# even though the paragraph as a whole clearly does, so splitting risked
# silently dropping real wanted content right after the heading -- caught
# live, before this shipped, by re-running the fix against the paper that
# motivated it. Removing just the glued substring keeps the paragraph
# intact as one gating unit while still stopping the heading text itself
# from ever being mistaken for a real extracted value.
_INLINE_HEADING_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"(?:[A-Za-z]{2,}\s+){1,3}(?:[IVXLC]{1,4}|\d{1,4})"
    r"\s+(?=[A-Z][a-z])"
)


def split_into_paragraphs(text: str) -> list[str]:
    """Blank-line-delimited paragraphs, references section dropped
    (see `_drop_reference_section`) and bare figure/table captions
    excluded one at a time (these can appear anywhere, not just at the
    end) -- both confirmed as real false-positive sources against a real
    PNAS supplementary-methods document (a cited paper's own title
    matching "amplicon"/"primer"; bare "Fig. S4. ..." caption lines
    matching "amplicon"). Also strips out any glued inline subsection
    heading (see _INLINE_HEADING_RE) so it can't survive into a
    downstream sentence and get mistaken for a real extracted value."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    paragraphs = _drop_reference_section(paragraphs)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        if _FIGURE_TABLE_CAPTION_RE.match(paragraph):
            continue
        stripped = _strip_leading_table_body(paragraph)
        if stripped:
            cleaned.append(_INLINE_HEADING_RE.sub(" ", stripped))
    return cleaned


def candidate_categories_for_paragraph(paragraph: str) -> frozenset[str]:
    """Stage 1: the cheap, deterministic keyword gate. A paragraph that
    matches zero categories here never reaches the LLM at all (Stage 2,
    not built in this module)."""
    return frozenset(
        category_name
        for category_name, patterns in _CATEGORY_PATTERNS.items()
        if any(pattern.search(paragraph) for pattern in patterns)
    )


# A short, standalone line with no terminal sentence punctuation -- the
# common shape of a methods subsection heading ("Sample collection",
# "2.2. DNA extraction and PCR amplification", "Library preparation and
# sequencing") once flattened to plain text and split into its own
# paragraph by split_into_paragraphs's blank-line rule. Deliberately
# conservative on both bounds: real methods sentences overwhelmingly end
# in a period and/or run well past 80 characters, so this rarely misfires
# on genuine content -- it also means a heading glued onto the start of
# its own following paragraph with no blank line between them (a real,
# separate known gap -- see group_sentences_into_category_runs's own
# bridging-limit comment) won't be recognized as a heading at all, since
# the combined text is no longer short.
_SECTION_HEADING_MAX_CHARS = 80
_SECTION_HEADING_MAX_WORDS = 12
_SENTENCE_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?]\s*$")


def _looks_like_section_heading(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped or len(stripped) > _SECTION_HEADING_MAX_CHARS:
        return False
    if _SENTENCE_TERMINAL_PUNCTUATION_RE.search(stripped):
        return False
    return len(stripped.split()) <= _SECTION_HEADING_MAX_WORDS


def paragraphs_before_next_section_heading(text: str) -> list[str]:
    """Stops yielding paragraphs the instant a heading-shaped paragraph
    appears that ISN'T itself about a real category -- per an explicit
    user instruction: "the majority of the time, sample collection and
    prep are the first thing in methods, and then won't be mentioned
    again. So, if the header/title of the section changes, the category
    is over by default." Only starts watching for that boundary AFTER
    the first paragraph that actually matches a real category's own
    keywords, so an early, unrelated heading common before the relevant
    one (e.g. "2.1 Study site" ahead of "2.2 Sample collection") can't
    cut things off before any relevant content is even reached. A
    heading that itself still matches a category's keywords (e.g. "2.2
    Sample collection and DNA extraction") is real content, not a
    boundary, and keeps the run going."""
    kept: list[str] = []
    seen_content = False
    for paragraph in split_into_paragraphs(text):
        candidates = candidate_categories_for_paragraph(paragraph)
        if seen_content and not candidates and _looks_like_section_heading(paragraph):
            break
        kept.append(paragraph)
        if candidates:
            seen_content = True
    return kept


# low_confidence_categories/_CATEGORY_MIN_DOCUMENT_MENTIONS (a whole-
# document minimum-mention-count gate) removed entirely: its only entry
# was targeted_qpcr_ddpcr_detection, which no longer exists as a category
# at all, per an explicit user request.


def group_sentences_into_category_runs(
    tagged_sentences: list[tuple[str, frozenset[str]]],
) -> dict[str, str]:
    """Stage 2.5: given a paragraph's sentences, each already tagged
    (Stage 2, LLM) with zero or more category names, groups each present
    category's own text. A run for category X starts at a sentence
    tagged X and extends through following untagged (bridging) sentences,
    ending the instant a sentence tagged with a DIFFERENT category (not
    including X, even if X is also present on that same sentence)
    appears -- or the instant more than _MAX_BRIDGING_SENTENCES untagged
    sentences appear in a row (see _runs_for_category's own docstring).
    Separate runs of the same category within one paragraph (category
    flips away and back) are concatenated together, in order, with the
    interrupting category's own sentences excluded throughout -- an
    explicit user specification: "if a paragraph goes from cat 1 to cat 2
    back to cat 1, the two cat 1 [runs] will be grouped consecutively
    with cat 2 removed."

    Computed independently per category (not as a single linear pass)
    so a sentence tagged with more than one category correctly continues
    every one of its own categories' runs while still breaking any OTHER
    category's currently-open run."""
    present_categories = {category for _, categories in tagged_sentences for category in categories}
    return {
        category: " ".join(_runs_for_category(tagged_sentences, category))
        for category in present_categories
    }


# A real live audit (10.7717/peerj.333) found a run for "sample_prep"
# started by one genuine sentence ("Samples were stored in seawater at 80
# C.") and then, via unlimited bridging, silently absorbed an entire
# table caption, two figure captions, and several paragraphs of coral-
# settlement-assay narrative -- none of it ever tagged with ANY category
# by Stage 2 (this source paper's own fulltext extraction apparently
# never inserts a blank line between what should be separate paragraphs,
# so split_into_paragraphs's own paragraph boundary never kicks in
# either). The SAME failure mode independently corrupted
# nucl_acid_ext_method_additional for a different real paper
# (10.1093/ismejo/wrae013): a run absorbed a whole unrelated incubation-
# experiment paragraph with nothing to do with nucleic acid extraction.
# A short, genuine connective sentence between two same-category
# sentences ("This was done as follows.") is common and should still
# bridge; an entire off-topic section should not. 2 consecutive untagged
# sentences is a deliberately conservative cutoff for the former without
# enabling the latter.
_MAX_BRIDGING_SENTENCES = 2


def _runs_for_category(tagged_sentences: list[tuple[str, frozenset[str]]], category: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    pending_bridge: list[str] = []
    in_run = False
    for sentence, categories in tagged_sentences:
        if category in categories:
            current.extend(pending_bridge)
            pending_bridge = []
            current.append(sentence)
            in_run = True
        elif not categories:
            if in_run:
                pending_bridge.append(sentence)
                if len(pending_bridge) > _MAX_BRIDGING_SENTENCES:
                    # Too many untagged sentences in a row to still be a
                    # brief connective gap -- close the run here, without
                    # ever including the unrelated bridge sentences.
                    if current:
                        runs.append(" ".join(current))
                    current = []
                    pending_bridge = []
                    in_run = False
        else:
            if current:
                runs.append(" ".join(current))
            current = []
            pending_bridge = []
            in_run = False
    if current:
        runs.append(" ".join(current))
    return runs


# A genuine PCR/qPCR/ddPCR mention -- bare "PCR" alone (word-boundary
# matched, so it never matches inside "qPCR"/"ddPCR") plus the
# qPCR/ddPCR/"polymerase chain reaction"/amplification vocabulary a
# bare-"PCR" match alone would miss (e.g. "A TaqMan qPCR assay used a FAM
# reporter dye" never says plain "PCR" anywhere).
_PCR_MENTION_RE = re.compile(
    r"\b(?:PCR|qPCR|ddPCR|digital\s+PCR|polymerase\s+chain\s+reaction|amplicon|amplified|amplification)\b",
    re.IGNORECASE,
)


def derive_pcr_0_1_from_category_detection(texts: list[tuple[str, str]]) -> RawFactCandidate | None:
    """`pcr_0_1` gates a broad swath of the existing PCR/assay checklist
    (extraction/faire_fields.py's "PCR / assay setup" group, several
    LLMJudgedSearchField entries). This used to derive from
    pcr1_primary_amplification_0_1/targeted_qpcr_ddpcr_detection_0_1
    category-presence detection instead of its own independent regex scan
    (an explicit user instruction, to avoid the two ever disagreeing), but
    both of those categories were removed entirely per a later explicit
    user request (Stage 2 scoped down to sample_prep only) -- this reverts
    to the original independent deterministic scan that mechanism had
    replaced. Still no LLM call, still gates the identical checklist
    swath as before."""
    for title, text in texts:
        match = _PCR_MENTION_RE.search(text)
        if match:
            return RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate="pcr_0_1",
                raw_field_name="pcr_0_1",
                raw_value="1",
                source_locator=f"section:{title}",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=text[max(0, match.start() - 60) : match.end() + 60],
                confidence_metadata={"detector": "pcr_mention_regex"},
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
        for paragraph in paragraphs_before_next_section_heading(text):
            for category_name in candidate_categories_for_paragraph(paragraph):
                found.setdefault(category_name, paragraph)
    facts: list[RawFactCandidate] = []
    for category in SECTION_CATEGORIES:
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
