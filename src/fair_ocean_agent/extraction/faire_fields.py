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


# Fields that remain part of the FAIRe registry, mappings, and exports but
# are not worth spending local-model context or generation time on --
# either because a structured adapter (NCBI BioSample/ENA/repository
# experiment records) is a far more reliable source for them than free
# prose, or because they vary sample-to-sample/run-to-run in a way a paper
# essentially never states explicitly (e.g. samp_category -- whether a
# given physical sample is a real sample or a control -- is not even in
# this taxonomy at all, precisely because it's not the kind of thing a
# paper's Methods section states per-sample). Structured adapters may
# still populate every field here. Keeping this policy beside the prompt
# taxonomy also prevents a future taxonomy expansion from silently adding
# one of these fields back to either paper or supplement LLM extraction.
#
# samp_collect_method/samp_store_method_additional (sampleMetadata) and
# assay_name (experimentRunMetadata, via `_target_table_for_faire_field`'s
# resolution -- assay_name resolves to projectMetadata, see mapping/
# rules.py's _TARGET_TABLE_OVERRIDES) were added per an explicit user
# review of which
# project/sample/experiment fields are realistically findable in prose
# versus structured-source-only. sampleMetadata's LLM checklist is now
# deliberately narrow: only collection_date/depth/coordinates (->
# eventDate, minimumDepthInMeters/maximumDepthInMeters,
# decimalLatitude/decimalLongitude) remain -- everything else about a
# sample, including which samples are controls, comes from APIs/structured
# supplementary data, never the LLM.
#
# platform/instrument were added in a follow-up review of a
# NOAA-specific FAIRe checklist that marks these (real projectMetadata
# fields, confirmed via `_target_table_for_faire_field`) "No LLM" --
# already covered by ENA's own instrument_platform/instrument_model facts
# wherever a study has them (see mapping/rules.py's EntityLevel.SEQUENCING_RUN
# rules). `lib_layout` is also No LLM, but is derived from FASTQ file counts
# in mapping/faire.py rather than from paper prose or ENA's declared
# library_layout. This corrects an earlier decision in
# this same module's history that kept `platform` LLM-askable after
# mis-scoping it as an "experiment" rather than "project" field.
LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS = frozenset(
    {
        "informationWithheld",
        "dataGeneralizations",
        "woce_sect",
        "inhibition_check_0_1",
        "inhibition_check",
        "samp_collect_method",
        "samp_store_method_additional",
        "assay_name",
        "platform",
        "instrument",
        "lib_layout",
        # sequencing_kit (native name) / seq_kit (hint): duplicates
        # search_flags.CONTROLLED_SEARCH_FIELDS's own "seq_kit" entry
        # (large curated list of real sequencing-kit names). Unlike the
        # PCR-adjacent overlaps below (gated, not excluded), sequencing
        # itself is universal to every eDNA paper -- there's no natural
        # boolean flag to gate this one on the way pcr_0_1 gates PCR
        # fields, so it's excluded outright rather than conditionally
        # asked. A known, accepted precision/recall trade-off for this one
        # field in the gold benchmark (see llm/benchmark.py).
        "seq_kit",
        # forward_sequencing_adapter/reverse_sequencing_adapter (native
        # names) / adapter_forward/adapter_reverse (hints): duplicate
        # search_flags.LLM_JUDGED_SEARCH_FIELDS's own "adapter_forward"/
        # "adapter_reverse" entries, a narrower, quote-anchored LLM pass
        # built specifically for these two fields (explicit instructions to
        # copy the sequence verbatim and only accept a fact tied to a real
        # quote_id). Running both would risk two independent LLM calls
        # producing conflicting facts for the same adapter sequence under
        # two different fact_type_candidate spellings. Excluded here in
        # favor of the more precise mechanism, same pattern as seq_kit.
        "adapter_forward",
        "adapter_reverse",
        # OTU/ASV clustering tool is a targeted quote-judged field: shared
        # bioinformatics terms like DADA2/QIIME/VSEARCH need a narrow,
        # evidence-cited decision so taxonomy-classification tools are not
        # confused with OTU/ASV generation.
        "otu_clust_tool",
        # otu_db is a targeted quote-judged field: it must avoid
        # classifier/software-only mentions.
        "otu_db",
        # targetTaxonomicAssay/targetTaxonomicScope (native names
        # assay_target_taxa/study_target_taxonomic_scope) are targeted
        # quote-judged fields: an ordered, priority-ranked search-term list
        # plus a required same-sentence assay/study-scope context word,
        # per an explicit user specification -- the broad checklist's
        # freeform prompt has no equivalent priority/context mechanism and
        # would risk a second, independently-worded LLM pass producing a
        # conflicting fact for the same FAIRe field.
        "targetTaxonomicAssay",
        "targetTaxonomicScope",
    }
)

# Narrative fallbacks do not carry a FAIRe hint, so target-field filtering
# cannot remove them. "collection_method"/"storage_conditions" are the
# fallback-narrative counterparts of the now-excluded
# sample_collection_method/sample_storage_conditions atomic fields (same
# samp_collect_method/samp_store_method_additional targets -- see
# mapping/rules.py) and are excluded for the same reason.
# "environmental_context" is deliberately unmapped by mapping/rules.py
# (genuinely ambiguous among env_broad_scale/env_local_scale/env_medium)
# and, being a sample-level narrative concept outside the now-narrow
# sampleMetadata checklist, is excluded too.
# "PCR_amplification_conditions" is no longer excluded here -- per an
# explicit user request it moved into the "PCR / assay setup" FIELD_GROUP
# instead (alongside a new "second_pcr_amplification_conditions"
# counterpart), since the shared framing every FALLBACK_NARRATIVE_FIELDS
# entry gets ("use ONLY IF no concept above applies") is the opposite of
# what was actually wanted: the raw PCR narrative captured regardless of
# whether the atomic fields also succeeded, specifically so a human can
# see which paragraph/assay a given atomic value came from.
LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS = frozenset(
    {
        "collection_method",
        "storage_conditions",
        "environmental_context",
    }
)


@dataclass(frozen=True)
class FaireExtractionField:
    native_name: str
    hint: str
    faire_hint: str | None = None
    example: str | None = None
    # Mirrors search_flags.ControlledSearchField's own field name/shape --
    # a deliberate consistency choice, not a coincidence. Empty (the
    # default) means "always shown" (today's behavior for every
    # pre-existing field). Non-empty means "only shown when at least one
    # of these deterministic boolean flags (extraction/search_flags.py's
    # TEXT_SEARCH_FLAGS, e.g. pcr_0_1) was detected true for this paper" --
    # see field_names_for_reference/render_field_reference's active_flags
    # parameter.
    required_any_flags: frozenset[str] = frozenset()


FIELD_GROUPS: dict[str, tuple[FaireExtractionField, ...]] = {
    "Sample collection / environment": (
        FaireExtractionField("collection_date", "date or date range when samples were collected", "eventDate", "2022-01-04"),
        FaireExtractionField(
            "depth",
            "sampling depth below the water/sediment/soil surface, including phrases like surface sediment, "
            "upper few millimeters, top 2 cm, or a depth range such as the epilimnion (0-20 m)",
            "minimumDepthInMeters",
            "upper few millimeters",
        ),
        FaireExtractionField("coordinates", "sampling latitude/longitude or coordinate pair", "decimalLatitude", "38.03 N, 122.15 W"),
        # Real gap found live (PMC10988111): a paper can name a real,
        # specific collection site ("Yantai Haichang Whale Shark Ocean
        # Park (Shandong, China)") without ever giving numeric coordinates
        # anywhere -- coordinates' own field above has nothing to extract
        # in that case, and lat/lon/depth/date are the highest-priority
        # facts this whole pipeline exists to capture, so a named site is
        # still worth capturing on its own rather than left blank just
        # because it isn't a coordinate pair.
        FaireExtractionField(
            "geo_loc_name",
            "the named geographic location where samples were collected -- a specific site, facility, "
            "city, region, or country (e.g. a named park, station, or city), NOT a coordinate pair "
            "(that's the separate coordinates field above) and not a lab/institution address unrelated "
            "to where the actual samples were collected",
            "geo_loc_name",
            "China: Yantai",
        ),
        FaireExtractionField("sample_collection_method", "how samples were physically collected", "samp_collect_method"),
        FaireExtractionField("sample_storage_conditions", "how samples were stored or preserved after collection", "samp_store_method_additional"),
        FaireExtractionField(
            "filter_name",
            "brand/product/model/name of the FILTER used to capture the sample (e.g. Sterivex filter or "
            "Millipore filter) -- the device the sample material is caught ON/IN. Never a syringe, "
            "pipette tip, collection tube, vial, or other labware used to transfer/subsample material "
            "(e.g. a 'cut-off syringe (Henke-Ject)' used to subsample sediment is a syringe, not a "
            "filter); do not return cross-reference placeholders like 'see below' or 'see above'",
            "filter_name",
            "Sterivex filter",
        ),
    ),
    "DNA extraction": (
        FaireExtractionField("dna_extraction_kit", "name of the extraction kit used", "nucl_acid_ext_kit", "DNeasy PowerWater Kit"),
        FaireExtractionField("dna_lysis_method", "lysis method (e.g. physical, chemical, enzymatic)", "nucl_acid_ext_lysis", "bead-beating"),
        FaireExtractionField("dna_separation_method", "how DNA was separated/purified (e.g. spin column, magnetic beads)", "nucl_acid_ext_sep"),
        FaireExtractionField("sample_volume_for_extraction", "volume or mass of sample processed for extraction", "samp_vol_we_dna_ext", "500 mL"),
        FaireExtractionField("sample_volume_for_extraction_unit", "unit for sample_volume_for_extraction", "samp_vol_we_dna_ext_unit", "mL"),
        FaireExtractionField(
            "dna_concentration",
            "concentration of the purified/extracted DNA sample itself (e.g. measured via Qubit, NanoDrop, or gel), "
            "never a PCR template DNA concentration added into a reaction mixture; preserve both the numeric value and "
            "unit together in this one field when reported (e.g. 12.4 ng/uL)",
            "concentration",
            "12.4 ng/uL",
        ),
        FaireExtractionField("dna_cleanup_method", "DNA clean-up/purification method or kit name", "dna_cleanup_method"),
    ),
    "PCR / assay setup": (
        # Every field in this group only applies to a paper that actually
        # performed PCR -- gated on pcr_0_1 (extraction/search_flags.py's
        # deterministic keyword detector for "this paper describes
        # PCR/qPCR/ddPCR/amplification"), matching a NOAA FAIRe checklist's
        # own per-field "if pcr_0_1 TRUE" conditional-requirement column.
        # assay_name is the one exception: it's fully excluded (not
        # gated) via LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS above, since
        # search_flags.CONTROLLED_SEARCH_FIELDS already covers it
        # deterministically end to end.
        FaireExtractionField("assay_name", "a short name/identifier the paper gives its assay", "assay_name", "18S-V4-eukaryote"),
        # target_gene/thermocycler/commercial_master_mix/assay_type also
        # have a search_flags.CONTROLLED_SEARCH_FIELDS entry with the same
        # identity, but that mechanism is literal-substring matching
        # against a curated term list, not free-text extraction -- checked
        # against real gold data and confirmed it demonstrably misses or
        # mangles real values a careful read of the prose would get right
        # (e.g. "MyFi Mix (Meridian Bioscience)" matches nothing in
        # commercial_mm's term list). Gated here rather than excluded, so
        # the LLM stays the richer source, just asked conditionally.
        # Real gap found live (10.3390/microorganisms10030558): this
        # field's own hint used to be so bare ("targeted, metabarcoding,
        # or other detection approach") that the model labeled a SECOND
        # marker-gene community-profiling assay (cbbL, amplified with its
        # own specific primers, OTU-clustered exactly like the paper's
        # own 16S assay) as "targeted" -- apparently reasoning "named,
        # specific primers = targeted", when FAIRe's own real distinction
        # is single-species/qPCR-style detection (targeted) vs.
        # marker-gene community profiling via amplicon sequencing +
        # OTU/ASV clustering (metabarcoding), regardless of how specific
        # the primers are.
        FaireExtractionField(
            "assay_type",
            "targeted (qPCR/ddPCR-style detection of one species or a small, named taxon set), "
            "metabarcoding (marker-gene community profiling via amplicon sequencing, OTU/ASV clustering, "
            "and taxonomic database assignment -- still metabarcoding even when specific, named primers "
            "are used, and even for a functional/non-taxonomic marker gene like cbbL/nifH/amoA profiled "
            "the same way), or other detection approach",
            "assay_type",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
        FaireExtractionField("target_gene", "targeted gene or locus", "target_gene", "16S rRNA", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("target_subfragment", "targeted hypervariable subregion", "target_subfragment", "V4", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("forward_primer_sequence", "forward primer sequence, 5' to 3'", "pcr_primer_forward", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("reverse_primer_sequence", "reverse primer sequence, 5' to 3'", "pcr_primer_reverse", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("forward_primer_name", "forward primer's name", "pcr_primer_name_forward", "515F", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("reverse_primer_name", "reverse primer's name", "pcr_primer_name_reverse", "926R", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("amplicon_size", "expected amplicon length in base pairs, excluding primers/adapters", "ampliconSize", "411 bp", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("annealing_temperature", "PCR annealing temperature -- if the text separately describes a second/index PCR, this is the FIRST PCR's own annealing temperature only, never the second PCR's", "annealingTemp", "55C", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("pcr_cycle_count", "number of PCR cycles -- if the text separately describes a second/index PCR, this is the FIRST PCR's own cycle count only, never the second PCR's", "pcr_cycles", "35", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("commercial_master_mix", "commercial master mix name/brand, if one was used -- if the text separately describes a second/index PCR using a different master mix, this is the FIRST PCR's own master mix only", "commercial_mm", required_any_flags=frozenset({"pcr_0_1"})),
        FaireExtractionField("custom_master_mix", "custom master mix composition, if a commercial one was not used -- if the text separately describes a second/index PCR using a different mixture, this is the FIRST PCR's own mixture only", "custom_mm", required_any_flags=frozenset({"pcr_0_1"})),
        # A two-step metabarcoding protocol (barcoding_pcr_appr =
        # "two-step PCR", e.g. search_flags.LLM_JUDGED_SEARCH_FIELDS'
        # barcoding_pcr_appr entry) runs a SECOND PCR to add
        # indices/barcodes/sequencing adapters to the first PCR's product --
        # confirmed on two real papers (PeerJ 10.7717/peerj.333's "second
        # PCR to incorporate 454-Titanium primers and unique barcodes";
        # PLOS ONE 10.1371/journal.pone.0303937's "...required for the
        # second PCR"). The real FAIRe checklist has a dedicated pcr2_*
        # field for every one of these second-PCR concepts, mirroring the
        # first PCR's own fields one for one -- gated on pcr_0_1 like the
        # rest of this group rather than a dedicated two-step-only flag,
        # since a one-step-PCR paper simply won't have any second-PCR text
        # for the model to (correctly) find nothing in.
        # pcr2_analysis_software and the first PCR's own
        # pcr_analysis_software are deliberately excluded per an explicit
        # user request; software/method details that matter for this
        # pipeline are captured in the targeted method-specific fields.
        #
        # No example values below (unlike their first-PCR counterparts):
        # confirmed live against a real paper's second-PCR text (PeerJ
        # 10.7717/peerj.333) that the model copied this module's own
        # example strings verbatim into raw_value when the real text didn't
        # state that quantity for the second PCR at all -- exactly the
        # "never copy an example into raw_value" failure mode
        # extraction/text.py's own prompt already warns against, just
        # triggered here in practice. Omitting the example removes the
        # temptation; every other field in this module already omits one
        # where a concise, unambiguous example wasn't obviously safe (e.g.
        # forward_primer_sequence).
        # Real gap found live (10.3389/fmicb.2017.01135): "index and
        # adapter were added to the purified product during the eight
        # cycles of second-round PCR using KAPA HiFi HotStart Ready mix"
        # left both fields blank -- the old "if a two-step PCR protocol
        # was used" hint asks the model to first classify the WHOLE
        # protocol before it's willing to extract a number, which is a
        # much less certain judgment than just recognizing that THIS
        # quote itself describes a distinct second amplification step.
        # Rewritten to name the concrete trigger phrasing directly rather
        # than requiring that prior classification.
        FaireExtractionField(
            "second_pcr_annealing_temperature",
            "annealing temperature of a distinct SECOND amplification/PCR step used to add an index, "
            "barcode, or sequencing adapter (recognizable from phrasing like a second-round PCR, second "
            "PCR, PCR2, indexing PCR, or index/adapter addition during another round of cycling) -- that "
            "second PCR's own annealing temperature only, never the first/original PCR's",
            "pcr2_annealingTemp",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
        FaireExtractionField(
            "second_pcr_cycle_count",
            "number of cycles in a distinct SECOND amplification/PCR step used to add an index, barcode, "
            "or sequencing adapter (recognizable from phrasing like a second-round PCR, second PCR, PCR2, "
            "indexing PCR, or index/adapter addition during another round of cycling) -- that second "
            "PCR's own cycle count only, never the first/original PCR's",
            "pcr2_cycles",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
        # Real gap found live: pcr_method_additional/pcr2_method_additional
        # were always blank. Root cause: "PCR_amplification_conditions"
        # (pcr_method_additional's own free-text source) was deliberately
        # excluded from the LLM checklist entirely (LLM_EXCLUDED_OPTIONAL_
        # NATIVE_FIELDS) on the theory that the atomic PCR fields above
        # already capture everything useful -- and pcr2_method_additional
        # never had ANY extraction path at all (its own CategoryTerm/
        # MappingRule were retired alongside the whole PCR2 section-
        # category, per an earlier explicit user request). Per an explicit
        # user request, this narrative text is independently useful for a
        # DIFFERENT reason than the atomic fields: when a paper describes
        # two separate assays each with their own PCR, the raw wording is
        # what actually shows which paragraph/PCR a given atomic value
        # came from -- something no atomic field alone can disambiguate.
        # Un-excluded and moved out of FALLBACK_NARRATIVE_FIELDS (whose
        # own shared prompt framing says "use ONLY IF no concept above
        # applies", the opposite of "capture this regardless, for
        # context") into this group instead, gated on pcr_0_1 like its
        # siblings, with a new pcr2 counterpart mirroring it.
        FaireExtractionField(
            "PCR_amplification_conditions",
            "the FIRST/original PCR's own methods text -- primers, mixture, thermal profile, or any other "
            "PCR-specific detail in the paper's own words, captured regardless of whether the atomic PCR "
            "fields above already captured a value, since this raw text is what shows which assay/PCR round "
            "it came from when a paper describes more than one",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
        FaireExtractionField(
            "second_pcr_amplification_conditions",
            "a distinct SECOND amplification/PCR step's own methods text (recognizable from phrasing like "
            "a second-round PCR, second PCR, PCR2, indexing PCR, or index/adapter addition during another "
            "round of cycling) -- that second PCR's own words only, never the first/original PCR's, "
            "captured regardless of whether pcr2_annealingTemp/pcr2_cycles above already captured a value",
            "pcr2_method_additional",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
        FaireExtractionField(
            "probe_sequence",
            "hydrolysis/TaqMan probe sequence, 5' to 3', if a probe-based qPCR/ddPCR assay was used",
            "probe_seq",
            required_any_flags=frozenset({"pcr_0_1", "probe_based_qPCR_ddPCR_assay_0_1"}),
        ),
        FaireExtractionField(
            "probe_concentration",
            "the probe's STOCK concentration (not the final in-reaction concentration), if a probe-based qPCR/ddPCR assay was used",
            "probe_conc",
            "5 uM",
            required_any_flags=frozenset({"pcr_0_1", "probe_based_qPCR_ddPCR_assay_0_1"}),
        ),
        FaireExtractionField(
            "assay_target_taxa",
            "the taxon/taxa or species name(s) targeted by the primers/probes/other approach used in the PCR",
            "targetTaxonomicAssay",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
        FaireExtractionField(
            "study_target_taxonomic_scope",
            "the broader taxonomic group(s) targeted by the STUDY as a whole, which can differ from assay_target_taxa (e.g. assay targets Chordata, but the study's scope is Chondrichthyes -- sharks and rays)",
            "targetTaxonomicScope",
            required_any_flags=frozenset({"pcr_0_1"}),
        ),
    ),
    "Controls & replicates": (
        # biological_replicate_count (this broad-checklist entry) and
        # search_flags.CONTROLLED_SEARCH_FIELDS's own "biological_rep"
        # deterministic text-regex entry were both removed per an explicit
        # user request, after a live audit of a real 5-paper run found
        # neither ever fired. biological_rep is now derived purely from
        # structured-API/supplement biological_rep_relation data
        # (mapping/faire.py::_apply_biological_rep_from_relations) -- the
        # paper's own text is deliberately never queried for it anymore.
        FaireExtractionField("pcr_replicate_count", "number of PCR technical replicates per sample", "pcr_rep", "3", required_any_flags=frozenset({"pcr_0_1"})),
    ),
    "qPCR / standard curve": (
        FaireExtractionField("quantification_cycle_threshold", "the fluorescence threshold value used for Cq/Ct", "thresholdQuantificationCycle"),
        FaireExtractionField("quantification_cycle", "a reported quantification cycle (Cq/Ct) value", "quantificationCycle"),
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
        FaireExtractionField("forward_sequencing_adapter", "forward sequencing adapter sequence", "adapter_forward"),
        FaireExtractionField("reverse_sequencing_adapter", "reverse sequencing adapter sequence", "adapter_reverse"),
        # phix_percentage was here as an LLM-askable field; per an explicit
        # user request ("just have a quick search ... for PhiX or its
        # variations"), it's now handled entirely by a deterministic
        # regex pass instead (search_flags.py's detect_phix_percentage_
        # facts) -- a percentage number co-occurring with "PhiX" in the
        # same sentence is unambiguous enough that no LLM judgment call
        # is needed, and the deterministic pass runs over both the main
        # paper text and supplementary text already.
    ),
    "Bioinformatics workflow": (
        FaireExtractionField("clustering_tool", "software (with version) used for OTU/ASV clustering", "otu_clust_tool"),
        FaireExtractionField("reference_database", "reference database(s) used for taxonomic assignment, with version", "otu_db", "SILVA 138"),
        # bioinformatics_sop_reference (-> sop_bioinformatics) deliberately
        # removed: an explicit user instruction to never populate it at all
        # (exports/faire.py's PROJECT_METADATA_SUPPRESSED_FIELDS also drops
        # its column entirely).
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


def faire_hints_probeable_from_text() -> frozenset[str]:
    """FAIRe fields the paper-text extractor can legitimately check.

    These are the overlap fields where a structured/API value should not
    suppress paper extraction. If paper evidence later disagrees with the
    API, mapping/faire.py keeps both pieces of evidence and marks the
    standardized value for human review.

    Static no-LLM fields stay excluded here: those are fields we have
    explicitly decided not to ask the general paper LLM about at all.
    """
    return all_faire_hints() - LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS


def suppress_resolved_faire_hints_for_text(resolved_hints: frozenset[str]) -> frozenset[str]:
    """Return only resolved FAIRe hints that should suppress paper LLM asks.

    Structured sources are still preferred in mapping, but their presence
    no longer hides fields that are also readable from paper prose. This
    lets API-vs-paper conflicts surface as review_required rows instead of
    disappearing before extraction.
    """
    return resolved_hints - faire_hints_probeable_from_text()


def native_name_to_faire_hint() -> dict[str, str]:
    """Maps a taxonomy native_name to its FAIRe hint, where one exists --
    lets a caller reconstruct the intended hint if a model omits
    `candidate_standard_fields` for a field this taxonomy knows a hint
    for."""
    return {f.native_name: f.faire_hint for fields in FIELD_GROUPS.values() for f in fields if f.faire_hint}


_ASSAY_SCOPED_GROUP_NAMES = frozenset({"PCR / assay setup", "qPCR / standard curve"})
_ASSAY_SCOPED_EXTRA_NAMES = frozenset({"pcr_replicate_count"})


def assay_scoped_field_names() -> frozenset[str]:
    """Native names describing a property of one specific assay (its
    primers, target gene, PCR/qPCR conditions, ...) rather than the whole
    study -- a paper can describe more than one distinct assay run on the
    same samples, each with its own values for these. `extraction/text.py`
    uses this to decide which facts an LLM-supplied `assay_tag` applies to;
    `mapping/rules.py` uses it to generate a parallel EntityLevel.ASSAY
    mapping rule alongside the existing study-level one.

    Deliberately excludes `biological_replicate_count` (a property of a
    sample/treatment, not an assay) and `negative_control_type`/
    `positive_control_type` -- both map to FAIRe's `sampleMetadata`, not
    `projectMetadata` (verified via `mapping.rules._target_table_for_faire_field`),
    so tagging them with an assay_tag would have no effect downstream; kept
    out of the taxonomy's assay-tagging surface to keep the prompt simpler
    for zero lost benefit. `pcr_replicate_count` stays in: it genuinely maps
    to `projectMetadata` and is a real per-assay setting."""
    names: set[str] = set()
    for group_name in _ASSAY_SCOPED_GROUP_NAMES:
        names.update(f.native_name for f in FIELD_GROUPS[group_name])
    names |= _ASSAY_SCOPED_EXTRA_NAMES
    return frozenset(names)


def field_names_for_reference(
    exclude_faire_hints: frozenset[str] = frozenset(),
    include_group_names: frozenset[str] | None = None,
    include_fallback_names: frozenset[str] | None = None,
    include_native_names: frozenset[str] | None = None,
    active_flags: frozenset[str] = frozenset(),
) -> frozenset[str]:
    excluded_hints = exclude_faire_hints | LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS
    names: set[str] = set()
    for group_name, fields in FIELD_GROUPS.items():
        if include_group_names is not None and group_name not in include_group_names:
            continue
        names.update(
            f.native_name
            for f in fields
            if f.faire_hint not in excluded_hints
            and (include_native_names is None or f.native_name in include_native_names)
            and (not f.required_any_flags or (f.required_any_flags & active_flags))
        )
    names.update(
        f.native_name
        for f in FALLBACK_NARRATIVE_FIELDS
        if f.native_name not in LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS
        if include_fallback_names is None or f.native_name in include_fallback_names
        if include_native_names is None or f.native_name in include_native_names
    )
    return frozenset(names)


def render_field_reference(
    exclude_faire_hints: frozenset[str] = frozenset(),
    include_group_names: frozenset[str] | None = None,
    include_fallback_names: frozenset[str] | None = None,
    include_native_names: frozenset[str] | None = None,
    active_flags: frozenset[str] = frozenset(),
) -> str:
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
    extraction/text.py's resolved_faire_fields_for_study). The static
    LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS policy is always applied as well.
    Most entries with no faire_hint stay, but narrative fallbacks that map
    only to an excluded optional field are removed through
    LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS. A group left with zero remaining
    entries is omitted rather than rendered as an empty heading.

    `active_flags` drops any field whose `required_any_flags` is non-empty
    and shares nothing with it -- e.g. a paper with no detected PCR
    evidence (extraction/search_flags.py's `pcr_0_1`) never even sees the
    "PCR / assay setup" checklist. Mirrors
    search_flags.detect_controlled_search_facts's own identical
    `required_any_flags & active_flags` check, for the same deterministic-
    flag vocabulary. Empty (the default) means every conditional field is
    hidden -- callers that care about a flag-gated group must pass real
    flags explicitly, never assume they're shown by default.

    `include_group_names` and `include_fallback_names` let extraction/text.py
    render smaller topic-focused checklists for local 4B models while still
    drawing every concept from this same taxonomy.
    """
    excluded_hints = exclude_faire_hints | LLM_EXCLUDED_OPTIONAL_FAIRE_FIELDS
    lines: list[str] = []
    for group_name, fields in FIELD_GROUPS.items():
        if include_group_names is not None and group_name not in include_group_names:
            continue
        remaining = [
            f
            for f in fields
            if f.faire_hint not in excluded_hints
            and (include_native_names is None or f.native_name in include_native_names)
            and (not f.required_any_flags or (f.required_any_flags & active_flags))
        ]
        if not remaining:
            continue
        lines.append(f"{group_name}:")
        for f in remaining:
            example = f" (e.g. \"{f.example}\")" if f.example else ""
            hint_note = f" [FAIRe hint: {f.faire_hint}]" if f.faire_hint else ""
            lines.append(f"- {f.native_name}: {f.hint}{example}{hint_note}")
    fallback_fields = [
        f
        for f in FALLBACK_NARRATIVE_FIELDS
        if f.native_name not in LLM_EXCLUDED_OPTIONAL_NATIVE_FIELDS
        if include_fallback_names is None or f.native_name in include_fallback_names
        if include_native_names is None or f.native_name in include_native_names
    ]
    if fallback_fields:
        lines.append("General narrative fallback (use only if no concept above applies; no FAIRe hint):")
    for f in fallback_fields:
        lines.append(f"- {f.native_name}: {f.hint}")
    return "\n".join(lines)
