"""Deterministic text flags that run before any LLM extraction.

These are not mappings and they do not decide downstream FAIRe
completeness on their own. They are cheap, evidence-bearing signals used
to decide whether a later targeted paper/supplement search is worth asking
about. Structured sources still get first chance to fill exact values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from fair_ocean_agent.config import MIN_LLM_MAX_OUTPUT_TOKENS
from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.sources.base import RawFactCandidate


@dataclass(frozen=True)
class TextSearchFlag:
    fact_type_candidate: str
    description: str
    positive_patterns: tuple[re.Pattern[str], ...]
    positive_value: str = "true"
    explicit_none_patterns: tuple[re.Pattern[str], ...] = ()
    explicit_none_value: str = "0"


@dataclass(frozen=True)
class ControlledSearchField:
    term_name: str
    section: str
    description: str
    search_terms: tuple[str, ...]
    required_any_flags: frozenset[str] = frozenset()
    value_strategy: str = "literal_matches"


@dataclass(frozen=True)
class LLMJudgedSearchField:
    term_name: str
    section: str
    description: str
    search_terms: tuple[str, ...]
    output_instructions: str
    allowed_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuoteCandidate:
    quote_id: str
    field_names: tuple[str, ...]
    title: str
    snippet_index: int
    text: str


TEXT_SEARCH_FLAGS: tuple[TextSearchFlag, ...] = (
    TextSearchFlag(
        fact_type_candidate="probe_based_qPCR_ddPCR_assay_0_1",
        description="probe-based qPCR/ddPCR assay keyword evidence",
        positive_patterns=(
            re.compile(r"\bTaqMan\b", re.IGNORECASE),
            re.compile(r"\bhydrolysis\s+probe(s)?\b", re.IGNORECASE),
            re.compile(r"\bprobe[-\s]+based\b", re.IGNORECASE),
            re.compile(r"\breporter\s+dye(s)?\b", re.IGNORECASE),
            re.compile(r"\bHRP[-\s]?labeled\b", re.IGNORECASE),
            re.compile(r"\bhorseradish\s+peroxidase\b", re.IGNORECASE),
            re.compile(r"\bquencher(s)?\b", re.IGNORECASE),
            re.compile(r"\bFAM\b"),
            re.compile(r"\bHEX\b"),
            re.compile(r"\bVIC\b"),
            re.compile(r"\bBHQ(?:-\d+)?\b", re.IGNORECASE),
            re.compile(r"\bZEN\b"),
            re.compile(r"\bMGB\b"),
        ),
    ),
    # pcr_0_1 used to be its own independent regex scan here -- removed
    # per an explicit user instruction ("this is essentially if category
    # PCR1=True... worth rewiring so we don't duplicate its process") and
    # now derived directly from extraction/section_categories.py's own
    # pcr1_primary_amplification_0_1 detection instead
    # (derive_pcr_0_1_from_category_detection), so the two can never
    # disagree.
    # neg_cont_0_1/pos_cont_0_1 used to be their own regex-only
    # positive/explicit-none pattern pairs here -- replaced per an
    # explicit user instruction ("use keyword 'control','blank' then
    # context in sentence to determine if positive or negative control or
    # not a control") with LLM_JUDGED_SEARCH_FIELDS entries below, which
    # let the model disambiguate a bare "control"/"blank" mention (e.g.
    # "quality control", "controlled for depth", "control treatment" in
    # an unrelated experimental-design sense) from a genuine sample
    # negative/positive control -- a judgment call a keyword regex alone
    # can't reliably make.
)

LLM_JUDGED_SEARCH_FIELDS: tuple[LLMJudgedSearchField, ...] = (
    LLMJudgedSearchField(
        term_name="barcoding_pcr_appr",
        section="Library preparation sequencing",
        description="How sequencing adapters and sample barcodes/indexes were added during metabarcoding library preparation.",
        allowed_values=("one-step PCR", "two-step PCR", "ligation-based"),
        output_instructions=(
            "Classify only the barcoding/indexing/library-construction approach for metabarcoding. "
            "Use one of: one-step PCR, two-step PCR, ligation-based. Map two-step PCR, two-stage PCR, "
            "two-round PCR, two-round PCR amplification strategy, second-round PCR, second round of PCR, "
            "second PCR, PCR2, indexing PCR, barcode PCR, or adapter PCR to two-step PCR -- including a "
            "quote that only describes adding an index/adapter during a distinct second amplification step "
            "(e.g. 'during the eight cycles of second-round PCR'), even if it never uses the literal phrase "
            "'two-step'. Map ligation-based, adapter ligation, barcode ligation, ligated adapters, ligation "
            "sequencing kit, or library adapters were ligated to ligation-based. Omit this field if the quote "
            "only mentions ordinary PCR without a library/barcoding/indexing context; a separate fallback "
            "handles one-step PCR."
        ),
        search_terms=(
            "one-step PCR",
            "single-step PCR",
            "one-step library",
            "fusion primer",
            "fusion primers",
            "tailed primer",
            "indexed primer",
            "two-step PCR",
            "two-stage PCR",
            "two-round PCR",
            "two-round PCR amplification strategy",
            # Real gap found live (10.3389/fmicb.2017.01135): "the eight
            # cycles of second-round PCR" never matched "second PCR" at
            # all -- the inserted "-round" broke the fixed phrase.
            "second-round PCR",
            "second round PCR",
            "second round of PCR",
            "second PCR",
            "indexing PCR",
            "barcode PCR",
            "adapter PCR",
            "adapter ligation",
            "barcode ligation",
            "ligation",
        ),
    ),
    LLMJudgedSearchField(
        term_name="assay_name",
        section="PCR",
        description=(
            "A short, machine-readable name identifying a specific named assay or marker-region assay used in the study."
        ),
        output_instructions=(
            "Return the published assay name when one exists. Otherwise use a concise marker-region name only "
            "when the marker and region are both explicit, such as 16S-V4 or 12S-V5. Return only the name, "
            "not a sentence. Do not return raw primer-pair names such as 515F/806R, and do not return bare "
            "functional gene targets such as hzsA as assay names; those belong in primer or target-gene fields."
        ),
        search_terms=(
            "assay",
            "assay name",
            "named assay",
            "MiFish-U",
            "MiFish-E",
            "MiFish",
            "Teleo",
            "Teleo01",
            "12S-V5",
            "MiBird",
            "MiMammal",
            "TAReuk",
            "Leray-XT",
            "Leray",
            "Folmer",
            "Uni18S",
            "16S V4",
            "16S V3-V4",
            "16S V4-V5",
            "18S V4",
            "18S V9",
            "18S V4-V9",
            "28S D1-D2",
            "28S D2-D3",
            "12S V5",
            "COI-5P",
            "ITS1",
            "ITS2",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V1-V2",
            "V2-V3",
            "V3-V4",
            "V3-V5",
            "V4-V5",
            "V4-V9",
            "V5-V7",
            "V6-V8",
            "V7-V9",
            "D1",
            "D2",
            "D1-D2",
            "D2-D3",
            "species-specific assay",
            "taxon-specific assay",
            "targeted assay",
            "qPCR assay",
            "ddPCR assay",
            "TaqMan assay",
            "hydrolysis probe assay",
            "assay developed by",
            "assay described by",
            "following the assay of",
            "modified assay",
        ),
    ),
    LLMJudgedSearchField(
        term_name="assay_target_taxa",
        section="PCR",
        description=(
            "The taxon or taxonomic group the PCR assay itself is designed to amplify/detect, based on the "
            "primers, probes, or other PCR approach -- the assay's technical capability, not necessarily the "
            "narrower organisms the study cares about (see study_target_taxonomic_scope for that)."
        ),
        output_instructions=(
            "Return the free-text taxonomic name(s) the assay/primers/probes were designed to amplify or "
            "detect, preserving the paper's own wording (e.g. 'vertebrates', 'teleost fishes', 'Eukaryota', "
            "'Atlantic salmon (Salmo salar)'). If multiple distinct target taxa are explicitly supported, "
            "return one object per value; final merged output is pipe-delimited. Do not infer a target taxon "
            "from a primer/assay name alone (e.g. 'MiFish' implies fish, but only return it if the quote "
            "itself states the target, not just the assay name). Never return the gene/marker/locus itself "
            "(e.g. '16S rRNA gene', '18S rRNA', 'COI', 'ITS') as the answer -- a gene name is not a taxon; "
            "if the quote only names the gene amplified without stating which organism(s) or taxonomic group "
            "it targets, there is no answer in that quote."
        ),
        search_terms=(
            "primers targeting",
            "primers targeted",
            "designed to amplify",
            "designed to target",
            "designed to detect",
            "assay targeting",
            "assay targeted",
            "assay specific for",
            "species-specific primers",
            "species-specific assay",
            "taxon-specific primers",
            "taxon-specific assay",
            "specific primers for",
            "probe targeting",
            "probe specific for",
            "designed for detection of",
            "used to detect",
            "amplifies",
            "amplification of",
            "universal primers for",
            "universal primer set",
            "broad-range primers",
            "metazoan primers",
            "vertebrate primers",
            "fish primers",
            "eukaryotic primers",
            "prokaryotic primers",
            "bacterial primers",
            "archaeal primers",
        ),
    ),
    # study_target_taxonomic_scope used to be a quote-judged search field here.
    # Temporarily disabled in favor of an abstract-level generative pass in
    # extraction/study_factor.py::generate_study_target_taxonomic_scope, because
    # real audits showed the quote path either kept whole evidence sentences
    # ("AOA's distribution was explored...") or returned "not found" for broad
    # abstract-level scopes like prokaryotic microorganisms/bacteria/archaea.
    LLMJudgedSearchField(
        term_name="adapter_forward",
        section="Library preparation sequencing",
        description="Forward sequencing adapter sequence.",
        output_instructions=(
            "Return only an explicit forward/read 1/P5/5-prime adapter sequence. Copy the sequence exactly "
            "from the quote, preserving letters and order. A named tag system fused ahead of the real PCR "
            "primer (e.g. Fluidigm CS1, Fluidigm Access Array CS1, a Nextera or TruSeq overhang) IS an "
            "adapter even if the quote never uses the word \"adapter\" -- return its own sequence, not the "
            "primer's. Omit this field if the quote names an adapter but does not give the sequence."
        ),
        search_terms=(
            "adapter-containing forward primer",
            "forward adapter",
            "adapter sequence",
            "read 1 adapter",
            "R1 adapter",
            "P5 adapter",
            "Illumina adapter",
            "sequencing adapter",
            "5' adapter",
            "5′ tail",
            "forward overhang",
            "Illumina overhang",
            "primer overhang",
            "primer tail",
            "overhang",
            "fusion primer",
            "fusion primer sequence",
            "tailed primer",
            "adapter-tailed primer",
            "adapter-tagged primer",
            "adapter-linked primer",
            "sequencing tail",
            "adapter",
            # Fluidigm CS1/CS2 ("common sequence") tags: a real, common
            # named adapter-tag system on their own Access Array PCR
            # platform, functionally identical to a Nextera/TruSeq
            # overhang, but named without the word "adapter" anywhere
            # nearby -- real gap found live: "Fluidigm CS1 + MiFish-U-F
            # ACACTGACGACATGGTTCTACA GTCGGTAAAACTCGTGCCAGC" never matched
            # any existing cue at all.
            "Fluidigm CS1",
            "Fluidigm CS2",
            "CS1 tag",
            "CS2 tag",
            "common sequence 1",
            "common sequence 2",
            "Access Array",
        ),
    ),
    LLMJudgedSearchField(
        term_name="adapter_reverse",
        section="Library preparation sequencing",
        description="Reverse sequencing adapter sequence.",
        output_instructions=(
            "Return only an explicit reverse/read 2/P7/3-prime adapter sequence. Copy the sequence exactly "
            "from the quote, preserving letters and order. A named tag system fused ahead of the real PCR "
            "primer (e.g. Fluidigm CS2, Fluidigm Access Array CS2, a Nextera or TruSeq overhang) IS an "
            "adapter even if the quote never uses the word \"adapter\" -- return its own sequence, not the "
            "primer's. Omit this field if the quote names an adapter but does not give the sequence."
        ),
        search_terms=(
            "adapter-containing reverse primer",
            "reverse adapter",
            "read 2 adapter",
            "R2 adapter",
            "P7 adapter",
            "Illumina adapter",
            "sequencing adapter",
            "3' adapter",
            "5′ tail",
            "reverse overhang",
            "Illumina overhang",
            "primer overhang",
            "primer tail",
            "overhang",
            "fusion primer",
            "fusion primer sequence",
            "tailed primer",
            "adapter-tailed primer",
            "adapter-tagged primer",
            "adapter-linked primer",
            "sequencing tail",
            "adapter",
            # See adapter_forward's own comment on Fluidigm CS1/CS2.
            "Fluidigm CS1",
            "Fluidigm CS2",
            "CS1 tag",
            "CS2 tag",
            "common sequence 1",
            "common sequence 2",
            "Access Array",
        ),
    ),
    LLMJudgedSearchField(
        term_name="sequencing_methodology",
        section="Library preparation sequencing",
        description=(
            "Relevant sequencing-method sentences from the paper, including the sequencing platform, "
            "instrument, chemistry/kit, read layout, read length, library loading/run setup, and the "
            "statement that libraries/amplicons/products were sequenced."
        ),
        output_instructions=(
            "Return the full relevant sequencing-method sentence or phrase, preserving the source wording. "
            "Include sentences that describe the actual sequencing run, platform/instrument, kit/chemistry, "
            "paired-end/single-end layout, read length, flow cell, loading, or library sequencing. Omit "
            "sample collection, PCR-only details with no sequencing/library-run information, and downstream "
            "bioinformatic analysis."
        ),
        search_terms=(
            "sequenced by",
            "sequenced on",
            "sequenced using",
            "sequencing was performed",
            "amplicon sequencing",
            "library was sequenced",
            "libraries were sequenced",
            "PCR product was sequenced",
            "PCR products were sequenced",
            "MiSeq",
            "HiSeq",
            "NextSeq",
            "NovaSeq",
            "Ion Torrent",
            "Ion PGM",
            "PacBio",
            "SMRT",
            "Oxford Nanopore",
            "MinION",
            "paired-end",
            "single-end",
            "2 x",
            "2x",
            "read length",
            "bp reads",
            "cycle kit",
            "reagent kit",
            "flow cell",
            "sequencing run",
        ),
    ),
    LLMJudgedSearchField(
        term_name="inhibition_check_0_1",
        section="PCR",
        description="Whether the study explicitly tested samples or DNA extracts for PCR inhibition.",
        allowed_values=("0", "1"),
        output_instructions=(
            "Return 1 only when the quote explicitly says PCR inhibition was checked, tested, assessed, "
            "evaluated, detected, or controlled using an inhibition-specific test/control. Return 0 only "
            "when the quote explicitly says inhibition was not checked/tested/assessed. Omit the field when "
            "there is no explicit inhibition-check statement. Do not return 1 from BSA, sample dilution, or "
            "an inhibitor-resistant master mix alone unless the quote explicitly connects it to checking or "
            "detecting inhibition."
        ),
        search_terms=(
            "PCR inhibition",
            "PCR inhibitor",
            "inhibition check",
            "inhibition test",
            "tested for inhibition",
            "internal amplification control",
            "internal positive control",
            "IAC",
            "IPC",
            "inhibition control",
            "exogenous internal control",
            "exogenous control",
            "spike-in control",
            "spiked control",
            "spiked sample",
            "DNA spike",
            "template spike",
            "serial dilution",
            "dilution series",
            "dilution test",
            "sample dilution",
            "undiluted",
            "1:5 dilution",
            "1:10 dilution",
            "1:100 dilution",
            "Cq shift",
            "Ct shift",
            "ΔCq",
            "delta Cq",
            "ΔCt",
            "delta Ct",
            "amplification efficiency",
            "reduced amplification",
            "delayed amplification",
            "matrix inhibition",
            "environmental inhibition",
            "inhibitory substances",
            "inhibition index",
            "inhibition assay",
            "inhibitor assay",
        ),
    ),
    LLMJudgedSearchField(
        term_name="inhibition_check",
        section="PCR",
        description=(
            "Method used to detect PCR inhibition, plus any action taken to reduce it, such as dilution, "
            "cleanup, additives, or inhibitor-tolerant reagents."
        ),
        output_instructions=(
            "Return the full explicit method phrase/sentence supported by the quote. Include the inhibition "
            "test/control and any mitigation action such as dilution, cleanup, additives, or inhibitor-tolerant "
            "reagents. Omit this field when the quote mentions BSA, dilution, or inhibitor-tolerant reagents "
            "without explicitly tying them to inhibition detection or mitigation."
        ),
        search_terms=(
            "PCR inhibition",
            "tested for inhibition",
            "inhibition was assessed",
            "inhibition was evaluated",
            "internal amplification control",
            "internal positive control",
            "IAC",
            "IPC",
            "spike-in",
            "spiked with",
            "exogenous DNA",
            "synthetic DNA control",
            "serial dilution",
            "dilution series",
            "dilution test",
            "Cq shift",
            "Ct shift",
            "ΔCq",
            "ΔCt",
            "amplification efficiency",
            "samples were diluted",
            "DNA was diluted",
            "extracts were diluted",
            "inhibitor removal",
            "PCR inhibitor removal",
            "additional purification",
            "DNA cleanup",
            "OneStep PCR Inhibitor Removal Kit",
            "PowerClean",
            "DNeasy PowerClean",
            "InhibitEX",
            "BSA",
            "bovine serum albumin",
            "T4 gene 32 protein",
            "PVP",
            "PVPP",
            "inhibitor-tolerant",
            "inhibitor-resistant polymerase",
            "Environmental Master Mix",
            "humic acid",
            "humic substances",
            "tannins",
            "polysaccharides",
            "phenolic compounds",
            "optimal dilution",
            "reduced inhibition",
            "improved amplification",
            "inhibition mitigation",
        ),
    ),
    LLMJudgedSearchField(
        term_name="amp_vis_method",
        section="Targeted detection",
        description="Method used to visualize or verify PCR amplification products.",
        output_instructions=(
            "Return only the PCR amplicon/product visualization or verification method, such as agarose gel "
            "electrophoresis, gel electrophoresis, capillary electrophoresis, Bioanalyzer, or TapeStation. "
            "Only accept it when the quote uses the method to inspect or verify amplified PCR products. Do "
            "not confuse microscopy/FISH visualization with PCR amplicon visualization."
        ),
        search_terms=(
            "agarose gel electrophoresis",
            "gel electrophoresis",
            "electrophoresis",
            "capillary electrophoresis",
            "Bioanalyzer",
            "TapeStation",
            "amplicons were visualized",
            "PCR products were visualized",
            "PCR products were checked",
            "amplicon bands",
            "PCR bands",
            "bands were observed",
        ),
    ),
    LLMJudgedSearchField(
        term_name="block_seq",
        section="Targeted detection",
        description="Nucleotide sequence of a blocking primer or blocking oligonucleotide.",
        output_instructions=(
            "Return only an explicit nucleotide sequence for a blocking primer, blocking oligo, blocker, "
            "host-blocking primer, or suppressor oligonucleotide. Do not return ordinary forward/reverse "
            "PCR primer sequences."
        ),
        search_terms=(
            "blocking primer",
            "blocking primers",
            "blocking oligo",
            "blocking oligonucleotide",
            "blocking oligonucleotides",
            "blocker",
            "host-blocking primer",
            "suppressor oligonucleotide",
        ),
    ),
    LLMJudgedSearchField(
        term_name="block_ref",
        section="Targeted detection",
        description="Citation/reference/provenance associated with a blocking primer or blocking oligonucleotide.",
        output_instructions=(
            "Return the citation, DOI, reference phrase, or provenance for the blocking primer/oligonucleotide. "
            "If it was designed in the current study, return that it was designed in the current study. Do not "
            "invent an external citation. Omit ordinary primer/probe references unless the quote explicitly "
            "describes a blocking primer, blocking oligo, blocker, or host-blocking/suppressor oligonucleotide."
        ),
        search_terms=(
            "blocking primer",
            "blocking oligo",
            "blocking oligonucleotide",
            "blocker",
            "host-blocking primer",
            "suppressor oligonucleotide",
        ),
    ),
    LLMJudgedSearchField(
        term_name="block_taxa",
        section="Targeted detection",
        description="Taxon/taxa whose amplification or hybridization the blocking oligonucleotide is intended to suppress.",
        output_instructions=(
            "Return the taxon or broad source the blocking oligo is intended to suppress, such as human, fish, "
            "host, chloroplast, mitochondrial, predator DNA, or another explicitly stated target. Do not infer "
            "the taxon from the sample type alone. Omit taxonomy-filtering or contaminant-removal sentences "
            "that mention chloroplasts, mitochondria, host reads, or non-target taxa but do not describe a "
            "blocking primer/oligo."
        ),
        search_terms=(
            "blocking primer",
            "blocking oligo",
            "blocking oligonucleotide",
            "suppress amplification",
            "inhibit amplification",
            "block host",
            "host DNA",
            "chloroplast",
            "mitochondrial",
            "predator DNA",
        ),
    ),
    LLMJudgedSearchField(
        term_name="detection_criteria",
        section="Targeted detection",
        description="Explicit criteria used to decide whether a target was detected/present/positive.",
        output_instructions=(
            "Return ONLY an explicit rule used to call a target positive/present/detected, preserving "
            "thresholds and replicate rules. Examples include Cq < 40 considered positive, target amplified "
            "in 2 of 3 technical replicates was accepted as detected, minimum positive droplets, fluorescence "
            "above threshold, minimum copy-number criterion, or explicit microscopy/hybridization positive "
            "criterion. Omit ordinary biological/PCR replicate counts, sample pooling, amplification success, "
            "statistical analysis of replicates, or a general statement that the target was detected unless "
            "the quote states the rule for calling the target positive/present/detected."
        ),
        search_terms=(
            "considered positive",
            "scored positive",
            "called positive",
            "accepted as positive",
            "counted as positive",
            "positive if",
            "positive when",
            "detection criteria",
            "detection criterion",
            "presence was defined",
            "presence was confirmed",
            "detected if",
            "detected when",
            "accepted as detected",
            "confirmed when",
            "Cq <",
            "Ct <",
            "quantification cycle below",
            "positive droplets",
            "fluorescence above threshold",
            "minimum copy number",
        ),
    ),
    LLMJudgedSearchField(
        term_name="lod_method",
        section="Targeted detection",
        description="How the assay limit of detection was determined.",
        output_instructions=(
            "Return the method used to determine LOD, such as dilution series, replicate detection probability, "
            "lowest standard consistently detected, probit/logistic model, synthetic target dilution, or "
            "genomic DNA dilution. Do not infer the method from an LOD value alone."
        ),
        search_terms=(
            "limit of detection",
            "LOD",
            "detection limit",
            "dilution series",
            "lowest standard",
            "consistently detected",
            "probit",
            "logistic model",
            "synthetic target dilution",
            "genomic DNA dilution",
        ),
    ),
    LLMJudgedSearchField(
        term_name="loq_method",
        section="Targeted detection",
        description="How the assay limit of quantification was determined.",
        output_instructions=(
            "Return the method used to determine LOQ, such as lowest standard meeting precision criteria, "
            "coefficient-of-variation threshold, standard-curve based determination, or replicate "
            "quantification criterion. Do not infer the method from an LOQ value alone."
        ),
        search_terms=(
            "limit of quantification",
            "LOQ",
            "quantification limit",
            "lowest standard",
            "precision criteria",
            "coefficient of variation",
            "coefficient-of-variation",
            "CV threshold",
            "standard curve",
            "replicate quantification",
        ),
    ),
    LLMJudgedSearchField(
        term_name="pcr_assay_lod",
        section="Targeted detection",
        description="Numerical assay limit of detection.",
        output_instructions=(
            "Return only the numerical LOD value explicitly reported by the quote, without the unit. Do not "
            "return the LOD method or infer an LOD from a dilution series."
        ),
        search_terms=("limit of detection", "LOD", "detection limit"),
    ),
    LLMJudgedSearchField(
        term_name="pcr_assay_lod_unit",
        section="Targeted detection",
        description="Unit corresponding to the assay limit of detection.",
        output_instructions=(
            "Return only the LOD unit explicitly reported by the quote, such as copies/reaction, copies/uL, "
            "copies/L, gene copies, or cells/reaction. Keep the paper's unit wording."
        ),
        search_terms=("limit of detection", "LOD", "detection limit", "copies/reaction", "copies/uL", "gene copies"),
    ),
    LLMJudgedSearchField(
        term_name="pcr_assay_loq",
        section="Targeted detection",
        description="Numerical assay limit of quantification.",
        output_instructions=(
            "Return only the numerical LOQ value explicitly reported by the quote, without the unit. Do not "
            "return the LOQ method or infer an LOQ from a standard curve."
        ),
        search_terms=("limit of quantification", "LOQ", "quantification limit"),
    ),
    LLMJudgedSearchField(
        term_name="pcr_assay_loq_unit",
        section="Targeted detection",
        description="Unit corresponding to the assay limit of quantification.",
        output_instructions=(
            "Return only the LOQ unit explicitly reported by the quote. Keep the paper's unit wording."
        ),
        search_terms=("limit of quantification", "LOQ", "quantification limit", "copies/reaction", "copies/uL"),
    ),
    LLMJudgedSearchField(
        term_name="probe_seq",
        section="Targeted detection",
        description="Nucleotide sequence of an explicitly identified detection or hybridization probe.",
        output_instructions=(
            "Return only the nucleotide sequence of an explicitly identified detection/hybridization probe, "
            "including qPCR, TaqMan/hydrolysis probe, molecular beacon, FISH probe, or CARD-FISH probe. Do "
            "not extract ordinary PCR primer sequences into this field."
        ),
        search_terms=(
            "probe sequence",
            "TaqMan probe",
            "hydrolysis probe",
            "molecular beacon",
            "FISH probe",
            "CARD-FISH probe",
            "oligonucleotide probe",
            "hybridization probe",
            "probe",
        ),
    ),
    LLMJudgedSearchField(
        term_name="probe_ref",
        section="Targeted detection",
        description="Citation/reference/provenance for the detection or hybridization probe.",
        output_instructions=(
            "Return the citation, DOI, reference phrase, or provenance for the detection/hybridization probe "
            "only when the same quote explicitly describes a reporter/probe-based assay or a FISH/CARD-FISH "
            "probe. If the probe was newly designed in the current paper, return that rather than fabricating "
            "a reference. Omit instrument names, environmental sensor citations, ordinary PCR primer "
            "references, and database/software citations."
        ),
        search_terms=(
            "probe described",
            "previously published probe",
            "probe was described",
            "probe designed",
            "designed in this study",
            "designed for this study",
            "modified from",
            "HRP-labeled",
            "horseradish peroxidase",
            "FISH probe",
            "CARD-FISH probe",
        ),
    ),
    LLMJudgedSearchField(
        term_name="probe_conc",
        section="Targeted detection",
        description="Explicitly reported concentration of a qPCR, FISH, CARD-FISH, or other detection probe.",
        output_instructions=(
            "Return ONLY the probe concentration with unit exactly as reported, such as 0.5 uM or 200 nM. "
            "The same quote must explicitly say this concentration belongs to a detection/hybridization "
            "probe. Do not return primer concentrations, DNA/RNA concentrations, filter pore sizes, or full "
            "sentences."
        ),
        search_terms=(
            "probe concentration",
            "probe was added",
            "probe final concentration",
            "probe at",
            "0.5 uM probe",
            "nM probe",
            "uM probe",
            "µM probe",
            "nM",
            "uM",
            "µM",
            "μM",
        ),
    ),
    LLMJudgedSearchField(
        term_name="std_source",
        section="Targeted detection",
        description="Source/material used to create an assay standard or standard curve.",
        output_instructions=(
            "Return the material/source used for assay calibration or the standard curve, such as plasmid "
            "containing target sequence, synthetic gBlock, genomic DNA, purified PCR product, cloned target "
            "sequence, or cultured organism DNA. Do not treat cloned material as a qPCR standard unless the "
            "quote explicitly uses it for assay calibration/quantification."
        ),
        search_terms=(
            "standard curve",
            "standard curves",
            "assay standard",
            "calibration standard",
            "plasmid standard",
            "synthetic gBlock",
            "gBlock",
            "genomic DNA standard",
            "purified PCR product",
            "cloned target",
            "serial dilution",
            "known copy number",
        ),
    ),
    LLMJudgedSearchField(
        term_name="thresholdQuantificationCycle",
        section="Targeted detection",
        description="Explicit fluorescence threshold parameter used to determine qPCR Cq/Ct.",
        output_instructions=(
            "Return only the actual fluorescence threshold parameter used to determine Cq/Ct, if explicitly "
            "stated. Do not populate with sample Cq values, Ct positivity cutoffs, number of PCR cycles, or "
            "detection criteria."
        ),
        search_terms=(
            "fluorescence threshold",
            "threshold quantification cycle",
            "quantification cycle threshold",
            "Ct threshold",
            "Cq threshold",
            "threshold cycle",
            "baseline threshold",
        ),
    ),
    LLMJudgedSearchField(
        term_name="targeted_detection_method_additional",
        section="Targeted detection",
        description=(
            "Useful targeted-detection details that do not fit cleanly elsewhere, including qPCR/ddPCR/FISH/"
            "CARD-FISH chemistry, probe names, hybridization conditions, labels, counterstains, validation, "
            "blocking oligos, detection rules, standard-curve details, and adapter/index addition when "
            "sequences are unavailable."
        ),
        output_instructions=(
            "Return a concise source-faithful sentence or semicolon-separated phrase preserving useful targeted "
            "detection methodology. Use this when the quote contains important qPCR, dPCR/ddPCR, FISH, CARD-FISH, "
            "blocking-oligo, assay-validation, standard-curve, detection-limit, inhibition-test, probe, or "
            "adapter/indexing detail that is not fully captured by another field. Do not include generic PCR, "
            "sequencing, sample collection, or downstream bioinformatics unless the sentence is specifically "
            "part of targeted detection or adapter/index addition without explicit adapter sequences."
        ),
        search_terms=(
            "qPCR",
            "quantitative PCR",
            "real-time PCR",
            "digital PCR",
            "dPCR",
            "ddPCR",
            "targeted PCR",
            "targeted detection",
            "probe-based detection",
            "FISH",
            "CARD-FISH",
            "fluorescence in situ hybridization",
            "hybridization buffer",
            "formamide",
            "hybridization probe",
            "TaqMan",
            "hydrolysis probe",
            "molecular beacon",
            "HRP-labeled",
            "horseradish peroxidase",
            "tyramide",
            "signal deposition",
            "counterstaining",
            "SYBR Green",
            "DAPI",
            "blocking oligo",
            "blocking primer",
            "detection criteria",
            "limit of detection",
            "limit of quantification",
            "standard curve",
            "PCR inhibition",
            "index adapter",
            "adapter were added",
            "adapters were added",
        ),
    ),
    LLMJudgedSearchField(
        term_name="otu_clust_tool",
        section="Bioinformatics",
        description=(
            "Software and version used to generate OTUs or ASVs from sequence reads. Do not confuse this "
            "with the taxonomic-classification tool; databases/classifiers such as SILVA, PR2, UNITE, RDP, "
            "BLAST, and naive Bayes generally assign taxonomy rather than generating OTUs/ASVs."
        ),
        output_instructions=(
            "Return the full OTU/ASV generation tool or pipeline component, including version/command when "
            "stated. When the quote gives both a fully spelled-out algorithm name and its abbreviation (e.g. "
            "'Markov Cluster algorithm (MCL)'), return the full form with the abbreviation, not the bare "
            "abbreviation alone. Accept ASV inference/denoising tools such as DADA2 or Deblur when the quote says "
            "ASVs/exact sequence variants were inferred/generated/denoised. Omit taxonomy-only classifiers or "
            "reference databases. If multiple tools are present, return the best option according to the "
            "configured search-term priority."
        ),
        search_terms=(
            "DADA2",
            "dada2",
            "q2-dada2",
            "QIIME 2",
            "QIIME2",
            "dada2 denoise-paired",
            "dada2 denoise-single",
            "QIIME 2 VSEARCH",
            "q2-vsearch",
            "cluster-features-de-novo",
            "cluster-features-closed-reference",
            "cluster-features-open-reference",
            "USEARCH",
            "UPARSE",
            "UCLUST",
            "VSEARCH",
            "mothur",
            "cluster.split",
            "cluster.classic",
            "OptiClust",
            "Deblur",
            "q2-deblur",
            "denoise-16S",
            "denoise-other",
            "UNOISE",
            "UNOISE2",
            "UNOISE3",
            "unoise3",
            "SWARM",
            "Swarm",
            "SUMACLUST",
            "sumaclust",
            "CD-HIT",
            "CD-HIT-EST",
            "cd-hit-est",
            "OBITools",
            "OBITools3",
            "CROP",
            "ESPRIT",
            "ESPRIT-Tree",
            "MICCA",
            "LotuS",
            "LotuS2",
            "USEARCH cluster_otus",
            "cluster_fast",
            "cluster_smallmem",
            "VSEARCH --cluster_fast",
            "--cluster_fast",
            "--cluster_size",
            "--cluster_unoise",
            "QIIME pick_otus.py",
            "pick_otus.py",
            "pick_open_reference_otus.py",
            "pick_closed_reference_otus.py",
            "BLASTClust",
            "DNACLUST",
            "GramCluster",
            "MCL",
            "Markov clustering",
            "OTU clustering was performed using",
            "sequences were clustered using",
            "ASVs were inferred using",
            "exact sequence variants were generated using",
            "denoised using",
        ),
    ),
    LLMJudgedSearchField(
        term_name="otu_db",
        section="Bioinformatics",
        description=(
            "Reference sequence database, including version/release/date when reported, used to assign "
            "taxonomy to OTUs or ASVs. If authors built their own database, record custom. "
            "Parallel-META3 and spelling variants count as otu_db because the pipeline carries a "
            "prepackaged taxonomy database even when a separate database such as SILVA is not named."
        ),
        output_instructions=(
            "Return the database name plus version/release/download/access date when stated. If multiple "
            "databases are used, return one value per database. Return custom plus a short description for a "
            "custom/in-house/local/curated database. Do not return assignment software alone, such as BLAST, "
            "RDP Classifier, QIIME 2, or naive Bayes, unless a reference database is also named. "
            "Exception: return Parallel-META3/Parallel META 3 as a database value when named, because it has "
            "a bundled taxonomy database. If another database is also named, return both as separate values."
        ),
        search_terms=(
            "reference database",
            "taxonomy database",
            "taxonomic database",
            "sequence database",
            "Parallel-META3",
            "Parallel META3",
            "Parallel META 3",
            "Parallel-META 3",
            "SILVA",
            "SILVA database",
            "SILVA_132",
            "PR2",
            "Protist Ribosomal Reference database",
            "FreshTrain",
            "FreshTrain database",
            "NCBI GenBank",
            "GenBank",
            "NCBI nucleotide",
            "NCBI nt",
            "NCBI nr",
            "NCBI database",
            "nonredundant NCBI database",
            "non-redundant NCBI database",
            "nonredundant (nr) NCBI database",
            "nr database",
            "nt database",
            "BOLD",
            "Barcode of Life Data System",
            "BOLD Systems",
            "UNITE",
            "UNITE database",
            "RDP",
            "Ribosomal Database Project",
            "Greengenes",
            "Greengenes2",
            "MIDORI",
            "MIDORI2",
            "MitoFish",
            "MitoFish database",
            "MetaZooGene",
            "MetaZooGene Atlas",
            "Diat.barcode",
            "DiatBarcode",
            "PhytoREF",
            "MaarjAM",
            "GTDB",
            "Genome Taxonomy Database",
            "MitoZoa",
            "EMBL",
            "ENA reference database",
            "custom database",
            "custom reference database",
            "curated database",
            "in-house database",
            "local database",
            "reference library",
            "sequence library",
            "barcode reference library",
            "trained classifier",
            "pretrained classifier",
            "classifier trained on",
            "release",
            "version",
            "v.",
            "accessed on",
            "downloaded on",
        ),
    ),
    # otu_seq_comp_appr's only other mechanism was extraction/
    # section_categories.py's CategoryTerm (taxonomic_assignment category,
    # Stage 3) -- added here as a companion per an explicit user request,
    # after a live 6-study audit found it's the one category-pipeline-
    # exclusive term (outside sample_prep) with no fallback anywhere else,
    # unlike its close sibling otu_db (also broad-checklist + this same
    # mechanism). Same search_terms as that CategoryTerm.
    LLMJudgedSearchField(
        term_name="otu_seq_comp_appr",
        section="Bioinformatics",
        description=(
            "The software, algorithm, or sequence-comparison tool used to compare OTU/ASV/feature "
            "sequences against reference sequences for taxonomic assignment, including version when "
            "reported. This includes alignment, similarity-search, or taxonomic sequence-comparison "
            "tools -- not software used only for OTU/ASV generation, read trimming, or assembly."
        ),
        output_instructions=(
            "Return the tool/algorithm name and version, if given. Only return it when the text shows "
            "it was used to assign or identify taxonomy, not for an unrelated sequence-comparison step."
        ),
        search_terms=(
            "sequences aligned using",
            "sequence comparison performed using",
            "alignment performed with",
            "queries aligned against",
            "reference sequences searched using",
            "sequence similarity search performed with",
            "alignment software",
            "query sequences compared using",
            "reference matching performed using",
            "taxonomic alignment performed with",
            "taxonomic classification",
            "taxonomic assignment",
            "taxonomy assigned",
            "assigned taxonomy",
            "taxonomic identification",
            "classified taxonomically",
            "sequences classified",
            "OTUs classified",
            "ASVs classified",
            "representative sequences classified",
            "compared against reference",
            "compared with reference sequences",
            "searched against",
            "aligned against",
            "matched against",
            "sequence similarity search",
            "best hit",
            "reference database",
            "reference sequences",
            "BLAST",
            "BLASTn",
            "MegaBLAST",
            "USEARCH",
            "VSEARCH",
            "CREST",
            "CREST4",
            "Kraken",
            "Kraken2",
            "Kraken 2",
        ),
    ),
    # screen_contam_method's only other mechanism was extraction/
    # section_categories.py's CategoryTerm (otu_asv_generation_filtering
    # category, Stage 3) -- added here as a companion per an explicit
    # user request, since that whole category is being retired alongside
    # every other non-sample_prep category, but the user separately
    # confirmed this specific field ("this is in the bioinformatics
    # pipeline, used to remove taxa found in control samples") must stay
    # working. Same cues/definition as that CategoryTerm.
    LLMJudgedSearchField(
        term_name="screen_contam_method",
        section="Bioinformatics",
        description=(
            "The method or rule used after sequencing to identify, flag, or remove sequences/OTUs/"
            "ASVs/taxa considered contaminants. This may include comparison with negative controls, "
            "prevalence- or frequency-based contaminant detection, removal of taxa enriched in blanks, "
            "or use of dedicated contamination-screening software."
        ),
        output_instructions=(
            "Include the software, threshold, or decision rule when stated."
        ),
        search_terms=(
            "contaminants screened using",
            "contaminant removal",
            "contamination filtering",
            "negative-control filtering",
            "blank-based filtering",
            "control-based contaminant removal",
            "features detected in blanks",
            "background contamination threshold",
            "contaminant sequences removed",
            "contamination screening method",
            "contaminant",
            "contamination",
            "blank",
            "negative control",
            "extraction blank",
            "PCR blank",
            "decontam",
            "prevalence",
            "frequency",
            "removed if present in controls",
        ),
    ),
    LLMJudgedSearchField(
        term_name="informationWithheld",
        section="Data management",
        description=(
            "Information that exists but was intentionally not provided in this record, usually "
            "for privacy, sensitivity, legal restrictions, agreements, endangered species, "
            "Indigenous/cultural considerations, or controlled access."
        ),
        output_instructions=(
            "Return the free-text reason information was withheld, preserving the paper's own "
            "wording (e.g. 'Exact coordinates were withheld to protect an endangered species.'). "
            "This also includes generic data-availability boilerplate that limits access to only "
            "SOME of the paper's data, such as 'All other data are available from the corresponding "
            "authors on reasonable request.' -- return that sentence verbatim even if the paper never "
            "states exactly what the withheld 'other data' is. If multiple distinct withheld items are "
            "explicitly supported, return one object per value; final merged output is pipe-delimited. "
            "Do not report a field as withheld just because this paper's own text simply never mentions "
            "it -- only report an EXPLICIT statement that something was intentionally not shared or that "
            "access to some data is conditional/restricted."
        ),
        search_terms=(
            "information withheld",
            "data withheld",
            "location withheld",
            "coordinates withheld",
            "exact location withheld",
            "not publicly available",
            "sensitive location",
            "sensitive data",
            "restricted access",
            "confidential",
            "confidentiality",
            "data sharing agreement",
            "available upon request",
            # A real paper (10.1038/s42003-024-06136-2) used this common
            # data-availability-statement boilerplate ("All other data are
            # available from the corresponding authors on reasonable
            # request.") -- confirmed missed entirely because none of the
            # cues above matched it.
            "reasonable request",
            "on request",
            "upon request",
            "cannot be shared",
            "not disclosed",
            "endangered species",
            "protected species",
            "culturally sensitive",
            "Indigenous knowledge",
            "human-identifiable",
        ),
    ),
    LLMJudgedSearchField(
        term_name="neg_cont_0_1",
        section="Controls",
        description=(
            "Whether a negative control (e.g. field blank, extraction blank, reagent blank, PCR blank, "
            "no-template control) was used to check for contamination or false positives."
        ),
        allowed_values=("1", "0"),
        output_instructions=(
            "Return \"1\" only if the quote explicitly describes a NEGATIVE control, blank, or "
            "no-template control genuinely used to check for contamination -- never a POSITIVE control, "
            "and never an unrelated use of the word \"control\" (e.g. \"quality control\", \"controlled "
            "for depth\", a \"control treatment\" in an experimental-design sense with no contamination-"
            "checking purpose). Return \"0\" only if the quote explicitly states NO negative control was "
            "used. If the quote is ambiguous or doesn't clearly support either, omit this field rather "
            "than guessing."
        ),
        search_terms=("control", "controls", "blank", "blanks"),
    ),
    LLMJudgedSearchField(
        term_name="pos_cont_0_1",
        section="Controls",
        description=(
            "Whether a positive control (e.g. mock community, reference/known/synthetic DNA, gBlock, "
            "plasmid control, positive amplification control) was used to confirm the assay worked."
        ),
        allowed_values=("1", "0"),
        output_instructions=(
            "Return \"1\" only if the quote explicitly describes a POSITIVE control genuinely used to "
            "confirm the assay worked -- never a NEGATIVE control, and never an unrelated use of the "
            "word \"control\" (e.g. \"quality control\", \"controlled for depth\", a \"control "
            "treatment\" in an experimental-design sense unrelated to assay validation). Return \"0\" "
            "only if the quote explicitly states NO positive control was used. If the quote is ambiguous "
            "or doesn't clearly support either, omit this field rather than guessing."
        ),
        # "control"/"controls"/"blank"/"blanks" alone miss the most common
        # real phrasing of a positive control: a real, live sentence
        # ("genomic DNA from a microbial mock community ... was included in
        # the final sample array to account for amplification and
        # sequencing error") never says "control" or "blank" at all, so the
        # candidate-quote step never even offered it to the judging LLM.
        # This field's own description already promises "mock community,
        # reference/known/synthetic DNA, gBlock" as detectable examples --
        # none of those phrases contain "control"/"blank" either, so all are
        # added here explicitly (mirroring the same vetted term list already
        # used for SAMPLE-level positive-control classification in
        # mapping/rules.py's _POSITIVE_CONTROL_TERMS).
        search_terms=(
            "control", "controls", "blank", "blanks",
            "mock community", "mock communities", "reference dna", "known dna",
            "synthetic dna", "gblock", "gblocks",
        ),
    ),
    # A real audit (10.1093/ismejo/wrae013, STUDY-295abf4a8f43) found rich
    # in-situ temperature/salinity measurements in the
    # paper's own text -- "In situ bottom water temperature (6.5C),
    # dissolved O2 (11.9 mg/L), and salinity (6.4 PSU) were measured with
    # a ProODO probe..." -- that never made it into sampleMetadata at all,
    # since temp/salinity previously only ever came from a
    # structured BioSample attribute (mapping/rules.py's SAMPLE-level
    # "temp"/"salinity" rules), never from text. The SAME
    # real paper also separately reports post-collection INCUBATION-
    # chamber daily-average temperature/oxygen/salinity ("Throughout the
    # acclimation phase oxygen, temperature, and salinity were measured
    # daily inside the two incubation chambers...") -- a genuinely
    # different concept FAIRe's temp/salinity fields don't
    # want (per temp's own definition: "Temperature of the sample AT THE
    # TIME OF SAMPLING"), so _SAMPLING_TIME_CONTEXT_RE below requires an
    # explicit collection-time/in-situ signal, not just the bare
    # measurement words, to tell the two contexts apart.
    LLMJudgedSearchField(
        term_name="in_situ_temp",
        section="Sample collection",
        description=(
            "The water/environment temperature measured in situ, at the time and site of sample "
            "collection -- not a later experimental, incubation, or laboratory-condition temperature."
        ),
        output_instructions=(
            "Return the numeric temperature value with its unit, verbatim from the quote (e.g. "
            "'6.5C'). Only accept a value explicitly described as measured in situ, at the time of "
            "sampling/collection, or at the sampling site -- never a later incubation, acclimation, "
            "or experimental-condition temperature. If multiple distinct in-situ temperature values "
            "are explicitly supported, return one object per value; final merged output is "
            "pipe-delimited."
        ),
        search_terms=(
            "in situ temperature",
            "in situ bottom water temperature",
            "temperature at collection",
            "site temperature",
            "temperature",
        ),
    ),
    LLMJudgedSearchField(
        term_name="in_situ_salinity",
        section="Sample collection",
        description=(
            "The salinity measured in situ, at the time and site of sample collection -- not a later "
            "experimental, incubation, or laboratory-condition salinity."
        ),
        output_instructions=(
            "Return the numeric salinity value with its unit, verbatim from the quote (e.g. '6.4 "
            "PSU'). Only accept a value explicitly described as measured in situ, at the time of "
            "sampling/collection, or at the sampling site -- never a later incubation, acclimation, "
            "or experimental-condition salinity. If multiple distinct in-situ values are explicitly "
            "supported, return one object per value; final merged output is pipe-delimited."
        ),
        search_terms=("salinity",),
    ),
)

SINGLE_BEST_LLM_JUDGED_FIELDS = frozenset(
    {
        "inhibition_check_0_1",
        "inhibition_check",
        "otu_clust_tool",
        "neg_cont_0_1",
        "pos_cont_0_1",
    }
)

CONTROLLED_SEARCH_FIELDS: tuple[ControlledSearchField, ...] = (
    # biological_rep/biological_rep_presence's own text-search entries were
    # removed entirely per an explicit user request: biological_rep is now
    # derived purely from structured-API/supplement biological_rep_relation
    # data (mapping/faire.py::_apply_biological_rep_from_relations) -- the
    # paper's own text is deliberately never queried for this field anymore.
    ControlledSearchField(
        term_name="assay_type",
        section="PCR",
        description="Fixed vocabulary classifier: targeted | metabarcoding | other | unknown.",
        value_strategy="assay_type_classifier",
        search_terms=(
            "quantitative PCR",
            "digital PCR",
            "species-specific",
            "taxon-specific",
            "targeted assay",
            "hydrolysis probe",
            "qPCR",
            "ddPCR",
            "TaqMan",
            "metabarcoding",
            "community profiling",
            "amplicon sequencing",
            "biodiversity assessment",
            "high-throughput sequencing",
            "multiple taxa",
            "universal primers",
            "HTS",
        ),
    ),
    ControlledSearchField(
        term_name="seq_kit",
        section="Library preparation sequencing",
        description="the name of sequencing kit, free text",
        value_strategy="sequencing_kit_phrase",
        search_terms=(
            "MiSeq Reagent Kit v3",
            "MiSeq Reagent Kit",
            "NEXTflex Rapid DNA-Seq kit",
            "NEXTflex DNA-Seq kit",
            "DNA-Seq kit",
            "Titanium chemistry",
            "NovaSeq kit",
            "ligation sequencing kit",
            "rapid sequencing kit",
            "sequencing kit",
            "reagent kit",
            "cycle kit",
            "v2 chemistry",
            "v3 chemistry",
            "SMRTbell",
            "flow cell",
            "cartridge",
        ),
    ),
    ControlledSearchField(
        term_name="checksum_method",
        section="Library preparation sequencing",
        description="Checksum algorithm used to verify sequencing file integrity.",
        value_strategy="checksum_method_algorithm",
        search_terms=(
            "MD5 checksum",
            "md5sum",
            "MD5",
            "md5",
            "SHA-256 checksum",
            "SHA-256",
            "SHA256",
            "CRC-32",
            "CRC32",
            "checksum method",
            "checksum algorithm",
            "SHA-1",
            "SHA1",
            "SHA-512",
            "xxHash",
        ),
    ),
    ControlledSearchField(
        term_name="target_gene",
        section="PCR",
        description="Field type: controlled vocabulary, list all that find in paper.",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="target_gene_phrase",
        search_terms=(
            "12S rRNA (SSU mitochondria)",
            "16S rRNA (LSU mitochondria)",
            "16S rRNA (SSU prokaryote)",
            "16S SSU rRNA",
            "16S rRNA SSU",
            "23S rRNA (LSU prokaryote)",
            "18S rRNA (SSU eukaryote)",
            "18S SSU rRNA",
            "18S rRNA SSU",
            "28S rRNA (LSU eukaryote)",
            "cytochrome c oxidase I",
            "cytochrome b",
            "12S rRNA",
            "16S rRNA",
            "18S rRNA",
            "12S",
            "16S",
            "18S",
            "23S",
            "28S",
            "rbcL",
            "CytB",
            "cytb",
            "COIII",
            "COII",
            "COI",
            "CO1",
            "cox1",
            "nifH",
            "ITS1",
            "ITS2",
            "ND1",
            "ND2",
            "ND3",
            "ND4",
            "ND5",
            "ND6",
            "amoA",
            "rpoB",
            "rpoC1",
            "rpoC2",
            "matK",
            "trnH",
            "trnL",
            "psbK",
            "D-loop",
        ),
    ),
    ControlledSearchField(
        term_name="probeReporter",
        section="PCR",
        description="Field type: free text, Type of fluorophore (reporter) used.",
        required_any_flags=frozenset({"pcr_0_1", "probe_based_qPCR_ddPCR_assay_0_1"}),
        search_terms=(
            "Texas Red",
            "fluorescent dye",
            "fluorophore",
            "reporter",
            "FAM",
            "HEX",
            "VIC",
            "Cy5",
            "TET",
            "TAMRA",
            "ROX",
            "JOE",
            "HRP",
            "HRP-labeled",
            "horseradish peroxidase",
        ),
    ),
    ControlledSearchField(
        term_name="probeQuencher",
        section="PCR",
        description="Field type: free text, Type of quencher used.",
        required_any_flags=frozenset({"pcr_0_1", "probe_based_qPCR_ddPCR_assay_0_1"}),
        search_terms=(
            "Zero-End Quencher (ZEN)",
            "Black Hole Quencher (BHQ)",
            "Minor Groove Binder (MGB)",
            "Iowa Black FQ",
            "Iowa Black",
            "lowa Black",
            "minor groove binder",
            "quencher",
            "BHQ-1",
            "BHQ-2",
            "BHQ1",
            "BHQ2",
            "BHQ",
            "ZEN",
            "TAMRA",
            "MGB",
        ),
    ),
    ControlledSearchField(
        term_name="commercial_mm",
        section="PCR",
        description="Name, brand, and manufacture of commercial, pre-made master mix",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="commercial_pcr_mixture_phrase",
        # No longer literal-substring matched -- see _match_pcr_mixture_phrase.
        # A bare brand mention (bare manufacturer names like "Thermo Fisher",
        # "Applied Biosystems", "Bio-Rad", "NEB", "KAPA" used to be literal
        # search_terms here) isn't reliable evidence on its own: confirmed
        # via a real gold paper (PeerJ 10.7717/peerj.333) that "Bio-Rad"
        # matched a mention of the thermocycler manufacturer ("DNA Engine
        # Tetrad2 Thermal Cycler (Bio-Rad, ...)"), not a master-mix product
        # -- that paper's actual PCR mixture was custom-assembled from
        # separate reagents (ExTaq buffer/polymerase, Pfu polymerase), which
        # `custom_mm` (below) now correctly captures instead. Detection is
        # now sentence-level: find a real PCR-mixture-description sentence
        # (via `_PCR_MIXTURE_MARKERS_RE`, e.g. "PCR mixture"/"polymerase
        # chain reaction (PCR) mixture"), then classify it as commercial
        # only if it also names a specific master-mix product/brand
        # (`_COMMERCIAL_MASTER_MIX_BRAND_RE`) -- otherwise it's `custom_mm`'s
        # job. The whole matched sentence becomes the value/evidence (a
        # bare brand/product name alone loses the actual reagent
        # composition a reviewer would want to see).
        search_terms=(
            "PCR Master Mix",
            "qPCR Master Mix",
            "master mix",
            "mastermix",
            "TaqMan",
            "SYBR",
            "Luna",
            "PowerUp",
            "QuantiTect",
            "SsoAdvanced",
        ),
    ),
    ControlledSearchField(
        term_name="custom_mm",
        section="PCR",
        description="Composition of a custom (non-commercial) PCR master mix, if a commercial one was not used",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="custom_pcr_mixture_phrase",
        search_terms=(
            "PCR mixture",
            "PCR mix",
            "reaction mixture",
            "polymerase chain reaction (PCR) mixture",
        ),
    ),
    ControlledSearchField(
        term_name="forward_primer_name",
        section="PCR",
        description="Forward primer name explicitly reported in the paper.",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="forward_primer_name_phrase",
        search_terms=("forward primer", "forward primers", "primer"),
    ),
    ControlledSearchField(
        term_name="reverse_primer_name",
        section="PCR",
        description="Reverse primer name explicitly reported in the paper.",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="reverse_primer_name_phrase",
        search_terms=("reverse primer", "reverse primers", "primer"),
    ),
    ControlledSearchField(
        term_name="forward_primer_sequence",
        section="PCR",
        description="Forward primer nucleotide sequence explicitly reported in the paper.",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="forward_primer_sequence_phrase",
        search_terms=("forward primer", "5'", "5′", "primer sequence"),
    ),
    ControlledSearchField(
        term_name="reverse_primer_sequence",
        section="PCR",
        description="Reverse primer nucleotide sequence explicitly reported in the paper.",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="reverse_primer_sequence_phrase",
        search_terms=("reverse primer", "5'", "5′", "primer sequence"),
    ),
    ControlledSearchField(
        term_name="thermocycler",
        section="PCR",
        description="The manufacturer and model of a thermocycler used.",
        required_any_flags=frozenset({"pcr_0_1"}),
        value_strategy="thermocycler_phrase",
        search_terms=(
            "DNA Engine Tetrad2 Thermal Cycler",
            "DNA Engine Tetrad 2 Thermal Cycler",
            "QuantStudio 12K Flex",
            "Thermal Cycler Dice Real Time System III",
            "Thermal Cycler Dice Real Time System II",
            "Thermal Cycler Dice Real Time System",
            "Eco Real-Time PCR System",
            "Naica Crystal Digital PCR",
            "GeneAmp PCR System 2400",
            "Automated Thermal Cycler",
            "PCR System 9700",
            "Applied Biosystems",
            "Magnetic Induction Cycler",
            "real-time PCR system",
            "LightCycler 480 II",
            "CFX96 Touch Deep Well",
            "Mastercycler nexus gradient",
            "Mastercycler realplex",
            "MIC qPCR Cycler",
            "Stratagene Mx3000P",
            "Stratagene Mx3005P",
            "qPCR system",
            "PCR machine",
            "thermal cycler",
            "thermocycler",
            "GeneAmp 2400",
            "GeneAmp 2700",
            "GeneAmp 2720",
            "GeneAmp 9600",
            "GeneAmp 9700",
            "VeritiPro",
            "Veriti",
            "SimpliAmp",
            "MiniAmp Plus",
            "MiniAmp",
            "ProFlex",
            "ABI 7900HT",
            "ABI 7500",
            "7500 Fast",
            "ABI 7300",
            "ABI 7000",
            "StepOnePlus",
            "StepOne",
            "ViiA 7",
            "QuantStudio Absolute Q",
            "QuantStudio Pro",
            "QuantStudio 1",
            "QuantStudio 3",
            "QuantStudio 5",
            "QuantStudio 6",
            "QuantStudio 7",
            "T100",
            "C1000 Touch",
            "C1000",
            "S1000",
            "MyCycler",
            "iCycler",
            "DNA Engine Tetrad",
            "DNA Engine",
            "PTC-100",
            "PTC-150",
            "PTC-200",
            "PTC-220",
            "PTC-225",
            "Tetrad 2",
            "Dyad",
            "CFX96 Touch",
            "CFX96",
            "CFX384 Touch",
            "CFX384",
            "CFX Opus 384",
            "CFX Opus 96",
            "CFX Opus",
            "CFX Duet",
            "iQ5",
            "MyiQ",
            "Chromo4",
            "Opticon 2",
            "Opticon",
            "MiniOpticon",
            "Mastercycler personal",
            "Mastercycler gradient",
            "Mastercycler pro 384",
            "Mastercycler pro S",
            "Mastercycler pro",
            "Mastercycler ep",
            "Mastercycler nexus X2",
            "Mastercycler nexus",
            "Mastercycler X2",
            "Mastercycler X40",
            "Mastercycler X50",
            "Mastercycler",
            "realplex",
            "LightCycler 1.0",
            "LightCycler 1.5",
            "LightCycler 2.0",
            "LightCycler Nano",
            "LightCycler 96",
            "LightCycler 480",
            "LightCycler PRO",
            "LightCycler",
            "Rotor-Gene 2000",
            "Rotor-Gene 3000",
            "Rotor-Gene 6000",
            "Rotor-Gene Q MDx",
            "Rotor-Gene Q",
            "Rotor-Gene",
            "RotorGene",
            "QIAquant 384",
            "QIAquant 96",
            "QIAquant",
            "QIAcuity Eight",
            "QIAcuity Four",
            "QIAcuity One",
            "QIAcuity",
            "SureCycler 8800",
            "SureCycler",
            "AriaMx",
            "AriaDx",
            "Mx3000P",
            "Mx3005P",
            "Mx4000",
            "Robocycler",
            "PicoMaxx",
            "Biometra",
            "TGradient",
            "TProfessional Basic",
            "TProfessional Standard",
            "TProfessional",
            "TAdvanced Twin",
            "TAdvanced 96 G",
            "TAdvanced 96",
            "TAdvanced",
            "TRobot II",
            "TRobot",
            "qTOWER 2.0",
            "qTOWER 2.2",
            "qTOWER3 G",
            "qTOWER3",
            "qTOWER iris",
            "qTOWER",
            "Thermal Cycler Dice",
            "Dice Touch",
            "CronoSTAR 96",
            "SmartChip",
            "Exicycler 96 Fast",
            "Exicycler 384",
            "Exicycler 96",
            "Exicycler V4",
            "Exicycler V5",
            "Exicycler",
            "ExiCycler",
            "Azure Cielo 3",
            "Azure Cielo 6",
            "Azure Cielo",
            "Cielo 3",
            "Cielo 6",
            "Cielo",
            "Mic qPCR",
            "Mic",
            "Smart Cycler",
            "SmartCycler",
            "Eco qPCR",
            "BioMark HD",
            "BioMark",
            "Juno",
            "48.48 Dynamic Array",
            "96.96 Dynamic Array",
            "Naica",
            "Nio",
            "Techne Prime",
            "PrimeG",
            "PrimeQ",
            "Prime",
            "TC-3000",
            "TC-4000",
            "TC-5000",
            "Touchgene",
            "Genius",
            "Labcycler Gradient",
            "Labcycler Basic",
            "Labcycler 48",
            "Labcycler 96",
            "Labcycler",
            "SuperCycler Trinity",
            "SuperCycler",
            "Swift MaxPro",
            "Swift",
            "Aerisbio Real-Time PCR",
            "GeneExplorer",
            "LifeExpress",
            "LifeTouch",
            "LineGene 9600 Plus",
            "LineGene 9600",
            "LineGene",
            "FQD-96A",
            "Primus 25",
            "Primus 96",
            "Primus",
            "peqSTAR",
            "PikoReal",
            "Piko",
            "Arktik",
        ),
    ),
    ControlledSearchField(
        term_name="sample_type",
        section="Project",
        description='Select one or more from "Water", "Soil", "Sediment", "Air", "HostAssociated", '
        '"MicrobialMatBiofilm", and "SymbiontAssociated", or "other".',
        search_terms=(
            "HostAssociated",
            "MicrobialMatBiofilm",
            "SymbiontAssociated",
            "Water",
            "Soil",
            "Sediment",
            "Air",
        ),
    ),
    ControlledSearchField(
        term_name="adapter_trimming_method",
        section="Bioinformatics",
        description="Primer/adapter trimming method, including software and version.",
        value_strategy="adapter_trim_tool",
        search_terms=("SeqPrep", "Trimmomatic"),
    ),
    ControlledSearchField(
        term_name="length_filtering_tool",
        section="Bioinformatics",
        description="Software used to filter reads by length.",
        value_strategy="length_quality_trim_tool",
        search_terms=("Trimmomatic", "USEARCH", "MINLEN"),
    ),
    ControlledSearchField(
        term_name="minimum_read_length",
        section="Bioinformatics",
        description="Minimum read length threshold used for filtering.",
        value_strategy="trimmomatic_minimum_read_length",
        search_terms=("MINLEN", "minimum read length", "reads below", "reads shorter than", "shorter than"),
    ),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_ASSAY_TYPE_CUES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "targeted",
        (
            re.compile(r"\bquantitative\s+PCR\b", re.IGNORECASE),
            re.compile(r"\bdigital\s+PCR\b", re.IGNORECASE),
            # "species-specific"/"taxon-specific" were removed as bare cues:
            # confirmed via a real gold paper (PeerJ 10.7717/peerj.333) that
            # both phrases appear routinely in non-assay contexts (e.g.
            # "coral species-specific cue preferences", an ecological
            # statement about coral behavior, not assay design) and falsely
            # triggered "targeted" with no PCR/probe/qPCR content anywhere
            # nearby. docs/architecture.md's Milestone 15 entry already
            # flagged this exact pair as unreliable for a different field
            # (pcr_0_1) in this same paper. "targeted\s+assay" below still
            # catches genuine "targeted assay" phrasing.
            re.compile(r"\btargeted\s+assay\b", re.IGNORECASE),
            re.compile(r"\bhydrolysis\s+probe(s)?\b", re.IGNORECASE),
            re.compile(r"\bqPCR\b", re.IGNORECASE),
            re.compile(r"\bddPCR\b", re.IGNORECASE),
            re.compile(r"\bTaqMan\b", re.IGNORECASE),
        ),
    ),
    (
        "metabarcoding",
        (
            re.compile(r"\bmetabarcoding\b", re.IGNORECASE),
            re.compile(r"\bcommunity\s+profiling\b", re.IGNORECASE),
            re.compile(r"\bamplicon\s+sequencing\b", re.IGNORECASE),
            re.compile(r"\bbiodiversity\s+assessment\b", re.IGNORECASE),
            re.compile(r"\bhigh[-\s]+throughput\s+sequencing\b", re.IGNORECASE),
            re.compile(r"\bmultiple\s+taxa\b", re.IGNORECASE),
            re.compile(r"\buniversal\s+primers\b", re.IGNORECASE),
            re.compile(r"\bHTS\b"),
            # Real gap found live (10.3390/microorganisms10030558): this
            # paper's own explicit "amplicon sequencing of 16S rRNA and
            # cbbL gene" framing sentence lives in the Introduction, which
            # select_relevant_sections excludes entirely (Methods-only
            # scoping) -- so the classifier never saw it in the real
            # pipeline, even though the *Methods* section alone still
            # describes unmistakably metabarcoding-shaped methodology for
            # BOTH its 16S marker AND its cbbL functional-gene marker:
            # OTU clustering and "gene amplicons" sequencing, neither
            # tied to a specific single-species/qPCR detection use. These
            # two cues are common Methods-section phrasing independent of
            # whether the paper ever uses the word "amplicon sequencing"
            # itself.
            re.compile(r"\boperational\s+taxonomic\s+units?\b", re.IGNORECASE),
            re.compile(r"\bgene\s+amplicons?\b", re.IGNORECASE),
        ),
    ),
    (
        # Real gap found live (PMC10988111, ISME Communications
        # 10.1093/ismeco/ycae036, "Metagenomic insights into
        # jellyfish-associated microbiome dynamics"): a shotgun
        # metagenomics paper's own assay_type was left blank because
        # neither existing bucket has anything for it, while its
        # Introduction/Discussion sections separately mention "16S rRNA
        # amplicon sequencing" and "16S rRNA gene sequencing" only to
        # contrast this paper's own method against *other* studies'
        # marker-gene approach -- those mentions still (correctly, per
        # this same bucket design) trip the "metabarcoding" bucket above,
        # since a bare-word/no-attribution regex classifier can't tell
        # "we did X" from "other studies did X" apart. Per explicit user
        # direction, that's an acceptable tradeoff here: list both rather
        # than have one crowd out the other (pipe-joined further down the
        # pipeline). assay_type_enum has no dedicated "metagenomic" member
        # (only targeted | metabarcoding | other:) -- "other:metagenomics"
        # follows the same "other:<free label>" convention as every other
        # FAIRe "other:" field, self-descriptive since there's no
        # companion assay_type_additional field to carry it separately.
        "other:metagenomics",
        (
            re.compile(r"\bshotgun\s+metagenomic(?:s)?\b", re.IGNORECASE),
            re.compile(r"\bmetagenomic\s+sequencing\b", re.IGNORECASE),
            re.compile(r"\bmetagenome\s+(?:assembly|assemblies|librar(?:y|ies))\b", re.IGNORECASE),
            re.compile(r"\bmetagenome-assembled\s+genomes?\b", re.IGNORECASE),
            re.compile(r"\bwhole[-\s]genome\s+shotgun\b", re.IGNORECASE),
            re.compile(r"\bshotgun\s+sequencing\b", re.IGNORECASE),
        ),
    ),
)
_SEQUENCING_KIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bMiSeq\s+Reagent\s+Kit(?:\s+v\d+)?\b", re.IGNORECASE),
    re.compile(r"\bNextera\s+XT\s+Index\s+Kit(?:\s*\([^)]+\))?", re.IGNORECASE),
    re.compile(
        r"\bNEXTflex(?:\W+|\s+)(?:Rapid\s+)?DNA[-\s]+Seq\s+[Kk]it(?:\s*\([^)]+\))?",
        re.IGNORECASE,
    ),
    re.compile(r"\bTitanium\s+chemistry\b", re.IGNORECASE),
    re.compile(r"\b(?:v2|v3)\s+chemistry\b", re.IGNORECASE),
    # TruSeq is a whole family of Illumina library prep kits (Stranded
    # mRNA, Nano, PCR-Free, DNA, ...) -- confirmed missing on a real paper
    # (ISME J 10.1093/ismejo/wrae013: "TruSeq Stranded mRNA kit (Illumina)").
    re.compile(r"\bTruSeq\s+\w+(?:\s+\w+){0,3}\s+[Kk]it\b"),
    re.compile(
        r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z0-9][A-Za-z0-9-]*){0,5}\s+"
        r"(?:Index|Library|Sequencing|Reagent|Amplicon|Barcode)\s+[Kk]it"
        r"(?:\s*\([^)]+\))?"
    ),
)
_THERMOCYCLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bDNA\s+Engine\s+Tetrad\s*2\s+Thermal\s+Cycler\b", re.IGNORECASE),
    re.compile(
        r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z0-9][A-Za-z0-9-]*){0,5}\s+"
        r"(?:Thermal\s+Cycler|thermocycler|Cycler|PCR\s+System)\b(?:\s*\([^)]+\))?",
        re.IGNORECASE,
    ),
)
_TRIMMOMATIC_RE = re.compile(
    r"\bTrimmomatic(?:\s+(?:v(?:ersion)?\.?\s*)?\d+(?:\.\d+)*)?\b",
    re.IGNORECASE,
)
# SeqPrep is a distinct, common adapter-trimming/read-merging tool --
# confirmed missing on a real paper (ISME J 10.1093/ismejo/wrae013) that
# uses SeqPrep specifically for adapter removal and Trimmomatic separately
# for quality/length trimming (LEADING/TRAILING/MINLEN) in the same
# pipeline. Before this fix, the (context-blind) Trimmomatic-only detector
# wrongly stamped "Trimmomatic" onto adapter_trimming_method too, even
# though that paper's own text explicitly attributes adapter removal to
# SeqPrep, not Trimmomatic.
_SEQPREP_RE = re.compile(r"\bSeqPrep(?:\s+(?:v(?:ersion)?\.?\s*)?\d+(?:\.\d+)*)?\b", re.IGNORECASE)
_USEARCH_RE = re.compile(r"\bUSEARCH(?:\s+(?:v(?:ersion)?\.?\s*)?\d+(?:\.\d+)*)?\b", re.IGNORECASE)
# A single trim-tool mention can serve either purpose (or both, if a tool
# genuinely does both in one invocation) -- disambiguated by what the same
# sentence says the tool was used FOR, not by which tool it happens to be.
_ADAPTER_TRIM_CONTEXT_RE = re.compile(
    r"\badapter(?:s)?\b|\bILLUMINACLIP\b|\bbarcodes?\b|\bprimers?\s+(?:removal|sequences)\b",
    re.IGNORECASE,
)
_LENGTH_QUALITY_TRIM_CONTEXT_RE = re.compile(
    r"\bquality\b|\bshort\s+reads?\b|\blength\b|\bMINLEN\b|\bLEADING\b|\bTRAILING\b|\bSLIDINGWINDOW\b",
    re.IGNORECASE,
)
_MINIMUM_READ_LENGTH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bMINLEN\s*:\s*(?P<value>\d+)\b", re.IGNORECASE),
    re.compile(
        r"\breads?\s+(?:that\s+)?(?:became|become|were|are)?\s*"
        r"shorter\s+than\s+(?P<value>\d+)\s*"
        r"(?P<unit>bp|base\s+pairs?|bases?|nt|nucleotides?)?"
        r"\b[^.]{0,120}\bdiscard(?:ed|ing)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:remov(?:e|ed|ing)|discard(?:ed|ing)?|filter(?:ed|ing)?)\s+"
        r"(?:all\s+)?reads?\s+(?:shorter\s+than|below|less\s+than|under)\s+"
        r"(?P<value>\d+)\s*(?P<unit>bp|base\s+pairs?|bases?|nt|nucleotides?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bminimum\s+read\s+length\s+(?:of|=|:)?\s*"
        r"(?P<value>\d+)\s*(?P<unit>bp|base\s+pairs?|bases?|nt|nucleotides?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btrim(?:med|ming)?\s+(?:reads?\s+)?to\s+"
        r"(?P<value>\d+)\s*(?P<unit>bp|base\s+pairs?|bases?|nt|nucleotides?)?\b",
        re.IGNORECASE,
    ),
)
_COORDINATED_SSU_RRNA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b16S\s+(?:and|/|,)\s+18S\s+SSU\s+rRNA\b", re.IGNORECASE),
    re.compile(r"\b16S\s+(?:and|/|,)\s+18S\s+rRNA\s+SSU\b", re.IGNORECASE),
)
_PRIMER_SEQUENCE_RE = re.compile(
    r"\b5\s*[′'’`]\s*(?P<sequence>[ACGTRYSWKMBDHVN\s*]+?)\s*3\s*[′'’`]",
    re.IGNORECASE,
)
_PRIMER_NAME_BEFORE_DIRECTION_RE = re.compile(
    r"\b(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]{1,50})\s+(?:forward|reverse)\s+primers?\b",
    re.IGNORECASE,
)
_PRIMER_NAME_MATCHES_RE = re.compile(
    r"\bmatches\s+(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]{1,50})\s+primers?\b",
    re.IGNORECASE,
)
_PRIMER_DIRECTIONAL_NAME_RE = re.compile(
    r"\b(?P<name>\d{1,2}S\s+rRNA\s+(?P<direction>[FR]))\b",
    re.IGNORECASE,
)
_PRIMER_NAME_EXCLUSIONS = frozenset({"unique", "universal", "indexed", "tailed"})
_PRIMER_PAIR_RE = re.compile(
    r"\b(?:primers?|primer\s+pairs?)\s+"
    r"(?P<forward>[A-Za-z0-9][A-Za-z0-9_.-]{1,50}F)\s*(?:/|[-–])\s*"
    r"(?P<reverse>[A-Za-z0-9][A-Za-z0-9_.-]{1,50}R)\b",
    re.IGNORECASE,
)
_PRIMER_PAIR_CONTINUATION_RE = re.compile(
    r"\b(?P<forward>[A-Za-z0-9][A-Za-z0-9_.-]{1,50}F)\s*(?:/|[-–])\s*"
    r"(?P<reverse>[A-Za-z0-9][A-Za-z0-9_.-]{1,50}R)\b",
    re.IGNORECASE,
)
_PRIMER_TABLE_ROW_RE = re.compile(
    r"\b(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]{1,50})\s*\|\s*"
    r"(?P<sequence>[ACGTRYSWKMBDHVN]{10,})\s*\|",
    re.IGNORECASE,
)
_PRIMER_TABLE_FLAT_ROW_RE = re.compile(
    r"\b(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]{1,50})\s+"
    r"(?P<sequence>[ACGTRYSWKMBDHVN]{10,})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _PrimerTableRow:
    name: str
    sequence: str
    evidence: str


@dataclass(frozen=True)
class _PrimerPairUse:
    forward_name: str
    reverse_name: str
    evidence: str
    source_locator: str

# A real bug found live (10.1093/ismejo/wrae013): a methods sentence
# describing three DIFFERENT tools for three DIFFERENT purposes as one
# semicolon-joined enumerated list -- "Quality trimming was conducted by:
# (i) removing Illumina adapters using SeqPrep 1.2...[64]; (ii) remove any
# leftover PhiX...using bowtie2 2.3.5.1 [65], and (iii) remove low quality
# and short reads using Trimmomatic 0.39..." -- has only ONE terminal
# period (at the very end), so the period-only split above offered this
# whole blob as a SINGLE candidate quote tagged for seven different
# fields at once (adapter_forward/reverse, error_rate_tool/type/cutoff,
# neg_cont_0_1/pos_cont_0_1) -- confirmed live, the model then attached
# "SeqPrep 1.2" (really the adapter-trimming tool) to error_rate_tool
# instead of "Trimmomatic 0.39" (the real quality-trimming tool), simply
# because both names were sitting in the same oversized quote. Mirrors
# section_category_extraction.py's own identical fix (independent
# duplicate, not a shared import -- these two modules have never
# cross-imported, and this is a small, self-contained helper either way).
_LIST_MARKER_COMMA_BOUNDARY_RE = re.compile(
    r",\s+and\s+(?=\((?:i{1,3}|iv|vi{0,3}|ix|x|\d{1,2})\)\s)"
)


def _split_top_level_semicolons(piece: str) -> list[str]:
    normalized = _LIST_MARKER_COMMA_BOUNDARY_RE.sub("; ", piece)
    parts: list[str] = []
    depth = 0
    start = 0
    for index, ch in enumerate(normalized):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            parts.append(normalized[start : index + 1])
            start = index + 1
    parts.append(normalized[start:])
    return [p.strip() for p in parts if p.strip()]


# "vol." (volume/volumes) is a real, common wet-lab-protocol abbreviation
# the naive [.!?]-then-whitespace split above mistakes for a sentence end
# -- confirmed live (10.1371/journal.pone.0303937): "1 vol. of phenol/
# chloroform/isoamyalcohol..." got shredded into "1 vol." and "of phenol/
# chloroform/isoamyalcohol...". "et al." is the other ubiquitous academic
# abbreviation with the same problem -- confirmed live while building
# primer-reference extraction, a citation like "Caporaso et al. (2011)"
# split right between the author name and its own year. Mirrors
# section_category_extraction.py's own identical fix.
_COMMON_ABBREVIATION_TAIL_RE = re.compile(r"\b(?:vol|et\s+al)\.$", re.IGNORECASE)


def _snippets(text: str) -> Iterable[tuple[int, str]]:
    """Yield short candidate evidence snippets with stable local positions."""
    normalized = " ".join(text.split())
    if not normalized:
        return
    index = 0
    pending: str | None = None
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        for sub_sentence in _split_top_level_semicolons(sentence):
            cleaned = sub_sentence.strip()
            if not cleaned:
                continue
            if pending is not None:
                cleaned = f"{pending} {cleaned}"
            if _COMMON_ABBREVIATION_TAIL_RE.search(cleaned):
                pending = cleaned
                continue
            pending = None
            yield index, cleaned
            index += 1
    if pending is not None:
        yield index, pending


def _snippet_matches(patterns: tuple[re.Pattern[str], ...], snippet: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                match.group(0)
                for pattern in patterns
                for match in pattern.finditer(snippet)
            },
            key=str.lower,
        )
    )


def _match_flag(flag: TextSearchFlag, text: str) -> tuple[int, str, str, tuple[str, ...]] | None:
    explicit_none_match: tuple[int, str, tuple[str, ...]] | None = None
    for index, snippet in _snippets(text):
        none_matches = _snippet_matches(flag.explicit_none_patterns, snippet)
        if none_matches and explicit_none_match is None:
            explicit_none_match = (index, snippet, none_matches)
        if none_matches:
            continue
        matches = _snippet_matches(flag.positive_patterns, snippet)
        if matches:
            return index, snippet, flag.positive_value, matches
    if explicit_none_match is None:
        return None
    index, snippet, matches = explicit_none_match
    return index, snippet, flag.explicit_none_value, matches


def _term_pattern(term: str) -> re.Pattern[str]:
    """Real gap found live (STUDY-012e2a73836d): "...potential
    contaminants..." and "...in the negative controls..." never matched
    the cues "contaminant"/"negative control" at all -- the strict
    right-boundary check (?![A-Za-z0-9]) correctly rejects a DIFFERENT
    word sharing a prefix (so "contaminant" doesn't wrongly match
    "contaminantly" or similar), but it was equally strict about a
    simple plural of the exact same word, which real prose uses
    constantly. `e?s?` optionally matches a trailing "s" or "es"
    (covering both "contaminant"->"contaminants" and "process"->
    "processes"/"analysis" staying as-is since irregular plurals aren't
    a simple suffix) without weakening the boundary check itself."""
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}e?s?(?![A-Za-z0-9])", re.IGNORECASE)


def _match_controlled_terms(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    patterns = [
        (term, _term_pattern(term))
        for term in sorted(field.search_terms, key=len, reverse=True)
        if term.strip()
    ]
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            spans: list[tuple[int, int, str, str]] = []
            for term, pattern in patterns:
                for match in pattern.finditer(snippet):
                    span = match.span()
                    if any(span[0] < existing[1] and existing[0] < span[1] for existing in spans):
                        continue
                    spans.append((span[0], span[1], match.group(0), term))
            for _start, _end, matched_value, configured_term in sorted(spans, key=lambda item: item[0]):
                key = matched_value.casefold()
                if key in seen_values:
                    continue
                seen_values.add(key)
                values.append(matched_value)
                if snippet not in evidence_quotes:
                    evidence_quotes.append(snippet)
                match_metadata.append(
                    {
                        "matched_value": matched_value,
                        "configured_search_term": configured_term,
                        "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    }
                )
    return values, evidence_quotes, match_metadata


def _canonical_target_gene(value: str) -> tuple[str, int, str] | None:
    normalized = value.casefold()
    for marker in ("16s", "18s"):
        if marker not in normalized:
            continue
        display_marker = marker.upper()
        if "ssu" in normalized:
            return display_marker, 3, f"{display_marker} rRNA SSU"
        if "rrna" in normalized:
            return display_marker, 2, f"{display_marker} rRNA"
        return display_marker, 1, display_marker
    return None


def _match_target_gene_terms(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    values, evidence_quotes, match_metadata = _match_controlled_terms(field, texts, locator_prefix)
    coordinated_values: list[str] = []
    coordinated_quotes: list[str] = []
    coordinated_metadata: list[dict] = []
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            for pattern in _COORDINATED_SSU_RRNA_PATTERNS:
                if pattern.search(snippet) is None:
                    continue
                for value in ("16S rRNA SSU", "18S rRNA SSU"):
                    coordinated_values.append(value)
                    coordinated_metadata.append(
                        {
                            "matched_value": value,
                            "matched_pattern": pattern.pattern,
                            "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                        }
                    )
                if snippet not in coordinated_quotes:
                    coordinated_quotes.append(snippet)
                break
    values = [*coordinated_values, *values]
    evidence_quotes = [*coordinated_quotes, *evidence_quotes]
    match_metadata = [*coordinated_metadata, *match_metadata]
    best_by_marker: dict[str, tuple[int, str, int]] = {}
    output_values: list[str | None] = []

    for index, value in enumerate(values):
        canonical = _canonical_target_gene(value)
        if canonical is None:
            output_values.append(value)
            continue
        marker, rank, canonical_value = canonical
        previous = best_by_marker.get(marker)
        if previous is None:
            best_by_marker[marker] = (rank, canonical_value, index)
            output_values.append(canonical_value)
            continue
        previous_rank, _previous_value, previous_index = previous
        if rank > previous_rank:
            output_values[previous_index] = None
            best_by_marker[marker] = (rank, canonical_value, index)
            output_values.append(canonical_value)
        else:
            output_values.append(None)

    collapsed_values: list[str] = []
    collapsed_metadata: list[dict] = []
    seen_values: set[str] = set()
    for value, metadata in zip(output_values, match_metadata):
        if value is None:
            continue
        key = value.casefold()
        if key in seen_values:
            continue
        seen_values.add(key)
        collapsed_values.append(value)
        collapsed_metadata.append({**metadata, "normalized_value": value})
    return collapsed_values, evidence_quotes, collapsed_metadata


def _match_controlled_sentences(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    patterns = [
        (term, _term_pattern(term))
        for term in sorted(field.search_terms, key=len, reverse=True)
        if term.strip()
    ]
    values: list[str] = []
    match_metadata: list[dict] = []
    seen_sentences: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            matched_terms = tuple(
                term
                for term, pattern in patterns
                if pattern.search(snippet)
            )
            if not matched_terms or snippet in seen_sentences:
                continue
            seen_sentences.add(snippet)
            values.append(snippet)
            match_metadata.append(
                {
                    "matched_terms": list(matched_terms),
                    "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                }
            )
    return values, values, match_metadata


def _match_regex_phrases(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
    patterns: tuple[re.Pattern[str], ...],
) -> tuple[list[str], list[str], list[dict]]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            accepted_spans: list[tuple[int, int]] = []
            for pattern in patterns:
                for match in pattern.finditer(snippet):
                    span = match.span()
                    if any(span[0] < end and start < span[1] for start, end in accepted_spans):
                        continue
                    value = (match.groupdict().get("value") or match.group(0)).strip(" .;,")
                    key = value.casefold()
                    if not value or key in seen_values:
                        continue
                    seen_values.add(key)
                    accepted_spans.append(span)
                    values.append(value)
                    if snippet not in evidence_quotes:
                        evidence_quotes.append(snippet)
                    match_metadata.append(
                        {
                            "matched_value": value,
                            "matched_pattern": pattern.pattern,
                            "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                        }
                    )
    return values, evidence_quotes, match_metadata


def _primer_window(snippet: str, direction: str) -> str | None:
    reverse_match = re.search(r"\breverse\s+primers?\b", snippet, re.IGNORECASE)
    if direction == "forward":
        window = snippet[: reverse_match.start()] if reverse_match else snippet
        return window if re.search(r"\bforward\s+primers?\b|\bSP[-_]?F\b|\b\d{1,2}S\s+rRNA\s+F\b", window, re.IGNORECASE) else None
    if reverse_match:
        window = snippet[reverse_match.start() :]
        return window if re.search(r"\breverse\s+primers?\b|\bSP[-_]?R\b|\b\d{1,2}S\s+rRNA\s+R\b", window, re.IGNORECASE) else None
    return snippet if re.search(r"\bSP[-_]?R\b|\b\d{1,2}S\s+rRNA\s+R\b", snippet, re.IGNORECASE) else None


def _clean_primer_sequence(value: str) -> str:
    return re.sub(r"[^ACGTRYSWKMBDHVN]", "", value.upper())


def _clean_primer_name(value: str) -> str:
    return value.strip(" .;,()[]")


def _parse_primer_table_rows(text: str) -> dict[str, _PrimerTableRow]:
    rows: dict[str, _PrimerTableRow] = {}
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        matches = list(_PRIMER_TABLE_ROW_RE.finditer(line))
        if not matches and "primer" in text.casefold():
            matches = list(_PRIMER_TABLE_FLAT_ROW_RE.finditer(line))
        for match in matches:
            name = _clean_primer_name(match.group("name"))
            sequence = _clean_primer_sequence(match.group("sequence"))
            if not name or len(sequence) < 10 or name.casefold() in _PRIMER_NAME_EXCLUSIONS:
                continue
            rows.setdefault(name.casefold(), _PrimerTableRow(name=name, sequence=sequence, evidence=line))
    return rows


def _primer_pair_uses(
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[_PrimerPairUse], dict[str, _PrimerTableRow]]:
    pairs: list[_PrimerPairUse] = []
    table_rows: dict[str, _PrimerTableRow] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for title, text in texts:
        table_rows.update(_parse_primer_table_rows(text))
        for snippet_index, snippet in _snippets(text):
            folded = snippet.casefold()
            if "primer" not in folded and "amplified" not in folded and "pcr" not in folded:
                continue
            pair_matches = [*_PRIMER_PAIR_RE.finditer(snippet), *_PRIMER_PAIR_CONTINUATION_RE.finditer(snippet)]
            for match in pair_matches:
                forward_name = _clean_primer_name(match.group("forward"))
                reverse_name = _clean_primer_name(match.group("reverse"))
                if not forward_name or not reverse_name:
                    continue
                key = (forward_name.casefold(), reverse_name.casefold())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(
                    _PrimerPairUse(
                        forward_name=forward_name,
                        reverse_name=reverse_name,
                        evidence=snippet,
                        source_locator=f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    )
                )
    return pairs, table_rows


def _match_primer_pairs_and_tables(
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
    *,
    direction: str,
    value_kind: str,
) -> tuple[list[str], list[str], list[dict]]:
    pairs, table_rows = _primer_pair_uses(texts, locator_prefix)
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for pair in pairs:
        primer_name = pair.forward_name if direction == "forward" else pair.reverse_name
        table_row = table_rows.get(primer_name.casefold())
        if value_kind == "sequence":
            if table_row is None:
                continue
            value = table_row.sequence
            evidence = f"{pair.evidence} | {table_row.evidence}"
            matched_pattern = _PRIMER_TABLE_ROW_RE.pattern
        else:
            value = primer_name
            evidence = pair.evidence
            matched_pattern = _PRIMER_PAIR_RE.pattern
        key = value.casefold()
        if not value or key in seen_values:
            continue
        seen_values.add(key)
        values.append(value)
        if evidence not in evidence_quotes:
            evidence_quotes.append(evidence)
        match_metadata.append(
            {
                "matched_value": value,
                "matched_pattern": matched_pattern,
                "primer_pair": f"{pair.forward_name}/{pair.reverse_name}",
                "source_locator": pair.source_locator,
            }
        )
    return values, evidence_quotes, match_metadata


def _match_primer_phrase(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
    *,
    direction: str,
    value_kind: str,
) -> tuple[list[str], list[str], list[dict]]:
    text_items = tuple(texts)
    pair_values, pair_quotes, pair_metadata = _match_primer_pairs_and_tables(
        text_items,
        locator_prefix,
        direction=direction,
        value_kind=value_kind,
    )
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = {value.casefold() for value in pair_values}

    for title, text in text_items:
        for snippet_index, snippet in _snippets(text):
            if "primer" not in snippet.casefold():
                continue
            window = _primer_window(snippet, direction)
            if not window:
                continue
            if value_kind == "sequence":
                matches = [
                    (_clean_primer_sequence(match.group("sequence")), _PRIMER_SEQUENCE_RE.pattern)
                    for match in _PRIMER_SEQUENCE_RE.finditer(window)
                ]
            else:
                name_matches: list[tuple[str, str]] = []
                for pattern in (_PRIMER_NAME_BEFORE_DIRECTION_RE, _PRIMER_NAME_MATCHES_RE):
                    for match in pattern.finditer(window):
                        name = match.group("name").strip(" .;,")
                        if name.casefold() not in _PRIMER_NAME_EXCLUSIONS:
                            name_matches.append((name, pattern.pattern))
                direction_letter = "F" if direction == "forward" else "R"
                for match in _PRIMER_DIRECTIONAL_NAME_RE.finditer(window):
                    if match.group("direction").casefold() == direction_letter.casefold():
                        name_matches.append((match.group("name").strip(" .;,"), _PRIMER_DIRECTIONAL_NAME_RE.pattern))
                matches = name_matches
            for value, pattern in matches:
                key = value.casefold()
                if not value or key in seen_values:
                    continue
                seen_values.add(key)
                values.append(value)
                if snippet not in evidence_quotes:
                    evidence_quotes.append(snippet)
                match_metadata.append(
                    {
                        "matched_value": value,
                        "matched_pattern": pattern,
                        "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    }
                )

    return [*pair_values, *values], [*pair_quotes, *evidence_quotes], [*pair_metadata, *match_metadata]


# Broad "there's a PCR-mixture-composition sentence here" trigger, shared
# by commercial_mm and custom_mm -- which field it belongs to is decided
# afterward by _COMMERCIAL_MASTER_MIX_BRAND_RE, not by this regex. Matches
# "PCR mixture"/"PCR mix"/"reaction mixture"/"master mix"/"mastermix" and
# the parenthetical "polymerase chain reaction (PCR) mixture" phrasing (a
# real gold paper, PeerJ 10.7717/peerj.333, uses exactly that shape, with
# "(PCR)" breaking a simpler "reaction\s+mixture" match).
_PCR_MIXTURE_MARKERS_RE = re.compile(
    r"\b(?:PCR|qPCR|reaction)\s+mix(?:ture)?\b"
    r"|\bmaster\s*mix\b"
    r"|\bpolymerase\s+chain\s+reaction\s*\(?PCR\)?\s+mixture\b",
    re.IGNORECASE,
)
# Some real papers list PCR reagents (polymerase, buffer, dNTPs) without
# ever using the word "mixture"/"mix" at all -- confirmed missing on a
# real paper (PLOS ONE 10.1371/journal.pone.0303937: "...performed with
# 0.02 U/ul of Phusion High Fidelity DNA polymerase, 1X Phusion HF Buffer
# and 200 uM of dNTPs (New England Biolabs, USA)."). A sentence naming a
# polymerase AND a buffer/dNTP is just as much a PCR-mixture-composition
# description as one using the word "mixture" -- both trigger the same
# commercial-vs-custom classification below.
_PCR_POLYMERASE_MENTION_RE = re.compile(
    r"\b(?:Taq|Phusion|Pfu|ExTaq|KAPA|Q5|GoTaq|Platinum|AmpliTaq|HotStar(?:Taq)?|iProof)\b"
    r"|\b\w+\s+(?:DNA\s+)?[Pp]olymerase\b",
    re.IGNORECASE,
)
_PCR_BUFFER_OR_DNTP_RE = re.compile(r"\bbuffer\b|\bdNTPs?\b", re.IGNORECASE)
# A PCR-mixture sentence is "commercial" only if it also names a specific
# master-mix product/brand or says "master mix"/"mastermix" outright --
# otherwise (buffer/dNTP/enzyme-by-enzyme composition, e.g. "3 ul 10X ExTaq
# buffer, 0.025 U ExTaq Polymerase...") it's custom_mm's job.
_COMMERCIAL_MASTER_MIX_BRAND_RE = re.compile(
    r"\bmaster\s*mix\b|\bTaqMan\b|\bSYBR\b|\bLuna\b|\bPowerUp\b|\bQuantiTect\b|\bSsoAdvanced\b",
    re.IGNORECASE,
)
_PCR_MIXTURE_VALUE_STOP_RE = re.compile(
    r"\s*,?\s+(?:and\s+)?(?:was|were)\s+amplified\s+using\b|"
    r"\s*,?\s+with\s+a\s+cycling\s+profile\b|"
    r"\s*,?\s+with\s+cycling\s+conditions\b|"
    r"\s*,?\s+under\s+(?:the\s+)?(?:following\s+)?cycling\s+conditions\b|"
    r"\s+PCR\s+conditions\s+were\b",
    re.IGNORECASE,
)


def _clean_pcr_mixture_value(snippet: str) -> str:
    match = _PCR_MIXTURE_VALUE_STOP_RE.search(snippet)
    if match is None:
        return snippet.strip()
    return snippet[: match.start()].strip(" ,;")


def _match_pcr_mixture_phrase(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
    *,
    classification: str,
) -> tuple[list[str], list[str], list[dict]]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            is_mixture_sentence = _PCR_MIXTURE_MARKERS_RE.search(snippet) or (
                _PCR_POLYMERASE_MENTION_RE.search(snippet) and _PCR_BUFFER_OR_DNTP_RE.search(snippet)
            )
            if not is_mixture_sentence:
                continue
            is_commercial = bool(_COMMERCIAL_MASTER_MIX_BRAND_RE.search(snippet))
            if ("commercial" if is_commercial else "custom") != classification:
                continue
            if snippet in evidence_quotes:
                continue
            value = _clean_pcr_mixture_value(snippet)
            values.append(value)
            evidence_quotes.append(snippet)
            match_metadata.append(
                {
                    "matched_value": value,
                    "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                }
            )
    return values, evidence_quotes, match_metadata


def _format_minimum_read_length(match: re.Match[str]) -> str:
    value = match.group("value")
    unit = (match.groupdict().get("unit") or "bp").strip().lower()
    if unit in {"base pair", "base pairs", "base", "bases", "nt", "nucleotide", "nucleotides"}:
        unit = "bp"
    return f"{value} {unit}"


_TRIM_TOOL_CONTEXT_WINDOW = 90


def _match_trim_tool(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
    *,
    purpose: str,
) -> tuple[list[str], list[str], list[dict]]:
    """`purpose` is "adapter" (adapter_trimming_method) or "length_quality"
    (length_filtering_tool) -- the same tool name (Trimmomatic especially,
    which can do both) is only attributed to a given field when context
    NEAR that specific mention (within _TRIM_TOOL_CONTEXT_WINDOW chars, not
    just anywhere in the same snippet) says what it was used for. A
    snippet-wide check is not tight enough: a real paper (ISME J 10.1093/
    ismejo/wrae013) describes adapter removal (SeqPrep) and quality/length
    trimming (Trimmomatic) as two numbered clauses "(i) ...; (ii) ...; and
    (iii) ..." inside ONE long compound sentence -- a snippet-wide context
    check would see both "adapter" and "Trimmomatic" in that one sentence
    and wrongly attribute Trimmomatic to adapter_trimming_method too, even
    though the two mentions are ~250 characters apart and describe
    different clauses. See _SEQPREP_RE's comment for the same real paper."""
    context_re = _ADAPTER_TRIM_CONTEXT_RE if purpose == "adapter" else _LENGTH_QUALITY_TRIM_CONTEXT_RE
    tool_patterns = (_SEQPREP_RE, _TRIMMOMATIC_RE) if purpose == "adapter" else (_TRIMMOMATIC_RE, _USEARCH_RE)
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            for pattern in tool_patterns:
                match = pattern.search(snippet)
                if match is None:
                    continue
                window_start = max(0, match.start() - _TRIM_TOOL_CONTEXT_WINDOW)
                window_end = match.end() + _TRIM_TOOL_CONTEXT_WINDOW
                if not context_re.search(snippet[window_start:window_end]):
                    continue
                value = match.group(0)
                key = value.casefold()
                if key in seen_values:
                    continue
                seen_values.add(key)
                values.append(value)
                if snippet not in evidence_quotes:
                    evidence_quotes.append(snippet)
                match_metadata.append(
                    {
                        "matched_value": value,
                        "matched_pattern": pattern.pattern,
                        "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    }
                )
    return values, evidence_quotes, match_metadata


def _match_trimmomatic_minimum_read_length(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            folded = snippet.casefold()
            if (
                "minlen" not in folded
                and not _TRIMMOMATIC_RE.search(snippet)
                and not re.search(r"\btrim(?:med|ming)?\s+(?:reads?\s+)?to\s+\d+", snippet, re.IGNORECASE)
                and not re.search(
                    r"\breads?\s+(?:that\s+)?(?:became|become|were|are)?\s*shorter\s+than\s+\d+",
                    snippet,
                    re.IGNORECASE,
                )
            ):
                continue
            for pattern in _MINIMUM_READ_LENGTH_PATTERNS:
                match = pattern.search(snippet)
                if match is None:
                    continue
                value = _format_minimum_read_length(match)
                key = value.casefold()
                if key in seen_values:
                    break
                seen_values.add(key)
                values.append(value)
                if snippet not in evidence_quotes:
                    evidence_quotes.append(snippet)
                match_metadata.append(
                    {
                        "matched_value": value,
                        "matched_pattern": pattern.pattern,
                        "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    }
                )
                break
    return values, evidence_quotes, match_metadata


_CHECKSUM_METHOD_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    ("MD5", (re.compile(r"\bMD5(?:\s+checksums?)?\b", re.IGNORECASE), re.compile(r"\bmd5sum\b", re.IGNORECASE))),
    ("SHA-256", (re.compile(r"\bSHA[-\s]?256(?:\s+checksums?)?\b", re.IGNORECASE),)),
    ("CRC-32", (re.compile(r"\bCRC[-\s]?32\b", re.IGNORECASE),)),
    (
        "other:",
        (
            re.compile(r"\bSHA[-\s]?1\b", re.IGNORECASE),
            re.compile(r"\bSHA[-\s]?512\b", re.IGNORECASE),
            re.compile(r"\bxxHash\b", re.IGNORECASE),
        ),
    ),
)


def _match_checksum_method(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            for value, patterns in _CHECKSUM_METHOD_PATTERNS:
                matches = _snippet_matches(patterns, snippet)
                if not matches or value in seen_values:
                    continue
                seen_values.add(value)
                values.append(value)
                if snippet not in evidence_quotes:
                    evidence_quotes.append(snippet)
                match_metadata.append(
                    {
                        "matched_value": value,
                        "matched_terms": list(matches),
                        "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    }
                )
    return values, evidence_quotes, match_metadata


def _classify_assay_type(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            for assay_type, patterns in _ASSAY_TYPE_CUES:
                matches = _snippet_matches(patterns, snippet)
                if not matches or assay_type in seen_values:
                    continue
                seen_values.add(assay_type)
                values.append(assay_type)
                if snippet not in evidence_quotes:
                    evidence_quotes.append(snippet)
                match_metadata.append(
                    {
                        "matched_value": assay_type,
                        "matched_terms": list(matches),
                        "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    }
                )
    return values, evidence_quotes, match_metadata


def _match_controlled_field(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    if field.value_strategy == "evidence_sentences":
        return _match_controlled_sentences(field, texts, locator_prefix)
    if field.value_strategy == "assay_type_classifier":
        return _classify_assay_type(field, texts, locator_prefix)
    if field.value_strategy == "commercial_pcr_mixture_phrase":
        return _match_pcr_mixture_phrase(field, texts, locator_prefix, classification="commercial")
    if field.value_strategy == "custom_pcr_mixture_phrase":
        return _match_pcr_mixture_phrase(field, texts, locator_prefix, classification="custom")
    if field.value_strategy == "sequencing_kit_phrase":
        return _match_regex_phrases(field, texts, locator_prefix, _SEQUENCING_KIT_PATTERNS)
    if field.value_strategy == "checksum_method_algorithm":
        return _match_checksum_method(field, texts, locator_prefix)
    if field.value_strategy == "thermocycler_phrase":
        return _match_regex_phrases(field, texts, locator_prefix, _THERMOCYCLER_PATTERNS)
    if field.value_strategy == "adapter_trim_tool":
        return _match_trim_tool(field, texts, locator_prefix, purpose="adapter")
    if field.value_strategy == "length_quality_trim_tool":
        return _match_trim_tool(field, texts, locator_prefix, purpose="length_quality")
    if field.value_strategy == "trimmomatic_minimum_read_length":
        return _match_trimmomatic_minimum_read_length(field, texts, locator_prefix)
    if field.value_strategy == "target_gene_phrase":
        return _match_target_gene_terms(field, texts, locator_prefix)
    if field.value_strategy == "forward_primer_name_phrase":
        return _match_primer_phrase(field, texts, locator_prefix, direction="forward", value_kind="name")
    if field.value_strategy == "reverse_primer_name_phrase":
        return _match_primer_phrase(field, texts, locator_prefix, direction="reverse", value_kind="name")
    if field.value_strategy == "forward_primer_sequence_phrase":
        return _match_primer_phrase(field, texts, locator_prefix, direction="forward", value_kind="sequence")
    if field.value_strategy == "reverse_primer_sequence_phrase":
        return _match_primer_phrase(field, texts, locator_prefix, direction="reverse", value_kind="sequence")
    return _match_controlled_terms(field, texts, locator_prefix)


# Shared by in_situ_temp/in_situ_salinity: their own
# search terms ("temperature", "salinity", ...) are deliberately broad, so
# this is the real gate that keeps a bare mention out -- confirmed live
# against a real paper that reports the SAME three measurements twice,
# once in situ at collection and once as later incubation-chamber daily
# averages (see LLM_JUDGED_SEARCH_FIELDS's own comment above these three
# fields).
_SAMPLING_TIME_CONTEXT_RE = re.compile(
    r"\bin[\s-]situ\b|at\s+the\s+time\s+of\s+(?:collection|sampling)|"
    r"at\s+the\s+sampling\s+site|upon\s+collection|prior\s+to\s+sampling|"
    r"at\s+collection|on\s+the\s+day\s+of\s+(?:collection|sampling)|"
    # Real gap found live (10.3390/microorganisms10030558): "Seawater
    # temperature was 28.1 C. Seawater samples were collected at the
    # surface layer..." states a plain collection sentence right next to
    # the measurement, with none of the above explicit qualifiers --
    # common enough on its own (paired with the new +/-1-sentence window
    # in _SAMPLING_TIME_WINDOW_FIELDS) to treat as equivalent in-situ
    # evidence, without needing "in situ"/"at the time of" spelled out.
    r"\bsamples?\s+(?:were|was)\s+collected\b",
    re.IGNORECASE,
)
_OTU_CLUSTER_TOOL_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"OTUs?|ASVs?|amplicon\s+sequence\s+variants?|exact\s+sequence\s+variants?|"
    r"cluster(?:ed|ing)?|cluster[-_]?features|pick_otus|pick_open_reference_otus|"
    r"pick_closed_reference_otus|denois(?:e|ed|ing)|inferred|generated|"
    r"unoise|deblur|swarm|sumaclust|cd-hit|optiClust"
    r")\b|--cluster",
    re.IGNORECASE,
)
_OTU_DB_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"(?:reference|taxonomy|taxonomic|sequence|barcode)\s+(?:database|library)|"
    r"SILVA(?:[_\s.-]?\d+(?:\.\d+)?)?|FreshTrain|Parallel[-\s]?META[-\s]?3|"
    r"PR2|Protist\s+Ribosomal\s+Reference|GenBank|"
    r"NCBI\s+(?:nucleotide|nt|nr|database)|non[-\s]?redundant\s+(?:\(?nr\)?\s+)?NCBI\s+database|"
    r"nt\s+database|BOLD|Barcode\s+of\s+Life|UNITE|Ribosomal\s+Database\s+Project|"
    r"Greengenes2?|MIDORI2?|MitoFish|MetaZooGene|Diat\.?Barcode|PhytoREF|"
    r"MaarjAM|GTDB|Genome\s+Taxonomy\s+Database|MitoZoa|EMBL|ENA\s+reference|"
    r"(?:custom|curated|in-house|local)\s+(?:reference\s+)?database|"
    r"(?:reference|sequence|barcode)\s+library|classifier\s+trained\s+on|"
    r"(?:trained|pretrained)\s+classifier"
    r")\b",
    re.IGNORECASE,
)
_OTU_DB_VALUE_RE = re.compile(
    r"\b(?:"
    r"SILVA(?:[_\s.-]?\d+(?:\.\d+)?)?|FreshTrain|Parallel[-\s]?META[-\s]?3|"
    r"PR2|Protist\s+Ribosomal\s+Reference|GenBank|"
    r"NCBI\s+(?:nucleotide|nt|nr|database)|non[-\s]?redundant\s+(?:\(?nr\)?\s+)?NCBI\s+database|"
    r"nt\s+database|BOLD|Barcode\s+of\s+Life|UNITE|Ribosomal\s+Database\s+Project|"
    r"Greengenes2?|MIDORI2?|MitoFish|MetaZooGene|Diat\.?Barcode|PhytoREF|"
    r"MaarjAM|GTDB|Genome\s+Taxonomy\s+Database|MitoZoa|EMBL|ENA|"
    r"custom|curated|in-house|local"
    r")\b",
    re.IGNORECASE,
)
_ASSAY_NAME_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"assay(?:\s+name)?|named\s+assay|primer(?:\s+(?:set|pair|mix))?|"
    r"amplicon|amplif(?:y|ied|ication)|PCR|qPCR|ddPCR|TaqMan|hydrolysis\s+probe|"
    r"(?:16S|18S|28S|12S|COI|ITS)\s+(?:V[1-9](?:[-\u2010-\u2015]V[1-9])?|D[1-3](?:[-\u2010-\u2015]D[1-3])?)|"
    r"(?:MiFish|Teleo|MiBird|MiMammal|TAReuk|Leray|Folmer|Uni18S|"
    r"515F|806R|926R|341F|785R|805R|1389F|EukB|mlCOIintF|jgHCO2198|"
    r"LCO1490|HCO2198|ITS1F|ITS2|fITS7|ITS4)"
    r")\b",
    re.IGNORECASE,
)
_BARE_FUNCTIONAL_GENE_ASSAY_NAME_RE = re.compile(
    r"^(?:hzs[ABC]?|hzo(?:F1|R1)?|narG|nir[SK]|amoA|nxr[AB]|nosZ|nifH|dsr[AB]|mcrA)$",
    re.IGNORECASE,
)
# A real audit found the model answering assay_target_taxa with the
# amplified marker gene itself ("16S rRNA gene") when a quote named the
# gene without stating which organism(s)/taxonomic group it targets -- a
# gene/marker name is never a valid taxon on its own, so a bare marker
# name (optionally suffixed with "gene"/"marker"/"locus"/"region") is
# rejected outright rather than accepted as the assay's target taxon.
_BARE_MARKER_GENE_TARGET_TAXA_RE = re.compile(
    r"^(?:1[268]S(?:\s*rRNA)?|28S(?:\s*rRNA)?|COI|cytochrome\s*b|rbcL|ITS1?|ITS2)"
    r"(?:\s+(?:gene|marker|locus|region))?$",
    re.IGNORECASE,
)
_TARGET_TAXONOMIC_ASSAY_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"primers?|probes?|assays?|PCR\s+assay|qPCR\s+assay|ddPCR\s+assay|targeted\s+assay|"
    r"designed\s+to\s+(?:amplify|detect|target)|specific\s+for|target(?:ed|ing)?|"
    r"amplif(?:y|ied|ies|ication)|detect(?:ed|ing|ion)?|universal\s+primers?|primer\s+set|primer\s+pair"
    r")\b",
    re.IGNORECASE,
)
_TARGET_TAXONOMIC_SCOPE_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"target(?:ed)?|target\s+taxa|target\s+species|focus(?:ed)?\s+on|aim(?:ed)?(?:\s+to)?|objective|"
    r"survey(?:ed)?|monitor(?:ing)?|detect(?:ed|ion\s+of)?|investigat(?:e|ed)|assess(?:ed)?|"
    r"characteriz(?:e|ed)|biodiversity\s+of|diversity\s+of|community\s+composition|"
    r"community\s+structure|distribution\s+of|occurrence\s+of|presence\s+of|species\s+of\s+interest|"
    r"taxa\s+of\s+interest"
    r")\b",
    re.IGNORECASE,
)
# screen_contam_method's own search_terms deliberately include bare
# "contaminant"/"contamination" (real papers phrase this many ways), but
# that alone is far too broad a gate on its own -- confirmed live via two
# real misses on the same paper: (1) CheckM's own standard output
# description ("assess genome completeness and contamination... using
# default bacterial marker genes") is about METAGENOME-ASSEMBLED-GENOME
# bin QC, not sequence/OTU/ASV contaminant screening, yet got merged with
# an unrelated CANU assembler mention into one bogus screen_contam_method
# value; (2) "The water sampler was sanitised and rinsed... to avoid
# contamination from previous sites" describes FIELD EQUIPMENT
# sterilization (sterilise_method's own concept), not sequence-level
# decontamination, yet got captured here instead, leaving
# sterilise_method blank. Requires one of the field's own real
# distinguishing signals (comparison against blanks/negative controls, a
# decontam-style statistical method, or an explicit contaminant sequence/
# OTU/ASV/taxon/read) rather than a bare "contaminant"/"contamination"
# mention anywhere.
_SCREEN_CONTAM_METHOD_CONTEXT_RE = re.compile(
    r"\bblank(?:s)?\b|\bnegative\s+control(?:s)?\b|\bcontrol\s+sample(?:s)?\b|\bdecontam|"
    r"\bprevalence\b|\bfrequency[-\s]based\b|"
    r"\bcontaminant\s+(?:sequences?|OTUs?|ASVs?|taxa|taxon|reads?)\b|"
    r"\b(?:sequences?|OTUs?|ASVs?|reads?|taxa|taxon)\s+(?:considered|flagged|identified|removed)\s+as\s+"
    r"(?:a\s+)?contaminant",
    re.IGNORECASE,
)
_BLOCKING_OLIGO_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"blocking\s+(?:primer|primers|oligo|oligos|oligonucleotide|oligonucleotides)|"
    r"host[-\s]+blocking\s+primer|"
    r"blocker(?:\s+(?:primer|oligo|oligonucleotide))?|"
    r"suppressor\s+oligonucleotide|"
    r"(?:suppress|inhibit|block)\s+(?:host\s+|predator\s+|non[-\s]?target\s+)?(?:DNA\s+)?amplification"
    r")\b",
    re.IGNORECASE,
)
_PROBE_ASSAY_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"TaqMan|hydrolysis\s+probe|molecular\s+beacon|reporter\s+dye|fluorophore|"
    r"FAM|HEX|VIC|Cy5|TET|TAMRA|ROX|JOE|BHQ|ZEN|MGB|"
    r"FISH|CARD[-\s]?FISH|oligonucleotide\s+probe|hybridization\s+probe|"
    r"HRP[-\s]?labeled|horseradish\s+peroxidase"
    r")\b",
    re.IGNORECASE,
)
_PROBE_REF_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"probe\s+(?:described|designed|modified|published)|"
    r"(?:described|designed|modified|published)\s+(?:by|for|in|from).{0,80}\bprobe|"
    r"previously\s+published\s+probe|"
    r"designed\s+in\s+this\s+study|"
    r"designed\s+for\s+this\s+study|"
    r"HRP[-\s]?labeled|horseradish\s+peroxidase|FISH\s+probe|CARD[-\s]?FISH\s+probe"
    r")\b",
    re.IGNORECASE,
)
_PROBE_CONC_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"probe\s+(?:concentration|final\s+concentration|was\s+added|at)|"
    r"(?:final\s+)?concentration\s+of\s+(?:the\s+)?probe|"
    r"probe\b.{0,100}\b\d+(?:\.\d+)?\s*(?:nM|uM|µM|μM)\b|"
    r"\d+(?:\.\d+)?\s*(?:nM|uM|µM|μM)\s+(?:[^.]{0,40}\s+)?probe"
    r")\b",
    re.IGNORECASE,
)
_PROBE_CONC_VALUE_RE = re.compile(
    r"^\s*(?:~|≈|about\s+|approximately\s+)?\d+(?:\.\d+)?\s*(?:nM|uM|µM|μM|nmol/L|umol/L|µmol/L|μmol/L)\s*$",
    re.IGNORECASE,
)
_DETECTION_CRITERIA_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"(?:considered|scored|called|accepted|counted)\s+(?:as\s+)?positive|"
    r"positive\s+(?:if|when)|"
    r"(?:detected|present)\s+(?:if|when)|"
    r"(?:accepted|confirmed|counted)\s+as\s+(?:detected|present)|"
    r"presence\s+was\s+(?:defined|confirmed)|"
    r"detection\s+criteri(?:on|a)|"
    r"(?:Cq|Ct)\s*[<≤]|"
    r"quantification\s+cycle\s+below|"
    r"positive\s+droplets?|"
    r"fluorescence\s+above\s+(?:the\s+)?threshold|"
    r"minimum\s+copy[-\s]?number\s+(?:criterion|criteria|threshold)"
    r")\b",
    re.IGNORECASE,
)
_DETECTION_CRITERIA_VALUE_RE = re.compile(
    r"\b(?:"
    r"positive|detected|present|presence|detection\s+criteri(?:on|a)|"
    r"Cq|Ct|quantification\s+cycle|positive\s+droplets?|fluorescence|copy[-\s]?number|threshold"
    r")\b|[<≤]",
    re.IGNORECASE,
)


# Real gap found live (10.3390/microorganisms10030558, STUDY-0049c7972ece):
# "Seawater temperature was 28.1 ± 0.2 °C. Seawater samples were
# collected at the surface layer (0.5 m depth) and the bottom..." states
# site temperature as its own sentence immediately before the collection
# sentence, never combining "in situ"/"at collection" wording with the
# measurement itself in one sentence -- a real, common Methods phrasing
# _SAMPLING_TIME_CONTEXT_RE's single-sentence check can't see, since the
# collection-time context sits in the NEXT sentence over. in_situ_temp/
# in_situ_salinity are the only two fields checked against a small
# +/-1-sentence window instead of the bare snippet, specifically because
# their own context regex was built to require explicit wording a
# same-sentence check can miss when a paper simply states conditions and
# then immediately describes collection right after.
_SAMPLING_TIME_WINDOW_FIELDS = frozenset({"in_situ_temp", "in_situ_salinity"})
_OLIGO_TABLE_CONTEXT_RE = re.compile(r"\b(?:oligonucleotide|primer|probe|blocking|blocker)\b", re.IGNORECASE)
_OLIGO_TABLE_ROW_CANDIDATE_RE = re.compile(
    r"\b(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]{1,50})\s*\|\s*"
    r"(?P<sequence>[ACGTRYSWKMBDHVN]{10,})\s*\|\s*"
    r"(?P<target>[^|.]{1,80})\s*\|\s*"
    r"(?P<use>[^|.]{1,20})\s*\|\s*"
    r"(?P<reference>[^.]{1,120})",
    re.IGNORECASE,
)


def _llm_judged_field_matches_snippet(field: LLMJudgedSearchField, snippet: str, window: str | None = None) -> bool:
    if not any(_term_pattern(term).search(snippet) for term in field.search_terms):
        return False
    if field.term_name in _SAMPLING_TIME_WINDOW_FIELDS:
        return bool(_SAMPLING_TIME_CONTEXT_RE.search(window or snippet))
    if field.term_name == "otu_clust_tool":
        return bool(_OTU_CLUSTER_TOOL_CONTEXT_RE.search(snippet))
    if field.term_name == "otu_db":
        return bool(_OTU_DB_CONTEXT_RE.search(snippet))
    if field.term_name == "assay_name":
        return bool(_ASSAY_NAME_CONTEXT_RE.search(snippet))
    if field.term_name == "screen_contam_method":
        return bool(_SCREEN_CONTAM_METHOD_CONTEXT_RE.search(snippet))
    if field.term_name in {"block_ref", "block_taxa"}:
        return bool(_BLOCKING_OLIGO_CONTEXT_RE.search(snippet))
    if field.term_name == "probe_ref":
        return bool(_PROBE_ASSAY_CONTEXT_RE.search(snippet) and _PROBE_REF_CONTEXT_RE.search(snippet))
    if field.term_name == "probe_conc":
        return bool(_PROBE_ASSAY_CONTEXT_RE.search(snippet) and _PROBE_CONC_CONTEXT_RE.search(snippet))
    if field.term_name == "detection_criteria":
        return bool(_DETECTION_CRITERIA_CONTEXT_RE.search(snippet))
    if field.term_name == "assay_target_taxa":
        return bool(_TARGET_TAXONOMIC_ASSAY_CONTEXT_RE.search(snippet))
    if field.term_name == "study_target_taxonomic_scope":
        return bool(_TARGET_TAXONOMIC_SCOPE_CONTEXT_RE.search(snippet))
    return True


def _candidate_fields_for_snippet(
    snippet: str, exclude_field_names: frozenset[str] = frozenset(), window: str | None = None
) -> tuple[str, ...]:
    field_names: list[str] = []
    for field in LLM_JUDGED_SEARCH_FIELDS:
        if field.term_name in exclude_field_names:
            continue
        if _llm_judged_field_matches_snippet(field, snippet, window):
            field_names.append(field.term_name)
    return tuple(field_names)


def _oligo_table_quote_candidates(
    text: str,
    *,
    title: str,
    existing_count: int,
    exclude_field_names: frozenset[str],
) -> tuple[QuoteCandidate, ...]:
    if not _OLIGO_TABLE_CONTEXT_RE.search(text):
        return ()
    candidates: list[QuoteCandidate] = []
    for match_index, match in enumerate(_OLIGO_TABLE_ROW_CANDIDATE_RE.finditer(" ".join(text.split()))):
        row_text = " | ".join(
            (
                match.group("name").strip(),
                match.group("sequence").strip(),
                match.group("target").strip(),
                match.group("use").strip(),
                match.group("reference").strip(),
            )
        )
        folded = row_text.casefold()
        field_names: list[str] = []
        if (
            "probe" in folded or match.group("use").strip().casefold() in {"c", "fish", "card-fish"}
        ):
            field_names.extend(["probe_seq", "probe_ref", "targeted_detection_method_additional"])
        if "block" in folded or "blocking" in folded or "blocker" in folded:
            field_names.extend(["block_seq", "block_ref", "block_taxa", "targeted_detection_method_additional"])
        field_names = [name for name in field_names if name not in exclude_field_names]
        if not field_names:
            continue
        candidates.append(
            QuoteCandidate(
                quote_id=f"Q{existing_count + len(candidates) + 1:03d}",
                field_names=tuple(dict.fromkeys(field_names)),
                title=title,
                snippet_index=match_index,
                text=row_text,
            )
        )
    return tuple(candidates)


def _neighboring_sentences_window(snippets_list: list[tuple[int, str]], position: int) -> str:
    """+/-1 sentence around snippets_list[position], joined -- see
    _SAMPLING_TIME_WINDOW_FIELDS's own comment for why this exists."""
    lo = max(0, position - 1)
    hi = min(len(snippets_list), position + 2)
    return " ".join(text for _, text in snippets_list[lo:hi])


def quote_candidates_for_llm_judged_search(
    texts: Iterable[tuple[str, str]],
    *,
    max_candidates: int = 40,
    exclude_field_names: frozenset[str] = frozenset(),
) -> tuple[QuoteCandidate, ...]:
    """Candidate source sentences for the small library-prep judgement LLM.

    The LLM never receives whole sections for these fields: only sentences
    that hit the user-supplied search terms (or, for
    _SAMPLING_TIME_WINDOW_FIELDS specifically, a small window around them --
    see that constant's own comment) -- final facts are accepted only
    when the model cites one of these quote IDs.
    """
    candidates: list[QuoteCandidate] = []
    seen_text: set[str] = set()
    for title, text in texts:
        for table_candidate in _oligo_table_quote_candidates(
            text,
            title=title,
            existing_count=len(candidates),
            exclude_field_names=exclude_field_names,
        ):
            if table_candidate.text in seen_text:
                continue
            seen_text.add(table_candidate.text)
            candidates.append(table_candidate)
            if len(candidates) >= max_candidates:
                return tuple(candidates)
        snippets_list = list(_snippets(text))
        for position, (snippet_index, snippet) in enumerate(snippets_list):
            window = _neighboring_sentences_window(snippets_list, position)
            field_names = _candidate_fields_for_snippet(snippet, exclude_field_names, window)
            if not field_names:
                continue
            candidate_text = window if _SAMPLING_TIME_WINDOW_FIELDS & set(field_names) else snippet
            if candidate_text in seen_text:
                continue
            seen_text.add(candidate_text)
            candidates.append(
                QuoteCandidate(
                    quote_id=f"Q{len(candidates) + 1:03d}",
                    field_names=field_names,
                    title=title,
                    snippet_index=snippet_index,
                    text=candidate_text,
                )
            )
            if len(candidates) >= max_candidates:
                return tuple(candidates)
    return tuple(candidates)


def build_llm_judged_search_prompt(candidates: tuple[QuoteCandidate, ...]) -> str:
    field_reference = "\n".join(
        (
            f"- {field.term_name}: {field.description} "
            f"{field.output_instructions}"
            + (f" Allowed values: {', '.join(field.allowed_values)}." if field.allowed_values else "")
        )
        for field in LLM_JUDGED_SEARCH_FIELDS
    )
    quotes = "\n".join(
        f"{candidate.quote_id} [{', '.join(candidate.field_names)}] {candidate.title}: {candidate.text}"
        for candidate in candidates
    )
    return f"""You are judging candidate source quotes for FAIRe projectMetadata targeted-search fields.

Use only the candidate quotes below. Do not use outside knowledge. Do not infer from a keyword alone.
Return a field only if a quote explicitly supports it. For free-text fields, keep raw_value as close as possible to
the quote text; copy exact phrases when possible. Do not rewrite adapter sequences. If multiple values for the same
field are explicitly supported, return one object per value, each citing its supporting quote_id. Each quote below
is labeled with the field name(s) in brackets it was pre-matched for -- a single dense quote can genuinely support
MORE THAN ONE of its own bracketed fields at once (e.g. one sentence naming both a database and a clustering tool);
return a separate object for each bracketed field the quote actually supports, not just one. Never attach a value
to a field that is NOT in that quote's own bracket list.

Fields:
{field_reference}

Return ONLY a JSON array. Each object must be:
{{"field": "<one listed field>", "raw_value": "<supported value>", "quote_id": "Q001"}}

Candidate quotes:
{quotes}
"""


def _allowed_field_lookup() -> dict[str, LLMJudgedSearchField]:
    return {field.term_name: field for field in LLM_JUDGED_SEARCH_FIELDS}


# adapter_forward/adapter_reverse's own verbatim-quote guard only checks
# that the value literally appears somewhere in its cited quote -- a real
# live bug (10.1093/ismejo/wrae013, "...targeting the adapter sequences
# [ ];") passed that guard while returning the literal citation-bracket
# placeholder "[ ]" as if it were a real adapter sequence, since "[ ]"
# does appear verbatim in the quote. IUPAC nucleotide alphabet only
# (ACGTU plus ambiguity codes), letters-only after stripping whitespace/
# dashes (real reported sequences sometimes have either, e.g. a dash
# joining a primer and its adapter overhang), at least 4 real bases so a
# stray single letter can't pass either.
_NUCLEOTIDE_SEQUENCE_RE = re.compile(r"^[ACGTURYSWKMBDHVN]{4,}$", re.IGNORECASE)


def _looks_like_nucleotide_sequence(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value)
    return bool(_NUCLEOTIDE_SEQUENCE_RE.fullmatch(compact))


def _valid_llm_judged_value(field: LLMJudgedSearchField, value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if field.term_name == "assay_name":
        return bool(_clean_assay_name_parts(stripped))
    if field.term_name == "assay_target_taxa":
        return not _BARE_MARKER_GENE_TARGET_TAXA_RE.fullmatch(stripped)
    if field.term_name == "otu_db":
        return bool(_OTU_DB_VALUE_RE.search(stripped))
    if field.term_name in ("adapter_forward", "adapter_reverse", "block_seq", "probe_seq"):
        return _looks_like_nucleotide_sequence(stripped)
    if field.term_name == "block_taxa":
        lowered = stripped.casefold()
        if any(term in lowered for term in ("otu", "asv", "abundance", "representative sequences", "unidentified sequences")):
            return False
        return len(stripped.split()) <= 8
    if field.term_name == "probe_conc":
        return bool(_PROBE_CONC_VALUE_RE.fullmatch(stripped))
    if field.term_name == "probe_ref":
        lowered = stripped.casefold()
        if any(term in lowered for term in ("ysi", "yellow springs", "http://www.drive5.com", "usearch")):
            return False
    if field.term_name in ("pcr_assay_lod", "pcr_assay_loq"):
        return bool(re.search(r"\d", stripped))
    if field.term_name == "detection_criteria":
        return bool(_DETECTION_CRITERIA_VALUE_RE.search(stripped))
    if not field.allowed_values:
        return True
    parts = [part.strip() for part in stripped.split("|")]
    return all(
        part in field.allowed_values
        or ("other:" in field.allowed_values and part.casefold().startswith("other:"))
        for part in parts
    )


def _clean_assay_name_parts(value: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in (p.strip() for p in value.split("|")):
        if not part:
            continue
        part = _normalize_assay_name_part(part)
        if "/" in part:
            continue
        if _BARE_FUNCTIONAL_GENE_ASSAY_NAME_RE.fullmatch(part):
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)
    return cleaned


_MOJIBAKE_DASH_RE = re.compile(r"(?:\u7ab6\u5929|\u7ab6\u96fb|\u7ab6\u642d|\u7ab6\u7763|\u7ab6\u6d9b)")
_ASSAY_DASH_NORMALIZE_RE = re.compile(r"[‐-―]")
_RRNA_REGION_ASSAY_NAME_RE = re.compile(
    r"^(?P<gene>1[68]\s*S)(?:\s*rRNA)?\s*[- ]\s*(?P<region>V\d(?:-V\d)?)$",
    re.IGNORECASE,
)


def _normalize_assay_name_part(value: str) -> str:
    part = _MOJIBAKE_DASH_RE.sub("-", value.strip())
    part = _ASSAY_DASH_NORMALIZE_RE.sub("-", part)
    part = re.sub(r"\s*-\s*", "-", part)
    part = re.sub(r"\s+", " ", part)
    marker_region = _RRNA_REGION_ASSAY_NAME_RE.fullmatch(part)
    if marker_region:
        gene = re.sub(r"\s+", "", marker_region.group("gene")).upper()
        return f"{gene}-{marker_region.group('region').upper()}"
    return part


def _llm_judged_value_priority(field: LLMJudgedSearchField, value: str, quote: str) -> int:
    if field.term_name == "otu_db" and not _OTU_DB_VALUE_RE.search(value):
        return len(field.search_terms) + 1
    # neg_cont_0_1/pos_cont_0_1: "1" always outranks "0" when the model
    # judges both across different candidate quotes -- real evidence a
    # control WAS used beats an unrelated quote that merely didn't
    # support one, matching the same "explicit positive evidence wins"
    # precedent as the retired TextSearchFlag pair this replaced.
    if field.term_name in ("neg_cont_0_1", "pos_cont_0_1"):
        return 0 if value == "1" else 1
    for index, term in enumerate(field.search_terms):
        if _term_pattern(term).search(value):
            return index
    for index, term in enumerate(field.search_terms):
        pattern = _term_pattern(term)
        if pattern.search(quote):
            return index
    return len(field.search_terms)


# Fields whose own output_instructions explicitly promise a verbatim copy
# from the quote (as opposed to a composed/summarized/classified answer,
# which several sibling fields in this same mechanism deliberately are --
# see _facts_from_llm_judgement's own comment at its call site).
_VERBATIM_REQUIRED_FIELDS = frozenset(
    {
        "informationWithheld",
        "in_situ_temp",
        "in_situ_salinity",
        "probe_seq",
        "block_seq",
    }
)

_OTU_SEQ_COMP_TOOL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("classify-consensus vsearch", re.compile(r"\bclassify[-_\s]+consensus[-_\s]+vsearch\b", re.IGNORECASE)),
    ("QIIME 2 feature-classifier", re.compile(r"\bQIIME\s*2\s+feature[-_\s]+classifier\b", re.IGNORECASE)),
    ("q2-feature-classifier", re.compile(r"\bq2[-_\s]+feature[-_\s]+classifier\b", re.IGNORECASE)),
    ("classify-sklearn", re.compile(r"\bclassify[-_\s]+sklearn\b", re.IGNORECASE)),
    ("RDP Classifier", re.compile(r"\bRDP\s+Classifier\b|\bRibosomal\s+Database\s+Project\s+classifier\b", re.IGNORECASE)),
    ("IDTAXA", re.compile(r"\bIDTAXA\b|\bDECIPHER\s+IdTaxa\b", re.IGNORECASE)),
    ("MegaBLAST", re.compile(r"\bMegaBLAST\b", re.IGNORECASE)),
    ("BLASTX", re.compile(r"\bBLASTX\b", re.IGNORECASE)),
    ("BLASTn", re.compile(r"\bBLASTn\b", re.IGNORECASE)),
    ("BLAST", re.compile(r"\bBLAST\b", re.IGNORECASE)),
    ("VSEARCH", re.compile(r"\bVSEARCH\b", re.IGNORECASE)),
    ("USEARCH", re.compile(r"\bUSEARCH\b", re.IGNORECASE)),
    ("CREST4", re.compile(r"\bCREST4\b", re.IGNORECASE)),
    ("CREST", re.compile(r"\bCREST\b", re.IGNORECASE)),
    ("Kraken2", re.compile(r"\bKraken\s*2(?:\.\d+(?:\.\d+)*)?\b|\bKraken2(?:\s+\d+(?:\.\d+)*)?\b", re.IGNORECASE)),
    ("Kraken", re.compile(r"\bKraken\b", re.IGNORECASE)),
    ("SINTAX", re.compile(r"\bSINTAX\b", re.IGNORECASE)),
    ("SEPP", re.compile(r"\bSEPP\b|\bq2[-_\s]+fragment[-_\s]+insertion\b", re.IGNORECASE)),
    ("EPA-ng", re.compile(r"\bEPA[-_\s]+ng\b", re.IGNORECASE)),
    ("pplacer", re.compile(r"\bpplacer\b", re.IGNORECASE)),
    ("PROTAX", re.compile(r"\bPROTAX\b", re.IGNORECASE)),
    ("TIPP", re.compile(r"\bTIPP\b", re.IGNORECASE)),
)


def _otu_seq_comp_tool_values(text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for canonical, pattern in _OTU_SEQ_COMP_TOOL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = " ".join(match.group(0).split())
        if canonical in {"Kraken2", "Kraken"}:
            value = value.replace("Kraken 2", "Kraken2")
        key = canonical.casefold()
        if key in seen:
            continue
        if canonical in {"BLAST", "VSEARCH", "Kraken", "CREST"}:
            lower_values = " | ".join(values).casefold()
            if canonical.casefold() in lower_values:
                continue
        seen.add(key)
        values.append(value)
    return values


def _format_otu_seq_comp_appr_value(entries: list[dict], quotes: list[str]) -> str:
    tools: list[str] = []
    seen_tools: set[str] = set()
    for text in [*(str(entry["raw_value"]) for entry in entries), *quotes]:
        for tool in _otu_seq_comp_tool_values(text):
            key = tool.casefold()
            if key in seen_tools:
                continue
            seen_tools.add(key)
            tools.append(tool)

    quote_parts: list[str] = []
    seen_quotes: set[str] = set()
    for text in [*(str(entry["raw_value"]) for entry in entries), *quotes]:
        cleaned = " ".join(str(text).split()).strip()
        if not cleaned:
            continue
        if any(cleaned.casefold() == tool.casefold() for tool in tools):
            continue
        key = cleaned.casefold()
        if key in seen_quotes:
            continue
        seen_quotes.add(key)
        quote_parts.append(cleaned)

    return " | ".join([*tools, *quote_parts])


def _format_control_value_with_quote(entry: dict) -> str:
    value = str(entry["raw_value"]).strip()
    quote = " ".join(str(entry.get("quote") or "").split()).strip()
    return f"{value} | {quote}" if quote else value


def _facts_from_llm_judgement(
    parsed,
    candidates: tuple[QuoteCandidate, ...],
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    if not isinstance(parsed, list):
        return []
    fields = _allowed_field_lookup()
    candidates_by_id = {candidate.quote_id: candidate for candidate in candidates}
    grouped: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field") or item.get("fact_type_candidate") or "").strip()
        value = str(item.get("raw_value") or "").strip()
        quote_id = str(item.get("quote_id") or item.get("evidence_id") or "").strip()
        field = fields.get(field_name)
        candidate = candidates_by_id.get(quote_id)
        if field is None or candidate is None or not _valid_llm_judged_value(field, value):
            continue
        # Mirrors the same fix in section_category_extraction.py's Stage 3
        # guard: the verbatim check alone doesn't stop the model from
        # attaching a value to a field this quote was never even offered
        # for, as long as the text happens to also appear in that quote.
        if field_name not in candidate.field_names:
            continue
        # A real live audit (10.1093/ismejo/wrae013) caught the model
        # returning a silently typo'd version of a quote's own tool name
        # -- the prompt's own "copy verbatim" instruction never actually
        # enforced it. Deliberately scoped to only the fields whose own
        # output_instructions promise a verbatim copy (_VERBATIM_REQUIRED_
        # FIELDS) -- several other free-text fields in this same mechanism
        # (assay_name's composed short name, inhibition_check's own
        # summary) are legitimately NOT verbatim by design, confirmed by
        # their own existing tests breaking when this check was first
        # tried unscoped against every field.
        if field_name in _VERBATIM_REQUIRED_FIELDS and value.casefold() not in candidate.text.casefold():
            continue
        if field_name == "assay_name":
            value = " | ".join(_clean_assay_name_parts(value))
            if not value:
                continue
        group = grouped.setdefault(
            field_name,
            {"entries": [], "quotes": [], "seen_values": set()},
        )
        key = value.casefold()
        if key in group["seen_values"]:
            continue
        group["seen_values"].add(key)
        if candidate.text not in group["quotes"]:
            group["quotes"].append(candidate.text)
        group["entries"].append(
            {
                "raw_value": value,
                "quote_id": quote_id,
                "source_locator": f"{locator_prefix}:llm_judged_search:{field_name}:{quote_id}",
                "priority": _llm_judged_value_priority(field, value, candidate.text),
                "quote": candidate.text,
            }
        )

    facts: list[RawFactCandidate] = []
    for field_name, group in grouped.items():
        field = fields[field_name]
        entries = sorted(group["entries"], key=lambda entry: entry["priority"])
        if field_name in SINGLE_BEST_LLM_JUDGED_FIELDS:
            entries = entries[:1]
        raw_value = (
            _format_otu_seq_comp_appr_value(entries, group["quotes"])
            if field_name == "otu_seq_comp_appr"
            else _format_control_value_with_quote(entries[0])
            if field_name in {"neg_cont_0_1", "pos_cont_0_1"}
            else " | ".join(entry["raw_value"] for entry in entries)
        )
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value=raw_value,
                source_locator=f"{locator_prefix}:llm_judged_search:{field_name}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=" | ".join(group["quotes"]),
                confidence_metadata={
                    "detector": "llm_judged_quote_search",
                    "section": field.section,
                    "description": field.description,
                    "matches": entries,
                },
            )
        )
    return facts


def _mirror_not_a_control_to_sibling_field(facts: list[RawFactCandidate]) -> list[RawFactCandidate]:
    """neg_cont_0_1/pos_cont_0_1 are always judged from the very same
    "control"/"blank" quote(s) -- a real quote the model judges as "0"
    (not a control at all, e.g. "quality control" in a data-processing
    sense) is equally not a positive control AND not a negative control.
    A real live-paper audit found the model reliably emitting only ONE of
    the two sibling fields for such a quote instead of both (the same
    kind of partial-completion miss seen elsewhere with this small local
    model), leaving the other blank rather than the correct "0". Only
    mirrors a "0" (a genuine "1" for one side says nothing definitive
    about the other -- a paper can have a real positive control mentioned
    separately from where its negative control is mentioned, so a "1" is
    never auto-mirrored)."""
    by_type = {fact.fact_type_candidate: fact for fact in facts}
    neg = by_type.get("neg_cont_0_1")
    pos = by_type.get("pos_cont_0_1")
    if neg is not None and neg.raw_value.split("|", 1)[0].strip() == "0" and pos is None:
        facts = [*facts, neg.model_copy(update={"fact_type_candidate": "pos_cont_0_1", "raw_field_name": "pos_cont_0_1"})]
    elif pos is not None and pos.raw_value.split("|", 1)[0].strip() == "0" and neg is None:
        facts = [*facts, pos.model_copy(update={"fact_type_candidate": "neg_cont_0_1", "raw_field_name": "neg_cont_0_1"})]
    return facts



# neg_cont_0_1's own search_terms ("control"/"controls"/"blank"/"blanks")
# are deliberately broad enough to also flag a POSITIVE control mention
# ("a positive control using synthetic DNA...") or an unrelated generic
# usage ("quality control was performed...") as a raw candidate -- the
# LLM's own judgement, not the candidate gate, is what tells those apart
# in the normal case. That means "the model returned nothing for this
# candidate" is genuinely ambiguous on its own: it could mean "correctly
# recognized this ISN'T a real negative control" (silence is the RIGHT
# answer, "0" is still the honest confident default) or "missed a real
# one" (STUDY-0161dd80b492: "Each batch of extractions included an
# extraction blank" is textbook negative-control terminology the model
# simply failed to report). Only "blank"/"blanks" -- unlike bare
# "control"/"controls", which both those false-positive-shaped
# candidates above also matched -- is unambiguous enough in this domain
# to trust as "a real negative control was very likely being described
# here", so only a missed BLANK-bearing candidate withholds the
# confident "0" default; a missed bare-"control" candidate still gets
# it, same as before.
_STRONG_NEG_CONT_EVIDENCE_RE = re.compile(r"\bblanks?\b", re.IGNORECASE)


def _control_not_found_fallback_facts(
    *,
    locator_prefix: str,
    existing_fact_types: frozenset[str],
    exclude_field_names: frozenset[str],
    candidates: tuple[QuoteCandidate, ...] = (),
) -> list[RawFactCandidate]:
    """Per an explicit user request ("i also see no mention of +/- controls
    ... I think they should both be 0"): when a paper's text never even
    raises a "control"/"blank" candidate for one or both of neg_cont_0_1/
    pos_cont_0_1, the honest, confident default is "0" (no control used),
    not a blank field indistinguishable from "never checked".

    Real gap found live (STUDY-0161dd80b492): a real, unambiguous
    "extraction blank" mention DID raise a neg_cont_0_1 candidate quote,
    but the model's own judgement call didn't return a value for it (a
    real partial-completion miss, the same class already seen elsewhere
    with this small local model) -- yet this fallback fired anyway and
    wrote a confident "0", silently converting a genuine negative
    control mention into the opposite answer. See
    _STRONG_NEG_CONT_EVIDENCE_RE's own comment for why this is scoped to
    "blank"-bearing candidates specifically rather than any candidate."""
    facts: list[RawFactCandidate] = []
    for field_name in ("neg_cont_0_1", "pos_cont_0_1"):
        if field_name in exclude_field_names or field_name in existing_fact_types:
            continue
        if field_name == "neg_cont_0_1" and any(
            field_name in candidate.field_names and _STRONG_NEG_CONT_EVIDENCE_RE.search(candidate.text)
            for candidate in candidates
        ):
            continue
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value="0 | no control/blank mention found",
                source_locator=f"{locator_prefix}:llm_judged_search:{field_name}:not_found_fallback",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=None,
                confidence_metadata={
                    "detector": "control_not_found_default",
                    "description": f"No control/blank mention was found supporting {field_name}.",
                },
            )
        )
    return facts


_BARCODING_TWO_STEP_KEYWORD_RE = re.compile(
    r"two[- ]step PCR|two[- ]stage PCR|two[- ]round PCR|"
    # Real gap found live (10.3389/fmicb.2017.01135): "the eight cycles
    # of second-round PCR" never matched the original "second PCR"
    # phrase at all -- the inserted "-round" broke the fixed 2-word
    # match, silently defaulting the whole study to one-step PCR and,
    # downstream, leaving the entire pcr2_* field family blank.
    r"second[- ]round(?:\s+of)?\s+PCR|second PCR|PCR ?2\b|indexing PCR|barcode PCR|adapter PCR",
    re.IGNORECASE,
)
_BARCODING_LIGATION_KEYWORD_RE = re.compile(
    r"ligation[- ]based|adapter ligation|barcode ligation|ligated adapters|ligation sequencing kit",
    re.IGNORECASE,
)


def _barcoding_pcr_appr_keyword_match(
    texts: tuple[tuple[str, str], ...],
) -> tuple[str, str] | None:
    """Deterministic keyword check for barcoding_pcr_appr, tried before
    defaulting to "one-step PCR". The quote-judged LLM path above already
    covers this same vocabulary (it's baked into that field's own
    search_terms/output_instructions), but a real live-paper audit showed
    a small local model can silently drop a candidate quote from a large
    combined judgement call -- letting the "no evidence found" default
    silently overwrite a real "two-round PCR amplification strategy"
    mention with "one-step PCR". This check never depends on the LLM
    actually applying its own mapping instruction correctly.
    """
    sentences = [
        sentence.strip()
        for _title, text in texts
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]
    for sentence in sentences:
        if _BARCODING_LIGATION_KEYWORD_RE.search(sentence):
            return "ligation-based", sentence
    for sentence in sentences:
        if _BARCODING_TWO_STEP_KEYWORD_RE.search(sentence):
            return "two-step PCR", sentence
    return None


def _barcoding_one_step_fallback_fact(
    *,
    texts: tuple[tuple[str, str], ...],
    locator_prefix: str,
    active_flags: frozenset[str],
    existing_fact_types: frozenset[str],
    exclude_field_names: frozenset[str],
) -> RawFactCandidate | None:
    if "barcoding_pcr_appr" in exclude_field_names or "barcoding_pcr_appr" in existing_fact_types:
        return None
    if "pcr_0_1" not in active_flags:
        return None
    keyword_match = _barcoding_pcr_appr_keyword_match(texts)
    if keyword_match is not None:
        value, evidence_sentence = keyword_match
        return RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="barcoding_pcr_appr",
            raw_field_name="barcoding_pcr_appr",
            raw_value=value,
            source_locator=f"{locator_prefix}:llm_judged_search:barcoding_pcr_appr:keyword_fallback",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
            evidence_quote=evidence_sentence,
            confidence_metadata={
                "detector": "barcoding_pcr_appr_keyword_fallback",
                "description": (
                    "pcr_0_1 was true and no quote-judged barcoding approach was "
                    "extracted by the LLM; matched deterministically via keyword instead."
                ),
            },
        )
    return RawFactCandidate(
        entity_level=EntityLevel.STUDY,
        fact_type_candidate="barcoding_pcr_appr",
        raw_field_name="barcoding_pcr_appr",
        raw_value="one-step PCR",
        source_locator=f"{locator_prefix}:llm_judged_search:barcoding_pcr_appr:pcr_flag_fallback",
        support_type=SupportType.DETERMINISTICALLY_DERIVED,
        evidence_quote=None,
        confidence_metadata={
            "detector": "pcr_flag_default",
            "description": (
                "pcr_0_1 was true and no quote-judged two-step or ligation-based "
                "barcoding approach was extracted"
            ),
        },
    )


def _not_found_fallback_facts(
    *,
    field_names: tuple[str, ...],
    candidates: tuple[QuoteCandidate, ...],
    locator_prefix: str,
    existing_fact_types: frozenset[str],
    exclude_field_names: frozenset[str],
) -> list[RawFactCandidate]:
    facts: list[RawFactCandidate] = []
    for field_name in field_names:
        if field_name in exclude_field_names or field_name in existing_fact_types:
            continue
        if any(field_name in candidate.field_names for candidate in candidates):
            continue
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value="not found",
                source_locator=f"{locator_prefix}:llm_judged_search:{field_name}:not_found_fallback",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=None,
                confidence_metadata={
                    "detector": "llm_judged_not_found_default",
                    "description": f"No qualifying candidate sentence was found for {field_name}.",
                },
            )
        )
    return facts


# A real, common notation: a paper reports its whole fusion-primer
# oligo ("adapter-tailed primer") as one sequence, e.g. "5'-
# TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG-CTCCTACGGGAGGCAGCAG-3'" -- the
# Illumina/Nextera sequencing-adapter overhang fused to the real PCR
# primer. adapter_forward/adapter_reverse correctly capture the whole
# thing verbatim, but the primer portion was never separately captured
# as pcr_primer_forward/pcr_primer_reverse at all -- confirmed live
# (10.1371/journal.pone.0303937): "5´–TCGTCGGCAGCGTCAGATGTGTAT
# AAGAGACAG-CTCCTACGGGAGGCAGCAG–3´" is a real Nextera forward overhang
# fused to the real 341F 16S primer, joined by one plain ASCII hyphen
# ("-", U+002D) -- distinct from the en-dash ("–", U+2013) the paper
# uses for the outer 5'/3' boundary markers, a reliable way to tell the
# fusion join apart from decorative punctuation.
_ADAPTER_TO_FUSED_PRIMER_FIELD = {"adapter_forward": "pcr_primer_forward", "adapter_reverse": "pcr_primer_reverse"}
# The digit/prime-mark order varies even within one paper's own notation
# (confirmed live: the same sentence uses "5´–" -- digit, then prime mark,
# then dash -- for the leading boundary, but "–3´" -- dash, then digit,
# then prime mark -- for the trailing one), so both orders are accepted on
# each end rather than assuming a fixed order.
_LEADING_PRIME_MARKER_RE = re.compile(
    r"^\s*(?:[53]\s*['’´]|['’´]\s*[53])\s*[-–—]?\s*", re.IGNORECASE
)
_TRAILING_PRIME_MARKER_RE = re.compile(
    r"\s*[-–—]?\s*(?:[53]\s*['’´]|['’´]\s*[53])\s*$", re.IGNORECASE
)
_NUCLEOTIDE_SEQUENCE_ONLY_RE = re.compile(r"^[ACGTURYSWKMBDHVN]{6,}$", re.IGNORECASE)


def _clean_fused_sequence_part(value: str) -> str:
    cleaned = _LEADING_PRIME_MARKER_RE.sub("", value)
    cleaned = _TRAILING_PRIME_MARKER_RE.sub("", cleaned)
    return re.sub(r"\s+", "", cleaned)  # real sequences never contain whitespace


def _fused_sequence_split(raw_value: str) -> tuple[str, str] | None:
    """Splits a fused adapter+primer sequence at its join point. A plain
    ASCII hyphen is the common, explicit notation (see this module's own
    history), but real gap found live: "Fluidigm CS1 + MiFish-U-F
    ACACTGACGACATGGTTCTACA GTCGGTAAAACTCGTGCCAGC" -- the CS1 adapter tag
    and the MiFish-U-F primer are simply printed side by side, space-
    separated, no hyphen anywhere between them. Tries a hyphen split
    first (the more explicit, intentional notation, so it wins whenever
    both a hyphen and incidental whitespace are present), then falls back
    to a plain-whitespace split when there's no hyphen at all -- either
    way, both resulting halves must independently validate as clean
    nucleotide-only sequences before the split is accepted, so this never
    guesses on something that isn't genuinely two fused sequences."""
    if "-" in raw_value:
        left, _, right = raw_value.rpartition("-")
        left, right = _clean_fused_sequence_part(left), _clean_fused_sequence_part(right)
        if _NUCLEOTIDE_SEQUENCE_ONLY_RE.match(left) and _NUCLEOTIDE_SEQUENCE_ONLY_RE.match(right):
            return left, right
    parts = raw_value.split()
    if len(parts) == 2:
        left, right = (_clean_fused_sequence_part(part) for part in parts)
        if _NUCLEOTIDE_SEQUENCE_ONLY_RE.match(left) and _NUCLEOTIDE_SEQUENCE_ONLY_RE.match(right):
            return left, right
    return None


def _split_fused_adapter_primer_facts(facts: list[RawFactCandidate]) -> list[RawFactCandidate]:
    result: list[RawFactCandidate] = []
    for fact in facts:
        primer_field = _ADAPTER_TO_FUSED_PRIMER_FIELD.get(fact.fact_type_candidate)
        split = _fused_sequence_split(fact.raw_value) if primer_field is not None else None
        if split is None:
            # Doesn't look like a genuine fusion-primer split (either no
            # separator found, or either side isn't a clean nucleotide
            # sequence) -- leave the fact as-is rather than risk mangling
            # something this pattern wasn't meant for.
            result.append(fact)
            continue
        adapter_part, primer_part = split
        result.append(fact.model_copy(update={"raw_value": adapter_part}))
        result.append(
            fact.model_copy(
                update={
                    "fact_type_candidate": primer_field,
                    "raw_field_name": primer_field,
                    "raw_value": primer_part,
                }
            )
        )
    return result


def detect_llm_judged_search_facts(
    backend: LLMBackend,
    texts: Iterable[tuple[str, str]],
    *,
    locator_prefix: str,
    # A real live audit (10.1371/journal.pone.0303937) caught a dense,
    # candidate-rich paper's response getting cut off mid-object at 512
    # tokens (finish_reason "length"), silently dropping an already-in-
    # progress field's answer along with it -- confirmed via the raw API
    # response, not model non-determinism.
    max_output_tokens: int | None = MIN_LLM_MAX_OUTPUT_TOKENS,
    active_flags: frozenset[str] = frozenset(),
    exclude_field_names: frozenset[str] = frozenset(),
) -> list[RawFactCandidate]:
    reusable_texts = tuple(texts)
    candidates = quote_candidates_for_llm_judged_search(reusable_texts, exclude_field_names=exclude_field_names)
    if not candidates:
        facts = []
        facts.extend(
            _not_found_fallback_facts(
                field_names=(
                    "assay_target_taxa",
                ),
                candidates=candidates,
                locator_prefix=locator_prefix,
                existing_fact_types=frozenset(),
                exclude_field_names=exclude_field_names,
            )
        )
        barcoding_fallback = _barcoding_one_step_fallback_fact(
            texts=reusable_texts,
            locator_prefix=locator_prefix,
            active_flags=active_flags,
            existing_fact_types=frozenset(),
            exclude_field_names=exclude_field_names,
        )
        if barcoding_fallback:
            facts.append(barcoding_fallback)
        facts.extend(
            _control_not_found_fallback_facts(
                locator_prefix=locator_prefix,
                existing_fact_types=frozenset(),
                exclude_field_names=exclude_field_names,
                candidates=candidates,
            )
        )
        return facts
    parsed, response = backend.generate_json(
        build_llm_judged_search_prompt(candidates),
        system="You extract FAIRe library-preparation facts from supplied quote IDs only.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(
            f"{backend.label}: library-prep quote judgement returned invalid JSON after retries"
        )
    facts = _facts_from_llm_judgement(parsed, candidates, locator_prefix=locator_prefix)
    facts = _mirror_not_a_control_to_sibling_field(facts)
    facts = _split_fused_adapter_primer_facts(facts)
    existing_fact_types = frozenset(fact.fact_type_candidate for fact in facts)
    not_found_fallbacks = _not_found_fallback_facts(
        field_names=(
            "assay_target_taxa",
        ),
        candidates=candidates,
        locator_prefix=locator_prefix,
        existing_fact_types=existing_fact_types,
        exclude_field_names=exclude_field_names,
    )
    if not_found_fallbacks:
        facts.extend(not_found_fallbacks)
        existing_fact_types = frozenset(fact.fact_type_candidate for fact in facts)
    barcoding_fallback = _barcoding_one_step_fallback_fact(
        texts=reusable_texts,
        locator_prefix=locator_prefix,
        active_flags=active_flags,
        existing_fact_types=existing_fact_types,
        exclude_field_names=exclude_field_names,
    )
    if barcoding_fallback:
        facts.append(barcoding_fallback)
        existing_fact_types = frozenset(fact.fact_type_candidate for fact in facts)
    facts.extend(
        _control_not_found_fallback_facts(
            locator_prefix=locator_prefix,
            existing_fact_types=existing_fact_types,
            exclude_field_names=exclude_field_names,
            candidates=candidates,
        )
    )
    return facts


# phix_perc: a percentage number and the word "PhiX" essentially always
# co-occurring in the same sentence is an unambiguous, purely mechanical
# signal ("15% PhiX", "PhiX (15%)", "spiked with 15% PhiX control") -- no
# LLM judgment call is needed for this one, per an explicit user request
# for "a quick search ... for PhiX or its variations". Deliberately a
# separate, self-contained deterministic pass (not a TEXT_SEARCH_FLAGS
# boolean, which can't carry a numeric value; not a CONTROLLED_SEARCH_
# FIELDS/LLMJudgedSearchField entry, which would spend an LLM call on
# something a regex already resolves outright).
_PHIX_MENTION_RE = re.compile(r"\bphix\b", re.IGNORECASE)
_PHIX_PERCENTAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def detect_phix_percentage_facts(
    texts: Iterable[tuple[str, str]],
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    """First sentence (across every supplied text, main paper or
    supplement) mentioning both PhiX and a percentage number wins -- same
    "first supporting sentence, one clear fact" convention as
    detect_text_search_flags below."""
    for title, text in texts:
        for index, snippet in _snippets(text):
            if not _PHIX_MENTION_RE.search(snippet):
                continue
            match = _PHIX_PERCENTAGE_RE.search(snippet)
            if not match:
                continue
            return [
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate="phix_perc",
                    raw_field_name="phix_perc",
                    raw_value=match.group(1),
                    source_locator=f"{locator_prefix}:{title}:sentence[{index}]",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=snippet,
                    confidence_metadata={"detector": "phix_percentage_regex"},
                )
            ]
    return []


def detect_text_search_flags(
    texts: Iterable[tuple[str, str]],
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    """Return one deterministic boolean flag per matched flag type.

    `texts` is an iterable of `(title, text)` pairs. The first supporting
    sentence wins so that a paper with many PCR mentions records one clear
    project-level flag instead of dozens of duplicate boolean facts.
    """
    facts: list[RawFactCandidate] = []
    seen: set[str] = set()
    for title, text in texts:
        for flag in TEXT_SEARCH_FLAGS:
            if flag.fact_type_candidate in seen:
                continue
            match = _match_flag(flag, text)
            if match is None:
                continue
            snippet_index, snippet, raw_value, matches = match
            seen.add(flag.fact_type_candidate)
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=flag.fact_type_candidate,
                    raw_field_name=flag.fact_type_candidate,
                    raw_value=raw_value,
                    source_locator=f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                    support_type=SupportType.DETERMINISTICALLY_DERIVED,
                    evidence_quote=snippet,
                    confidence_metadata={
                        "detector": "keyword_text_search_flag",
                        "matched_terms": list(matches),
                        "description": flag.description,
                    },
                )
            )
    return facts


def detect_controlled_search_facts(
    texts: Iterable[tuple[str, str]],
    *,
    locator_prefix: str,
    active_flags: frozenset[str],
) -> list[RawFactCandidate]:
    """Extract projectMetadata controlled-search values activated by flags.

    Matches are literal source substrings. When more than one unique match
    is found for a term, the raw value is joined with " | " in first-seen
    source order, matching the FAIRe spreadsheet convention the user
    supplied.
    """
    reusable_texts = tuple(texts)
    facts: list[RawFactCandidate] = []
    for field in CONTROLLED_SEARCH_FIELDS:
        if field.required_any_flags and not (field.required_any_flags & active_flags):
            continue
        values, evidence_quotes, match_metadata = _match_controlled_field(
            field,
            reusable_texts,
            locator_prefix,
        )
        if not values:
            continue
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field.term_name,
                raw_field_name=field.term_name,
                raw_value=" | ".join(values),
                source_locator=f"{locator_prefix}:controlled_search:{field.term_name}",
                support_type=SupportType.DETERMINISTICALLY_DERIVED,
                evidence_quote=" | ".join(evidence_quotes),
                confidence_metadata={
                    "detector": "controlled_text_search",
                    "section": field.section,
                    "description": field.description,
                    "activated_by_flags": sorted(field.required_any_flags & active_flags),
                    "matches": match_metadata,
                },
            )
        )
    return facts


# A real live audit (10.1093/ismejo/wrae013, STUDY-295abf4a8f43) found a
# BioSample-submitted "elevation = 34 m" attribute for a benthic sediment
# sample, while the paper's own text says "at a site with 34 m water
# depth". FAIRe's own elev definition is "height above ... mean sea
# level ... i.e. 0m for seawater sample" -- a nonzero elev on a seafloor
# sample is itself a strong signal of exactly this kind of real-world
# BioSample submitter error (entering water/sample depth under the
# "elevation" field). Per an explicit user request, this is verified with
# a genuine LLM judgement call (not a blanket regex substitution) and
# ONLY invoked by the caller when the sample is soil/sediment AND elev is
# populated AND depth is not -- see extraction/api_verification.py, the
# only caller.
def confirm_value_described_as_depth(
    backend: LLMBackend,
    numeric_value: str,
    section_texts: Iterable[tuple[str, str]],
    *,
    max_candidates: int = 10,
) -> str | None:
    """Returns the confirming quote (verbatim, from the source text) if
    the paper's own text explicitly ties `numeric_value` to a water/sample
    depth concept, or None if no candidate sentence supports that."""
    value_pattern = re.compile(rf"(?<!\d){re.escape(numeric_value)}(?!\d)")
    depth_phrase_pattern = re.compile(
        r"\b(?:water\s+depth|depth\s+of|m\s+depth|meters?\s+depth|deep\s+water|"
        r"water\s+column\s+depth|sampling\s+depth|site\s+depth)\b",
        re.IGNORECASE,
    )
    candidates: list[str] = []
    seen: set[str] = set()
    for _title, text in section_texts:
        for _index, sentence in _snippets(text):
            if sentence in seen:
                continue
            if value_pattern.search(sentence) and depth_phrase_pattern.search(sentence):
                seen.add(sentence)
                candidates.append(sentence)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        return None

    numbered = "\n".join(f"Q{index + 1:03d}: {quote}" for index, quote in enumerate(candidates))
    prompt = f"""A structured database record for a sediment/soil sample reports its "elevation" as {numeric_value} m. \
This is very likely a data-entry error -- elevation is a site's height above sea level (almost always 0m for an \
underwater site), so a real "{numeric_value} m" value under "elevation" for a seafloor sample usually means a \
submitter actually recorded WATER DEPTH or SAMPLE DEPTH there instead.

Below are candidate sentences from the paper's own text that mention the number {numeric_value}. Determine whether \
any sentence explicitly confirms this number as the site's water depth, sampling depth, or a similar "how deep \
below the surface" measurement -- NOT elevation, altitude, or height above sea level.

Return ONLY a JSON object: {{"confirmed": true or false, "quote_id": "<id of the confirming quote, or empty string>"}}

Candidate quotes:
{numbered}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You verify whether a specific number in a paper's own text confirms a suspected data-labeling error.",
        temperature=0,
        max_tokens=128,
    )
    if not isinstance(parsed, dict) or not parsed.get("confirmed"):
        return None
    match = re.match(r"Q0*(\d+)$", str(parsed.get("quote_id") or "").strip())
    if not match:
        return None
    index = int(match.group(1)) - 1
    if 0 <= index < len(candidates):
        return candidates[index]
    return None
