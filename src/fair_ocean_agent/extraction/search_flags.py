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
            "two-round PCR, two-round PCR amplification strategy, "
            "second PCR, PCR2, indexing PCR, barcode PCR, or adapter PCR to two-step PCR. Map "
            "ligation-based, adapter ligation, barcode ligation, ligated adapters, ligation sequencing kit, "
            "or library adapters were ligated to ligation-based. Omit this field if the quote only mentions "
            "ordinary PCR without a library/barcoding/indexing context; a separate fallback handles one-step PCR."
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
            "A short, machine-readable name identifying a specific assay or primer/probe set used in the study."
        ),
        output_instructions=(
            "Return the published assay name when one exists. Otherwise generate a stable concise name from the "
            "marker and primer/probe set, such as 16S-V4, 515F/806R, or 12S-V5. Return only the name, not a "
            "sentence. If multiple distinct assay/primer/probe sets are explicitly supported, return one object "
            "per value; final merged output is pipe-delimited."
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
            "515F/806R",
            "515F-Y/926R",
            "341F/785R",
            "341F/805R",
            "TAReuk454FWD1/TAReukREV3",
            "1389F/EukB",
            "mlCOIintF/jgHCO2198",
            "LCO1490/HCO2198",
            "ITS1F/ITS2",
            "fITS7/ITS4",
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
            "primer set",
            "primer pair",
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
            "itself states the target, not just the assay name)."
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
        term_name="lib_screen",
        section="Library preparation sequencing",
        description=(
            "Wet-lab methods used to screen, enrich, clean, size-select, quantify, normalize, "
            "or otherwise check prepared libraries before or after library creation. Not "
            "bioinformatic filtering of sequencing reads."
        ),
        output_instructions=(
            "Return the best explicit sentence or phrase for wet-lab library screening/QC/enrichment/cleanup/"
            "size-selection/quantification/normalization/pooling/dilution/loading. Do not return read filtering, "
            "quality trimming, denoising, or other bioinformatic filtering of sequencing reads."
        ),
        search_terms=(
            "library screening",
            "library QC",
            "library quality control",
            "library enrichment",
            "size selection",
            "size-selected",
            "fragment selection",
            "selected for fragments",
            "library purification",
            "library cleanup",
            "purified libraries",
            "cleaned libraries",
            "AMPure",
            "AMPure XP",
            "SPRI beads",
            "SPRIselect",
            "magnetic bead purification",
            "Pippin Prep",
            "BluePippin",
            "E-Gel SizeSelect",
            "gel extraction",
            "gel purification",
            "QIAquick PCR Purification Kit",
            "MinElute",
            "DNA Clean & Concentrator",
            "Bioanalyzer",
            "TapeStation",
            "Fragment Analyzer",
            "LabChip",
            "Qubit",
            "PicoGreen",
            "Quant-iT",
            "fluorometric quantification",
            "KAPA Library Quantification Kit",
            "library qPCR",
            "quantified by qPCR",
            "library concentration",
            "library molarity",
            "fragment distribution",
            "fragment profile",
            "normalized",
            "library normalization",
            "equimolar",
            "equimolar pooling",
            "pooled libraries",
            "library pooling",
            "diluted to",
            "loaded onto the flow cell",
            "target enrichment",
            "hybridization capture",
            "probe capture",
            "capture enrichment",
        ),
    ),
    LLMJudgedSearchField(
        term_name="adapter_forward",
        section="Library preparation sequencing",
        description="Forward sequencing adapter sequence.",
        output_instructions=(
            "Return only an explicit forward/read 1/P5/5-prime adapter sequence. Copy the sequence exactly "
            "from the quote, preserving letters and order. Omit this field if the quote names an adapter but "
            "does not give the sequence."
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
        ),
    ),
    LLMJudgedSearchField(
        term_name="adapter_reverse",
        section="Library preparation sequencing",
        description="Reverse sequencing adapter sequence.",
        output_instructions=(
            "Return only an explicit reverse/read 2/P7/3-prime adapter sequence. Copy the sequence exactly "
            "from the quote, preserving letters and order. Omit this field if the quote names an adapter but "
            "does not give the sequence."
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
        ),
    ),
    LLMJudgedSearchField(
        term_name="trim_method",
        section="Bioinformatics",
        description=(
            "Software or named method used specifically to remove PCR primers, sequencing adapters, "
            "adapter tails/overhangs, or other technical sequences from sequencing reads before downstream analysis."
        ),
        output_instructions=(
            "Return the tool or named method exactly as reported, ideally including version, such as "
            "Cutadapt v4.2, QIIME 2 q2-cutadapt v2023.5, or Trimmomatic v0.39. If no software is named "
            "but primer/adapter/technical-sequence removal is explicit, return the explicit method phrase. "
            "Do not return general quality filtering, minimum-length filtering, demultiplexing, denoising, "
            "chimera removal, or OTU/ASV clustering. Do not invent a tool from the described operation."
        ),
        search_terms=(
            "Cutadapt",
            "q2-cutadapt",
            "QIIME 2 cutadapt",
            "QIIME2 cutadapt",
            "Trimmomatic",
            "fastp",
            "BBDuk",
            "BBTools",
            "AdapterRemoval",
            "Trim Galore",
            "Atropos",
            "Skewer",
            "Flexbar",
            "Porechop",
            "Dorado trim",
            "Guppy",
            "SeqPrep",
            "USEARCH",
            "VSEARCH",
            "mothur",
            "trim.seqs",
            "OBITools",
            "OBITools3",
            "FASTX Toolkit",
            "fastx_clipper",
            "primer trimming",
            "primer removal",
            "adapter trimming",
            "adapter removal",
            "primers were removed",
            "adapters were removed",
            "primer sequences were removed",
            "adapter sequences were removed",
            "technical sequences were removed",
        ),
    ),
    LLMJudgedSearchField(
        term_name="trim_param",
        section="Bioinformatics",
        description=(
            "Specific settings used when removing primers/adapters/technical sequences, such as allowed "
            "mismatches, error rate, overlap, adapter/primer sequence, discarded untrimmed reads, or indel handling."
        ),
        output_instructions=(
            "Return the parameter/value phrase for the primer/adapter/technical-sequence trimming operation, "
            "preserving source syntax when possible, such as -e 0.1 -O 5 --discard-untrimmed, "
            "ILLUMINACLIP:TruSeq3-PE.fa:2:30:10, maximum 2 primer mismatches, or no indels permitted. "
            "Do not return unrelated quality filtering or length filtering parameters that happen to use the "
            "same software."
        ),
        search_terms=(
            "minimum overlap",
            "min overlap",
            "minimum adapter overlap",
            "maximum error rate",
            "error rate",
            "allowed error rate",
            "allowed mismatches",
            "maximum mismatches",
            "mismatch",
            "discard-untrimmed",
            "discard untrimmed",
            "untrimmed reads discarded",
            "no-indels",
            "indels",
            "anchored adapter",
            "anchored primer",
            "adapter sequence",
            "primer sequence",
            "front adapter",
            "forward adapter",
            "reverse adapter",
            "5' adapter",
            "3' adapter",
            "match-read-wildcards",
            "wildcards",
            "-g",
            "-G",
            "-a",
            "-A",
            "-e",
            "--error-rate",
            "-O",
            "--overlap",
            "--discard-untrimmed",
            "--no-indels",
            "ILLUMINACLIP",
            "seedMismatches",
            "palindromeClipThreshold",
            "simpleClipThreshold",
            "ktrim",
            "mink",
            "hdist",
            "hdist2",
            "adapter_sequence",
            "adapter_sequence_r2",
            "detect_adapter_for_pe",
            "minimum trimmed length",
        ),
    ),
    LLMJudgedSearchField(
        term_name="demux_tool",
        section="Bioinformatics",
        description=(
            "Software and version used to assign multiplexed sequencing reads to their correct "
            "samples using index, barcode, or MID sequences."
        ),
        output_instructions=(
            "Return the detailed demultiplexing pipeline phrase supported by the quote, including software, "
            "version, command, and barcode/index mismatch parameters when stated. If multiple demultiplexing "
            "tools are present, return the best option according to the configured search-term priority."
        ),
        search_terms=(
            "QIIME 2",
            "QIIME2",
            "qiime demux",
            "demux emp-paired",
            "demux emp-single",
            "QIIME",
            "split_libraries_fastq.py",
            "split_libraries.py",
            "bcl2fastq",
            "Illumina bcl2fastq",
            "BCL Convert",
            "bcl-convert",
            "Illumina BCL Convert",
            "MiSeq Reporter",
            "BaseSpace",
            "BaseSpace Sequence Hub",
            "CASAVA",
            "Cutadapt",
            "cutadapt demultiplex",
            "OBITools",
            "ngsfilter",
            "obi ngsfilter",
            "mothur",
            "trim.seqs",
            "USEARCH",
            "VSEARCH",
            "Sabre",
            "deML",
            "Je",
            "Je-demultiplex",
            "fastq-multx",
            "ea-utils",
            "Flexbar",
            "Stacks",
            "process_radtags",
            "Guppy barcoder",
            "guppy_barcoder",
            "Dorado demux",
            "dorado demux",
            "qcat",
            "Porechop",
            "Lima",
            "PacBio Lima",
            "demultiplexed using",
            "demultiplexing software",
            "barcode splitting",
            "index-based separation",
            "reads assigned to samples",
            "barcode mismatch",
            "index mismatch",
            "maximum mismatch",
            "--p-golay-error-correction",
            "--barcode-mismatches",
            "--no-index",
        ),
    ),
    LLMJudgedSearchField(
        term_name="error_rate_tool",
        section="Bioinformatics",
        description=(
            "Software, function, or pipeline step that removes or trims sequencing reads based on a "
            "stated quality or error threshold."
        ),
        output_instructions=(
            "Return the detailed quality/error filtering pipeline phrase supported by the quote, including "
            "software, function, version, and threshold parameters when stated. If multiple tools are present, "
            "return the best option according to the configured search-term priority."
        ),
        search_terms=(
            "DADA2",
            "filterAndTrim",
            "QIIME 2",
            "QIIME2",
            "q2-dada2",
            "dada2 denoise-paired",
            "dada2 denoise-single",
            "QIIME",
            "split_libraries_fastq.py",
            "quality-filter q-score",
            "USEARCH",
            "UPARSE",
            "fastq_filter",
            "VSEARCH",
            "--fastq_filter",
            "Cutadapt",
            "fastp",
            "Trimmomatic",
            "mothur",
            "trim.seqs",
            "BBDuk",
            "BBTools",
            "Sickle",
            "PRINSEQ",
            "FASTX Toolkit",
            "fastq_quality_filter",
            "fastq_quality_trimmer",
            "SolexaQA",
            "DynamicTrim",
            "CLC Genomics Workbench",
            "CLC quality trim",
            "NanoFilt",
            "Filtlong",
            "Dorado",
            "Guppy",
            "PacBio CCS",
            "pbccs",
            "SMRT Link",
            "Trim Galore",
            "Atropos",
            "Skewer",
            "Flexbar",
            "AdapterRemoval",
            "SeqPrep",
            "SOAPnuke",
            "FaQCs",
            "NGS QC Toolkit",
            "maxEE",
            "truncQ",
            "fastq_maxee",
            "fastq_maxee_rate",
            "--p-max-ee-f",
            "--p-max-ee-r",
            "--p-trunc-q",
            "--p-min-quality",
            "--quality-cutoff",
            "qualified_quality_phred",
            "SLIDINGWINDOW",
            "AVGQUAL",
            "qtrim",
            "trimq",
            "min_qscore",
        ),
    ),
    LLMJudgedSearchField(
        term_name="error_rate_type",
        section="Bioinformatics",
        description="Type of quality/error measurement used to decide whether reads or bases should be removed or trimmed.",
        allowed_values=("expected error rate", "Phred score", "quality filtered", "other:"),
        output_instructions=(
            "Classify the quality/error measurement as one of: expected error rate, Phred score, quality filtered, "
            "or other:<source phrase>. "
            "Use expected error rate for expected error/maxEE/max expected error/maxEE-rate parameters. "
            "Use Phred score for Phred/Q-score/Q20/Q25/Q30/quality-score/truncQ/min-quality style thresholds. "
            "Use quality filtered for generic quality-filter/quality-trimmed wording when no more specific "
            "expected-error or Phred/Q-score measurement is stated. "
            "Use other:<source phrase> only when the quote gives a different explicit quality/error measurement. "
            "If multiple measurements are explicitly supported, return one object per value; final merged output "
            "is ordered by the configured search-term priority."
        ),
        search_terms=(
            "expected error",
            "expected errors",
            "expected error rate",
            "maximum expected error",
            "maxEE",
            "Phred score",
            "Phred quality",
            "quality filtered",
            "quality filter",
            "quality filtering",
            "quality trimmed",
            "quality trimming",
            "quality score",
            "Q score",
            "Q-score",
            "Q20",
            "Q25",
            "Q30",
            "bases below Q",
            "reads below Q",
            "truncQ",
            "fastq_maxee",
            "fastq_maxee_rate",
            "qualified_quality_phred",
            "AVGQUAL",
            "SLIDINGWINDOW",
            "trimq",
            "min_qscore",
            "--p-max-ee-f",
            "--p-max-ee-r",
            "--p-trunc-q",
            "--p-min-quality",
        ),
    ),
    LLMJudgedSearchField(
        term_name="chimera_check_method",
        section="Bioinformatics",
        description=(
            "How chimeric PCR sequences were identified or removed, including the approach "
            "(de novo, reference-based, or both) and the software/version used."
        ),
        output_instructions=(
            "Return a compact method phrase supported by the quote, preserving software/command/version and "
            "approach details such as de novo, reference-based, consensus, pooled, per-sample, or reference "
            "database. If multiple distinct chimera-checking methods are explicitly supported, return one "
            "object per value; final merged output is pipe-delimited."
        ),
        search_terms=(
            "chimera",
            "chimeric",
            "chimera removal",
            "chimera checking",
            "chimera detection",
            "remove chimeras",
            "DADA2",
            "removeBimeraDenovo",
            "removeBimeraDenovo()",
            "QIIME 2 DADA2",
            "q2-dada2",
            "dada2 denoise-paired",
            "dada2 denoise-single",
            "VSEARCH",
            "uchime_denovo",
            "uchime_ref",
            "--uchime_denovo",
            "--uchime_ref",
            "USEARCH",
            "UCHIME",
            "UCHIME2",
            "UCHIME3",
            "UPARSE",
            "cluster_otus",
            "unoise3",
            "mothur",
            "chimera.vsearch",
            "chimera.uchime",
            "chimera.slayer",
            "QIIME",
            "identify_chimeric_seqs.py",
            "DECIPHER",
            "FindChimeras",
            "FindChimeras()",
            "ChimeraSlayer",
            "Bellerophon",
            "Pintail",
            "de novo",
            "denovo",
            "de-novo",
            "reference-based",
            "reference based",
            "reference-guided",
            "consensus",
            "pooled",
            "per-sample",
            "gold database",
            "Gold database",
            "SILVA reference",
            "RDP reference",
            "UNITE reference",
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
        term_name="min_reads_tool",
        section="Bioinformatics",
        description=(
            "Software and version used to remove low-abundance reads, ASVs, OTUs, or detections "
            "based on a minimum read-count threshold, relative-abundance threshold, or background "
            "detected in blanks."
        ),
        output_instructions=(
            "Return the supported software/version/function/script phrase used for minimum-read, low-abundance, "
            "relative-abundance, singleton/doubleton, prevalence, or blank/background filtering. If the quote "
            "does not name software, return the explicit method phrase. If multiple tools are explicitly "
            "supported, return one object per value; final merged output is pipe-delimited."
        ),
        search_terms=(
            "DADA2",
            "QIIME 2",
            "QIIME2",
            "feature-table filter-features",
            "feature-table filter-samples",
            "QIIME",
            "filter_otus_from_otu_table.py",
            "OBITools3",
            "OBITools",
            "obigrep",
            "obi grep",
            "obiclean",
            "USEARCH",
            "UPARSE",
            "sortbysize",
            "minsize",
            "VSEARCH",
            "--sortbysize",
            "--minsize",
            "mothur",
            "remove.rare",
            "remove.seqs",
            "phyloseq",
            "prune_taxa",
            "filter_taxa",
            "decontam",
            "isContaminant",
            "microDecon",
            "metabaR",
            "LULU",
            "R",
            "custom R script",
            "custom script",
            "Python script",
            "minimum read count",
            "minimum reads",
            "read-count threshold",
            "low-abundance sequences",
            "low abundance ASVs",
            "low abundance OTUs",
            "rare ASVs",
            "rare OTUs",
            "rare sequences",
            "singletons removed",
            "doubletons removed",
            "singleton filtering",
            "fewer than 10 reads",
            "less than 10 reads",
            "<10 reads",
            "relative abundance threshold",
            "relative read abundance",
            "% abundance cutoff",
            "noise detected in blanks",
            "blank threshold",
            "control threshold",
            "background reads",
            "removed if present below",
            "discarded below",
            "filtered below",
            "min_count",
            "min_reads",
            "abundance_threshold",
            "prevalence",
            "threshold",
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
            "stated. Accept ASV inference/denoising tools such as DADA2 or Deblur when the quote says ASVs/"
            "exact sequence variants were inferred/generated/denoised. Omit taxonomy-only classifiers or "
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
        term_name="otu_clust_cutoff",
        section="Bioinformatics",
        description=(
            "Sequence-similarity percentage used to group sequences into the same OTU. For ASV workflows, "
            "FAIRe treats exact variants as approximately 100% similarity, but do not claim the paper "
            "explicitly reported 100 unless this is a derived ASV normalization policy."
        ),
        output_instructions=(
            "Return only the numeric percent value, without a percent sign. Map 97% similarity or --id 0.97 "
            "to 97; 0.03 distance or 3% divergence to 97; 0.01 distance to 99. For DADA2/ASVs/exact sequence "
            "variants with no stated similarity threshold, omit this field unless the quote explicitly supports "
            "a derived ASV-as-100 normalization; if returned, use 100. If multiple cutoffs are present, return "
            "the best option according to the configured search-term priority."
        ),
        search_terms=(
            "OTU clustering cutoff",
            "OTU similarity threshold",
            "clustering threshold",
            "97% similarity",
            "99% similarity",
            "98% similarity",
            "95% similarity",
            "100% similarity",
            "sequence identity threshold",
            "percent identity",
            "percentage identity",
            "clustered at 97%",
            "clustered at 99%",
            "grouped at 97%",
            "operational taxonomic units",
            "OTUs were clustered",
            "de novo clustering",
            "closed-reference clustering",
            "open-reference clustering",
            "ASV",
            "ASVs",
            "amplicon sequence variant",
            "exact sequence variant",
            "100% identity",
            "distance cutoff",
            "genetic distance",
            "0.03 distance",
            "3% divergence",
            "--p-perc-identity",
            "--id",
            "id=",
            "--id 0.97",
            "cutoff=0.03",
            "similarity=0.97",
            "threshold=0.97",
            "-c 0.97",
            "radius",
            "d=1",
        ),
    ),
    LLMJudgedSearchField(
        term_name="otu_db",
        section="Bioinformatics",
        description=(
            "Reference sequence database, including version/release/date when reported, used to assign "
            "taxonomy to OTUs or ASVs. If authors built their own database, record custom."
        ),
        output_instructions=(
            "Return the database name plus version/release/download/access date when stated. Return custom "
            "for a custom/in-house/local/curated database. Do not return assignment software alone, such as "
            "BLAST, RDP Classifier, QIIME 2, or naive Bayes, unless a reference database is also named."
        ),
        search_terms=(
            "reference database",
            "taxonomy database",
            "taxonomic database",
            "sequence database",
            "SILVA",
            "SILVA database",
            "PR2",
            "Protist Ribosomal Reference database",
            "NCBI GenBank",
            "GenBank",
            "NCBI nucleotide",
            "NCBI nt",
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
    LLMJudgedSearchField(
        term_name="tax_assign_cat",
        section="Bioinformatics",
        description=(
            "High-level computational approach used to assign taxonomy to sequences, OTUs, or ASVs. "
            "This is a CLASSIFICATION into one of a fixed set of categories, not a quote -- the real "
            "FAIRe field is a controlled enum, and a paper virtually never states the category name "
            "itself (it names a tool like CREST4/BLAST/VSEARCH instead)."
        ),
        # allowed_values are the real tax_assign_cat_enum members
        # (schemas/faire/enums.yaml), including the literal trailing colon
        # on 'other:' that field expects.
        allowed_values=("sequence similarity", "sequence composition", "phylogeny", "probabilistic", "other:"),
        output_instructions=(
            "Classify the method used to assign taxonomy to OTUs/ASVs/features into one FAIRe category: "
            "sequence similarity, sequence composition, phylogeny, probabilistic, or other. Base the "
            "classification on the actual taxonomic-assignment algorithm or software, not on the reference "
            "database. A method that compares query sequences directly with reference sequences based on "
            "alignment or sequence identity is sequence similarity; a classifier based on sequence "
            "composition or k-mer features is sequence composition; placement into a reference phylogeny "
            "is phylogeny; explicitly probabilistic taxonomic inference is probabilistic. Return ONLY one "
            "of: sequence similarity, sequence composition, phylogeny, probabilistic, other: <description> "
            "-- never a quoted sentence."
        ),
        search_terms=(
            "BLAST",
            "BLASTn",
            "MegaBLAST",
            "megablast",
            "VSEARCH",
            "USEARCH",
            "CREST",
            "CREST4",
            "taxonomic classification",
            "taxonomic assignment",
            "taxonomy assigned",
            "assigned taxonomy",
            "taxonomic identification",
            "classified taxonomically",
            "global alignment",
            "sequence similarity",
            "naive Bayes",
            "Naive Bayesian",
            "naive Bayes classifier",
            "QIIME 2 feature-classifier",
            "classify-sklearn",
            "q2-feature-classifier",
            "RDP Classifier",
            "Ribosomal Database Project classifier",
            "SINTAX",
            "Kraken",
            "Kraken2",
            "CLARK",
            "CLARK-S",
            "SEPP",
            "fragment-insertion",
            "q2-fragment-insertion",
            "EPA-ng",
            "EPA",
            "evolutionary placement",
            "pplacer",
            "PROTAX",
            "TIPP",
            "IDTAXA",
            "DECIPHER IdTaxa",
            "lowest common ancestor",
            "LCA",
            "nearest neighbor",
            "nearest-neighbour",
            "best hit",
            "top hit",
            "sequence identity",
            "percent identity",
            "percentage identity",
            "k-mer",
            "kmer",
            "k-mer classifier",
            "sequence composition",
            "phylogenetic placement",
            "phylogenetic assignment",
            "reference tree",
            "probabilistic taxonomic assignment",
            "posterior probability",
        ),
    ),
    # tax_class_other used to be a quote-judged field here -- replaced per
    # an explicit user request ("tax_class_other can be all classified
    # 'TAXONOMIC ASSIGNMENT'. can ask the LLM to summarize based on the
    # section classified 'TAXONOMIC ASSIGNMENT'.") with a generative
    # mechanism: extraction/section_category_extraction.py's
    # _generate_tax_class_other_fact, which summarizes Stage 2's already-
    # classified taxonomic_assignment run-text in the model's own words
    # instead of quoting one narrow sentence verbatim.
    # otu_raw_description used to be a quote-judged field here -- replaced
    # per an explicit user request ("i'd prefer if the LLM generates 1-2
    # sentences of its own description of the OTU process, the quotes
    # captured are not meaningful for either paper") with a generative
    # mechanism: extraction/section_category_extraction.py's
    # _generate_otu_raw_description_fact, which summarizes the already-
    # categorized otu_asv_generation_filtering run-text in the model's own
    # words instead of quoting a real paper's often-unhelpful cross-
    # reference sentence (e.g. "we ... employed the same data analysis
    # pipeline" with no further detail) verbatim.
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
        search_terms=("control", "controls", "blank", "blanks"),
    ),
)

SINGLE_BEST_LLM_JUDGED_FIELDS = frozenset(
    {
        "demux_tool",
        "error_rate_tool",
        "inhibition_check_0_1",
        "inhibition_check",
        "lib_screen",
        "trim_method",
        "otu_clust_tool",
        "otu_clust_cutoff",
        "otu_db",
        "tax_assign_cat",
        "neg_cont_0_1",
        "pos_cont_0_1",
    }
)

CONTROLLED_SEARCH_FIELDS: tuple[ControlledSearchField, ...] = (
    ControlledSearchField(
        term_name="sterilise_method",
        section="Project",
        description="Explicit contamination-minimization procedures, retained as direct source text.",
        value_strategy="sterilise_method_sentences",
        search_terms=(
            "contamination control",
            "single-use equipment",
            "separate pre-PCR",
            "decontamination",
            "decontaminate",
            "cleaned with",
            "rinsed with",
            "sodium hypochlorite",
            "flame sterilised",
            "sterilise",
            "sterilize",
            "DNA Away",
            "DNAZap",
            "clean room",
            "bleach",
            "ethanol",
            "UV-C",
            "autoclave",
            "UV",
            "sterile bags",
            "sterile tubes",
            "sterile syringes",
            "sterile containers",
            "sterile vials",
            "sterile bottles",
        ),
    ),
    ControlledSearchField(
        term_name="biological_rep",
        section="Sample collection",
        description=(
            "Number of independently collected biological/environmental samples at each sampling "
            "point or treatment; never PCR, extraction, filtration, sequencing, or analytical replicates."
        ),
        value_strategy="biological_replicate_integer",
        search_terms=(
            "biological replicates",
            "biological replicate",
            "independent replicate",
            "replicate samples",
            "samples per site",
            "samples per station",
            "replicates per site",
            "replicates per treatment",
            "collected in duplicate",
            "collected in triplicate",
            "three independent samples",
            "replicate water samples",
            "replicate sediment samples",
            "n =",
            "n=",
        ),
    ),
    ControlledSearchField(
        term_name="biological_rep_presence",
        section="Sample collection",
        description="Explicit source statement that biological/environmental replicates were or were not present.",
        value_strategy="biological_replicate_presence",
        search_terms=(
            "without replicates",
            "no replicates",
            "no biological replicates",
            "no environmental replicates",
            "without biological replicates",
            "without environmental replicates",
        ),
    ),
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
        term_name="library_layout",
        section="Library preparation sequencing",
        description="single-end or paired-end sequencing read layout.",
        value_strategy="library_layout_phrase",
        search_terms=(
            "paired-end",
            "paired end",
            "paired-end reads",
            "paired end reads",
            "2 x 300",
            "2x300",
            "2 × 300",
            "2 x 250",
            "2x250",
            "2 × 250",
            "single-end",
            "single end",
            "single-end reads",
            "single end reads",
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
        search_terms=("MINLEN", "minimum read length", "reads below", "reads shorter than"),
    ),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BIOLOGICAL_REPLICATE_EXCLUSION_RE = re.compile(
    r"\b(PCR|technical|extraction|filtration|sequencing|analytical|library|qPCR|ddPCR|well)\s+replicates?\b",
    re.IGNORECASE,
)
_BIOLOGICAL_REPLICATE_PATTERNS: tuple[tuple[re.Pattern[str], dict[str, str]], ...] = (
    (re.compile(r"\bcollected\s+in\s+duplicate\b", re.IGNORECASE), {"value": "2"}),
    (re.compile(r"\bcollected\s+in\s+triplicate\b", re.IGNORECASE), {"value": "3"}),
    (re.compile(r"\bthree\s+independent\s+samples?\b", re.IGNORECASE), {"value": "3"}),
    (re.compile(r"\btwo\s+independent\s+samples?\b", re.IGNORECASE), {"value": "2"}),
    (re.compile(r"\b(?P<num>\d+)\s+(?:biological|environmental|independent)\s+replicates?\b", re.IGNORECASE), {}),
    (re.compile(r"\b(?:biological|environmental|independent)\s+replicates?\s*(?:\(?\s*n\s*=\s*)?(?P<num>\d+)\b", re.IGNORECASE), {}),
    (re.compile(r"\b(?P<num>\d+)\s+replicate\s+(?:water|sediment|environmental\s+)?samples?\b", re.IGNORECASE), {}),
    (re.compile(r"\breplicate\s+(?:water|sediment|environmental\s+)?samples?\s*(?:\(?\s*n\s*=\s*)?(?P<num>\d+)\b", re.IGNORECASE), {}),
    (re.compile(r"\b(?P<num>\d+)\s+samples?\s+per\s+(?:site|station|treatment)\b", re.IGNORECASE), {}),
    (re.compile(r"\bsamples?\s+per\s+(?:site|station|treatment)\s*(?:=|:|was|were)?\s*(?P<num>\d+)\b", re.IGNORECASE), {}),
    (re.compile(r"\b(?P<num>\d+)\s+replicates?\s+per\s+(?:site|station|treatment)\b", re.IGNORECASE), {}),
    (re.compile(r"\breplicates?\s+per\s+(?:site|station|treatment)\s*(?:=|:|was|were)?\s*(?P<num>\d+)\b", re.IGNORECASE), {}),
    # A bare "n = <number>" / "N = <number>" pattern (gated only by a loose
    # context check requiring the word "sample(s)" ANYWHERE in the same
    # sentence) used to live here and was removed: confirmed via
    # a real gold paper (PeerJ 10.7717/peerj.333) that it produced two
    # unrelated false positives in the same methods paragraph -- "(n = 4
    # well replicates per cue...)" (a technical well count, not a biological
    # replicate) and "...x N-72 C 10 min, with N = 17-24 depending on the
    # sample" (a PCR cycle count) -- both satisfied the loose context check
    # via the bare word "sample"/"cue...samples" appearing elsewhere in the
    # sentence, producing a nonsensical joined raw_value ("4 | 17"). No
    # simple proximity regex reliably distinguishes a genuine "(n = 4
    # biological replicates)" from these unrelated "n ="/"N =" usages that
    # are extremely common in PCR/qPCR methods prose (well counts, cycle
    # counts, dilution series); removed rather than further loosened or
    # tightened. The LLM's unconditional biological_replicate_count
    # checklist field remains the complementary source for well-phrased
    # cases this narrower deterministic set no longer covers.
)
_BIOLOGICAL_REPLICATE_NEGATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwithout\s+(?:biological\s+|environmental\s+)?replicates?\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:biological\s+|environmental\s+)?replicates?\b", re.IGNORECASE),
)

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
        ),
    ),
)
_SEQUENCING_KIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bMiSeq\s+Reagent\s+Kit(?:\s+v\d+)?\b", re.IGNORECASE),
    re.compile(r"\bNextera\s+XT\s+Index\s+Kit(?:\s*\([^)]+\))?", re.IGNORECASE),
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


def _snippets(text: str) -> Iterable[tuple[int, str]]:
    """Yield short candidate evidence snippets with stable local positions."""
    normalized = " ".join(text.split())
    if not normalized:
        return
    for index, sentence in enumerate(_SENTENCE_SPLIT_RE.split(normalized)):
        cleaned = sentence.strip()
        if cleaned:
            yield index, cleaned


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
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


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


_STERILE_CONTAINER_TOOL_RE = re.compile(
    r"\bsterile\s+(?:[\w.+\-]+\s+){0,5}?"
    r"(?:bags?|tubes?|syringes?|containers?|vials?|bottles?)\b",
    re.IGNORECASE,
)


def _match_sterilise_method_sentences(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    values, evidence_quotes, match_metadata = _match_controlled_sentences(field, texts, locator_prefix)
    seen_sentences = {value.casefold() for value in values}
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            if snippet.casefold() in seen_sentences or not _STERILE_CONTAINER_TOOL_RE.search(snippet):
                continue
            seen_sentences.add(snippet.casefold())
            values.append(snippet)
            evidence_quotes.append(snippet)
            match_metadata.append(
                {
                    "matched_terms": ["sterile container/tool"],
                    "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                }
            )
    unique_values: list[str] = []
    unique_metadata: list[dict] = []
    seen_unique: set[str] = set()
    for value, metadata in zip(values, match_metadata):
        key = value.casefold()
        if key in seen_unique:
            continue
        seen_unique.add(key)
        unique_values.append(value)
        unique_metadata.append(metadata)
    return unique_values, unique_values, unique_metadata


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


def _match_primer_phrase(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
    *,
    direction: str,
    value_kind: str,
) -> tuple[list[str], list[str], list[dict]]:
    values: list[str] = []
    evidence_quotes: list[str] = []
    match_metadata: list[dict] = []
    seen_values: set[str] = set()

    for title, text in texts:
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

    return values, evidence_quotes, match_metadata


def _match_biological_replicates(
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
            if _BIOLOGICAL_REPLICATE_EXCLUSION_RE.search(snippet):
                continue
            for pattern, options in _BIOLOGICAL_REPLICATE_PATTERNS:
                match = pattern.search(snippet)
                if match is None:
                    continue
                value = options.get("value") or match.group("num")
                if value in seen_values:
                    continue
                seen_values.add(value)
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


def _match_biological_replicate_presence(
    field: ControlledSearchField,
    texts: Iterable[tuple[str, str]],
    locator_prefix: str,
) -> tuple[list[str], list[str], list[dict]]:
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            if _BIOLOGICAL_REPLICATE_EXCLUSION_RE.search(snippet):
                continue
            for pattern in _BIOLOGICAL_REPLICATE_NEGATIVE_PATTERNS:
                match = pattern.search(snippet)
                if match is None:
                    continue
                return (
                    ["FALSE"],
                    [snippet],
                    [
                        {
                            "matched_value": "FALSE",
                            "matched_pattern": pattern.pattern,
                            "source_locator": f"{locator_prefix}:{title}:sentence[{snippet_index}]",
                        }
                    ],
                )
    return [], [], []


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
            values.append(snippet)
            evidence_quotes.append(snippet)
            match_metadata.append(
                {
                    "matched_value": snippet,
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


_PAIRED_END_LAYOUT_PATTERNS = (
    re.compile(r"\bpaired[-\s]+end(?:\s+reads?)?\b", re.IGNORECASE),
    re.compile(r"\b2\s*(?:x|×)\s*(?:150|250|300)\b", re.IGNORECASE),
)
_SINGLE_END_LAYOUT_PATTERNS = (
    re.compile(r"\bsingle[-\s]+end(?:\s+reads?)?\b", re.IGNORECASE),
)


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


def _match_library_layout(
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
            for value, patterns in (
                ("paired end", _PAIRED_END_LAYOUT_PATTERNS),
                ("single end", _SINGLE_END_LAYOUT_PATTERNS),
            ):
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
    if field.value_strategy == "sterilise_method_sentences":
        return _match_sterilise_method_sentences(field, texts, locator_prefix)
    if field.value_strategy == "biological_replicate_integer":
        return _match_biological_replicates(field, texts, locator_prefix)
    if field.value_strategy == "biological_replicate_presence":
        return _match_biological_replicate_presence(field, texts, locator_prefix)
    if field.value_strategy == "assay_type_classifier":
        return _classify_assay_type(field, texts, locator_prefix)
    if field.value_strategy == "commercial_pcr_mixture_phrase":
        return _match_pcr_mixture_phrase(field, texts, locator_prefix, classification="commercial")
    if field.value_strategy == "custom_pcr_mixture_phrase":
        return _match_pcr_mixture_phrase(field, texts, locator_prefix, classification="custom")
    if field.value_strategy == "sequencing_kit_phrase":
        return _match_regex_phrases(field, texts, locator_prefix, _SEQUENCING_KIT_PATTERNS)
    if field.value_strategy == "library_layout_phrase":
        return _match_library_layout(field, texts, locator_prefix)
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


_MIN_READS_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"minimum\s+reads?|read[-\s]+count\s+threshold|low[-\s]+abundance|"
    r"rare\s+(?:ASVs?|OTUs?|sequences?)|singletons?|doubletons?|"
    r"fewer\s+than\s+\d+\s+reads?|less\s+than\s+\d+\s+reads?|"
    r"relative\s+(?:read\s+)?abundance|blank\s+threshold|control\s+threshold|"
    r"background\s+reads?|noise\s+detected\s+in\s+blanks|"
    r"removed\s+if\s+present\s+below|discarded\s+below|filtered\s+below|"
    r"minsize|min_count|min_reads|abundance_threshold|prevalence|"
    r"prune_taxa|filter_taxa|isContaminant|decontam|microDecon|metabaR|LULU"
    r")\b|<\s*\d+\s+reads?",
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
_OTU_CLUSTER_CUTOFF_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"OTU\s+clustering\s+cutoff|OTU\s+similarity\s+threshold|clustering\s+threshold|"
    r"\d+(?:\.\d+)?%\s+(?:similarity|identity|divergence)|"
    r"sequence\s+identity\s+threshold|percent(?:age)?\s+identity|"
    r"clustered\s+at\s+\d+|grouped\s+at\s+\d+|"
    r"OTUs?\s+were\s+clustered|(?:de\s+novo|closed-reference|open-reference)\s+clustering|"
    r"100%\s+identity|distance\s+cutoff|genetic\s+distance|"
    r"0\.\d+\s+distance|\d+%\s+divergence|"
    r"--p-perc-identity|--id|id=|cutoff=|similarity=|threshold=|-c\s+0\.\d+|d=1"
    r")\b",
    re.IGNORECASE,
)
_CHIMERA_REQUIRED_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"chimeras?|chimeric|bimeras?|spurious\s+biological\s+variants?|"
    r"fusion\s+transcripts?|gene\s+fusions?|hybrid\s+genes?|intergenic\s+splicing|"
    r"PCR\s+artifacts?|hybrid\s+sequences?|hybrid\s+amplicons?|"
    r"artificial\s+recombinants?|split\s+reads?|supplementary\s+alignments?"
    r")\b",
    re.IGNORECASE,
)
_TRIM_METHOD_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"primer\s+trimming|primer\s+removal|primers\s+were\s+removed|"
    r"primer\s+sequences\s+(?:were\s+)?(?:removed|trimmed)|trimmed\s+primers?|"
    r"primer\s+clipping|adapter\s+trimming|adapter\s+removal|"
    r"adapters\s+were\s+removed|adapter\s+sequences\s+(?:were\s+)?(?:removed|trimmed)|"
    r"trimmed\s+adapters?|adapter\s+clipping|adaptor\s+trimming|adaptor\s+removal|"
    r"sequencing\s+adapter\s+removal|sequencing\s+adapters\s+removed|"
    r"PCR\s+primer\s+removal|amplicon\s+primer\s+removal|"
    r"technical\s+sequence\s+removal|technical\s+sequences\s+removed|"
    r"fusion\s+primer\s+removal|primer\s+tail\s+removal|adapter\s+tail\s+removal|"
    r"overhang\s+removal|sequencing\s+tail\s+removal|barcode\s+removal|"
    r"barcodes\s+were\s+trimmed"
    r")\b",
    re.IGNORECASE,
)
_TRIM_PARAM_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"primer\s+trimming|primer\s+removal|primer\s+sequences\s+(?:were\s+)?(?:removed|trimmed)|"
    r"adapter\s+trimming|adapter\s+removal|adapter\s+sequences\s+(?:were\s+)?(?:removed|trimmed)|"
    r"adaptor\s+trimming|adaptor\s+removal|sequencing\s+adapter\s+removal|"
    r"PCR\s+primer\s+removal|amplicon\s+primer\s+removal|fusion\s+primer\s+removal|"
    r"primer\s+tail\s+removal|adapter\s+tail\s+removal|overhang\s+removal|"
    r"technical\s+sequence\s+removal|barcode\s+removal|"
    r"Cutadapt\s+was\s+used\s+to\s+remove|q2-cutadapt\s+was\s+used\s+to\s+trim|"
    r"ILLUMINACLIP|adapter_sequence|ktrim"
    r")\b",
    re.IGNORECASE,
)
_OTU_DB_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"(?:reference|taxonomy|taxonomic|sequence|barcode)\s+(?:database|library)|"
    r"SILVA|PR2|Protist\s+Ribosomal\s+Reference|GenBank|NCBI\s+(?:nucleotide|nt)|"
    r"nt\s+database|BOLD|Barcode\s+of\s+Life|UNITE|Ribosomal\s+Database\s+Project|"
    r"Greengenes2?|MIDORI2?|MitoFish|MetaZooGene|Diat\.?Barcode|PhytoREF|"
    r"MaarjAM|GTDB|Genome\s+Taxonomy\s+Database|MitoZoa|EMBL|ENA\s+reference|"
    r"(?:custom|curated|in-house|local)\s+(?:reference\s+)?database|"
    r"(?:reference|sequence|barcode)\s+library|classifier\s+trained\s+on|"
    r"(?:trained|pretrained)\s+classifier"
    r")\b",
    re.IGNORECASE,
)
_TAX_ASSIGNMENT_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"taxonomy|taxonomic|taxon(?:\s+assignment)?|taxonomic\s+assignment|"
    r"assigned\s+taxonomy|taxonomy\s+was\s+assigned|assign\s+taxonomy|"
    r"taxonomy\s+assignment|taxonomic\s+classification|classified\s+taxonomically|"
    r"taxonomic\s+identification|identified\s+taxonomically|sequence\s+identification|"
    r"species\s+identification|taxonomic\s+annotation|annotated\s+against|"
    r"classified\s+against|matched\s+against\s+a\s+reference\s+database|"
    r"(?:OTUs?|ASVs?)\s+were\s+(?:assigned|classified)|"
    r"sequences\s+were\s+(?:classified|identified|assigned)|"
    r"classified\s+sequences|reference\s+taxonomy|reference\s+database|classifier"
    r")\b",
    re.IGNORECASE,
)
_OTU_DB_VALUE_RE = re.compile(
    r"\b(?:"
    r"SILVA|PR2|Protist\s+Ribosomal\s+Reference|GenBank|NCBI\s+(?:nucleotide|nt)|"
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


def _llm_judged_field_matches_snippet(field: LLMJudgedSearchField, snippet: str) -> bool:
    if not any(_term_pattern(term).search(snippet) for term in field.search_terms):
        return False
    if field.term_name == "min_reads_tool":
        return bool(_MIN_READS_CONTEXT_RE.search(snippet))
    if field.term_name == "otu_clust_tool":
        return bool(_OTU_CLUSTER_TOOL_CONTEXT_RE.search(snippet))
    if field.term_name == "otu_clust_cutoff":
        return bool(_OTU_CLUSTER_CUTOFF_CONTEXT_RE.search(snippet))
    if field.term_name == "chimera_check_method":
        return bool(_CHIMERA_REQUIRED_CONTEXT_RE.search(snippet))
    if field.term_name == "trim_method":
        return bool(_TRIM_METHOD_CONTEXT_RE.search(snippet))
    if field.term_name == "trim_param":
        return bool(_TRIM_PARAM_CONTEXT_RE.search(snippet))
    if field.term_name == "otu_db":
        return bool(_OTU_DB_CONTEXT_RE.search(snippet))
    if field.term_name == "tax_assign_cat":
        return bool(_TAX_ASSIGNMENT_CONTEXT_RE.search(snippet))
    if field.term_name == "assay_name":
        return bool(_ASSAY_NAME_CONTEXT_RE.search(snippet))
    if field.term_name == "assay_target_taxa":
        return bool(_TARGET_TAXONOMIC_ASSAY_CONTEXT_RE.search(snippet))
    if field.term_name == "study_target_taxonomic_scope":
        return bool(_TARGET_TAXONOMIC_SCOPE_CONTEXT_RE.search(snippet))
    return True


def _candidate_fields_for_snippet(snippet: str, exclude_field_names: frozenset[str] = frozenset()) -> tuple[str, ...]:
    field_names: list[str] = []
    for field in LLM_JUDGED_SEARCH_FIELDS:
        if field.term_name in exclude_field_names:
            continue
        if _llm_judged_field_matches_snippet(field, snippet):
            field_names.append(field.term_name)
    return tuple(field_names)


def quote_candidates_for_llm_judged_search(
    texts: Iterable[tuple[str, str]],
    *,
    max_candidates: int = 40,
    exclude_field_names: frozenset[str] = frozenset(),
) -> tuple[QuoteCandidate, ...]:
    """Candidate source sentences for the small library-prep judgement LLM.

    The LLM never receives whole sections for these fields: only sentences
    that hit the user-supplied search terms. Final facts are accepted only
    when the model cites one of these quote IDs.
    """
    candidates: list[QuoteCandidate] = []
    seen_text: set[str] = set()
    for title, text in texts:
        for snippet_index, snippet in _snippets(text):
            field_names = _candidate_fields_for_snippet(snippet, exclude_field_names)
            if not field_names or snippet in seen_text:
                continue
            seen_text.add(snippet)
            candidates.append(
                QuoteCandidate(
                    quote_id=f"Q{len(candidates) + 1:03d}",
                    field_names=field_names,
                    title=title,
                    snippet_index=snippet_index,
                    text=snippet,
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
field are explicitly supported, return one object per value, each citing its supporting quote_id.

Fields:
{field_reference}

Return ONLY a JSON array. Each object must be:
{{"field": "<one listed field>", "raw_value": "<supported value>", "quote_id": "Q001"}}

Candidate quotes:
{quotes}
"""


def _allowed_field_lookup() -> dict[str, LLMJudgedSearchField]:
    return {field.term_name: field for field in LLM_JUDGED_SEARCH_FIELDS}


def _valid_llm_judged_value(field: LLMJudgedSearchField, value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if not field.allowed_values:
        return True
    parts = [part.strip() for part in stripped.split("|")]
    return all(
        part in field.allowed_values
        or ("other:" in field.allowed_values and part.casefold().startswith("other:"))
        for part in parts
    )


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
            }
        )

    facts: list[RawFactCandidate] = []
    for field_name, group in grouped.items():
        field = fields[field_name]
        entries = sorted(group["entries"], key=lambda entry: entry["priority"])
        if field_name in SINGLE_BEST_LLM_JUDGED_FIELDS:
            entries = entries[:1]
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value=" | ".join(entry["raw_value"] for entry in entries),
                source_locator=f"{locator_prefix}:llm_judged_search:{field_name}",
                # tax_assign_cat is the one field in this mechanism that's
                # a genuine classification, not a quote -- honestly tagged
                # INFERRED rather than EXPLICIT, per an explicit user
                # confirmation that the category label itself is never
                # actually stated in the source text.
                support_type=SupportType.INFERRED if field_name == "tax_assign_cat" else SupportType.EXPLICIT,
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
    if neg is not None and neg.raw_value == "0" and pos is None:
        facts = [*facts, neg.model_copy(update={"fact_type_candidate": "pos_cont_0_1", "raw_field_name": "pos_cont_0_1"})]
    elif pos is not None and pos.raw_value == "0" and neg is None:
        facts = [*facts, pos.model_copy(update={"fact_type_candidate": "neg_cont_0_1", "raw_field_name": "neg_cont_0_1"})]
    return facts


def _control_not_found_fallback_facts(
    *, locator_prefix: str, existing_fact_types: frozenset[str], exclude_field_names: frozenset[str]
) -> list[RawFactCandidate]:
    """Per an explicit user request ("i also see no mention of +/- controls
    ... I think they should both be 0"): when a paper's text never even
    raises a "control"/"blank" candidate for one or both of neg_cont_0_1/
    pos_cont_0_1, the honest, confident default is "0" (no control used),
    not a blank field indistinguishable from "never checked"."""
    facts: list[RawFactCandidate] = []
    for field_name in ("neg_cont_0_1", "pos_cont_0_1"):
        if field_name in exclude_field_names or field_name in existing_fact_types:
            continue
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value="0",
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
    r"two[- ]step PCR|two[- ]stage PCR|two[- ]round PCR|second PCR|PCR ?2\b|indexing PCR|barcode PCR|adapter PCR",
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


def _chimera_not_recorded_fallback_fact(
    *,
    candidates: tuple[QuoteCandidate, ...],
    locator_prefix: str,
    existing_fact_types: frozenset[str],
    exclude_field_names: frozenset[str],
) -> RawFactCandidate | None:
    if "chimera_check_method" in exclude_field_names or "chimera_check_method" in existing_fact_types:
        return None
    if any("chimera_check_method" in candidate.field_names for candidate in candidates):
        return None
    return RawFactCandidate(
        entity_level=EntityLevel.STUDY,
        fact_type_candidate="chimera_check_method",
        raw_field_name="chimera_check_method",
        raw_value="no chimeric recorded.",
        source_locator=f"{locator_prefix}:llm_judged_search:chimera_check_method:not_recorded_fallback",
        support_type=SupportType.DETERMINISTICALLY_DERIVED,
        evidence_quote=None,
        confidence_metadata={
            "detector": "chimera_required_context_default",
            "description": (
                "No sentence matched the required chimera/artifact/fusion context terms for "
                "chimera_check_method."
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


def detect_llm_judged_search_facts(
    backend: LLMBackend,
    texts: Iterable[tuple[str, str]],
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 512,
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
                    "trim_method",
                    "trim_param",
                    "tax_assign_cat",
                    "assay_target_taxa",
                ),
                candidates=candidates,
                locator_prefix=locator_prefix,
                existing_fact_types=frozenset(),
                exclude_field_names=exclude_field_names,
            )
        )
        chimera_fallback = _chimera_not_recorded_fallback_fact(
            candidates=candidates,
            locator_prefix=locator_prefix,
            existing_fact_types=frozenset(),
            exclude_field_names=exclude_field_names,
        )
        if chimera_fallback:
            facts.append(chimera_fallback)
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
    existing_fact_types = frozenset(fact.fact_type_candidate for fact in facts)
    not_found_fallbacks = _not_found_fallback_facts(
        field_names=(
            "trim_method",
            "trim_param",
            "tax_assign_cat",
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
    chimera_fallback = _chimera_not_recorded_fallback_fact(
        candidates=candidates,
        locator_prefix=locator_prefix,
        existing_fact_types=existing_fact_types,
        exclude_field_names=exclude_field_names,
    )
    if chimera_fallback:
        facts.append(chimera_fallback)
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
        )
    )
    return facts


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
