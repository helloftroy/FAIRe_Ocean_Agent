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
    TextSearchFlag(
        fact_type_candidate="pcr_0_1",
        description="PCR/amplification keyword evidence",
        positive_value="1",
        positive_patterns=(
            re.compile(r"\bPCR\b", re.IGNORECASE),
            re.compile(r"\bqPCR\b", re.IGNORECASE),
            re.compile(r"\bddPCR\b", re.IGNORECASE),
            # \bamplification\b alone missed a real gold case ("...was
            # amplified using primers...") -- amplif\w* also matches
            # amplify/amplified/amplifying/amplifies, every PCR-relevant
            # verb form, with no real false-positive risk (no common
            # English word besides these starts with "amplif").
            re.compile(r"\bamplif\w*\b", re.IGNORECASE),
            re.compile(r"\bpolymerase\s+chain\s+reaction\b", re.IGNORECASE),
        ),
    ),
    TextSearchFlag(
        fact_type_candidate="neg_cont_0_1",
        description="explicit evidence that negative controls were or were not used",
        positive_value="1",
        positive_patterns=(
            re.compile(r"\bnegative\s+controls?\b", re.IGNORECASE),
            re.compile(r"\bfield\s+blanks?\b", re.IGNORECASE),
            re.compile(r"\bequipment\s+blanks?\b", re.IGNORECASE),
            re.compile(r"\bfiltration\s+blanks?\b", re.IGNORECASE),
            re.compile(r"\bextraction\s+blanks?\b", re.IGNORECASE),
            re.compile(r"\breagent\s+blanks?\b", re.IGNORECASE),
            re.compile(r"\bPCR\s+blanks?\b", re.IGNORECASE),
            re.compile(r"\bno[-\s]+template\s+controls?\b", re.IGNORECASE),
            re.compile(r"\bcontrol\s+wells?\b", re.IGNORECASE),
            re.compile(r"\b(?:FSW|filtered\s+seawater)\s+controls?\b", re.IGNORECASE),
            re.compile(r"\bcontrol\s+treatments?\b", re.IGNORECASE),
            re.compile(r"\bNTC(?:s)?\b"),
        ),
        explicit_none_patterns=(
            re.compile(r"\bno\s+negative\s+controls?\s+(?:were\s+)?(?:used|included|performed|run|analy[sz]ed)\b", re.IGNORECASE),
            re.compile(r"\bwithout\s+negative\s+controls?\b", re.IGNORECASE),
            re.compile(r"\bnegative\s+controls?\s+(?:were\s+)?(?:not\s+used|not\s+included|absent|omitted)\b", re.IGNORECASE),
            re.compile(r"\bno\s+(?:field|equipment|filtration|extraction|reagent|PCR)\s+blanks?\s+(?:were\s+)?(?:used|included|performed|run|analy[sz]ed)\b", re.IGNORECASE),
            re.compile(r"\bno\s+no[-\s]+template\s+controls?\s+(?:were\s+)?(?:used|included|performed|run|analy[sz]ed)\b", re.IGNORECASE),
            re.compile(r"\bno\s+NTCs?\s+(?:were\s+)?(?:used|included|performed|run|analy[sz]ed)\b", re.IGNORECASE),
        ),
    ),
    TextSearchFlag(
        fact_type_candidate="pos_cont_0_1",
        description="explicit evidence that positive controls were or were not used",
        positive_value="1",
        positive_patterns=(
            re.compile(r"\bpositive\s+controls?\b", re.IGNORECASE),
            re.compile(r"\bmock\s+communit(?:y|ies)\b", re.IGNORECASE),
            re.compile(r"\breference\s+DNA\b", re.IGNORECASE),
            re.compile(r"\bknown\s+DNA\b", re.IGNORECASE),
            re.compile(r"\bsynthetic\s+DNA\b", re.IGNORECASE),
            re.compile(r"\bgBlock(?:s)?\b", re.IGNORECASE),
            re.compile(r"\bplasmid\s+controls?\b", re.IGNORECASE),
            re.compile(r"\breference\s+tissue\b", re.IGNORECASE),
            re.compile(r"\bpositive\s+template\b", re.IGNORECASE),
            re.compile(r"\bpositive\s+amplification\s+controls?\b", re.IGNORECASE),
        ),
        explicit_none_patterns=(
            re.compile(r"\bno\s+positive\s+controls?\s+(?:were\s+)?(?:used|included|performed|run|analy[sz]ed)\b", re.IGNORECASE),
            re.compile(r"\bwithout\s+positive\s+controls?\b", re.IGNORECASE),
            re.compile(r"\bpositive\s+controls?\s+(?:were\s+)?(?:not\s+used|not\s+included|absent|omitted)\b", re.IGNORECASE),
            re.compile(r"\bno\s+mock\s+communit(?:y|ies)\s+(?:were\s+)?(?:used|included)\b", re.IGNORECASE),
            re.compile(r"\bno\s+(?:reference|known|synthetic)\s+DNA\s+(?:was\s+|were\s+)?(?:used|included)\b", re.IGNORECASE),
            re.compile(r"\bno\s+gBlocks?\s+(?:were\s+)?(?:used|included)\b", re.IGNORECASE),
            re.compile(r"\bno\s+plasmid\s+controls?\s+(?:were\s+)?(?:used|included)\b", re.IGNORECASE),
        ),
    ),
)

LLM_JUDGED_SEARCH_FIELDS: tuple[LLMJudgedSearchField, ...] = (
    LLMJudgedSearchField(
        term_name="barcoding_pcr_appr",
        section="Library preparation sequencing",
        description="PCR approach for metabarcoding/library construction.",
        allowed_values=("one-step PCR", "two-step PCR", "ligation-based", "other"),
        output_instructions=(
            "Classify only the barcoding/indexing/library-construction approach for metabarcoding. "
            "Use one of: one-step PCR, two-step PCR, ligation-based, other. If both one-step and "
            "two-step or ligation approaches are explicitly described, join values with ' | '. Omit "
            "the field if the quote only mentions ordinary PCR without a library/barcoding/indexing context."
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
        term_name="lib_screen",
        section="Library preparation sequencing",
        description="Specific enrichment or screening methods applied before and/or after creating libraries.",
        output_instructions=(
            "Return only explicit library screening, cleanup, enrichment, quantification, size-selection, "
            "normalization, pooling, or QC methods. Copy the relevant phrase or sentence as close to the "
            "quote text as possible; do not paraphrase."
        ),
        search_terms=(
            "library screening",
            "library QC",
            "quality checked",
            "quality control",
            "size selected",
            "fragment selection",
            "gel purified",
            "bead purified",
            "AMPure",
            "Bioanalyzer",
            "TapeStation",
            "Fragment Analyzer",
            "Qubit",
            "qPCR quantified",
            "normalized",
            "pooled",
            "enriched",
            "capture",
            "hybridization",
            "cleaned",
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
        ),
    ),
)

CONTROLLED_SEARCH_FIELDS: tuple[ControlledSearchField, ...] = (
    ControlledSearchField(
        term_name="sterilise_method",
        section="Project",
        description="Explicit contamination-minimization procedures, retained as direct source text.",
        value_strategy="evidence_sentences",
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
        term_name="sequencing_location",
        section="Library preparation sequencing",
        description="Facility, laboratory, or center where sequencing was performed.",
        value_strategy="sequencing_location_phrase",
        search_terms=(
            "sequenced at",
            "sequenced using",
            "Genome Sequencing and Analysis Facility",
            "GSAF",
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
        term_name="assay_name",
        section="PCR",
        description="A brief, concise identifier for assay with no spaces or special characters.",
        required_any_flags=frozenset({"pcr_0_1"}),
        search_terms=(
            "Earth Microbiome Project 16S",
            "TAReuk454FWD1/TAReukREV3",
            "mlCOIintF/jgHCO2198",
            "mICOIintF/jgHCO2198",
            "ZBJ-ArtF1c/ZBJ-ArtR2c",
            "hydrolysis probe assay",
            "species-specific assay",
            "taxon-specific assay",
            "metabarcoding assay",
            "COI minibarcode",
            "SYBR Green assay",
            "TaqMan assay",
            "universal assay",
            "marker assay",
            "MacDonald-18S",
            "MacDonald_18S",
            "MiFish-Elasmo",
            "MiFish-12S",
            "MiFish_12S",
            "MiFish-U/E",
            "MiFish-U2",
            "MiMammal",
            "MiFish-E",
            "MiFish-U",
            "MiFish",
            "MiBird",
            "MiEel",
            "Teleo01",
            "Teleo",
            "teleo01",
            "teleo",
            "12S-V5",
            "Vert01",
            "Riaz-12S",
            "Riaz_12S",
            "Valentini-12S",
            "Valentini_12S",
            "AcMDB07",
            "Berry-16S",
            "Berry_16S",
            "Ve16S1",
            "Ve16S3",
            "515F-Y/806R",
            "515FY/806RB",
            "515F-Y/926R",
            "515FY/926R",
            "515F/806R",
            "515F-806R",
            "341F/785R",
            "341F/805R",
            "27F/1492R",
            "V1-V2",
            "V3-V4",
            "V4-V5",
            "EMP 16S",
            "Arch519F/Arch915R",
            "Arch349F/Arch806R",
            "TAReuk",
            "1389F/EukB",
            "Euk1391f/EukBr",
            "V4 18S",
            "18S V4",
            "V9 18S",
            "18S V9",
            "Uni18S",
            "Leray-XT",
            "Leray",
            "BF2/BR2",
            "BF3/BR2",
            "fwhF2/EPTDr2n",
            "Uni-Minibar",
            "LCO1490/HCO2198",
            "Folmer",
            "Zeale",
            "ITS1F/ITS2",
            "ITS1/ITS2",
            "ITS3/ITS4",
            "fITS7/ITS4",
            "gITS7/ITS4",
            "UNITE ITS",
            "trnL g/h",
            "trnL P6 loop",
            "rbcL-a",
            "trnH-psbA",
            "ITS2 plant",
            "Plant ITS2",
            "qPCR assay",
            "ddPCR assay",
            "primer set",
            "primer pair",
            "V4",
            "rbcL",
            "matK",
            "ITS2",
            "ITS1",
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
        value_strategy="trimmomatic_tool",
        search_terms=("Trimmomatic",),
    ),
    ControlledSearchField(
        term_name="length_filtering_tool",
        section="Bioinformatics",
        description="Software used to filter reads by length.",
        value_strategy="trimmomatic_tool",
        search_terms=("Trimmomatic", "MINLEN"),
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
    re.compile(r"\bTitanium\s+chemistry\b", re.IGNORECASE),
    re.compile(r"\b(?:v2|v3)\s+chemistry\b", re.IGNORECASE),
    # TruSeq is a whole family of Illumina library prep kits (Stranded
    # mRNA, Nano, PCR-Free, DNA, ...) -- confirmed missing on a real paper
    # (ISME J 10.1093/ismejo/wrae013: "TruSeq Stranded mRNA kit (Illumina)").
    re.compile(r"\bTruSeq\s+\w+(?:\s+\w+){0,3}\s+[Kk]it\b"),
)
_THERMOCYCLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bDNA\s+Engine\s+Tetrad\s*2\s+Thermal\s+Cycler\b", re.IGNORECASE),
    re.compile(
        r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z0-9][A-Za-z0-9-]*){0,5}\s+"
        r"(?:Thermal\s+Cycler|thermocycler|Cycler|PCR\s+System)\b(?:\s*\([^)]+\))?",
        re.IGNORECASE,
    ),
)
_SEQUENCING_LOCATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bsequenced\s+(?:using\s+[^.]*?\s+)?at\s+the\s+"
        r"(?P<value>[A-Z][^.]*?(?:Facility|Center|Centre|Core|Institute|University)[^.]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<value>Genome\s+Sequencing\s+and\s+Analysis\s+Facility\s+\(GSAF\)\s+"
        r"at\s+the\s+University\s+of\s+Texas\s+at\s+Austin)\b",
        re.IGNORECASE,
    ),
)
_TRIMMOMATIC_RE = re.compile(
    r"\bTrimmomatic(?:\s+(?:v(?:ersion)?\.?\s*)?\d+(?:\.\d+)*)?\b",
    re.IGNORECASE,
)
_TRIMMOMATIC_MINLEN_PATTERNS: tuple[re.Pattern[str], ...] = (
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
        return window if re.search(r"\bforward\s+primers?\b|\bSP[-_]?F\b", window, re.IGNORECASE) else None
    if reverse_match:
        window = snippet[reverse_match.start() :]
        return window if re.search(r"\breverse\s+primers?\b|\bSP[-_]?R\b", window, re.IGNORECASE) else None
    return snippet if re.search(r"\bSP[-_]?R\b", snippet, re.IGNORECASE) else None


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
            if not _PCR_MIXTURE_MARKERS_RE.search(snippet):
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


def _match_trimmomatic_tool(
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
            match = _TRIMMOMATIC_RE.search(snippet)
            if match is None:
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
                    "matched_pattern": _TRIMMOMATIC_RE.pattern,
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
            if "minlen" not in snippet.casefold() and not _TRIMMOMATIC_RE.search(snippet):
                continue
            for pattern in _TRIMMOMATIC_MINLEN_PATTERNS:
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
    if field.value_strategy == "biological_replicate_integer":
        return _match_biological_replicates(field, texts, locator_prefix)
    if field.value_strategy == "assay_type_classifier":
        return _classify_assay_type(field, texts, locator_prefix)
    if field.value_strategy == "commercial_pcr_mixture_phrase":
        return _match_pcr_mixture_phrase(field, texts, locator_prefix, classification="commercial")
    if field.value_strategy == "custom_pcr_mixture_phrase":
        return _match_pcr_mixture_phrase(field, texts, locator_prefix, classification="custom")
    if field.value_strategy == "sequencing_kit_phrase":
        return _match_regex_phrases(field, texts, locator_prefix, _SEQUENCING_KIT_PATTERNS)
    if field.value_strategy == "thermocycler_phrase":
        return _match_regex_phrases(field, texts, locator_prefix, _THERMOCYCLER_PATTERNS)
    if field.value_strategy == "sequencing_location_phrase":
        return _match_regex_phrases(field, texts, locator_prefix, _SEQUENCING_LOCATION_PATTERNS)
    if field.value_strategy == "trimmomatic_tool":
        return _match_trimmomatic_tool(field, texts, locator_prefix)
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


def _candidate_fields_for_snippet(snippet: str) -> tuple[str, ...]:
    field_names: list[str] = []
    for field in LLM_JUDGED_SEARCH_FIELDS:
        if any(_term_pattern(term).search(snippet) for term in field.search_terms):
            field_names.append(field.term_name)
    return tuple(field_names)


def quote_candidates_for_llm_judged_search(
    texts: Iterable[tuple[str, str]],
    *,
    max_candidates: int = 40,
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
            field_names = _candidate_fields_for_snippet(snippet)
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
    return f"""You are judging candidate source quotes for FAIRe projectMetadata library-preparation fields.

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
    return all(part in field.allowed_values for part in parts)


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
            {"values": [], "quotes": [], "matches": [], "seen_values": set()},
        )
        key = value.casefold()
        if key in group["seen_values"]:
            continue
        group["seen_values"].add(key)
        group["values"].append(value)
        if candidate.text not in group["quotes"]:
            group["quotes"].append(candidate.text)
        group["matches"].append(
            {
                "raw_value": value,
                "quote_id": quote_id,
                "source_locator": f"{locator_prefix}:llm_judged_search:{field_name}:{quote_id}",
            }
        )

    facts: list[RawFactCandidate] = []
    for field_name, group in grouped.items():
        field = fields[field_name]
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value=" | ".join(group["values"]),
                source_locator=f"{locator_prefix}:llm_judged_search:{field_name}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=" | ".join(group["quotes"]),
                confidence_metadata={
                    "detector": "llm_judged_quote_search",
                    "section": field.section,
                    "description": field.description,
                    "matches": group["matches"],
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
) -> list[RawFactCandidate]:
    reusable_texts = tuple(texts)
    candidates = quote_candidates_for_llm_judged_search(reusable_texts)
    if not candidates:
        return []
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
    return _facts_from_llm_judgement(parsed, candidates, locator_prefix=locator_prefix)


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
