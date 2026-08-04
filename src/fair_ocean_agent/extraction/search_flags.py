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
        positive_patterns=(
            re.compile(r"\bPCR\b", re.IGNORECASE),
            re.compile(r"\bqPCR\b", re.IGNORECASE),
            re.compile(r"\bddPCR\b", re.IGNORECASE),
            re.compile(r"\bamplification\b", re.IGNORECASE),
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
        search_terms=(
            "MiSeq Reagent Kit v3",
            "MiSeq Reagent Kit",
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
            "chemistry",
        ),
    ),
    ControlledSearchField(
        term_name="target_gene",
        section="PCR",
        description="Field type: controlled vocabulary, list all that find in paper.",
        required_any_flags=frozenset({"pcr_0_1"}),
        search_terms=(
            "12S rRNA (SSU mitochondria)",
            "16S rRNA (LSU mitochondria)",
            "16S rRNA (SSU prokaryote)",
            "23S rRNA (LSU prokaryote)",
            "18S rRNA (SSU eukaryote)",
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
            "ITS",
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
        search_terms=(
            "PCR Master Mix",
            "qPCR Master Mix",
            "Thermo Fisher",
            "Applied Biosystems",
            "master mix",
            "mastermix",
            "PCR mix",
            "qPCR mix",
            "TaqMan",
            "SYBR",
            "Luna",
            "PowerUp",
            "QuantiTect",
            "SsoAdvanced",
            "KAPA",
            "NEB",
            "Bio-Rad",
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
        term_name="thermocycler",
        section="PCR",
        description="The manufacturer and model of a thermocycler used.",
        required_any_flags=frozenset({"pcr_0_1"}),
        search_terms=(
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
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BIOLOGICAL_REPLICATE_EXCLUSION_RE = re.compile(
    r"\b(PCR|technical|extraction|filtration|sequencing|analytical|library|qPCR|ddPCR)\s+replicates?\b",
    re.IGNORECASE,
)
_BIOLOGICAL_REPLICATE_CONTEXT_RE = re.compile(
    r"\b(biological|environmental|independent|collected|samples?|sites?|stations?|treatments?|water|sediment)\b",
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
    (re.compile(r"\bn\s*=\s*(?P<num>\d+)\b", re.IGNORECASE), {"needs_context": "1"}),
)

_ASSAY_TYPE_CUES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "targeted",
        (
            re.compile(r"\bquantitative\s+PCR\b", re.IGNORECASE),
            re.compile(r"\bdigital\s+PCR\b", re.IGNORECASE),
            re.compile(r"\bspecies[-\s]+specific\b", re.IGNORECASE),
            re.compile(r"\btaxon[-\s]+specific\b", re.IGNORECASE),
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
                if options.get("needs_context") and not _BIOLOGICAL_REPLICATE_CONTEXT_RE.search(snippet):
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
    return _match_controlled_terms(field, texts, locator_prefix)


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
