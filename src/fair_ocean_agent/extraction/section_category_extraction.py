"""Stage 2 (LLM sentence-level categorization) and Stage 3 (category-scoped
term extraction): the two real LLM calls in the PCR/library-prep/
bioinformatics methods categorize-then-extract pipeline. extraction/
section_categories.py holds the deterministic building blocks this module
calls into (Stage 1's paragraph keyword gate, Stage 2.5's run-grouping);
this module is where the LLM actually gets invoked.

Both calls follow the same "quote-candidate-then-judge" discipline already
proven in extraction/search_flags.py's LLMJudgedSearchField mechanism: the
LLM is never shown free-running whole-document text, only already
keyword-gated candidates, and every one of the 106 term definitions
carries the SAME shared instruction to copy values verbatim rather than
generate them -- an explicit user instruction ("In all cases, have the
program take word for word from text and not generate."), applied via one
shared prompt template rather than 106 bespoke output instructions, plus a
programmatic guard (Stage 3 discards any extracted value that doesn't
literally appear in its own cited quote) since a prompt instruction alone
is never trusted to be self-enforcing anywhere else in this codebase.

Deliberately additive, not a replacement: every one of these 106 fields
already has some existing extraction path (the broad `FaireExtractionField`
checklist, or a standalone `LLMJudgedSearchField` entry) that keeps running
unchanged. Per an explicit user request, this new pipeline's own output is
meant to be compared against the old mechanisms' output before any of that
existing machinery is retired -- this module does not exclude or suppress
any existing field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from fair_ocean_agent.config import MIN_LLM_MAX_OUTPUT_TOKENS
from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.extraction.section_categories import (
    SECTION_CATEGORIES,
    TERM_PHASES_BY_CATEGORY,
    CategoryTerm,
    SectionCategory,
    _term_pattern,
    candidate_categories_for_paragraph,
    group_sentences_into_category_runs,
    paragraphs_before_next_section_heading,
)
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.logging_setup import get_logger
from fair_ocean_agent.sources.base import RawFactCandidate

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_ID_RE = re.compile(r"^S(\d+)\.(\d+)$")
_AMOUNT_WITH_UNIT_RE = re.compile(
    r"(?:[~∼≈]\s*)?\d+(?:\.\d+)?\s*(?:mg|g|kg|mL|ml|L|liters?|litres?|µL|uL)\b",
    re.IGNORECASE,
)
_DNA_EXTRACTION_AMOUNT_CONTEXT_RE = re.compile(
    r"\b(?:DNA|RNA|nucleic\s+acid).{0,80}\bextract(?:ed|ion)\b|\b(?:used|processed)\s+for\s+(?:DNA|RNA|nucleic\s+acid)\s+extraction\b",
    re.IGNORECASE,
)
_SAMPLE_SIZE_CONTEXT_RE = re.compile(
    r"\b(?:samples?\s+were\s+)?(?:collected|sampled|obtained)\b|"
    r"\b(?:total\s+)?(?:volume|mass|amount)\s+(?:collected|sampled)\b|"
    r"\b(?:L|mL|g|kg)\s+of\s+(?:water|sediment)\s+were\s+collected\b",
    re.IGNORECASE,
)
_POOLING_ACTION_RE = re.compile(r"\b(?:pool(?:ed|ing)?|combin(?:ed|ing))\b", re.IGNORECASE)
# Real gap found live (10.3390/microorganisms10030558): "Three replicates
# were pooled and purified..." names what got pooled as "replicates", not
# any of the original words below -- a real, common way of describing
# multiple parallel PCR reactions/aliquots from the same sample being
# combined before cleanup, missed entirely before this was added.
_POOLING_SAMPLE_CONTEXT_RE = re.compile(
    r"\b(?:DNA|RNA|nucleic\s+acids?|extracts?|samples?|subsamples?|replicates?|aliquots?|reactions?|amplicons?)\b",
    re.IGNORECASE,
)
_SIZE_FRAC_FIELD = "size_frac"
_SIZE_FRAC_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*[-\s]?\s*[µμu]m\b", re.IGNORECASE)
_FILTER_PASSIVE_ACTIVE_FIELD = "filter_passive_active_0_1"
_ACTIVE_FILTER_NAME_RE = re.compile(r"\bSterivex\b", re.IGNORECASE)
_INTERNAL_EXPEDITION_ID_FIELD = "internal_expedition_id"
_ODP_LEG_WITH_PREFIX_RE = re.compile(
    r"\b(?:Ocean\s+Drilling\s+Program\s+\(ODP\)|ODP)\s+Leg\s+(?P<number>\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)
_BARE_ODP_LEG_RE = re.compile(r"\bLeg\s+(?P<number>\d+[A-Za-z]?)\b", re.IGNORECASE)
_IODP_EXPEDITION_WITH_PREFIX_RE = re.compile(
    r"\b(?:Integrated\s+Ocean\s+Drilling\s+Program\s+\(IODP\)|IODP)\s+Expedition\s+(?P<number>\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)
_BARE_IODP_EXPEDITION_RE = re.compile(r"\bExpedition\s+(?P<number>\d+[A-Za-z]?)\b", re.IGNORECASE)
_CRUISE_ID_RE = re.compile(
    r"\b(?:shakedown\s+)?(?:cruise|voyage)\s+(?:number\s+|no\.\s*|#\s*)?(?P<id>[A-Z]{1,8}\d{2,}(?:[-_/][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)
_RV_CRUISE_RE = re.compile(
    r"\b(?:R/V|RV|M/V|D/V)\s+[A-Z][A-Za-z0-9._-]*(?:\s+[A-Z][A-Za-z0-9._-]*){0,3}\s+"
    r"(?:cruise|voyage)\s+(?:number\s+|no\.\s*|#\s*)?(?P<id>[A-Z]{1,8}\d{2,}(?:[-_/][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)
_FILTER_EVIDENCE_FIELDS = frozenset(
    {
        "filter_material",
        "filter_name",
        "filter_diameter",
        "filter_surface_area",
        "size_frac",
        "prefilter_material",
    }
)
_ACTIVE_FILTRATION_CONTEXT_RE = re.compile(
    r"\b(?:active(?:ly)?\s+filter|pumped?|pumping|pump|vacuum|suction|peristaltic|"
    r"pressure|overpressure|pressuri[sz]ed|compressed\s+air|syringe\s+pressure|"
    r"forced\s+through|fan-driven|flowmeter|flow\s*rate|flowrate)\b",
    re.IGNORECASE,
)
_NUCL_ACID_EXT_LYSIS_FIELD = "nucl_acid_ext_lysis"
_UNIT_COMPANION_FIELDS = {
    "samp_vol_we_dna_ext_unit": "samp_vol_we_dna_ext",
    "samp_size_unit": "samp_size",
}


@dataclass(frozen=True)
class _NormalizationSpec:
    field_name: str
    prompt: str
    system: str


_NORMALIZATION_SUFFIX = "_normalized"
_FIELD_NORMALIZATION_SPECS: tuple[_NormalizationSpec, ...] = (
    _NormalizationSpec(
        field_name="nucl_acid_ext_lysis",
        system="You normalize nucleic-acid extraction lysis evidence to short controlled technique names.",
        prompt=(
            "From the supplied nucleic-acid extraction text, identify the techniques or reagents explicitly used "
            "to lyse cells or disrupt biological material to release nucleic acids. Return only short standardized "
            "technique names, separated by | when multiple methods were used. Extract the specific technique, "
            "not a broad category. Do not infer chemistry from a kit name unless it is stated. Use "
            "kit-proprietary lysis only when a commercial extraction kit is explicitly used but the lysis "
            "mechanism is not stated.\n\nCanonical examples:\n"
            "bead beating, bead milling, mechanical homogenization, rotor-stator homogenization, manual grinding, "
            "mortar-and-pestle grinding, cryogenic grinding, sonication, ultrasonication, French press, "
            "high-pressure homogenization, freeze-thaw lysis, heat lysis, boiling lysis, osmotic lysis, alkaline "
            "lysis, SDS lysis, CTAB lysis, guanidinium thiocyanate lysis, guanidinium hydrochloride lysis, "
            "Triton X-100 lysis, NP-40 lysis, Tween lysis, sodium deoxycholate lysis, "
            "N-lauroylsarcosine/Sarkosyl lysis, lysozyme lysis, proteinase K digestion, lyticase lysis, "
            "zymolyase lysis, mutanolysin lysis, achromopeptidase lysis, cellulase digestion, chitinase "
            "digestion, enzymatic digestion, kit-proprietary lysis. Use other:<short literal technique> for a "
            "clearly stated lysis technique absent from this list."
        ),
    ),
    _NormalizationSpec(
        field_name="nucl_acid_ext_sep",
        system="You normalize nucleic-acid extraction separation/purification evidence to short controlled technique names.",
        prompt=(
            "From the supplied nucleic-acid extraction text, identify the technique(s) used to separate or purify "
            "nucleic acids from the lysed sample. Return only short standardized technique names, separated by | "
            "when multiple methods were used. Extract the specific separation principle. Do not classify "
            "centrifugation alone as a separation technique unless it is clearly part of a named separation "
            "principle.\n\nCanonical examples:\n"
            "silica spin-column, silica membrane, silica adsorption, glass-fiber adsorption, "
            "magnetic-bead purification, SPRI bead purification, solid-phase extraction, "
            "phenol-chloroform extraction, phenol-chloroform-isoamyl alcohol extraction, chloroform extraction, "
            "organic phase extraction, TRIzol phase separation, ethanol precipitation, isopropanol precipitation, "
            "alcohol precipitation, PEG precipitation, sodium acetate precipitation, ammonium acetate "
            "precipitation, lithium chloride precipitation, salting-out, anion-exchange column, ion-exchange "
            "chromatography, hydroxyapatite purification, density-gradient ultracentrifugation, CsCl gradient "
            "purification, ultrafiltration, centrifugal-filter purification, dialysis, precipitation-only. Use "
            "other:<short literal technique> for a clearly stated separation technique absent from this list."
        ),
    ),
    _NormalizationSpec(
        field_name="prep_method_additional",
        system="You normalize sample-preparation evidence to short controlled operation names.",
        prompt=(
            "From the supplied sample-preparation text, identify important preparation operations performed before "
            "nucleic-acid extraction that are not already captured by specific collection, filtration, storage, "
            "precipitation, extraction, PCR, or sequencing fields. Return short standardized technique names "
            "separated by |. Use this as a small residual field, not a duplicate dump of samp_mat_process.\n\n"
            "Canonical examples: filter handling, filter subdivision, filter folding, aseptic handling, mixing, "
            "vortexing, shaking/agitation, incubation, thawing, pressure-assisted processing, vacuum-assisted "
            "processing, sequential processing, flow monitoring, manual transfer, settling. Use other:<short "
            "literal technique> for a clearly stated residual preparation operation absent from this list."
        ),
    ),
    _NormalizationSpec(
        field_name="nucl_acid_ext_method_additional",
        system="You normalize additional nucleic-acid extraction evidence to short controlled procedure names.",
        prompt=(
            "From the supplied nucleic-acid extraction text, identify additional extraction procedures not already "
            "represented by the extraction kit, lysis, or separation method. Return short standardized technique "
            "names separated by |. Do not include generic incubation temperatures, reagent volumes, or "
            "centrifugation durations just because they occur in the extraction protocol.\n\nCanonical examples: "
            "RNase treatment, DNase treatment, inhibitor removal, humic-acid removal, desalting, buffer exchange, "
            "nucleic-acid concentration, carrier-assisted precipitation, glycogen-assisted precipitation, "
            "linear-polyacrylamide-assisted precipitation, repeated extraction, sequential extraction, multiple "
            "elution, heated elution, extended digestion, nuclease inactivation, RNA stabilization, DNA "
            "stabilization. Use other:<short literal technique> for a clearly stated extraction-specific procedure "
            "absent from this list."
        ),
    ),
    _NormalizationSpec(
        field_name="samp_collect_device",
        system="You normalize sampling-device evidence to short controlled device names.",
        prompt=(
            "From the supplied sampling text, identify the physical device(s) used to obtain the environmental or "
            "biological sample from its source. Return short standardized device names separated by | when multiple "
            "devices were used. Do not include containers or equipment used only after collection for storage, "
            "processing, filtration, or laboratory analysis.\n\nCanonical examples: Niskin bottle, Nansen bottle, "
            "GO-FLO bottle, Van Dorn bottle, Kemmerer bottle, CTD rosette, water sampling bottle, underway intake, "
            "flow-through sampler, pump, peristaltic pump, submersible pump, syringe, box corer, multicorer, "
            "gravity corer, piston corer, push core, hand corer, vibracorer, sediment corer, Van Veen grab, Ekman "
            "grab, Ponar grab, Smith-McIntyre grab, sediment grab, sediment trap, porewater sampler, Rhizon "
            "sampler, plankton net, bongo net, ring net, WP2 net, neuston net, manta net, MOCNESS, trawl, beam "
            "trawl, otter trawl, epibenthic sled, dredge, swab, biopsy punch, scraper, spatula, scoop, forceps, "
            "filter cartridge, in-situ filtration sampler, passive eDNA sampler. Use other:<short literal device> "
            "for a clearly stated collection device absent from this list."
        ),
    ),
    _NormalizationSpec(
        field_name="samp_collect_method",
        system="You normalize sampling-method evidence to short controlled method names.",
        prompt=(
            "From the supplied sampling text, identify the method used to obtain the sample from its original "
            "environment or source. Return short standardized method names separated by |. Distinguish the method "
            "from the physical device, e.g. samp_collect_device = Niskin bottle but samp_collect_method = "
            "depth-specific water sampling.\n\nCanonical examples: discrete water sampling, depth-specific water "
            "sampling, integrated-depth water sampling, surface-water sampling, bottom-water sampling, underway "
            "water sampling, flow-through water sampling, pumped water sampling, sediment coring, push coring, "
            "grab sampling, porewater sampling, sediment-trap sampling, hand collection, net sampling, vertical "
            "net tow, horizontal net tow, oblique net tow, trawling, dredging, epibenthic sled sampling, swabbing, "
            "scraping, biopsy sampling, tissue excision, dissection sampling, whole-organism collection, transect "
            "sampling, quadrat sampling, active filtration sampling, passive sampling, biofilm sampling, benthic "
            "substrate collection, ice coring, snow sampling. Use other:<short literal method> for a clearly "
            "stated collection method absent from this list."
        ),
    ),
    _NormalizationSpec(
        field_name="samp_mat_process",
        system="You normalize sample-processing evidence to short controlled technique names.",
        prompt=(
            "From the supplied sample-processing text, identify physical or chemical processing applied to the "
            "collected sample before nucleic-acid extraction. Return short standardized technique names separated "
            "by |. Do not collapse specific processing into a vague term when the specific method is known.\n\n"
            "Canonical examples: filtration, prefiltration, vacuum filtration, pressure filtration, pump "
            "filtration, syringe filtration, tangential-flow filtration, ultrafiltration, cascade filtration, "
            "sequential filtration, size fractionation, sieving, wet sieving, dry sieving, subsampling, aliquoting, "
            "cutting, slicing, dissection, scraping, homogenization, grinding, cryogenic grinding, milling, "
            "crushing, tissue maceration, centrifugation, pelleting, sedimentation, settling, decanting, washing, "
            "rinsing, resuspension, precipitation, density separation, density-gradient separation, flotation, "
            "concentration, freeze-drying, lyophilization, air drying, oven drying, dehydration, cell sorting, "
            "flow-cytometric sorting, magnetic cell separation, manual picking, decalcification. Use other:<short "
            "literal processing step> for a clearly stated processing step absent from this list."
        ),
    ),
)
# concentration is meant to hold a short numeric measurement (e.g.
# "25.4 ng/uL"), not a description of how it was measured -- confirmed
# live, a real bug (10.1002/ece3.6071): a sentence describing the kit/
# instrument used ("DNA concentration was measured with the Quant-iT
# dsDNA HS assay kit ... using a Qubit 2.0 Fluorometer ...") was accepted
# as the concentration VALUE itself. A plain "contains a digit" check
# wouldn't have caught it, since "Qubit 2.0" has one. A generous length cap
# does: no real reported concentration value runs anywhere near this long.
_SHORT_NUMERIC_VALUE_MAX_CHARS = {"concentration": 30}
# A dotted, multi-letter abbreviation/brand name (e.g. "E.Z.N.A.", "U.S.A.")
# at the very end of a split piece: the naive [.!?] split above treats each
# of its internal periods as a sentence end, real problem confirmed live --
# "resuspended in the lysis buffer of the E.Z.N.A.® Soil DNA kit ..." was
# shredded into "...the E.Z.N.A." and "® Soil DNA kit ...", corrupting both
# nucl_acid_ext_lysis (truncated) and nucl_acid_ext_kit (missing its own
# brand prefix) in the same real sentence. 2+ single-letter-dot groups is
# deliberately the threshold (not 1) so an ordinary sentence ending in a
# single initial ("... call it X.") is left alone.
_DOTTED_ABBREVIATION_TAIL_RE = re.compile(r"(?:\b[A-Za-z]\.){2,}$")
# "vol." (volume/volumes) is a real, common wet-lab-protocol abbreviation
# the naive [.!?]-then-whitespace splitter above mistakes for a sentence
# end -- confirmed live (10.1371/journal.pone.0303937): "1 vol. of
# phenol/chloroform/isoamyalcohol..." got shredded into "1 vol." and "of
# phenol/chloroform/isoamyalcohol...", corrupting the DNA-cleanup methods
# paragraph's own sentence boundaries and, downstream, its Stage 2
# categorization.
# "et al." is the other ubiquitous academic abbreviation this naive
# period-based splitter mistakes for a sentence boundary -- a real problem
# confirmed live while building primer-reference extraction: "...primer
# 515F, originally described by Caporaso et al. (2011), and the reverse
# primer 806R..." was shredded right between the author name and its own
# year, permanently separating a primer's name from the citation meant to
# identify it (the exact adjacency pcr_primer_reference_forward/reverse
# depends on).
_COMMON_ABBREVIATION_TAIL_RE = re.compile(r"\b(?:vol|et\s+al)\.$", re.IGNORECASE)
# A real bug found live (10.1093/ismejo/wrae013): a methods paragraph
# describing three DIFFERENT tools for three DIFFERENT purposes as one
# semicolon-joined enumerated list -- "...using SeqPrep 1.2...[64]; (ii)
# ...using bowtie2 2.3.5.1 [65], and (iii) ...using Trimmomatic 0.39..." --
# never got split at all by the period-only splitter above (its only
# terminal period is at the very end), so every downstream field pulled
# from this text saw all three tools/purposes jumbled into one candidate,
# corrupting trim_method/error_rate_tool alike. A top-level (non-
# parenthetical) semicolon is a common, genuine sentence-equivalent
# boundary in academic writing; ", and (iii)" is the same list-marker
# convention joined by a comma instead, normalized to a semicolon first so
# one splitter handles both. Never applied INSIDE parentheses/brackets --
# a semicolon-joined citation list like "(Smith 2020; Jones 2021)" must
# never be split.
_LIST_MARKER_COMMA_BOUNDARY_RE = re.compile(
    r",\s+and\s+(?=\((?:i{1,3}|iv|vi{0,3}|ix|x|\d{1,2})\)\s)"
)


def _split_top_level_semicolons(piece: str) -> list[str]:
    # Keeps the semicolon attached to the END of its own piece (mirroring
    # how the period-based split above keeps its own terminal "."/"!"/"?")
    # rather than discarding it -- Stage 2.5's own run-assembly
    # (group_sentences_into_category_runs) rejoins same-category sentences
    # with a plain space, so a split boundary that isn't preserved in the
    # text itself would be silently lost the moment Stage 3 re-splits the
    # assembled run text, right back into the same unsplit blob this fix
    # is for.
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


def _split_into_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.split(paragraph):
        piece = piece.strip()
        if not piece:
            continue
        for sub_piece in _split_top_level_semicolons(piece):
            if sentences and (
                _DOTTED_ABBREVIATION_TAIL_RE.search(sentences[-1])
                or _COMMON_ABBREVIATION_TAIL_RE.search(sentences[-1])
            ):
                sentences[-1] = f"{sentences[-1]} {sub_piece}"
            else:
                sentences.append(sub_piece)
    return sentences


def _build_categorization_prompt(
    indexed_paragraphs: list[tuple[int, list[str], frozenset[str]]],
) -> str:
    category_reference = "\n".join(f"- {category.name}: {category.label}" for category in SECTION_CATEGORIES)
    lines: list[str] = []
    for paragraph_index, sentences, candidates in indexed_paragraphs:
        lines.append(f"Paragraph {paragraph_index} (candidate categories: {', '.join(sorted(candidates))}):")
        for sentence_index, sentence in enumerate(sentences):
            lines.append(f"  S{paragraph_index}.{sentence_index}: {sentence}")
    sentences_block = "\n".join(lines)
    return f"""You are categorizing sentences from a scientific paper's methods text into topic categories.

Categories:
{category_reference}

For each sentence below, decide which of ITS OWN PARAGRAPH's listed candidate categories it genuinely belongs to
(zero, one, or more than one). A sentence may belong to zero categories even though its paragraph was flagged, if
that particular sentence isn't actually about any of them.

Critical rule: a software/tool name (e.g. DADA2, QIIME2, Cutadapt, UCHIME, UPARSE) only helps locate relevant
text -- it never determines the category by itself, since the same tool can perform different operations in
different papers. Classify based on the OPERATION the sentence describes (the verb/action), not the tool name
alone. For example "chimeras were removed using DADA2" is chimera removal, while "reads were clustered into OTUs
using DADA2" is OTU/ASV generation, even though both name the same tool.

Sentences:
{sentences_block}

Return ONLY a JSON array. Each object must be:
{{"sentence_id": "S<paragraph>.<sentence>", "categories": ["<category_name>", ...]}}
Omit a sentence entirely if it belongs to none of its paragraph's own candidate categories.
"""


# A real live audit (10.1371/journal.pone.0303937) caught this: one call
# for the whole paper's worth of candidate sentences (112 here) needed
# 2443 completion tokens and got silently cut off at 1024 (finish_reason
# "length"), tagging only 47 of them -- everything past the cutoff,
# including the sentence naming SILVA_132/FreshTrain for otu_db and the
# chimera-filtration sentence feeding otu_asv_generation_filtering, was
# never categorized at all, so nothing downstream in Stage 3 ever saw it
# as a candidate. Even 2048 (this codebase's usual floor,
# MIN_LLM_MAX_OUTPUT_TOKENS) still cut off at 92/112 sentences on this
# same paper; 4096 was the first budget that reached finish_reason
# "stop" for that one paper -- but a denser or longer paper could still
# exceed even that, and per an explicit user request to prioritize
# thoroughness over speed, categorize_paragraphs now implements the
# "real long-term fix" this comment used to just call out as unfinished:
# chunking, capped by sentence count (not paragraph count, since that's
# the actual unit both the prompt and the model's per-item JSON output
# scale with) rather than one shot for the whole paper. 60 was chosen
# with real headroom: the confirmed incident's own density (112
# sentences / 2443 tokens ~= 21.8 completion tokens/sentence) puts a
# 60-sentence batch at ~1300 tokens against a 4096 budget, over 3x
# headroom even before accounting for this being one of the more
# candidate-dense real papers seen. A single paragraph large enough to
# exceed this alone still goes out as its own (correspondingly larger)
# batch -- sentences are never split across a paragraph's own boundary,
# since gated_paragraphs' candidate-category list is per-paragraph, not
# per-sentence, and splitting mid-paragraph would need that same list
# repeated (and kept consistent) across whichever batches sentence 1 vs
# sentence 40 landed in, for no benefit an unmerged paragraph -- these
# are vanishingly rare in real methods prose -- doesn't already get for
# free.
_MAX_SENTENCES_PER_CATEGORIZATION_CALL = 60


def _batched_by_sentence_count(
    indexed: list[tuple[int, list[str], frozenset[str]]], max_sentences: int
) -> list[list[tuple[int, list[str], frozenset[str]]]]:
    batches: list[list[tuple[int, list[str], frozenset[str]]]] = []
    current: list[tuple[int, list[str], frozenset[str]]] = []
    current_count = 0
    for item in indexed:
        _, sentences, _ = item
        if current and current_count + len(sentences) > max_sentences:
            batches.append(current)
            current = []
            current_count = 0
        current.append(item)
        current_count += len(sentences)
    if current:
        batches.append(current)
    return batches


def categorize_paragraphs(
    backend: LLMBackend,
    gated_paragraphs: list[tuple[str, frozenset[str]]],
    *,
    max_output_tokens: int | None = 4096,
) -> dict[int, list[tuple[str, frozenset[str]]]]:
    """Stage 2. `gated_paragraphs` is (paragraph_text, candidate_categories)
    pairs, already filtered by Stage 1's keyword gate -- typically
    `paragraphs_before_next_section_heading` + `candidate_categories_for_paragraph`
    applied across a document's texts. Returns {paragraph_index:
    tagged_sentences}, `paragraph_index` matching `gated_paragraphs`'s own
    position, ready for `group_sentences_into_category_runs`.

    One call per _MAX_SENTENCES_PER_CATEGORIZATION_CALL-sized batch (see
    its own comment for why), not one call for the whole paper -- still
    far fewer calls than one per paragraph, the efficiency the
    categorize-then-extract redesign was for in the first place."""
    if not gated_paragraphs:
        return {}
    indexed = [
        (index, _split_into_sentences(paragraph), candidates)
        for index, (paragraph, candidates) in enumerate(gated_paragraphs)
    ]
    indexed = [(index, sentences, candidates) for index, sentences, candidates in indexed if sentences]
    if not indexed:
        return {}

    valid_category_names = {category.name for category in SECTION_CATEGORIES}
    tags_by_sentence: dict[tuple[int, int], frozenset[str]] = {}
    for batch in _batched_by_sentence_count(indexed, _MAX_SENTENCES_PER_CATEGORIZATION_CALL):
        prompt = _build_categorization_prompt(batch)
        parsed, _response = backend.generate_json(
            prompt,
            system=(
                "You categorize methods-text sentences into topic categories using only the listed "
                "candidate categories per paragraph."
            ),
            temperature=0,
            max_tokens=max_output_tokens,
        )
        if parsed is None:
            raise LLMBackendError(f"{backend.label}: section-category sentence tagging returned invalid JSON after retries")
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if not isinstance(item, dict):
                continue
            match = _SENTENCE_ID_RE.match(str(item.get("sentence_id") or ""))
            if not match:
                continue
            categories = item.get("categories")
            if not isinstance(categories, list):
                continue
            valid = frozenset(c for c in categories if isinstance(c, str) and c in valid_category_names)
            if valid:
                tags_by_sentence[(int(match.group(1)), int(match.group(2)))] = valid

    return {
        paragraph_index: [
            (sentence, tags_by_sentence.get((paragraph_index, sentence_index), frozenset()))
            for sentence_index, sentence in enumerate(sentences)
        ]
        for paragraph_index, sentences, _candidates in indexed
    }


@dataclass(frozen=True)
class _TermQuoteCandidate:
    quote_id: str
    term_names: tuple[str, ...]
    text: str


def _term_candidate_quotes(terms: tuple[CategoryTerm, ...], run_text: str) -> tuple[_TermQuoteCandidate, ...]:
    candidates: list[_TermQuoteCandidate] = []
    seen_text: set[str] = set()
    for sentence in _split_into_sentences(run_text):
        if sentence in seen_text:
            continue
        matched_terms = tuple(
            term.native_name
            for term in terms
            if any(_term_pattern(cue).search(sentence) for cue in term.search_cues)
        )
        if not matched_terms:
            continue
        seen_text.add(sentence)
        candidates.append(_TermQuoteCandidate(quote_id=f"Q{len(candidates) + 1:03d}", term_names=matched_terms, text=sentence))
    return tuple(candidates)


def _searchable_terms(category: SectionCategory) -> tuple[CategoryTerm, ...]:
    return tuple(term for term in category.terms if not term.fallback_only)


def _term_phases(category: SectionCategory) -> tuple[tuple[str, tuple[CategoryTerm, ...]], ...]:
    """Splits a category's searchable terms into smaller, ordered groups
    for Stage 3 (see extraction/section_categories.py's
    TERM_PHASES_BY_CATEGORY for why -- smaller, workflow-ordered passes
    with only their own field definitions listed, instead of one call
    listing every field in the category regardless of relevance). A
    category with no phase mapping there falls back to one phase holding
    every searchable term, i.e. today's original unsplit behavior."""
    all_terms = _searchable_terms(category)
    phase_names = TERM_PHASES_BY_CATEGORY.get(category.name)
    if phase_names is None:
        return ((category.name, all_terms),)
    terms_by_name = {term.native_name: term for term in all_terms}
    return tuple(
        (phase_name, tuple(terms_by_name[name] for name in native_names if name in terms_by_name))
        for phase_name, native_names in phase_names
    )


def _build_term_extraction_prompt(
    category_label: str, terms: tuple[CategoryTerm, ...], candidates: tuple[_TermQuoteCandidate, ...]
) -> str:
    fields_reference = "\n".join(f"- {term.native_name}: {term.definition}" for term in terms)
    quotes = "\n".join(f"{c.quote_id} [{', '.join(c.term_names)}]: {c.text}" for c in candidates)
    return f"""You are extracting FAIRe "{category_label}" fields from supplied quote IDs only.

Use only the candidate quotes below. Do not use outside knowledge, do not infer, and do not paraphrase -- copy
the relevant value WORD FOR WORD from the quote text exactly as reported (numbers, units, software names,
sequences, and phrases must all be verbatim, not generated or reworded). Return a field only if a quote
explicitly supports it. If multiple distinct values for the same field are explicitly supported by different
quotes, return one object per value.

Each quote below is labeled with the field name(s) in brackets it was pre-matched for -- only extract one of
those bracketed field(s) for that quote, even if the quote's text also happens to contain a number or phrase
that could look relevant to a different, unbracketed field.

Fields:
{fields_reference}

Return ONLY a JSON array. Each object must be:
{{"field": "<one listed field>", "raw_value": "<verbatim value copied from the quote>", "quote_id": "Q001"}}

Candidate quotes:
{quotes}
"""


def _is_valid_short_numeric_value(field_name: str, value: str) -> bool:
    max_chars = _SHORT_NUMERIC_VALUE_MAX_CHARS.get(field_name)
    return max_chars is None or len(value) <= max_chars


def _valid_context_for_amount_field(field_name: str, quote: str) -> bool:
    if field_name in {"samp_vol_we_dna_ext", "samp_vol_we_dna_ext_unit"}:
        return bool(_DNA_EXTRACTION_AMOUNT_CONTEXT_RE.search(quote))
    if field_name in {"samp_size", "samp_size_unit"}:
        return bool(_SAMPLE_SIZE_CONTEXT_RE.search(quote))
    if field_name == "pool_dna_num":
        return bool(_POOLING_ACTION_RE.search(quote) and _POOLING_SAMPLE_CONTEXT_RE.search(quote))
    return True


# adapter_forward/adapter_reverse also exist in extraction/search_flags.py's
# LLMJudgedSearchField taxonomy, which found the same real bug this guards
# against (10.1093/ismejo/wrae013): a citation-bracket artifact ("[ ]")
# passing the verbatim-quote check as if it were a real adapter sequence,
# since it does appear literally in its own quote. Stage 3's own cue list
# for these two terms is narrower than quote-judged's, so this hasn't been
# observed to fire here yet, but the same failure mode is equally possible
# whenever a paper names an adapter without giving its actual sequence.
_ADAPTER_SEQUENCE_FIELDS = frozenset({"adapter_forward", "adapter_reverse"})
_NUCLEOTIDE_SEQUENCE_RE = re.compile(r"^[ACGTURYSWKMBDHVN]{4,}$", re.IGNORECASE)


def _valid_sequence_value(field_name: str, value: str) -> bool:
    if field_name not in _ADAPTER_SEQUENCE_FIELDS:
        return True
    compact = re.sub(r"[\s-]", "", value)
    return bool(_NUCLEOTIDE_SEQUENCE_RE.fullmatch(compact))


def _amount_with_unit_from_quote(quote: str) -> str | None:
    match = _AMOUNT_WITH_UNIT_RE.search(quote)
    if not match:
        return None
    return " ".join(match.group(0).split())


_BOOLEAN_FIELD_SUFFIX = "_0_1"


def _resolve_boolean_field_entries(group: dict) -> None:
    """"1" always outranks "0" when different candidate quotes disagree --
    real, explicit evidence the condition holds (e.g. "connected to
    compressed air and an overpressure of 2 bar was applied", clearly
    active filtration) beats a separate, merely-uninformative quote that
    didn't happen to state it either way. Mirrors the same precedent
    already established for neg_cont_0_1/pos_cont_0_1 in
    extraction/search_flags.py's own _llm_judged_value_priority."""
    entries = group.get("entries") or []
    winner = "1" if "1" in entries else ("0" if entries else None)
    if winner is None:
        return
    group["entries"] = [winner]
    # Narrow the evidence quotes down to only the ones that actually
    # supported the winning value -- a losing "0" quote (e.g. a mini-
    # flowmeter's own model number, uninformative on its own) shouldn't
    # be cited as evidence for a final "1" answer.
    winning_quotes = group.get("entry_quotes", {}).get(winner)
    if winning_quotes:
        group["quotes"] = winning_quotes


def _add_missing_amount_companions(grouped: dict[str, dict]) -> None:
    for unit_field, amount_field in _UNIT_COMPANION_FIELDS.items():
        if amount_field in grouped or unit_field not in grouped:
            continue
        for quote in grouped[unit_field]["quotes"]:
            if not _valid_context_for_amount_field(amount_field, quote):
                continue
            amount = _amount_with_unit_from_quote(quote)
            if not amount:
                continue
            grouped[amount_field] = {"entries": [amount], "quotes": [quote]}
            break


def _add_filter_passive_active_fallback(grouped: dict[str, dict]) -> None:
    """Default ordinary filtration evidence to passive/0 unless an active
    driving mechanism or active-by-design filter is stated.

    The LLM stage only emits this boolean when the field is offered as a
    candidate and the model chooses to answer it. Real papers often say only
    "filtered through a 0.22 um cartridge", which is enough to fill
    filter_name/size_frac but not enough to contain words like active,
    passive, pump, or pressure. For the FAIRe output this still should not be
    blank: filtration with no stated active driving mechanism is the practical
    passive/default case. Sterivex is an exception: those cartridge filters
    are treated as active even when the paper omits pump/pressure wording.
    """
    quotes: list[str] = []
    for field_name in _FILTER_EVIDENCE_FIELDS:
        group = grouped.get(field_name)
        if not group:
            continue
        for entry in group.get("entries") or []:
            if entry not in quotes:
                quotes.append(entry)
        for quote in group.get("quotes") or []:
            if quote not in quotes:
                quotes.append(quote)
    existing_group = grouped.get(_FILTER_PASSIVE_ACTIVE_FIELD)
    if existing_group:
        if any(_ACTIVE_FILTER_NAME_RE.search(text) for text in quotes):
            existing_group["entry_quotes"].setdefault("1", [])
            for quote in quotes:
                if quote not in existing_group["entry_quotes"]["1"]:
                    existing_group["entry_quotes"]["1"].append(quote)
                if quote not in existing_group["quotes"]:
                    existing_group["quotes"].append(quote)
            if "1" not in existing_group["entries"]:
                existing_group["entries"].append("1")
        return
    if not quotes:
        return

    value = "1" if any(
        _ACTIVE_FILTRATION_CONTEXT_RE.search(quote) or _ACTIVE_FILTER_NAME_RE.search(quote)
        for quote in quotes
    ) else "0"
    grouped[_FILTER_PASSIVE_ACTIVE_FIELD] = {
        "entries": [value],
        "quotes": quotes,
        "entry_quotes": {value: quotes},
    }


def _add_size_frac_values_from_candidate_quotes(
    grouped: dict[str, dict],
    candidates: tuple[_TermQuoteCandidate, ...],
) -> None:
    """Capture every micrometer pore size in size_frac candidate quotes.

    Real filtration sentences often describe a filter cascade, e.g. 180-um,
    5.0-um, and 0.2-um filters in one sentence. The LLM can return only the
    first value even though the quote supports all three; once the sentence
    has already passed the size_frac cue gate, extracting the simple
    micrometer values deterministically is safer than another prompt tweak.
    """
    group = grouped.setdefault(_SIZE_FRAC_FIELD, {"entries": [], "quotes": [], "entry_quotes": {}})
    for candidate in candidates:
        if _SIZE_FRAC_FIELD not in candidate.term_names:
            continue
        for match in _SIZE_FRAC_VALUE_RE.finditer(candidate.text):
            value = " ".join(match.group(0).split())
            key = value.casefold()
            group["entry_quotes"].setdefault(key, [])
            if candidate.text not in group["entry_quotes"][key]:
                group["entry_quotes"][key].append(candidate.text)
            if not any(entry.casefold() == key for entry in group["entries"]):
                group["entries"].append(value)
            if candidate.text not in group["quotes"]:
                group["quotes"].append(candidate.text)
    if not group["entries"]:
        grouped.pop(_SIZE_FRAC_FIELD, None)


def _extract_internal_expedition_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = " ".join(value.strip(" ;,.").split())
        if not value:
            return
        key = value.casefold()
        if key in seen:
            return
        seen.add(key)
        ids.append(value)

    for match in _ODP_LEG_WITH_PREFIX_RE.finditer(text):
        add(f"ODP Leg {match.group('number')}")
    for match in _IODP_EXPEDITION_WITH_PREFIX_RE.finditer(text):
        add(f"IODP Expedition {match.group('number')}")
    if re.search(r"\bOcean\s+Drilling\s+Program\s+\(ODP\)\b", text, re.IGNORECASE):
        for match in _BARE_ODP_LEG_RE.finditer(text):
            add(f"ODP Leg {match.group('number')}")
    if re.search(r"\bIntegrated\s+Ocean\s+Drilling\s+Program\s+\(IODP\)\b", text, re.IGNORECASE):
        for match in _BARE_IODP_EXPEDITION_RE.finditer(text):
            add(f"IODP Expedition {match.group('number')}")
    for pattern in (_RV_CRUISE_RE, _CRUISE_ID_RE):
        for match in pattern.finditer(text):
            add(match.group("id"))
    return ids


def _prepend_expedition_ids_to_raw_value(grouped: dict[str, dict]) -> None:
    group = grouped.get(_INTERNAL_EXPEDITION_ID_FIELD)
    if not group:
        return
    text = " ".join([*(group.get("entries") or []), *(group.get("quotes") or [])])
    ids = _extract_internal_expedition_ids(text)
    if not ids:
        return
    entries = [*(group.get("entries") or []), *(group.get("quotes") or [])]
    id_keys = {identifier.casefold() for identifier in ids}
    retained_seen: set[str] = set()
    retained_entries: list[str] = []
    for entry in entries:
        key = entry.casefold()
        if key in id_keys or key in retained_seen:
            continue
        retained_seen.add(key)
        retained_entries.append(entry)
    group["entries"] = [*ids, *retained_entries]


# A real live audit (10.7717/peerj.9857) caught dna_cleanup_0_1 left blank
# while dna_cleanup_method resolved to a real value ("PCR clean-up kit
# (Fermentas)") from the exact same quote -- the boolean's own cue list
# ("DNA was cleaned", "cleanup was performed", ...) is narrower than the
# method field's ("cleaned using", "cleanup kit", ...) and simply didn't
# happen to match "Amplicons were cleaned using PCR clean-up kit
# (Fermentas)". Rather than keep the two cue lists in permanent lockstep,
# derive the boolean directly from the method field resolving at all: a
# real cleanup method being named is definitionally "yes, a cleanup
# happened" -- same "derive the companion from evidence that already
# resolved" shape as _add_filter_passive_active_fallback above.
_BOOLEAN_COMPANION_FROM_METHOD_FIELDS = {"dna_cleanup_method": "dna_cleanup_0_1"}


def _add_missing_boolean_companions_from_method(grouped: dict[str, dict]) -> None:
    for method_field, boolean_field in _BOOLEAN_COMPANION_FROM_METHOD_FIELDS.items():
        if boolean_field in grouped:
            continue
        method_group = grouped.get(method_field)
        if not method_group or not method_group.get("quotes"):
            continue
        quotes = method_group["quotes"]
        grouped[boolean_field] = {"entries": ["1"], "quotes": quotes, "entry_quotes": {"1": quotes}}


# Same rationale/precedent as _MAX_SENTENCES_PER_CATEGORIZATION_CALL
# above (Stage 2's own, confirmed-live version of this same truncation
# risk): per an explicit user request to prioritize thoroughness over
# speed, this dense-category call is chunked too, even without its own
# confirmed incident yet -- "no direct evidence yet of this one
# truncating" was never a claim that it couldn't, and matches
# quote_candidates_for_llm_judged_search's own existing max_candidates=40
# precedent (search_flags.py) for the same "candidate quote judging"
# call shape.
_MAX_CANDIDATES_PER_TERM_EXTRACTION_CALL = 40


def extract_category_terms(
    backend: LLMBackend,
    category: SectionCategory,
    run_text: str,
    *,
    locator_prefix: str,
    # Raised from 1024 to this codebase's usual floor alongside the same
    # truncation bug's more severe Stage 2 instance (see categorize_
    # paragraphs).
    max_output_tokens: int | None = MIN_LLM_MAX_OUTPUT_TOKENS,
) -> list[RawFactCandidate]:
    """Stage 3 for one category. `run_text` is Stage 2.5's assembled
    per-category run text (`group_sentences_into_category_runs`'s
    output). `pcr_method_additional`-style fallback_only terms are never
    part of this term-level candidate search (an explicit user
    instruction: "do not independently keyword-search") -- they're left
    for a future fallback pass, not built here.

    Per an explicit user request, this runs one smaller LLM call per
    workflow phase (see extraction/section_categories.py's
    TERM_PHASES_BY_CATEGORY) instead of one call listing every field in
    the category -- each phase's own prompt only lists ITS OWN field
    definitions, in the order these facts naturally appear in a paper's
    Methods narrative, rather than diluting a small model's attention
    with every field regardless of relevance to that batch's quotes.
    Still sub-batched by _MAX_CANDIDATES_PER_TERM_EXTRACTION_CALL within
    a phase if that phase alone has a lot of matching quotes."""
    grouped: dict[str, dict] = {}
    all_candidates: list[_TermQuoteCandidate] = []

    for phase_name, phase_terms in _term_phases(category):
        candidates = _term_candidate_quotes(phase_terms, run_text)
        if not candidates:
            continue
        all_candidates.extend(candidates)

        terms_by_name = {term.native_name: term for term in phase_terms}
        candidates_by_id = {c.quote_id: c for c in candidates}

        for batch_start in range(0, len(candidates), _MAX_CANDIDATES_PER_TERM_EXTRACTION_CALL):
            batch = candidates[batch_start : batch_start + _MAX_CANDIDATES_PER_TERM_EXTRACTION_CALL]
            prompt = _build_term_extraction_prompt(category.label, phase_terms, batch)
            try:
                parsed, _response = backend.generate_json(
                    prompt,
                    system=f'You extract FAIRe "{category.label}" ({phase_name}) facts from supplied quote IDs only.',
                    temperature=0,
                    max_tokens=max_output_tokens,
                )
            except LLMBackendError:
                # A failure in ONE phase's own call (bad JSON, timeout, ...)
                # must not lose every OTHER phase's already-succeeded facts
                # -- splitting one category-wide call into several smaller,
                # independent phase calls (per an explicit user request)
                # otherwise multiplies the odds that *something* fails while
                # making that failure's blast radius the same "lose
                # everything" as before splitting. Confirmed live this was a
                # real regression: a single phase failing on a real cluster
                # run silently zeroed out sample_prep facts (samp_size,
                # pool_dna_num, concentration, x_env_var_block, ...) that
                # used to survive under the old, unsplit single-call design.
                logger.warning(
                    "%s: %s:%s term extraction failed for one batch -- skipping this phase, "
                    "other phases' facts are unaffected",
                    backend.label, category.name, phase_name,
                )
                break
            if parsed is None:
                logger.warning(
                    "%s: %s:%s term extraction returned invalid JSON after retries -- "
                    "skipping this phase, other phases' facts are unaffected",
                    backend.label, category.name, phase_name,
                )
                break
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                field_name = str(item.get("field") or "").strip()
                value = str(item.get("raw_value") or "").strip()
                quote_id = str(item.get("quote_id") or "").strip()
                term = terms_by_name.get(field_name)
                candidate = candidates_by_id.get(quote_id)
                if term is None or candidate is None or not value:
                    continue
                # A real live audit caught the model answering a quote_id with a
                # field name that quote was never even offered for (e.g. Q002 was
                # only tagged/candidate-listed for screen_other, but the model
                # attached its value to min_reads_cutoff instead, since that
                # value happened to also appear verbatim in the same quote text).
                # The verbatim guard below only checks the TEXT, not which field
                # the quote was actually tagged for -- this closes that gap.
                if field_name not in candidate.term_names:
                    continue
                if not _valid_sequence_value(field_name, value):
                    continue
                if not _valid_context_for_amount_field(field_name, candidate.text):
                    continue
                if not _is_valid_short_numeric_value(field_name, value):
                    continue
                if field_name.endswith(_BOOLEAN_FIELD_SUFFIX):
                    # A real live audit (10.1371/journal.pone.0303937) caught
                    # this: filter_passive_active_0_1 is a JUDGED classification
                    # ("1"/"0"), not a copied value -- the verbatim check below is
                    # meaningless for it (a single digit character coincidentally
                    # appears in almost any real quote, e.g. a catalog/model
                    # number), and worse, actively harmful here: the one quote
                    # that unambiguously supports "1" ("...connected to
                    # compressed air and an overpressure of 2 bar was applied.")
                    # has no literal "1" anywhere in it, while an unrelated quote
                    # that only names a flowmeter MODEL NUMBER happened to
                    # contain a stray "0" and won by pure chance. Boolean fields
                    # are validated by shape instead (only "0"/"1" survive), with
                    # the actual value chosen after the loop by _resolve_boolean_
                    # field_entries below.
                    if value not in ("0", "1"):
                        continue
                # Verbatim-only guard: a prompt instruction alone is never trusted
                # to be self-enforcing anywhere else in this codebase -- discard
                # any value that doesn't literally appear in its own cited quote,
                # rather than trusting the model's self-report of verbatim-ness.
                elif value.casefold() not in candidate.text.casefold():
                    continue
                group = grouped.setdefault(field_name, {"entries": [], "quotes": [], "entry_quotes": {}})
                key = value.casefold()
                group["entry_quotes"].setdefault(key, [])
                if candidate.text not in group["entry_quotes"][key]:
                    group["entry_quotes"][key].append(candidate.text)
                if any(entry.casefold() == key for entry in group["entries"]):
                    continue
                group["entries"].append(value)
                if candidate.text not in group["quotes"]:
                    group["quotes"].append(candidate.text)

    candidates = tuple(all_candidates)
    _add_missing_amount_companions(grouped)
    if category.name == "sample_prep":
        _add_size_frac_values_from_candidate_quotes(grouped, candidates)
        _add_filter_passive_active_fallback(grouped)
        _add_missing_boolean_companions_from_method(grouped)
        _prepend_expedition_ids_to_raw_value(grouped)
    for field_name, group in grouped.items():
        if field_name.endswith(_BOOLEAN_FIELD_SUFFIX):
            _resolve_boolean_field_entries(group)

    facts: list[RawFactCandidate] = []
    for field_name, group in grouped.items():
        facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=field_name,
                raw_field_name=field_name,
                raw_value=" | ".join(group["entries"]),
                source_locator=f"{locator_prefix}:section_category_terms:{category.name}:{field_name}",
                support_type=SupportType.EXPLICIT,
                evidence_quote=" | ".join(group["quotes"]),
                confidence_metadata={"detector": "section_category_term_extraction", "category": category.name},
            )
        )
    return facts


def extract_section_category_facts(
    backend: LLMBackend,
    texts: list[tuple[str, str]],
    *,
    locator_prefix: str,
) -> list[RawFactCandidate]:
    """The full Stage 1 -> 2 -> 2.5 -> 3 pipeline across every supplied
    (title, text) pair. Additive alongside every existing extraction
    mechanism -- see module docstring."""
    gated_paragraphs: list[tuple[str, frozenset[str]]] = []
    for _title, text in texts:
        for paragraph in paragraphs_before_next_section_heading(text):
            candidates = candidate_categories_for_paragraph(paragraph)
            if candidates:
                gated_paragraphs.append((paragraph, candidates))
    if not gated_paragraphs:
        return []

    tagged_by_paragraph = categorize_paragraphs(backend, gated_paragraphs)

    run_texts_by_category: dict[str, list[str]] = {}
    for tagged_sentences in tagged_by_paragraph.values():
        for category_name, run_text in group_sentences_into_category_runs(tagged_sentences).items():
            if run_text:
                run_texts_by_category.setdefault(category_name, []).append(run_text)

    categories_by_name = {category.name: category for category in SECTION_CATEGORIES}
    facts: list[RawFactCandidate] = []
    for category_name, run_texts in run_texts_by_category.items():
        category = categories_by_name[category_name]
        combined_run_text = " ".join(run_texts)
        facts.extend(
            extract_category_terms(backend, category, combined_run_text, locator_prefix=locator_prefix)
        )
    facts.extend(_fallback_only_leftover_facts(run_texts_by_category, locator_prefix=locator_prefix))
    return facts


def _quote_text_for_fact(fact: RawFactCandidate) -> str:
    return " ".join((fact.evidence_quote or fact.raw_value or "").split())


def _split_pipe_values(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


_LEAKED_PLACEHOLDER_RE = re.compile(r"<[^<>]*>")


def _normalization_parts(value: object) -> list[str]:
    if isinstance(value, list):
        pieces = [str(piece).strip() for piece in value]
    else:
        pieces = [piece.strip() for piece in str(value or "").split("|")]
    normalized: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        piece = piece.strip(" ;,.")
        if not piece or piece.casefold() in {"none", "not found", "n/a", "not applicable"}:
            continue
        # Real gap found live (PMC10988111): several of these fields' own
        # normalization prompts instruct "Use other:<short literal
        # technique/device/method>" as a FORMAT template for the model to
        # fill in -- a weak/confused model can echo that placeholder back
        # verbatim ("other:<short literal technique>") instead of actually
        # substituting a real value. A real extracted term never contains
        # literal angle brackets, so this is an unambiguous leak signal,
        # not a real value worth keeping under any interpretation.
        if _LEAKED_PLACEHOLDER_RE.search(piece):
            continue
        key = piece.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(piece)
    return normalized


def _sort_terms_by_first_occurrence(terms: list[str], reference_text: str) -> list[str]:
    """Orders normalized terms by where each one's evidence actually first
    appears in the source quotes, instead of leaving them in whatever order
    the normalization LLM call happened to emit its JSON list in -- there
    was previously no ordering policy at all here, confirmed live: a real
    nucl_acid_ext_sep value had its terms in an order that matched neither
    the paper's own text nor any obvious technique sequence. A term whose
    exact text isn't found verbatim in the quotes (e.g. the model
    paraphrased it) sorts after every term that was found, keeping its
    original relative position among other not-found terms."""
    reference = reference_text.casefold()

    aliases = {
        "sonication": ("sonication", "sonicated", "ultrasonic", "ultrasonication"),
        "ultrasonication": ("ultrasonication", "ultrasonic", "sonication", "sonicated"),
        "bead beating": ("bead beating", "bead-beating", "bead beaten", "bead beat"),
        "freeze-thaw lysis": ("freeze-thaw", "freeze thaw"),
        "proteinase k digestion": ("proteinase k digestion", "proteinase k"),
        "lysozyme lysis": ("lysozyme lysis", "lysozyme"),
        "sds lysis": ("sds lysis", "sds"),
        "ctab lysis": ("ctab lysis", "ctab"),
        "phenol-chloroform extraction": ("phenol-chloroform extraction", "phenol/chloroform", "phenol chloroform"),
        "phenol-chloroform-isoamyl alcohol extraction": (
            "phenol-chloroform-isoamyl alcohol",
            "phenol/chloroform/isoamyl alcohol",
            "phenol chloroform isoamyl alcohol",
        ),
        "silica spin-column": ("silica spin-column", "spin column", "spin-column"),
        "magnetic-bead purification": ("magnetic-bead", "magnetic bead"),
        "freeze-drying": ("freeze-drying", "freeze dried", "freeze-dried", "lyophilized", "lyophilised"),
        "lyophilization": ("lyophilization", "lyophilized", "lyophilised", "freeze-drying", "freeze-dried"),
        "filtration": ("filtration", "filtered", "filtering"),
        "prefiltration": ("prefiltration", "pre-filtered", "prefiltered", "pre-filtering", "prefiltering"),
        "grinding": ("grinding", "ground"),
        "manual grinding": ("manual grinding", "ground"),
        "mortar-and-pestle grinding": ("mortar and pestle", "mortar-and-pestle"),
        "integrated-depth water sampling": ("integrated-depth", "integrated depth", "integrated water"),
        "depth-specific water sampling": ("depth-specific", "depth specific"),
    }

    def _position(term: str) -> int:
        positions = [
            reference.find(alias)
            for alias in aliases.get(term.casefold(), (term.casefold(),))
            if reference.find(alias) >= 0
        ]
        return min(positions) if positions else len(reference)

    return sorted(terms, key=_position)


def normalize_controlled_sample_prep_facts(
    backend: LLMBackend,
    facts: list[RawFactCandidate],
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
) -> list[RawFactCandidate]:
    """Second pass over already-extracted long-form sample-prep quotes only.

    The first-pass extractors intentionally keep source-faithful sentences.
    This refinement turns selected fields into short comparable technique
    names, while temporarily appending the original source sentences after
    the normalized names so audits can judge the normalization quickly.
    """
    facts_by_field: dict[str, list[str]] = {}
    for spec in _FIELD_NORMALIZATION_SPECS:
        quotes: list[str] = []
        seen: set[str] = set()
        for fact in facts:
            if fact.fact_type_candidate != spec.field_name:
                continue
            quote = _quote_text_for_fact(fact)
            if not quote:
                continue
            for part in _split_pipe_values(quote):
                cleaned = " ".join(part.split())
                key = cleaned.casefold()
                if cleaned and key not in seen:
                    seen.add(key)
                    quotes.append(cleaned)
        if quotes:
            facts_by_field[spec.field_name] = quotes

    normalized_facts: list[RawFactCandidate] = []
    for spec in _FIELD_NORMALIZATION_SPECS:
        quotes = facts_by_field.get(spec.field_name)
        if not quotes:
            continue

        quote_block = "\n".join(f"Q{index:03d}: {quote}" for index, quote in enumerate(quotes, start=1))
        prompt = f"""{spec.prompt}

Use only these supplied quotes:
{quote_block}

Return ONLY a JSON object: {{"{spec.field_name}": "term1 | term2"}}
"""
        parsed, _response = backend.generate_json(
            prompt,
            system=spec.system,
            temperature=0,
            max_tokens=max_output_tokens,
        )
        if parsed is None:
            raise LLMBackendError(f"{backend.label}: {spec.field_name} normalization returned invalid JSON after retries")
        value = parsed.get(spec.field_name) if isinstance(parsed, dict) else ""
        normalized = _normalization_parts(value)
        if not normalized:
            continue
        normalized = _sort_terms_by_first_occurrence(normalized, " ".join(quotes))

        normalized_field = f"{spec.field_name}{_NORMALIZATION_SUFFIX}"
        normalized_facts.append(
            RawFactCandidate(
                entity_level=EntityLevel.STUDY,
                fact_type_candidate=normalized_field,
                raw_field_name=normalized_field,
                raw_value=" | ".join([*normalized, *quotes]),
                source_locator=f"{locator_prefix}:{spec.field_name}:normalized_from_extracted_quotes",
                support_type=SupportType.EXPLICIT,
                evidence_quote=" | ".join(quotes),
                confidence_metadata={
                    "detector": f"llm_normalized_{spec.field_name}",
                    "source_quote_count": len(quotes),
                },
            )
        )
    return normalized_facts


# A category can have more than one fallback_only term when a specific
# sub-topic warrants its own catch-all separate from the category's
# general one -- per an explicit user request: nucl_acid_ext_method_
# additional should only catch nucleic-acid-extraction-shaped leftover
# text within sample_prep, not the whole category's leftovers (that's
# prep_method_additional's own job). A leftover sentence matching a
# listed native_name's own predicate is claimed by that term FIRST; only
# sentences no entry here claims fall through to the category's other
# (unscoped) fallback_only term(s).
_NUCL_ACID_EXT_GENERIC_PROCEDURE_RE = re.compile(
    r"\b(?:"
    r"bead[-\s]?beat(?:ing|en)?|zirconium\s+beads?|sonicat(?:ed|ion)|ultrasonic|"
    r"centrifug(?:ed|ation)|wash(?:ed|ing)?\s+step|"
    r"precipitat(?:ed|ion)|elut(?:ed|ion)|reagent|repeated\s+(?:treatment|extraction)|"
    r"extraction\s+(?:step|procedure|protocol)|DNA\s+(?:was\s+)?extract\w*|"
    r"RNA\s+(?:was\s+)?extract\w*|nucleic\s+acid\w*\s+extract\w*"
    r")\b",
    re.IGNORECASE,
)
# "incubat(?:ed|ion)" alone was a real bug found live (10.1093/ismejo/
# wrae013): a paragraph entirely about a flux-incubation EXPERIMENT
# ("submersed... inside two incubation chambers... Throughout the
# acclimation phase...") -- nothing to do with nucleic acid extraction at
# all -- got swept into nucl_acid_ext_method_additional purely because it
# mentions "incubation chambers". Unlike the other generic procedure
# words above (bead-beating, centrifugation, reagent, ...), "incubated"
# is used constantly across unrelated wet-lab/experimental contexts, so
# it's scoped separately here, requiring an actual nucleic-acid/
# extraction-specific anchor word nearby before it counts.
_NUCL_ACID_EXT_INCUBATION_RE = re.compile(r"\bincubat(?:ed|ion)\b", re.IGNORECASE)
_NUCL_ACID_EXT_CONTEXT_ANCHOR_RE = re.compile(
    r"\bDNA\b|\bRNA\b|nucleic\s+acid|\blysis\b|\bextraction\b|\bkit\b|\bbuffer\b|\bpellet\b|\belution\b",
    re.IGNORECASE,
)


def _matches_nucl_acid_ext_method_additional(sentence: str) -> bool:
    if _NUCL_ACID_EXT_GENERIC_PROCEDURE_RE.search(sentence):
        return True
    return bool(
        _NUCL_ACID_EXT_INCUBATION_RE.search(sentence) and _NUCL_ACID_EXT_CONTEXT_ANCHOR_RE.search(sentence)
    )


_SCOPED_FALLBACK_KEYWORD_FILTERS: dict[str, Callable[[str], bool]] = {
    "nucl_acid_ext_method_additional": _matches_nucl_acid_ext_method_additional,
}


def _fallback_only_terms_for_category(category: SectionCategory) -> list[CategoryTerm]:
    return [term for term in category.terms if term.fallback_only]


def _fallback_only_leftover_facts(
    run_texts_by_category: dict[str, list[str]], *, locator_prefix: str
) -> list[RawFactCandidate]:
    """Populates each category's own fallback_only term(s) (prep_method_
    additional, nucl_acid_ext_method_additional, pcr_method_additional,
    pcr2_method_additional, seq_method_additional, targeted_detection_
    method_additional) with whatever in-category sentences didn't match
    any other term's own cues -- this is the ORIGINAL design intent for
    fallback_only terms ("captures whatever in-category text didn't
    match any other term's own cues... do not independently keyword-
    search", see CategoryTerm's own docstring), confirmed live to have
    never actually been built: extract_category_terms's own terms_by_name
    excludes fallback_only terms entirely, so nothing ever populated them
    (a real gap found live, 10.1371/journal.pone.0303937: prep_method_
    additional stayed empty for a paper whose Methods described a
    compressed-air/overpressure filtration rig, an Arduino flowmeter, and
    threshold-based filtration stopping in real detail that fit no other
    sample_prep field). Deliberately separate from bioinfo_method_
    additional above, which is a different, purpose-built keyword-driven
    mechanism scoped to the bioinformatics categories specifically, not
    this generic one."""
    categories_by_name = {category.name: category for category in SECTION_CATEGORIES}
    facts: list[RawFactCandidate] = []
    for category_name, run_texts in run_texts_by_category.items():
        category = categories_by_name.get(category_name)
        if category is None:
            continue
        fallback_terms = _fallback_only_terms_for_category(category)
        if not fallback_terms:
            continue
        scoped_terms = [t for t in fallback_terms if t.native_name in _SCOPED_FALLBACK_KEYWORD_FILTERS]
        unscoped_terms = [t for t in fallback_terms if t.native_name not in _SCOPED_FALLBACK_KEYWORD_FILTERS]
        searchable_terms = tuple(term for term in category.terms if not term.fallback_only)
        leftover_by_term: dict[str, list[str]] = {term.native_name: [] for term in fallback_terms}
        seen_sentences: set[str] = set()
        for run_text in run_texts:
            for sentence in _split_into_sentences(run_text):
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                if any(
                    _term_pattern(cue).search(sentence)
                    for term in searchable_terms
                    for cue in term.search_cues
                ):
                    continue
                claimed_by = next(
                    (t for t in scoped_terms if _SCOPED_FALLBACK_KEYWORD_FILTERS[t.native_name](sentence)),
                    None,
                )
                if claimed_by is not None:
                    leftover_by_term[claimed_by.native_name].append(sentence)
                elif unscoped_terms:
                    leftover_by_term[unscoped_terms[0].native_name].append(sentence)
        for term_name, leftover_sentences in leftover_by_term.items():
            if not leftover_sentences:
                continue
            combined = " ".join(leftover_sentences)
            facts.append(
                RawFactCandidate(
                    entity_level=EntityLevel.STUDY,
                    fact_type_candidate=term_name,
                    raw_field_name=term_name,
                    raw_value=combined,
                    source_locator=f"{locator_prefix}.{category_name}.fallback_leftover",
                    support_type=SupportType.EXPLICIT,
                    evidence_quote=combined,
                )
            )
    return facts


# x_pulled_env_var: a dedicated second pass over x_env_var_block's OWN raw
# quotes (never re-reads the paper), per an explicit user request --
# x_env_var_block stays exactly as-is (a broad, cue-gated capture of every
# candidate environmental-measurement sentence), and this field extracts
# only the genuine "name = value" pairs out of it. Two real, live-audit-
# confirmed failure modes motivate this: (1) a candidate sentence can
# match x_env_var_block's own cues (e.g. "chlorophyll maxima") while
# stating no actual measured number at all -- a bare mention, not a
# value; (2) a candidate sentence can report a real number that belongs to
# a LATER experimental manipulation (a climate room, an incubation
# chamber, an acclimation phase) rather than the original sample's own
# in-situ condition at collection, and x_env_var_block's cue-gate alone
# can't distinguish the two since both use the same variable names. Same
# "dedicated single-purpose function" mechanism as study_factor.py's
# generate_study_factor -- one field, one custom prompt -- but grounded
# like every other quote-gated field in this module: the numeric value
# itself must appear verbatim in its own cited quote (only the variable
# NAME is allowed to be a light paraphrase/abbreviation of the quote's own
# wording), so SupportType.EXPLICIT applies, not INFERRED.
def extract_pulled_env_var_facts(
    backend: LLMBackend,
    section_category_term_facts: list[RawFactCandidate],
    *,
    locator_prefix: str,
    max_output_tokens: int | None = MIN_LLM_MAX_OUTPUT_TOKENS,
) -> list[RawFactCandidate]:
    env_var_fact = next(
        (fact for fact in section_category_term_facts if fact.fact_type_candidate == "x_env_var_block"),
        None,
    )
    if env_var_fact is None or not env_var_fact.evidence_quote:
        return []

    quotes = [q.strip() for q in env_var_fact.evidence_quote.split(" | ") if q.strip()]
    if not quotes:
        return []

    quote_lines = "\n".join(f"Q{i + 1:03d}: {quote}" for i, quote in enumerate(quotes))
    prompt = f"""You are extracting genuine measured environmental-variable values that describe the ORIGINAL
environmental sample itself (the water, sediment, soil, or other material at its point of collection) -- its
in-situ condition at the time and place the sample was actually taken.

From each candidate quote below, return one object per genuine "variable = value" pair, but ONLY when both of
these hold:
- The quote states an explicit measured number together with its variable (temperature, salinity, pH,
  chlorophyll, dissolved oxygen, a nutrient concentration, etc.). A quote that only mentions a variable's name in
  passing, with no actual number attached to it, supports nothing -- return nothing for that quote.
- The value describes the ORIGINAL sample's own in-situ condition at collection. Do NOT extract a value from a
  LATER experimental manipulation, incubation, acclimation, climate room, laboratory chamber, or other
  controlled/testing condition, even if it reuses the same variable name and even if a real number is present
  (e.g. a "climate room set to an in situ bottom water temperature of ~6C" or an "acclimation phase" measurement
  is NOT the original sample and must be excluded).

Do not invent a value. Copy the numeric value and its unit exactly as it appears in the quote.

Candidate quotes:
{quote_lines}

Return ONLY a JSON array. Each object must be:
{{"variable": "<short name or chemical formula/abbreviation for the variable>", "value": "<value copied verbatim from the quote, including its unit>", "quote_id": "Q001"}}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You extract only genuine, in-situ, verbatim-valued environmental measurements from supplied quote IDs.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: x_pulled_env_var extraction returned invalid JSON after retries")
    if not isinstance(parsed, list):
        return []

    quotes_by_id = {f"Q{i + 1:03d}": quote for i, quote in enumerate(quotes)}
    pairs: list[str] = []
    used_quotes: list[str] = []
    seen_pairs: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        variable = str(item.get("variable") or "").strip()
        value = str(item.get("value") or "").strip()
        quote_id = str(item.get("quote_id") or "").strip()
        quote = quotes_by_id.get(quote_id)
        if not variable or not value or quote is None:
            continue
        if value.casefold() not in quote.casefold():
            continue
        pair = f"{variable} = {value}"
        if pair.casefold() in seen_pairs:
            continue
        seen_pairs.add(pair.casefold())
        pairs.append(pair)
        if quote not in used_quotes:
            used_quotes.append(quote)

    if not pairs:
        return []

    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="x_pulled_env_var",
            raw_field_name="x_pulled_env_var",
            raw_value=" | ".join(pairs),
            source_locator=f"{locator_prefix}:x_pulled_env_var:llm_refined_from_env_var_block",
            support_type=SupportType.EXPLICIT,
            evidence_quote=" | ".join(used_quotes),
            confidence_metadata={"detector": "x_pulled_env_var_refinement"},
        )
    ]
