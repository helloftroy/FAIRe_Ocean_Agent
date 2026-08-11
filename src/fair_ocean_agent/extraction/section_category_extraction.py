"""Stage 2 (LLM sentence-level categorization) and Stage 3 (category-scoped
term extraction): the two real LLM calls in the PCR/library-prep/
bioinformatics methods categorize-then-extract pipeline. extraction/
section_categories.py holds the deterministic building blocks this module
calls into (Stage 1's paragraph keyword gate, Stage 2.5's run-grouping);
this module is where the LLM actually gets invoked.

Both calls follow the same "quote-candidate-then-judge" discipline already
proven in extraction/search_flags.py's LLMJudgedSearchField mechanism: the
LLM is never shown free-running whole-document text, only already
keyword-gated candidates, and every one of the 114 term definitions
carries the SAME shared instruction to copy values verbatim rather than
generate them -- an explicit user instruction ("In all cases, have the
program take word for word from text and not generate."), applied via one
shared prompt template rather than 114 bespoke output instructions, plus a
programmatic guard (Stage 3 discards any extracted value that doesn't
literally appear in its own cited quote) since a prompt instruction alone
is never trusted to be self-enforcing anywhere else in this codebase.

Deliberately additive, not a replacement: every one of these 114 fields
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

from fair_ocean_agent.database.enums import EntityLevel, SupportType
from fair_ocean_agent.extraction.section_categories import (
    SECTION_CATEGORIES,
    SectionCategory,
    _term_pattern,
    candidate_categories_for_paragraph,
    group_sentences_into_category_runs,
    low_confidence_categories,
    split_into_paragraphs,
)
from fair_ocean_agent.llm.base import LLMBackend, LLMBackendError
from fair_ocean_agent.sources.base import RawFactCandidate

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_ID_RE = re.compile(r"^S(\d+)\.(\d+)$")


def _split_into_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


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


def categorize_paragraphs(
    backend: LLMBackend,
    gated_paragraphs: list[tuple[str, frozenset[str]]],
    *,
    max_output_tokens: int | None = 1024,
) -> dict[int, list[tuple[str, frozenset[str]]]]:
    """Stage 2. `gated_paragraphs` is (paragraph_text, candidate_categories)
    pairs, already filtered by Stage 1's keyword gate -- typically
    `split_into_paragraphs` + `candidate_categories_for_paragraph` applied
    across a document's texts. Returns {paragraph_index:
    tagged_sentences}, `paragraph_index` matching `gated_paragraphs`'s own
    position, ready for `group_sentences_into_category_runs`.

    One call for the whole batch (not one per paragraph) -- the
    efficiency the categorize-then-extract redesign was for in the first
    place."""
    if not gated_paragraphs:
        return {}
    indexed = [
        (index, _split_into_sentences(paragraph), candidates)
        for index, (paragraph, candidates) in enumerate(gated_paragraphs)
    ]
    indexed = [(index, sentences, candidates) for index, sentences, candidates in indexed if sentences]
    if not indexed:
        return {}

    prompt = _build_categorization_prompt(indexed)
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

    tags_by_sentence: dict[tuple[int, int], frozenset[str]] = {}
    if isinstance(parsed, list):
        valid_category_names = {category.name for category in SECTION_CATEGORIES}
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


def _term_candidate_quotes(category: SectionCategory, run_text: str) -> tuple[_TermQuoteCandidate, ...]:
    candidates: list[_TermQuoteCandidate] = []
    seen_text: set[str] = set()
    searchable_terms = tuple(term for term in category.terms if not term.fallback_only)
    for sentence in _split_into_sentences(run_text):
        if sentence in seen_text:
            continue
        matched_terms = tuple(
            term.native_name
            for term in searchable_terms
            if any(_term_pattern(cue).search(sentence) for cue in term.search_cues)
        )
        if not matched_terms:
            continue
        seen_text.add(sentence)
        candidates.append(_TermQuoteCandidate(quote_id=f"Q{len(candidates) + 1:03d}", term_names=matched_terms, text=sentence))
    return tuple(candidates)


def _build_term_extraction_prompt(category: SectionCategory, candidates: tuple[_TermQuoteCandidate, ...]) -> str:
    fields_reference = "\n".join(
        f"- {term.native_name}: {term.definition}" for term in category.terms if not term.fallback_only
    )
    quotes = "\n".join(f"{c.quote_id} [{', '.join(c.term_names)}]: {c.text}" for c in candidates)
    return f"""You are extracting FAIRe "{category.label}" fields from supplied quote IDs only.

Use only the candidate quotes below. Do not use outside knowledge, do not infer, and do not paraphrase -- copy
the relevant value WORD FOR WORD from the quote text exactly as reported (numbers, units, software names,
sequences, and phrases must all be verbatim, not generated or reworded). Return a field only if a quote
explicitly supports it. If multiple distinct values for the same field are explicitly supported by different
quotes, return one object per value.

Fields:
{fields_reference}

Return ONLY a JSON array. Each object must be:
{{"field": "<one listed field>", "raw_value": "<verbatim value copied from the quote>", "quote_id": "Q001"}}

Candidate quotes:
{quotes}
"""


def extract_category_terms(
    backend: LLMBackend,
    category: SectionCategory,
    run_text: str,
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 1024,
) -> list[RawFactCandidate]:
    """Stage 3 for one category. `run_text` is Stage 2.5's assembled
    per-category run text (`group_sentences_into_category_runs`'s
    output). `pcr_method_additional`-style fallback_only terms are never
    part of this term-level candidate search (an explicit user
    instruction: "do not independently keyword-search") -- they're left
    for a future fallback pass, not built here."""
    candidates = _term_candidate_quotes(category, run_text)
    if not candidates:
        return []

    prompt = _build_term_extraction_prompt(category, candidates)
    parsed, _response = backend.generate_json(
        prompt,
        system=f'You extract FAIRe "{category.label}" facts from supplied quote IDs only.',
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: {category.name} term extraction returned invalid JSON after retries")
    if not isinstance(parsed, list):
        return []

    terms_by_name = {term.native_name: term for term in category.terms if not term.fallback_only}
    candidates_by_id = {c.quote_id: c for c in candidates}
    grouped: dict[str, dict] = {}
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
        # Verbatim-only guard: a prompt instruction alone is never trusted
        # to be self-enforcing anywhere else in this codebase -- discard
        # any value that doesn't literally appear in its own cited quote,
        # rather than trusting the model's self-report of verbatim-ness.
        if value.casefold() not in candidate.text.casefold():
            continue
        group = grouped.setdefault(field_name, {"entries": [], "quotes": []})
        key = value.casefold()
        if any(entry.casefold() == key for entry in group["entries"]):
            continue
        group["entries"].append(value)
        if candidate.text not in group["quotes"]:
            group["quotes"].append(candidate.text)

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
    suppressed = low_confidence_categories(texts)
    gated_paragraphs: list[tuple[str, frozenset[str]]] = []
    for _title, text in texts:
        for paragraph in split_into_paragraphs(text):
            candidates = candidate_categories_for_paragraph(paragraph) - suppressed
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
    facts.extend(_bioinfo_method_additional_fact(run_texts_by_category, locator_prefix=locator_prefix))
    facts.extend(
        _generate_otu_raw_description_fact(backend, run_texts_by_category, locator_prefix=locator_prefix)
    )
    facts.extend(
        _generate_tax_class_other_fact(backend, run_texts_by_category, locator_prefix=locator_prefix)
    )
    return facts


# The categories that make up "the bioinformatics pipeline section" for
# bioinfo_method_additional's purposes -- raw-read preprocessing through
# OTU/ASV generation through taxonomic assignment, deliberately excluding
# the wet-lab categories (assay definition, PCR1/PCR2, library prep) that
# precede sequencing.
_BIOINFORMATICS_CATEGORY_NAMES = (
    "raw_read_preprocessing",
    "otu_asv_generation_filtering",
    "taxonomic_assignment",
)

_BIOINFO_PIPELINE_SENTENCE_RE = re.compile(
    r"\b(?:"
    r"raw\s+(?:paired[-\s]+end\s+)?reads?|sequence\s+reads?|clean\s+reads?|"
    r"quality\s+(?:check|control|filter(?:ed|ing)?)|trim(?:med|ming)|"
    r"demultiplex(?:ed|ing)?|merge(?:d|ing)?|denois(?:ed|ing)?|dereplicat(?:ed|ion)|"
    r"FastQC|Trimmomatic|Cutadapt|fastp|QIIME2?|DADA2|UCHIME|VSEARCH|USEARCH|UPARSE|"
    r"ASVs?|amplicon\s+sequence\s+variants?|OTUs?|operational\s+taxonomic\s+units?|"
    r"feature\s+table|raref(?:ied|action)|chimer(?:a|as|ic)|bimeras?|"
    r"cluster(?:ed|ing)?|reference\s+(?:database|library|sequences?)|"
    r"SILVA|PR2|GenBank|UNITE|RDP|BOLD|MIDORI|taxonom(?:y|ic|ically)|"
    r"classified\s+taxonomically|taxonomy\s+was\s+assigned|assigned\s+taxonomy|BLASTn?|"
    r"classify-sklearn|feature-classifier|naive\s+Bayes"
    r")\b",
    re.IGNORECASE,
)

_DOWNSTREAM_COMMUNITY_STATS_RE = re.compile(
    r"\b(?:"
    r"correlation\s+plots?|correlation\s+between|PAST|SIMPER|similarity\s+percentage\s+analysis|"
    r"alpha\s+diversity|diversity\s+indices|Shannon|Simpson|Pielou|richness|observed\s+species|"
    r"physicochemical|vertical\s+(?:comparison|distribution)|prokaryotic\s+distribution|"
    r"differences\s+between|dissimilarity|compared\s+(?:against|to)|comparison\s+of|"
    r"sample\s+groups?|water\s+depths?|depth\s+across|taxa\s+change\s+with\s+depth|"
    r"abundances?\s+between\s+the\s+samples?"
    r")\b",
    re.IGNORECASE,
)


def _bioinfo_pipeline_sentences(run_text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in _split_into_sentences(run_text):
        if not _BIOINFO_PIPELINE_SENTENCE_RE.search(sentence):
            continue
        if _DOWNSTREAM_COMMUNITY_STATS_RE.search(sentence):
            continue
        sentences.append(sentence)
    return sentences


def _bioinfo_method_additional_fact(
    run_texts_by_category: dict[str, list[str]], *, locator_prefix: str
) -> list[RawFactCandidate]:
    """Per an explicit user request, now that the bioinformatics section
    is classified (Stage 1/2/2.5 above): capture source-faithful
    bioinformatics-pipeline sentences from Stage 2.5's assembled runs. A
    final deterministic sentence filter keeps raw-read/feature/taxonomic
    assignment processing and drops downstream ecological/statistical
    analyses that were over-tagged by the sentence classifier."""
    sections = [
        " ".join(
            sentence
            for run_text in run_texts_by_category[category_name]
            for sentence in _bioinfo_pipeline_sentences(run_text)
        )
        for category_name in _BIOINFORMATICS_CATEGORY_NAMES
        if run_texts_by_category.get(category_name)
    ]
    sections = [section for section in sections if section]
    if not sections:
        return []
    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="bioinfo_method_additional",
            raw_field_name="bioinfo_method_additional",
            raw_value=" || ".join(sections),
            source_locator=f"{locator_prefix}:section_category_terms:bioinfo_method_additional",
            support_type=SupportType.DETERMINISTICALLY_DERIVED,
            confidence_metadata={
                "detector": "bioinformatics_category_run_text_capture",
                "categories": [
                    category_name for category_name in _BIOINFORMATICS_CATEGORY_NAMES if run_texts_by_category.get(category_name)
                ],
            },
        )
    ]


def _generate_otu_raw_description_fact(
    backend: LLMBackend,
    run_texts_by_category: dict[str, list[str]],
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
) -> list[RawFactCandidate]:
    """otu_raw_description is deliberately GENERATED, not extracted, per
    an explicit user instruction: "i'd prefer if the LLM generates 1-2
    sentences of its own description of the OTU process, the quotes
    captured are not meaningful for either paper" -- confirmed against a
    real paper whose only "raw OTU table" sentence was an unhelpful
    cross-reference to another paper's methods ("we ... employed the same
    data analysis pipeline") rather than any real description. Mirrors
    extraction/study_factor.py's generative pattern (SupportType.INFERRED,
    evidence_quote holds the source material rather than a literal
    supporting quote), but summarizes this category's own run-text
    instead of the paper's abstract.
    """
    run_texts = run_texts_by_category.get("otu_asv_generation_filtering")
    if not run_texts:
        return []
    source_text = " ".join(run_texts)
    prompt = f"""Write your own concise 1-2 sentence description of how the initial/raw OTU, ASV, or feature
table was generated and initially processed, based on the source text below. Summarize in your own words --
do not copy sentences verbatim from the source text. If the source text does not actually describe this
processing (for example, it only refers to another paper's methods without giving any real detail), briefly
say so rather than inventing detail.

Return ONLY a JSON object: {{"otu_raw_description": "<your 1-2 sentence description>"}}

Source text:
{source_text}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You summarize a paper's own OTU/ASV/feature-table generation process in your own words from its methods text.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: otu_raw_description generation returned invalid JSON after retries")
    sentence = str(parsed.get("otu_raw_description") or "").strip() if isinstance(parsed, dict) else ""
    if not sentence:
        return []
    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="otu_raw_description",
            raw_field_name="otu_raw_description",
            raw_value=sentence,
            source_locator=f"{locator_prefix}:section_category_terms:otu_raw_description:llm_generated",
            support_type=SupportType.INFERRED,
            evidence_quote=source_text,
            confidence_metadata={"detector": "llm_generated_otu_raw_description"},
        )
    ]


def _generate_tax_class_other_fact(
    backend: LLMBackend,
    run_texts_by_category: dict[str, list[str]],
    *,
    locator_prefix: str,
    max_output_tokens: int | None = 256,
) -> list[RawFactCandidate]:
    """tax_class_other is deliberately GENERATED, not extracted, per an
    explicit user instruction: "tax_class_other can be all classified
    'TAXONOMIC ASSIGNMENT'. can ask the LLM to summarize based on the
    section classified 'TAXONOMIC ASSIGNMENT'." -- rather than requiring
    one narrow quote to explicitly state "additional parameters/cutoffs"
    verbatim, this summarizes the whole taxonomic_assignment category's
    already-classified (Stage 2) run-text in the model's own words.
    Mirrors _generate_otu_raw_description_fact's generative pattern."""
    run_texts = run_texts_by_category.get("taxonomic_assignment")
    if not run_texts:
        return []
    source_text = " ".join(run_texts)
    prompt = f"""Write your own concise 1-2 sentence summary of the taxonomic-assignment parameters, cutoffs,
thresholds, ambiguity-handling rules, or other taxonomic-assignment details described in the source text below
-- the FAIRe field this feeds is defined as "additional information on parameters and cutoffs used for
taxonomic assignment". Summarize in your own words -- do not copy sentences verbatim from the source text. If
the source text does not describe any such additional parameters/cutoffs/rules (for example, it only names the
tool and reference database with no further detail), return an empty string rather than inventing detail.

Return ONLY a JSON object: {{"tax_class_other": "<your 1-2 sentence summary, or empty string>"}}

Source text:
{source_text}
"""
    parsed, _response = backend.generate_json(
        prompt,
        system="You summarize a paper's own taxonomic-assignment parameters/cutoffs in your own words from its methods text.",
        temperature=0,
        max_tokens=max_output_tokens,
    )
    if parsed is None:
        raise LLMBackendError(f"{backend.label}: tax_class_other generation returned invalid JSON after retries")
    sentence = str(parsed.get("tax_class_other") or "").strip() if isinstance(parsed, dict) else ""
    if not sentence:
        return []
    return [
        RawFactCandidate(
            entity_level=EntityLevel.STUDY,
            fact_type_candidate="tax_class_other",
            raw_field_name="tax_class_other",
            raw_value=sentence,
            source_locator=f"{locator_prefix}:section_category_terms:tax_class_other:llm_generated",
            support_type=SupportType.INFERRED,
            evidence_quote=source_text,
            confidence_metadata={"detector": "llm_generated_tax_class_other"},
        )
    ]
