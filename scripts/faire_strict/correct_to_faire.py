#!/usr/bin/env python3
"""Make a strict-FAIRe copy of the FAIRe export, for the NOAA fill-in.

Reads the original export (never modified) and writes, to OUTPUT_DIR:

- projectMetadata.csv, sampleMetadata.csv, experimentRunMetadata.csv:
  only FAIRe v1.0.2 terms (plus internal_study_id, kept on purpose),
  in the FAIRe template's column order, values corrected by the rules
  below.
- extras_<table>.csv: every value that left the FAIRe columns (dropped
  non-FAIRe columns, evidence text split off a 0/1 field), keyed by
  internal_study_id (+ samp_name / lib_id), so nothing is lost.
- changes_<table>.csv: every distinct change, aggregated:
  column, rule, original value, new value, number of cells.
- review_other.csv: every value that could not be mapped without
  guessing and was written as "other: <original>" (user decision
  2026-11-10: use "other: ..." for these even where FAIRe does not
  allow "other", and review them by hand).

The rules are documented one by one in correct_to_faire.md (repo root),
using the rule ids below. Reference: reference/faire/VERSIONS.md.

Usage:
    python scripts/faire_strict/correct_to_faire.py \\
        data/snapshots/faire_2026-11 data/snapshots/faire_2026-11_strict
"""
from __future__ import annotations

import collections
import csv
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checklist import (  # noqa: E402
    LIST_SEPARATOR,
    MISSING_VALUE_TERMS,
    REPO,
    TABLES,
    Term,
    is_number,
    load_checklist,
)

csv.field_size_limit(10**9)

FULL_TEMPLATE = REPO / "reference/faire/FAIRe_checklist/FAIRe_checklist_v1.0.2_FULLtemplate.xlsx"
KEEP_EXTRA_COLUMNS = ("internal_study_id",)  # user decision 2026-11-10
SIDECAR_KEYS = {
    "projectMetadata": ("internal_study_id",),
    "sampleMetadata": ("internal_study_id", "samp_name"),
    "experimentRunMetadata": ("internal_study_id", "samp_name", "lib_id"),
}

# --- R03 missing values -------------------------------------------------
# Exact INSDC wording (FAIRe guidelines). Case variants and the bare
# category words our sources use are mapped to the matching category.
_MISSING_ALIASES = {
    "missing": "missing: not provided",
    "not provided": "missing: not provided",
    "not collected": "missing: not collected",
    "restricted access": "missing: restricted access",
    "missing: not reported": "missing: not provided",
}

# --- R05 units that equal a term's standard unit ------------------------
_NUM_FWD = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNIT_PATTERNS = {
    "degree Celsius": r"(?:°|º|◦|˚|o)?\s*C|℃|degrees?\s*C(?:elsius)?",
    "basepairs": r"bp|bps|base[\s-]?pairs?",
    "cycles": r"cycles?",
    "meter": r"m|meters?|metres?",
    "mm": r"mm|millimet(?:er|re)s?",
    "mm2": r"mm2|mm²",
    "µm": r"µm|μm|um|microns?|micromet(?:er|re)s?",
    "nM": r"nM",
    "%": r"%",
    "reads": r"reads?",
}
_CASE_INSENSITIVE_UNITS = frozenset({"degree Celsius", "basepairs", "cycles", "reads", "%"})
# R05b: a filter pore size given in µM (micromolar) can only be a typo for
# µm. Separate rule id so it can be rejected on review.
_SIZE_FRAC_MICROMOLAR = re.compile(rf"^({_NUM_FWD})(?:\s*-\s*({_NUM_FWD}))?\s*-?\s*(?:µM|μM|uM)$")
# Short lead-in words seen before a value ("stored at -80 °C"). Only these
# exact phrases are removed; anything longer stays unmapped.
_LEAD_IN = r"(?:(?:stored|kept|frozen|maintained|store)\s+at\s+|at\s+)?"
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


def _normalise_number_text(text: str) -> str:
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"(^|\s)-\s+(?=\d)", r"\1-", text)  # "- 80" -> "-80"
    return text.strip()


def _number_with_standard_unit(term: Term, item: str) -> str | None:
    pattern = _UNIT_PATTERNS.get(term.unit)
    if pattern is None:
        return None
    text = _normalise_number_text(item)
    match = re.fullmatch(
        rf"{_LEAD_IN}({_NUM})(?:\s*-\s*({_NUM}))?\s*-?\s*(?:{pattern})\.?",
        text,
        # Case matters for metric units (µm is not µM, m is not M).
        flags=re.IGNORECASE if term.unit in _CASE_INSENSITIVE_UNITS else 0,
    )
    if match is None:
        return None
    low, high = match.group(1), match.group(2)
    return f"{low}-{high}" if high else low


# --- R08 platform ---------------------------------------------------------
# Keyword -> FAIRe platform (INSDC platform names). Checked in order.
_PLATFORM_KEYWORDS = (
    (r"illumina|miseq|hiseq|novaseq|nextseq|iseq|genome analy[sz]er|hiscan|solexa", "ILLUMINA"),
    (r"\b454\b|gs ?flx|gs junior|roche|pyroseq|titanium", "LS454"),
    (r"ion torrent|ion ?pgm|ion s5|ion proton|ion gene ?studio|\bpgm\b", "ION_TORRENT"),
    (r"nanopore|minion|gridion|promethion|flongle|\bont\b", "OXFORD_NANOPORE"),
    (r"pacbio|pacific biosciences|sequel|\brs ?ii\b|\bsmrt", "PACBIO_SMRT"),
    (r"dnbseq|dnb-?seq|mgiseq|mgi ?seq|\bmgi\b", "DNBSEQ"),
    (r"bgiseq|bgi-?seq", "BGISEQ"),
    (r"sanger|capillary|abi ?3\d{2,3}|3730|3130|3500|applied biosystems", "CAPILLARY"),
    (r"element|aviti", "ELEMENT"),
    (r"helicos", "HELICOS"),
    (r"genapsys", "GENAPSYS"),
    (r"ultima", "ULTIMA"),
)


def _map_platform(item: str) -> str | None:
    low = item.lower()
    hits = {platform for pattern, platform in _PLATFORM_KEYWORDS if re.search(pattern, low)}
    return hits.pop() if len(hits) == 1 else None  # two platforms named: do not guess


# --- column-specific controlled vocabulary maps --------------------------
# original (lowercased, stripped) -> FAIRe value. Only unambiguous ones.
_VOCAB_MAPS: dict[str, dict[str, str]] = {
    "assay_type": {
        "qpcr": "targeted",
        "quantitative pcr": "targeted",
        "ddpcr": "targeted",
        "digital pcr": "targeted",
    },
    "amp_vis_method": {
        "agarose gel electrophoresis": "gel electrophoresis",
        "agarose gel": "gel electrophoresis",
        "gel electrophoresis": "gel electrophoresis",
        "electrophoresis": "gel electrophoresis",
        "1% agarose gel electrophoresis": "gel electrophoresis",
        "1.5% agarose gel electrophoresis": "gel electrophoresis",
        "2% agarose gel electrophoresis": "gel electrophoresis",
        "quantitative pcr": "qPCR",
        "real-time pcr": "qPCR",
        "droplet digital pcr": "dPCR",
        "ddpcr": "dPCR",
    },
    "lib_layout": {
        "paired": "paired end",
        "paired-end": "paired end",
        "single": "single end",
        "single-end": "single end",
    },
}

# R04: evidence text appended to 0/1 project fields by our extraction.
_EVIDENCE_BOOLEAN_COLUMNS = ("neg_cont_0_1", "pos_cont_0_1")
# Decision D3 (pending user review): "0 | no control/blank mention found"
# means the paper does not mention controls, not that none were used.
_NO_MENTION_EVIDENCE = "no control/blank mention found"
_NO_MENTION_VALUE = "missing: not provided"

def _float_at_least(limit: float):
    def check(value: str) -> bool:
        try:
            return float(value) >= limit
        except ValueError:
            return False
    return check


# F01: flagged for review, values left unchanged (no guessing).
_PLAUSIBILITY_FLAGS = (
    ("samp_store_temp", _float_at_least(50), "storage temperature >= 50 degC; minus sign possibly lost in PDF text"),
)

_BOOLEAN_WORDS = {"yes": "1", "true": "1", "no": "0", "false": "0"}


# R13: reviewed phrase -> FAIRe term rules, one per line with a reason,
# in maps/controlled_terms.csv. Matched against the whole item after
# normalising (lowercase, hyphens/dashes to spaces, spaces collapsed,
# trailing period removed), and only for short items: a sentence is never
# reduced to a term.
_MAP_FILE = Path(__file__).resolve().parent / "maps/controlled_terms.csv"
_MAX_MAPPED_WORDS = 8


def _load_term_maps() -> dict[str, list[tuple[re.Pattern, str, str]]]:
    maps: dict[str, list[tuple[re.Pattern, str, str]]] = collections.defaultdict(list)
    with _MAP_FILE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            maps[row["column"]].append((re.compile(row["pattern"]), row["faire_value"], row["rule"]))
    return maps


_TERM_MAPS = _load_term_maps()


def _normalise_phrase(item: str) -> str:
    text = item.lower().replace("\u2010", " ").replace("\u2011", " ").replace("\u2013", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def _map_phrase(column: str, item: str) -> tuple[str, str] | None:
    if column not in _TERM_MAPS or len(item.split()) > _MAX_MAPPED_WORDS:
        return None
    phrase = _normalise_phrase(item)
    for pattern, value, rule in _TERM_MAPS[column]:
        if pattern.fullmatch(phrase):
            return value, rule
    return None


class Log:
    def __init__(self) -> None:
        self.changes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.other: collections.Counter = collections.Counter()
        self.flags: collections.Counter = collections.Counter()

    def change(self, table: str, column: str, rule: str, old: str, new: str) -> None:
        if old != new:
            self.changes[table][(column, rule, old, new)] += 1


# --- R14 number + unit pairs ----------------------------------------------
# FAIRe keeps the number and its unit in separate columns. "15 ml" in the
# value column becomes 15 and unit "mL" -- only when every number in the
# cell has the same unit and the unit column is empty or already agrees.
# No unit conversion (e.g. µL is not in FAIRe's volume list: decision D6).
_UNIT_PAIRS = {
    "sampleMetadata": (
        ("samp_vol_we_dna_ext", "samp_vol_we_dna_ext_unit"),
        ("samp_size", "samp_size_unit"),
        ("pump_flow_rate", "pump_flow_rate_unit"),
        ("concentration", "concentration_unit"),
        ("host_tot_mass", "host_tot_mass_unit"),
        ("host_height", "host_height_unit"),
        ("host_length", "host_length_unit"),
    ),
}
# FAIRe terms we add as columns even though the original export lacks them.
_ADDED_COLUMNS = {"sampleMetadata": ("concentration_unit",)}
_UNIT_SYNONYMS = {
    "mL": ("ml", "milliliter", "milliliters", "millilitre", "millilitres"),
    "L": ("l", "liter", "liters", "litre", "litres"),
    "mg": ("mg", "milligram", "milligrams"),
    "g": ("g", "gram", "grams", "gr"),
    "kg": ("kg", "kilogram", "kilograms"),
    "cm2": ("cm2", "cm²"),
    "m2": ("m2", "m²"),
    "cm3": ("cm3", "cm³", "cc"),
    "m3": ("m3", "m³"),
    "mm": ("mm",),
    "cm": ("cm",),
    "m": ("m",),
    "L/min": ("l/min", "l min-1", "l min−1", "l min^-1", "liters per minute", "litres per minute", "lpm"),
    "L/h": ("l/h", "l/hr", "l h-1", "l h−1", "liters per hour", "litres per hour"),
    "L/s": ("l/s", "l s-1", "l s−1"),
    "m3/s": ("m3/s",), "m3/min": ("m3/min",), "m3/h": ("m3/h",),
    "ng/µl": ("ng/µl", "ng/μl", "ng/ul", "ng µl-1", "ng μl-1", "ng µl−1", "ng μl−1", "ng/µl-1"),
    "copies/µl": ("copies/µl", "copies/μl", "copies/ul", "copies µl-1", "copies μl-1"),
}
_UNIT_LOOKUP = {syn: canonical for canonical, syns in _UNIT_SYNONYMS.items() for syn in (canonical, *syns)}
# Matched case-insensitively except where case changes the meaning
# (m vs M, ml vs ML).
_CASE_SENSITIVE_UNIT_KEYS = frozenset({"g", "kg", "m", "ml", "mm", "cm", "mg", "m2", "m3", "cm2", "cm3"})


def _canonical_unit(text: str, allowed: tuple[str, ...]) -> str | None:
    """FAIRe spelling of a unit, if it is one of this unit term's allowed
    values. Case is kept for the metric prefixes that need it (mL vs ML)."""
    key = re.sub(r"\s+", " ", text.strip()).rstrip(".")
    canonical = _UNIT_LOOKUP.get(key)
    if canonical is None and key.lower() not in _CASE_SENSITIVE_UNIT_KEYS:
        canonical = _UNIT_LOOKUP.get(key.lower())
    if canonical is None and key in ("mL", "Ml"):  # ml spelled with capital L
        canonical = "mL"
    return canonical if canonical in allowed else None


_NUM_UNIT = re.compile(rf"^(?:approximately |approx\.? |~ ?|ca\.? )?({_NUM_FWD})(?:\s*-\s*({_NUM_FWD}))?\s*-?\s*(.+)$")


def _split_unit_pairs(table: str, row: dict[str, str], terms: dict[str, Term], log: Log) -> None:
    for value_col, unit_col in _UNIT_PAIRS.get(table, ()):
        value = row.get(value_col) or ""
        if not value or value_col not in terms or unit_col not in terms:
            continue
        allowed = terms[unit_col].vocabulary
        items = _split_items(value)
        numbers, units = [], set()
        for item in items:
            text = _normalise_number_text(item)
            if is_number(text):
                numbers.append(text)
                continue
            match = _NUM_UNIT.match(text)
            unit = _canonical_unit(match.group(3), allowed) if match else None
            if unit is None:
                numbers = None
                break
            low, high = match.group(1), match.group(2)
            numbers.append(f"{low}-{high}" if high else low)
            units.add(unit)
        if not numbers or not units or len(units) != 1:
            continue
        unit = units.pop()
        existing = row.get(unit_col) or ""
        existing_units = {_canonical_unit(u, allowed) or u for u in _split_items(existing)}
        if existing and existing_units != {unit}:
            continue  # unit column disagrees: leave both for review
        new_value = LIST_SEPARATOR.join(dict.fromkeys(numbers))
        log.change(table, value_col, "R14_number_unit_split", value, new_value)
        if not existing:
            log.change(table, unit_col, "R14_number_unit_split", "(empty)", unit)
        row[value_col] = new_value
        row[unit_col] = unit


# --- R15 target_gene ---------------------------------------------------------
# Unambiguous gene names -> FAIRe target_gene terms (whole item, normalised
# like R13). 16S alone is ambiguous in FAIRe (SSU prokaryote vs LSU
# mitochondria) and is decided from the row's target taxa (R15b/R15c), else
# left to D0. ITS1/ITS2 subregions also go to target_subfragment (R15d).
_TARGET_GENE_PATTERNS = (
    (r"(18 ?s|ssu eukaryot\w*)( rrna| rdna| ribosomal rna)?( gene)?( ssu)?( rrna)?( gene)?", "18S rRNA (SSU eukaryote)"),
    (r"28 ?s( rrna| rdna| ribosomal rna)?( gene)?", "28S rRNA (LSU eukaryote)"),
    (r"(mt ?)?12 ?s( rrna| rdna| ribosomal rna)?( gene)?", "12S rRNA (SSU mitochondria)"),
    (r"23 ?s( rrna| rdna| ribosomal rna)?( gene)?", "23S rRNA (LSU prokaryote)"),
    (r"(mt ?16 ?s|mitochondrial 16 ?s)( rrna| rdna| ribosomal rna)?( gene)?", "16S rRNA (LSU mitochondria)"),
    (r"16 ?s( rrna| rdna)? ssu|ssu 16 ?s( rrna| rdna)?", "16S rRNA (SSU prokaryote)"),
    (r"co ?1|cox ?1|coi|cytochrome (c )?oxidase( subunit)? (i|1)|mitochondrial cytochrome c oxidase subunit i( \(coi\))?", "COI"),
    (r"cyt ?b|cytochrome b", "CytB"),
    (r"its ?[12]?( region| rdna| rrna)?|internal transcribed spacer( [12])?( region)?", "ITS"),
    (r"(bacterial|archaeal|comammox) ?amoa", "amoA"),
)
_TARGET_GENE_RE = [(re.compile(p), v) for p, v in _TARGET_GENE_PATTERNS]
_PLAIN_16S = re.compile(r"16 ?s( rrna| rdna| ribosomal rna| ribsosomal rna)?( gene| genes)?")
_PROKARYOTE_WORDS = re.compile(r"bacteri|archae|prokaryot|microb|cyanobacter", re.IGNORECASE)
_ANIMAL_WORDS = re.compile(
    r"vertebrat|fish|amphib|mammal|animal|metazoa|invertebrat|mollus|crustac|insect|arthropod|bird|reptil|"
    r"chordat|elasmobranch|shark|teleost|actinopter|cetacea|turtle",
    re.IGNORECASE,
)
_ITS_SUBREGION = re.compile(r"its ?-?([12])|internal transcribed spacer ([12])", re.IGNORECASE)


def _target_gene_item(item: str, row: dict[str, str]) -> tuple[str, str] | None:
    phrase = _normalise_phrase(item)
    for pattern, value in _TARGET_GENE_RE:
        if pattern.fullmatch(phrase):
            return value, "R15_target_gene_name"
    if _PLAIN_16S.fullmatch(phrase):
        assay, scope = row.get("targetTaxonomicAssay", ""), row.get("targetTaxonomicScope", "")
        prokaryote = bool(_PROKARYOTE_WORDS.search(assay) or _PROKARYOTE_WORDS.search(scope))
        animal_any = bool(_ANIMAL_WORDS.search(assay) or _ANIMAL_WORDS.search(scope))
        if prokaryote and not animal_any:
            return "16S rRNA (SSU prokaryote)", "R15b_16S_prokaryote_from_target_taxa"
        if _ANIMAL_WORDS.search(scope) and not prokaryote:
            return "16S rRNA (LSU mitochondria)", "R15c_16S_mitochondrial_from_target_taxa"
    return None


def _template_order() -> dict[str, list[str]]:
    wb = openpyxl.load_workbook(FULL_TEMPLATE, read_only=True, data_only=True)
    order: dict[str, list[str]] = {}
    ws = wb["projectMetadata"]
    ws.reset_dimensions()
    order["projectMetadata"] = [r[2] for r in list(ws.iter_rows(values_only=True))[1:] if r[2]]
    for table in ("sampleMetadata", "experimentRunMetadata"):
        ws = wb[table]
        ws.reset_dimensions()
        rows = list(ws.iter_rows(values_only=True))
        order[table] = [c for c in rows[2] if c]
    return order


def _correct_item(term: Term, item: str) -> tuple[str, str]:
    """(new item, rule id) for one list item of a FAIRe column."""
    # R03 missing values
    if item in MISSING_VALUE_TERMS:
        return item, "keep"
    low = item.lower().strip()
    if low in MISSING_VALUE_TERMS:
        return low, "R03_missing_value_case"
    if low in _MISSING_ALIASES:
        return _MISSING_ALIASES[low], "R03_missing_value_alias"

    # User decisions 2026-11-10 (correct_to_faire.md, D1/D2).
    if term.name == "checkls_ver" and item == "1.0.2":
        return item, "keep"  # D1: the real checklist version, kept as text
    if term.name == "lib_layout" and low == "no files":
        return "none", "R12_lib_layout_no_files"  # D2

    t = term.term_type
    if t in ("controlled vocabulary", "numeric or controlled vocabulary"):
        if item in term.vocabulary:
            return item, "keep"
        other = re.fullmatch(r"other\s*:\s*(.+)", item, flags=re.IGNORECASE | re.DOTALL)
        if other:
            return f"other: {other.group(1).strip()}", "R06_other_format"
        by_case = {v.lower(): v for v in term.vocabulary}
        if low in by_case:
            return by_case[low], "R06_vocab_case"
        if term.name == "platform":
            mapped = _map_platform(item)
            if mapped:
                return mapped, "R08_platform_keyword"
        if term.name.endswith("_unit"):
            unit = _canonical_unit(item, term.vocabulary)
            if unit:
                return unit, "R14b_unit_spelling"
        mapped = _VOCAB_MAPS.get(term.name, {}).get(low)
        if mapped:
            return mapped, f"R09_vocab_map_{term.name}"
        phrase = _map_phrase(term.name, item)
        if phrase:
            return phrase
        if t.startswith("numeric"):
            if is_number(item):
                return item, "keep"
            number = _number_with_standard_unit(term, item)
            if number is not None:
                return number, "R05_strip_standard_unit"
        return f"other: {item}", "R07_unmapped_other"
    if t in ("numeric", "integer"):
        if is_number(item, integer=(t == "integer")):
            return item, "keep"
        cleaned = _normalise_number_text(item)
        if cleaned != item and is_number(cleaned, integer=(t == "integer")):
            return cleaned, "R05_number_characters"
        number = _number_with_standard_unit(term, item)
        if number is not None and is_number(number, integer=(t == "integer")):
            return number, "R05_strip_standard_unit"
        if term.name == "size_frac":
            typo = _SIZE_FRAC_MICROMOLAR.match(_normalise_number_text(item))
            if typo:
                low, high = typo.group(1), typo.group(2)
                return (f"{low}-{high}" if high else low), "R05b_size_frac_micromolar_typo"
        return f"other: {item}", "R07_unmapped_other"
    if t == "Boolean":
        if item in ("0", "1"):
            return item, "keep"
        if low in _BOOLEAN_WORDS:
            return _BOOLEAN_WORDS[low], "R10_boolean_words"
        return f"other: {item}", "R07_unmapped_other"
    return item, "keep"  # free text, fixed format, DOI: later stages


def _split_items(value: str) -> list[str]:
    return [p.strip() for p in value.split("|") if p.strip()]


def correct_cell(
    table: str, term: Term, value: str, log: Log, extras: dict[str, str], row: dict[str, str] | None = None
) -> str:
    if not value:
        return value
    if table == "projectMetadata" and term.name in _EVIDENCE_BOOLEAN_COLUMNS:
        flag, _, evidence = value.partition("|")
        flag, evidence = flag.strip(), evidence.strip()
        if flag in ("0", "1") and evidence:
            extras[f"{term.name}__evidence"] = evidence
            new = flag
            rule = "R04_split_boolean_evidence"
            if flag == "0" and evidence == _NO_MENTION_EVIDENCE:
                new, rule = _NO_MENTION_VALUE, "R04_no_mention_is_missing"
            log.change(table, term.name, rule, value, new)
            return new
    if table == "projectMetadata" and term.name == "target_gene" and row is not None:
        new_items = []
        for item in _split_items(value):
            mapped = _target_gene_item(item, row)
            if mapped:
                new_item, rule = mapped
                log.change(table, term.name, rule, item, new_item)
                subregion = _ITS_SUBREGION.search(item)
                if subregion and rule == "R15_target_gene_name":
                    sub = f"ITS{subregion.group(1) or subregion.group(2)}"
                    existing = _split_items(row.get("target_subfragment", ""))
                    if sub not in existing:
                        row["target_subfragment"] = LIST_SEPARATOR.join([*existing, sub])
                        log.change(table, "target_subfragment", "R15d_its_subregion", "(from target_gene)", sub)
            else:
                new_item, rule = _correct_item(term, item)
                if rule != "keep":
                    log.change(table, term.name, rule, item, new_item)
                if rule == "R07_unmapped_other":
                    log.other[(table, term.name, item)] += 1
            if new_item not in new_items:
                new_items.append(new_item)
        new_value = LIST_SEPARATOR.join(new_items)
        if new_value != LIST_SEPARATOR.join(_split_items(value)):
            extras["target_gene__original"] = value
        return new_value
    if table == "sampleMetadata" and term.name == "samp_category":
        # R11: our pipeline used samp_category as a sample-label log
        # (BioSample source_material_id / sample_name / title; see
        # sources/ncbi.py::_sample_category_from_title_or_name). Labels are
        # not FAIRe categories: move them to extras. What goes in the
        # field instead is decision D4 in correct_to_faire.md.
        kept, labels = [], []
        for item in _split_items(value):
            new_item, rule = _correct_item(term, item)
            (kept if rule != "R07_unmapped_other" else labels).append(new_item if rule != "R07_unmapped_other" else item)
        if labels:
            extras["samp_category__sample_label"] = LIST_SEPARATOR.join(labels)
            new_value = LIST_SEPARATOR.join(dict.fromkeys(kept))
            log.change(table, term.name, "R11_sample_label_to_extras", "(sample label)", new_value or "(blank, see D4)")
            return new_value
    items = _split_items(value)
    new_items: list[str] = []
    for item in items:
        new_item, rule = _correct_item(term, item)
        if rule != "keep":
            log.change(table, term.name, rule, item, new_item)
        if rule == "R07_unmapped_other":
            log.other[(table, term.name, item)] += 1
        if new_item not in new_items:
            new_items.append(new_item)
    for check_column, check, message in _PLAUSIBILITY_FLAGS:
        if term.name == check_column:
            for new_item in new_items:
                if check(new_item):
                    log.flags[(table, term.name, message, new_item)] += 1
    new_value = LIST_SEPARATOR.join(new_items)
    joined_old = LIST_SEPARATOR.join(items)
    if len(new_items) < len(items):
        log.change(table, term.name, "R02_dedupe_after_mapping", joined_old, new_value)
    if value != joined_old and "|" in value:
        log.change(table, term.name, "R02_list_separator", "(any list)", "(' | ' separator)")
    return new_value


def process_table(table: str, src: Path, out: Path, terms: dict[str, Term], order: list[str], log: Log) -> int:
    with (src / f"{table}.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        added = _ADDED_COLUMNS.get(table, ())
        faire_cols = [c for c in order if (c in header or c in added) and c in terms]
        misplaced = [c for c in header if c in terms and c not in order]
        dropped = [c for c in header if c not in terms and c not in KEEP_EXTRA_COLUMNS]
        out_cols = [*[c for c in KEEP_EXTRA_COLUMNS if c in header], *faire_cols]
        keys = SIDECAR_KEYS[table]
        extra_cols: list[str] = list(keys)
        extra_rows: list[dict[str, str]] = []
        n = 0
        with (out / f"{table}.csv").open("w", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=out_cols, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                n += 1
                extras: dict[str, str] = {}
                for col in dropped + misplaced:
                    if row.get(col):
                        extras[col] = row[col]
                _split_unit_pairs(table, row, terms, log)
                new_row = {c: row.get(c, "") for c in out_cols if c in KEEP_EXTRA_COLUMNS}
                for col in faire_cols:
                    new_row[col] = correct_cell(table, terms[col], row.get(col, ""), log, extras, row)
                writer.writerow(new_row)
                if extras:
                    for col in extras:
                        if col not in extra_cols:
                            extra_cols.append(col)
                    extra_rows.append({**{k: row.get(k, "") for k in keys}, **extras})
    with (out / f"extras_{table}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=extra_cols)
        writer.writeheader()
        writer.writerows(extra_rows)
    for col in dropped + misplaced:
        log.change(table, col, "R01_column_moved_to_extras", "(column)", f"extras_{table}.csv")
    print(
        f"{table}: {n} rows; {len(out_cols)} columns kept, {len(dropped)} non-FAIRe and "
        f"{len(misplaced)} misplaced column(s) moved to extras_{table}.csv ({len(extra_rows)} rows)",
        flush=True,
    )
    return n


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if src.resolve() == out.resolve():
        sys.exit("Output directory must differ from the original export.")
    out.mkdir(parents=True, exist_ok=True)
    terms = load_checklist()
    order = _template_order()
    log = Log()
    for table in TABLES:
        process_table(table, src, out, terms, order[table], log)
        with (out / f"changes_{table}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["column", "rule", "original", "new", "cells"])
            for (column, rule, old, new), count in sorted(
                log.changes[table].items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])
            ):
                writer.writerow([column, rule, old, new, count])
    with (out / "review_other.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "column", "kind", "original", "cells"])
        for (table, column, item), count in sorted(log.other.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])):
            # phrase: a short value not in the FAIRe list (often a correct
            # "other:"); sentence: pasted method text, likely wrong field.
            kind = "phrase" if len(item.split()) <= _MAX_MAPPED_WORDS else "sentence"
            writer.writerow([table, column, kind, item, count])
    print(f"review_other.csv: {len(log.other)} distinct values written as 'other: ...'")
    with (out / "review_flags.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "column", "flag", "value", "cells"])
        for (table, column, message, value), count in sorted(log.flags.items()):
            writer.writerow([table, column, message, value, count])
    print(f"review_flags.csv: {len(log.flags)} distinct values flagged for review")


if __name__ == "__main__":
    main()
