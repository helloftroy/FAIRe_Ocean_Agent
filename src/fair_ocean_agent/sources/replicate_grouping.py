"""Pure detection of biological-replicate sample groups from sample-name
suffix patterns (e.g. "Site_A_rep1"/"Site_A_rep2", "Sample_A"/"Sample_B"/
"Sample_C") -- feeds FAIRe's real `biological_rep_relation` sampleMetadata
slot, whose value lists the samp_name of every sibling replicate in a group,
pipe-joined, including the sample itself (schemas/faire/schema.yaml's own
worked example: "S01_1 | S01_2 | S01_3").

Shared by sources/supplement_parsing.py (a table's own sample-id cell
strings) and sources/ncbi.py (a BioSample's sample_name attribute or title)
rather than duplicated in each -- entity_external_id conventions differ per
adapter (a supplement table's raw cell string vs. a BioSample accession), so
each adapter builds its own RawFactCandidates from the groups this module
returns; only the grouping/regex logic itself is shared.

Deliberately NOT used by sources/ena.py: an ENA run's sample_accession is an
INSDC accession (e.g. "SAMEA1234567"), never a free-text name -- no suffix
pattern to detect there.

Two independent, deliberately conservative signals:
  1. EXPLICIT_REP_MARKER -- an explicit "rep"/"replicate" token plus a
     digit (e.g. "_rep1", "-REP_2", "_replicate3", or
     "Site1Rep1Valsecchi16S"). High confidence: the token is
     unambiguous. Embedded forms keep any trailing constant assay/sample
     suffix as part of the grouping key, so "Site1Rep1Valsecchi16S" and
     "Site1Rep2Valsecchi18S" are not merged.
  2. TRAILING_NUMBER_SUFFIX -- a bare number after an underscore or space
     (e.g. "LM_1", "LM 2"). This is intentionally separator-scoped and
     grouped by exact prefix, so "LM_1" and "LM_2" group, but "LM_1" and
     "LMM_2" do not.
  3. TRAILING_LETTER_SUFFIX -- a single trailing letter after a separator
     (e.g. "_A", "-b"). Lower confidence -- letters are also commonly used
     for genuinely different sites/stations (e.g. "Station_A"/"Station_B"),
     so this signal is gated by a minimum group size and a
     consecutive-letters check: a lone A/B pair never groups, and
     non-consecutive letters (e.g. only A and F) never group either.

Bare trailing digits without a separator on a GENERIC full-word base (e.g.
"Sample1" vs "Sample12" vs "Sample2") are never treated as a replicate
signal -- they're the single most common false-positive shape (arbitrary/
incrementing sample numbering, not same-site replication); see
_GENERIC_NUMBERED_SAMPLE_BASE_RE.

  4. SHORT_PREFIX_NUMBER_SUFFIX -- opt-in only (include_short_prefix_signal,
     default off), a bare digit directly after a letter/underscore base of
     any length, no separator (e.g. "E2"/"E3", "AS1"/"AS2",
     "PB_Biofilm1"/"PB_Biofilm2"/"PB_Biofilm3", or "T_C1P"/
     "T_C2P"/"T_C3P" where a constant trailing letter is part of the
     sample-series code). Real gap found live
     (PMC10988111): a developmental-stage time series named samples by a
     short stage-code prefix (P polyp, ES early strobila, AS advanced
     strobila, E ephyra) plus a bare replicate number, no separator at
     all. A second real gap (another gold paper) needed the same
     no-separator matching for a much longer, descriptive base
     ("PB_Biofilm") -- so despite the name, this signal's base is no
     longer length-capped, just excluded when it's "S" (a real, common
     generic sequential sample-ID convention in supplement tables) or one
     of the same generic filler words TRAILING_NUMBER_SUFFIX already
     excludes ("Sample"/"Isolate"/... -- what keeps "Sample1"/"Sample12"
     excluded even now that the length cap is gone). Confirmed live this
     signal is still NOT safe to enable unconditionally: the caller
     (sources/ncbi.py only, for BioSample sample_name/title text
     specifically -- never supplement_parsing.py's own table-row IDs, an
     entirely different and much more collision-prone namespace) opts in
     explicitly rather than this being on by default for every consumer
     of this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ReplicateSignal(str, Enum):
    EXPLICIT_REP_MARKER = "explicit_rep_marker"
    TRAILING_NUMBER_SUFFIX = "trailing_number_suffix"
    TRAILING_LETTER_SUFFIX = "trailing_letter_suffix"
    SHORT_PREFIX_NUMBER_SUFFIX = "short_prefix_number_suffix"


@dataclass(frozen=True)
class ReplicateGroup:
    # Sample *identifiers* (the caller's own dict keys -- never the raw name
    # strings used only to detect the pattern), deterministically ordered:
    # ascending rep number for EXPLICIT_REP_MARKER, ascending letter for
    # TRAILING_LETTER_SUFFIX.
    members: tuple[str, ...]
    signal: ReplicateSignal


# Matches "_rep1", "-REP_2", "_replicate3" (case-insensitive on the "rep"/
# "replicate" token only -- `base` is captured verbatim, case preserved, so
# "Site_A" and "site_a" are never silently merged).
_EXPLICIT_REP_RE = re.compile(r"^(?P<base>.+?)[-_](?:rep(?:licate)?)[-_]?(?P<num>\d+)$", re.IGNORECASE)
_EXPLICIT_EMBEDDED_REP_RE = re.compile(
    r"^(?P<base>.+?)(?:rep(?:licate)?)[-_]?(?P<num>\d+)(?P<suffix>[A-Za-z][A-Za-z0-9_.-]*)$",
    re.IGNORECASE,
)

# Matches "LM 6" and "LM_6", but deliberately not "LM-6" or "LM6".
_TRAILING_NUMBER_RE = re.compile(r"^(?P<base>.+?)[_ ]+(?P<num>\d+)$")
_GENERIC_NUMBERED_SAMPLE_BASE_RE = re.compile(
    r"^(?:bio)?samples?|specimens?|isolates?|libraries?|runs?$",
    re.IGNORECASE,
)

# Matches "E2"/"E3", "AS1"/"AS2", (real gap found live, another gold
# paper) "PB_Biofilm1"/"PB_Biofilm2"/"PB_Biofilm3", and names with a
# constant trailing letter after the replicate number ("T_C1P"/"T_C2P") --
# a bare digit directly after a letter/underscore base of ANY length, no
# separator before the number. Originally
# capped at 1-2 letters, relaxed to any length once a real descriptive
# multi-word base ("PB_Biofilm") turned up needing the same treatment --
# the base can never contain a digit itself (so "Sample1" splits cleanly
# as base="Sample", not swallowed into a longer base), which is what keeps
# this from re-admitting the "Sample1"/"Sample12" false-positive shape:
# that base still gets caught by _GENERIC_NUMBERED_SAMPLE_BASE_RE below.
# "S" alone is excluded regardless, since "S1"/"S2" is itself a common
# generic sequential sample-ID convention, not a replicate signal -- see
# module docstring's SHORT_PREFIX_NUMBER_SUFFIX section.
_SHORT_PREFIX_NUMBER_RE = re.compile(r"^(?P<base>[A-Za-z][A-Za-z_]*)(?P<num>\d+)(?P<suffix>[A-Za-z]*)$")
_SHORT_PREFIX_EXCLUDED_BASES = frozenset({"S"})
_MIN_SHORT_PREFIX_GROUP_SIZE = 2

# Matches "Site_A", "Site-b" -- exactly one letter directly after a
# separator at the end of the string. Does NOT match "Station_Alpha" (the
# trailing token is "Alpha", not a single character).
_TRAILING_LETTER_RE = re.compile(r"^(?P<base>.+)[-_](?P<letter>[A-Za-z])$")

# Minimum group size for the letter-suffix signal, chosen so a lone
# "Station_A"/"Station_B" pair (plausibly two different sites, not
# replicates of one site) is never grouped on its own.
_MIN_LETTER_SUFFIX_GROUP_SIZE = 3


def detect_replicate_groups(
    sample_names_by_id: dict[str, str],
    *,
    include_letter_suffix_signal: bool = True,
    include_short_prefix_signal: bool = False,
) -> list[ReplicateGroup]:
    """`sample_names_by_id` maps each sample's own identifier (a supplement
    table's raw sample-id cell string, or a BioSample accession) to the
    free-text name string to run suffix-pattern detection against -- for
    supplement tables the identifier IS the name; for NCBI BioSample the
    identifier is the accession and the name is the `sample_name` attribute
    or BioSample title.

    include_short_prefix_signal defaults to False: confirmed live it isn't
    safe for every caller (supplement_parsing.py's own table-row IDs are a
    much more collision-prone namespace, e.g. real "S1"/"S2" sequential
    IDs) -- only sources/ncbi.py's real BioSample sample_name/title text
    opts in.

    Returns disjoint groups (an id matched by the explicit-marker signal is
    never re-considered for the letter-suffix signal). A group of size 1
    (no sibling found) is never returned -- nothing to link.
    """
    consumed: set[str] = set()
    groups: list[ReplicateGroup] = []

    explicit_buckets: dict[str, list[tuple[str, int]]] = {}
    for sample_id, name in sample_names_by_id.items():
        stripped = name.strip()
        match = _EXPLICIT_REP_RE.match(stripped)
        if match:
            key = match.group("base")
        else:
            match = _EXPLICIT_EMBEDDED_REP_RE.match(stripped)
            if not match:
                continue
            key = f"{match.group('base')}\0{match.group('suffix')}"
        explicit_buckets.setdefault(key, []).append(
            (sample_id, int(match.group("num")))
        )
    for members in explicit_buckets.values():
        if len(members) < 2:
            continue
        ordered = tuple(sample_id for sample_id, _ in sorted(members, key=lambda pair: pair[1]))
        groups.append(ReplicateGroup(members=ordered, signal=ReplicateSignal.EXPLICIT_REP_MARKER))
        consumed.update(ordered)

    numeric_buckets: dict[str, list[tuple[str, int]]] = {}
    for sample_id, name in sample_names_by_id.items():
        if sample_id in consumed:
            continue
        match = _TRAILING_NUMBER_RE.match(name.strip())
        if match:
            if _GENERIC_NUMBERED_SAMPLE_BASE_RE.match(match.group("base").strip()):
                continue
            numeric_buckets.setdefault(match.group("base"), []).append(
                (sample_id, int(match.group("num")))
            )
    for members in numeric_buckets.values():
        if len(members) < 2:
            continue
        ordered = tuple(sample_id for sample_id, _ in sorted(members, key=lambda pair: pair[1]))
        groups.append(ReplicateGroup(members=ordered, signal=ReplicateSignal.TRAILING_NUMBER_SUFFIX))
        consumed.update(ordered)

    if include_short_prefix_signal:
        short_prefix_buckets: dict[str, list[tuple[str, int]]] = {}
        for sample_id, name in sample_names_by_id.items():
            if sample_id in consumed:
                continue
            match = _SHORT_PREFIX_NUMBER_RE.match(name.strip())
            if not match:
                continue
            base = match.group("base")
            suffix = match.group("suffix")
            if base.upper() in _SHORT_PREFIX_EXCLUDED_BASES or _GENERIC_NUMBERED_SAMPLE_BASE_RE.match(base):
                continue
            short_prefix_buckets.setdefault(f"{base}\0{suffix}", []).append((sample_id, int(match.group("num"))))
        for members in short_prefix_buckets.values():
            if len(members) < _MIN_SHORT_PREFIX_GROUP_SIZE:
                continue
            ordered = tuple(sample_id for sample_id, _ in sorted(members, key=lambda pair: pair[1]))
            groups.append(ReplicateGroup(members=ordered, signal=ReplicateSignal.SHORT_PREFIX_NUMBER_SUFFIX))
            consumed.update(ordered)

    if include_letter_suffix_signal:
        letter_buckets: dict[str, list[tuple[str, str]]] = {}
        for sample_id, name in sample_names_by_id.items():
            if sample_id in consumed:
                continue
            match = _TRAILING_LETTER_RE.match(name.strip())
            if match:
                letter_buckets.setdefault(match.group("base"), []).append(
                    (sample_id, match.group("letter").upper())
                )
        for members in letter_buckets.values():
            if len(members) < _MIN_LETTER_SUFFIX_GROUP_SIZE:
                continue
            letters = [letter for _, letter in members]
            if len(set(letters)) != len(letters):
                continue  # duplicate letter within one base -- ambiguous, skip
            sorted_pairs = sorted(members, key=lambda pair: pair[1])
            sorted_letters = [letter for _, letter in sorted_pairs]
            if any(
                ord(sorted_letters[i + 1]) - ord(sorted_letters[i]) != 1
                for i in range(len(sorted_letters) - 1)
            ):
                continue  # non-consecutive letters -- e.g. only A and Z
            groups.append(
                ReplicateGroup(
                    members=tuple(sample_id for sample_id, _ in sorted_pairs),
                    signal=ReplicateSignal.TRAILING_LETTER_SUFFIX,
                )
            )
    return groups
