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
     digit (e.g. "_rep1", "-REP_2", "_replicate3"). High confidence: the
     token is unambiguous.
  2. TRAILING_LETTER_SUFFIX -- a single trailing letter after a separator
     (e.g. "_A", "-b"). Lower confidence -- letters are also commonly used
     for genuinely different sites/stations (e.g. "Station_A"/"Station_B"),
     so this signal is gated by a minimum group size and a
     consecutive-letters check: a lone A/B pair never groups, and
     non-consecutive letters (e.g. only A and F) never group either.

Bare trailing digits (e.g. "Sample1" vs "Sample12" vs "Sample2") are never
treated as a replicate signal by either tier -- they're the single most
common false-positive shape (arbitrary/incrementing sample numbering, not
same-site replication) and match neither regex below. This means FAIRe's own
worked example ("S01_1"/"S01_2"/"S01_3", a bare numeric suffix) is not
detected by this module unless the real sample names also carry an explicit
"rep" token or a letter suffix -- an intentional recall/precision tradeoff.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ReplicateSignal(str, Enum):
    EXPLICIT_REP_MARKER = "explicit_rep_marker"
    TRAILING_LETTER_SUFFIX = "trailing_letter_suffix"


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
) -> list[ReplicateGroup]:
    """`sample_names_by_id` maps each sample's own identifier (a supplement
    table's raw sample-id cell string, or a BioSample accession) to the
    free-text name string to run suffix-pattern detection against -- for
    supplement tables the identifier IS the name; for NCBI BioSample the
    identifier is the accession and the name is the `sample_name` attribute
    or BioSample title.

    Returns disjoint groups (an id matched by the explicit-marker signal is
    never re-considered for the letter-suffix signal). A group of size 1
    (no sibling found) is never returned -- nothing to link.
    """
    consumed: set[str] = set()
    groups: list[ReplicateGroup] = []

    explicit_buckets: dict[str, list[tuple[str, int]]] = {}
    for sample_id, name in sample_names_by_id.items():
        match = _EXPLICIT_REP_RE.match(name.strip())
        if match:
            explicit_buckets.setdefault(match.group("base"), []).append(
                (sample_id, int(match.group("num")))
            )
    for members in explicit_buckets.values():
        if len(members) < 2:
            continue
        ordered = tuple(sample_id for sample_id, _ in sorted(members, key=lambda pair: pair[1]))
        groups.append(ReplicateGroup(members=ordered, signal=ReplicateSignal.EXPLICIT_REP_MARKER))
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
