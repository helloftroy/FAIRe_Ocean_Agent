"""R21: geo_loc_name against the INSDC /geo_loc_name vocabulary.

Format (INSDC, FAIRe): "<country or sea>[:<region>][, <locality>]".
The vocabulary is the pinned copy in reference/insdc/ (revised
2024-07-08). Only the country/sea part is checked or moved; the rest of
the text is kept exactly as written.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_DIR = REPO / "reference/insdc"
CURRENT = tuple(l.strip() for l in (_DIR / "geo_loc_name_current.txt").read_text().splitlines() if l.strip())
HISTORICAL = tuple(l.strip() for l in (_DIR / "geo_loc_name_historical.txt").read_text().splitlines() if l.strip())
_BY_LOWER = {c.lower(): c for c in CURRENT}

# Other names for a vocabulary entry. Only names that cannot mean
# anything else; checked against the vocabulary in the dry run.
ALIASES = {
    "united states": "USA",
    "united states of america": "USA",
    "u.s.a.": "USA",
    "u.s.a": "USA",
    "u.s.": "USA",
    "us": "USA",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "russian federation": "Russia",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "vietnam": "Viet Nam",
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    "people's republic of china": "China",
    "pr china": "China",
    "p.r. china": "China",
    "czechia": "Czechia",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
}
# Historical names (INSDC list) whose country has one current name.
RENAMED = {
    "Czech Republic": "Czechia",
    "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "Burma": "Myanmar",
    "East Timor": "Timor-Leste",
    "Zaire": "Democratic Republic of the Congo",
    "Belgian Congo": "Democratic Republic of the Congo",
    "British Guiana": "Guyana",
    "Siam": "Thailand",
    "Micronesia": "Micronesia, Federated States of",
}
_GAZ = re.compile(r"^GAZ\s*:\s*(?=\D)", re.IGNORECASE)

# Parts of a country written as the country: kept as the region.
REGION_OF = {
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
}


def _country(text: str) -> tuple[str, str] | None:
    """(vocabulary entry, rule) for a candidate country/sea name."""
    if text in CURRENT:
        return text, "keep"
    low = text.lower().strip().rstrip(".")
    if low in _BY_LOWER:
        return _BY_LOWER[low], "R21a_geo_case"
    if low in ALIASES and ALIASES[low] in CURRENT:
        return ALIASES[low], "R21b_geo_alias"
    return None


def resolve_geo(item: str) -> tuple[str, str] | None:
    """(new item, rule) or None when no single INSDC name can be found."""
    text = item.strip()
    gaz = _GAZ.match(text)
    if gaz:  # "GAZ:Norway" (GAZ ontology label without an ID)
        inner = resolve_geo(text[gaz.end():])
        if inner:
            return inner[0], "R21f_geo_gaz_prefix"
        return None
    head, colon, rest = text.partition(":")
    head = head.strip()
    if colon:
        hit = _country(head)
        if hit:
            country, rule = hit
            if rule == "keep" and head == text.partition(":")[0]:
                return item, "keep"
            return f"{country}:{rest}", rule
        if head in RENAMED:
            return f"{RENAMED[head]}:{rest}", "R21g_geo_renamed_country"
        if head in HISTORICAL:
            return item, "R21h_geo_historical_name"
        region = REGION_OF.get(head.lower())
        if region:
            return f"{region}: {head},{rest}" if rest.strip() else f"{region}: {head}", "R21e_geo_region_of"
        return None
    if text in RENAMED:
        return RENAMED[text], "R21g_geo_renamed_country"
    if text in HISTORICAL:
        return item, "R21h_geo_historical_name"
    parts = [p.strip() for p in text.split(",")]
    found = []
    for i, part in enumerate(parts):
        hit = _country(part)
        if hit:
            found.append((i, hit[0]))
        elif part.lower() in REGION_OF:
            found.append((i, REGION_OF[part.lower()]))
    if len({c for _, c in found}) != 1 or len(found) != 1:
        return None
    i, country = found[0]
    region_word = parts[i] if parts[i].lower() in REGION_OF else None
    others = [p for j, p in enumerate(parts) if j != i and p]
    if region_word:
        others = [region_word, *others] if i == 0 else [*others, region_word]
    if not others:
        _, rule = _country(text) or (None, "R21e_geo_region_of")
        return country, rule
    if region_word and len(parts) == 1:
        return f"{country}: {region_word}", "R21e_geo_region_of"
    if i == 0:
        return f"{country}: {', '.join(others)}", "R21c_geo_country_first_to_prefix"
    if i == len(parts) - 1:
        return f"{country}: {', '.join(others)}", "R21d_geo_country_last_to_prefix"
    return None
