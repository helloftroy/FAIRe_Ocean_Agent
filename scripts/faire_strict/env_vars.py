"""R25: environmental variables from the env-var extras columns into FAIRe
sampleMetadata columns (user request 2026-11-11).

The export collects physicochemical measurements into two non-FAIRe columns
(mapping/rules.py _PHYSICOCHEMICAL_ENV_VAR_ATTRIBUTES; the FAIRe columns are
suppressed at export by an earlier user request):
  x_env_var_block   "<name>: <value>" per BioSample attribute, mixed with
                    methods sentences from the paper
  x_pulled_env_var  "<name> = <value>", the same attributes plus values
                    pulled from the paper text
R25a takes an entry from x_env_var_block; R25b from x_pulled_env_var, only
for a variable the block did not give. An entry is used only when its
value is ONE plain number in a plausible range, with no unit or a unit that
means FAIRe's unit (fixed-unit terms) or a recognised unit (terms with a
_unit column, which also allow "other: <unit>"). Ranges, inequalities,
words ("in situ", "room temperature"), and missing tokens are left in the
extras columns, which stay as they are.
"""
from __future__ import annotations

import re

# display name (as written by mapping/rules.py) -> FAIRe sampleMetadata term
DISPLAY_TO_TERM = {
    "temperature": "temp",
    "salinity": "salinity",
    "ph": "ph",
    "chlorophyll": "chlorophyll",
    "dissolved oxygen": "diss_oxygen",
    "dissolved inorganic carbon": "diss_inorg_carb",
    "dissolved inorganic nitrogen": "diss_inorg_nitro",
    "dissolved organic carbon": "diss_org_carb",
    "dissolved organic nitrogen": "diss_org_nitro",
    "nitrate": "nitrate",
    "nitrite": "nitrite",
    "organic matter": "org_matter",
    "particulate organic carbon": "part_org_carb",
    "particulate organic nitrogen": "part_org_nitro",
    "suspended particulate matter": "suspend_part_matter",
    "total carbon": "tot_carb",
    "total dissolved nitrogen": "tot_diss_nitro",
    "total nitrogen": "tot_nitro",
    "total organic carbon": "tot_org_carb",
    "total particulate carbon": "tot_part_carb",
}
# Fixed-unit terms: accepted unit spellings (lower case, spaces removed) and a
# plausible range. ppt / ‰ / g/kg are numerically equal to psu for seawater;
# µg/L equals mg/m3.
FIXED = {
    "temp": ({"", "c", "°c", "degc", "degreesc", "degreec", "celsius", "degreecelsius", "degreescelsius", "℃"}, -5, 120),
    "salinity": ({"", "psu", "ppt", "‰", "g/kg", "pss", "pss-78", "practicalsalinityunit", "practicalsalinityunits"}, 0, 400),
    "ph": ({""}, 0, 14),
    "chlorophyll": ({"", "mg/m3", "mg/m³", "mgm-3", "µg/l", "ug/l", "μg/l"}, 0, 10_000),
    "suspend_part_matter": ({"", "mg/l", "mgl-1"}, 0, 1_000_000),
}
# Unit spellings -> FAIRe unit vocabulary value (case-sensitive keys first).
_UNIT_EXACT = {"uM": "µM", "µM": "µM", "μM": "µM", "mM": "other: mM", "nM": "other: nM"}
_UNIT_LOWER = {
    "µmol/l": "µmol/L", "umol/l": "µmol/L", "μmol/l": "µmol/L", "µmoll-1": "µmol/L", "umoll-1": "µmol/L",
    "µmol/kg": "µmol/kg", "umol/kg": "µmol/kg", "μmol/kg": "µmol/kg", "µmolkg-1": "µmol/kg", "umolkg-1": "µmol/kg",
    "mmol/l": "mmol/L", "mol/l": "mol/L", "mmol/kg": "mmol/kg",
    "mol/m3": "mol/m3", "mmol/m3": "mmol/m3", "µmol/m3": "µmol/m3", "umol/m3": "µmol/m3",
    "mg/l": "mg/L", "mgl-1": "mg/L", "µg/l": "µg/L", "ug/l": "µg/L", "μg/l": "µg/L",
    "ml/l": "mL/L", "ppm": "parts per million", "partspermillion": "parts per million",
}
_MISSING = re.compile(r"^(not applicable|not collected|not provided|missing|unknown|na|n/a|none|bdl|nd|-)\b", re.I)
_NUMBER = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*(.*)$")
_ENTRY = re.compile(
    r"^\s*(" + "|".join(sorted(map(re.escape, DISPLAY_TO_TERM), key=len, reverse=True)) + r")\s*([:=])\s*(.+?)\s*$",
    re.I,
)


def _normalise(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("◦", "°").replace("º", "°").replace("˚", "°")
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)          # 6,5 -> 6.5
    value = re.sub(r"\s*±\s*\d+(?:\.\d+)?\s*\S*", "", value)  # drop "± 0.5°C"
    value = value.replace("_mol", "µmol")                  # mangled µ
    return value.strip()


def parse_value(term: str, raw: str) -> tuple[str, str | None] | None:
    """(number, FAIRe unit or None) for one entry value, or None if unusable.
    unit None = a fixed-unit term (no unit column)."""
    value = _normalise(raw)
    if not value or _MISSING.match(value):
        return None
    m = _NUMBER.match(value)
    if not m:
        return None
    number, rest = m.group(1), m.group(2).strip()
    if re.search(r"\d", rest):          # a range, a second number, "1:2"
        return None
    x = float(number)
    if term in FIXED:
        accepted, low, high = FIXED[term]
        if rest.lower().replace(" ", "") not in accepted or not low <= x <= high:
            return None
        return number, None
    if x < 0:
        return None
    if not rest:
        return number, "missing: not provided"
    unit = _UNIT_EXACT.get(rest) or _UNIT_LOWER.get(rest.lower().replace(" ", ""))
    if unit is None:
        if re.search(r"[a-z]{4,}", rest.lower()) and "/" not in rest:  # words, not a unit
            return None
        unit = f"other: {rest}"
    return number, unit


def entries(cell: str, separator: str) -> list[tuple[str, str]]:
    """(FAIRe term, raw value) for every "<name><separator> <value>" item."""
    out = []
    for item in (cell or "").split("|"):
        m = _ENTRY.match(item)
        if m and m.group(2) == separator:
            out.append((DISPLAY_TO_TERM[m.group(1).lower()], m.group(3)))
    return out


def env_values(block: str, pulled: str) -> dict[str, tuple[str, str | None, str, str]]:
    """term -> (number, unit, rule, source item) for one sample row. The first
    usable entry per term wins; R25b only for terms R25a did not fill."""
    found: dict[str, tuple[str, str | None, str, str]] = {}
    for rule, cell, sep in (("R25a_env_var_from_block", block, ":"), ("R25b_env_var_from_pulled", pulled, "=")):
        for term, raw in entries(cell, sep):
            if term in found:
                continue
            parsed = parse_value(term, raw)
            if parsed:
                found[term] = (*parsed, rule, raw)
    return found


ALL_COLUMNS = tuple(
    col for term in DISPLAY_TO_TERM.values() for col in ((term,) if term in FIXED else (term, f"{term}_unit"))
)
