"""The pinned FAIRe v1.0.2 checklist (reference/faire/), parsed into one
spec per term. The checklist spreadsheet is the authority; see
reference/faire/VERSIONS.md for the exact version and why."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[2]
CHECKLIST_XLSX = REPO / "reference/faire/FAIRe_checklist/FAIRe_checklist_v1.0.2.xlsx"
TABLES = ("projectMetadata", "sampleMetadata", "experimentRunMetadata")

# FAIRe guidelines, "Missing values" (reference/faire/FAIRe_guidelines):
# the only allowed wording, from the INSDC missing value vocabulary.
MISSING_VALUE_TERMS = (
    "not applicable: control sample",
    "not applicable: sample group",
    "not applicable",
    "missing: not collected: synthetic construct",
    "missing: not collected: lab stock",
    "missing: not collected: third party data",
    "missing: not collected",
    "missing: not provided: data agreement established pre-2023",
    "missing: not provided",
    "missing: restricted access: endangered species",
    "missing: restricted access: human-identifiable",
    "missing: restricted access",
)
LIST_SEPARATOR = " | "


@dataclass
class Term:
    name: str
    tables: tuple[str, ...]
    section: str
    requirement: str  # M, HR, R, O
    condition: str
    term_type: str
    unit: str
    fixed_format: str
    vocabulary: tuple[str, ...] = field(default_factory=tuple)
    allows_other: bool = False
    example: str = ""
    description: str = ""


def load_checklist(path: Path = CHECKLIST_XLSX) -> dict[str, Term]:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["checklist"]
    ws.reset_dimensions()
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    terms: dict[str, Term] = {}
    for raw in rows[1:]:
        row = {k: ("" if v is None else str(v).strip()) for k, v in zip(header, raw)}
        if not row.get("term_name"):
            continue
        options = [o.strip() for o in row["controlled_vocabulary_options"].split("|") if o.strip()]
        allows_other = any(o.lower().startswith("other") for o in options)
        vocabulary = tuple(o for o in options if not o.lower().startswith("other") and o != "numeric value")
        terms[row["term_name"]] = Term(
            name=row["term_name"],
            tables=tuple(t.strip() for t in row["data_type"].split("|")),
            section=row["section"],
            requirement=row["requirement_level_code"],
            condition=row["requirement_level_condition"],
            term_type=row["term_type"],
            unit=row["unit"],
            fixed_format=row["fixed_format"],
            vocabulary=vocabulary,
            allows_other=allows_other,
            example=row["example"],
            description=row["description"],
        )
    # Post-v1.0.2 slot fix upstream (reference/faire/VERSIONS.md):
    # pcr2_plate_id belongs to experimentRunMetadata.
    if "pcr2_plate_id" in terms:
        terms["pcr2_plate_id"].tables = ("experimentRunMetadata",)
    return terms


_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_RANGE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)\s*-\s*[+-]?(\d+(\.\d*)?|\.\d+)$")


def is_number(value: str, *, integer: bool = False) -> bool:
    if integer:
        return bool(re.fullmatch(r"[+-]?\d+", value)) or bool(re.fullmatch(r"[+-]?\d+\s*-\s*[+-]?\d+", value))
    return bool(_NUMBER.match(value) or _RANGE.match(value))
