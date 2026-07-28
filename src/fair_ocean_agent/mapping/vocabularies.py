"""Loads FAIRe's controlled vocabularies from the vendored `enums.yaml` and
checks whether a standardized value is a member.

Some FAIRe enums are genuinely closed sets (`platform_enum`, 16 values;
`target_gene_enum`, 29 values; `assay_type_enum`, 3 values) -- upstream
enumerates every allowed term. Others (`env_broad_scale_enum`,
`env_local_scale_enum`, `env_medium_enum`, ...) list exactly one
illustrative ENVO purl as an *example* of the expected shape, not an
exhaustive set -- FAIRe's own convention (confirmed by inspecting the
vendored schema.yaml/enums.yaml directly) is that these fields accept any
term from the relevant ontology. `enums.yaml` carries no machine-readable
flag distinguishing the two cases, so this module uses the only reliable
signal actually present in the vendored file: an enum with exactly one
permissible value is treated as illustrative/open (checked only for
CURIE/purl shape); an enum with more than one is treated as closed
(checked for exact membership).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from fair_ocean_agent.config import REPO_ROOT

FAIRE_SCHEMA_DIR = REPO_ROOT / "schemas" / "faire"

_ONTOLOGY_TERM_PATTERN = "http://purl.obolibrary.org/obo/"


@dataclass(frozen=True)
class VocabCheckResult:
    is_valid: bool
    is_closed_vocab: bool
    message: str


@lru_cache(maxsize=1)
def _load_enums() -> dict:
    with (FAIRE_SCHEMA_DIR / "enums.yaml").open() as f:
        data = yaml.safe_load(f)
    return data.get("enums", data)


def enum_permissible_values(enum_name: str) -> tuple[str, ...]:
    enums = _load_enums()
    enum = enums.get(enum_name)
    if enum is None:
        return ()
    return tuple(enum.get("permissible_values", {}).keys())


def is_closed_vocab(enum_name: str) -> bool:
    """True if `enum_name` is a fully-enumerated closed set rather than a
    single illustrative example. See module docstring for the heuristic."""
    return len(enum_permissible_values(enum_name)) > 1


def check_value(enum_name: str, value: str) -> VocabCheckResult:
    values = enum_permissible_values(enum_name)
    if not values:
        return VocabCheckResult(True, False, f"No enum {enum_name!r} found; not checked")

    if is_closed_vocab(enum_name):
        if value in values:
            return VocabCheckResult(True, True, f"{value!r} is a valid {enum_name} value")
        return VocabCheckResult(
            False, True, f"{value!r} is not one of the {len(values)} allowed {enum_name} values"
        )

    # Open/illustrative enum: only check that it looks like an ontology term.
    if value.startswith(_ONTOLOGY_TERM_PATTERN) or ":" in value:
        return VocabCheckResult(True, False, f"{value!r} looks like an ontology term for {enum_name}")
    return VocabCheckResult(
        False, False, f"{value!r} does not look like an ontology term/CURIE expected for {enum_name}"
    )
