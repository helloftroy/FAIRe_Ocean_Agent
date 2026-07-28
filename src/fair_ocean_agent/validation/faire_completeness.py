"""Which FAIRe fields this pipeline can actually check for completeness
today.

FAIRe's `requirement_level_condition` (compiled into
`standards.faire_registry`'s `faire_terms`, e.g. env_broad_scale:
"Mandatory unless samp_category = negative control, positive control, or
PCR standard") mostly references *other* FAIRe fields
(`samp_category`, `assay_type`, `pcr_0_1`, `dna_cleanup_0_1`, ...) that
this pipeline doesn't populate anywhere -- no source adapter or
extraction step produces them. A completeness check can't evaluate "is
this exempt because samp_category = negative control" when samp_category
itself is never known; guessing would risk exactly the false positives
(flagging a control sample for "missing" a field it was never required to
have) this check exists to avoid.

So this module deliberately covers only **unconditionally** Mandatory
fields -- `requirement_level_code == "M"` and `requirement_level_condition
is None`. Conditionally-Mandatory fields are excluded outright, not
approximated: `validation_handlers.py`'s `populate_faire_missingness_for_study`
only ever asks about the fields this module returns, so a conditional
field is simply never checked, not silently checked wrong.
"""
from __future__ import annotations

from fair_ocean_agent.standards.faire_registry import build_faire_registry

# The only two FAIRe classes this pipeline has real per-row data for --
# see that module's own scope note. ampData/stdData/experimentRunMetadata/
# eLowQuantData/taxaRaw/taxaFinal have no entity to check completeness
# against yet (exports/faire.py writes them header-only), so checking
# their Mandatory fields would just report 100% missing uniformly, which
# isn't a meaningful completeness signal.
CHECKABLE_CLASSES = ("projectMetadata", "sampleMetadata")


def unconditionally_mandatory_faire_fields() -> dict[str, list[dict]]:
    """Returns {class_name: [term, ...]} for every FAIRe term that is
    Mandatory with no requirement_level_condition, restricted to
    CHECKABLE_CLASSES and grouped by which of those classes it belongs to
    (a term can belong to more than one class, e.g. `assay_name`)."""
    terms = build_faire_registry()
    by_class: dict[str, list[dict]] = {name: [] for name in CHECKABLE_CLASSES}
    for term in terms:
        if term.get("requirement_level_code") != "M":
            continue
        if term.get("requirement_level_condition"):
            continue
        for class_name in term.get("data_type") or []:
            if class_name in by_class:
                by_class[class_name].append(term)
    return by_class
