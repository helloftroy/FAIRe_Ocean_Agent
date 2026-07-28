"""Cross-source agreement checks (section 16): compare independently
extracted values for the same conceptual field across different sources
for the same study.

Crossref/Europe PMC/OpenAlex are genuinely independent metadata pipelines
for the same publication (different APIs, different underlying
processing) -- agreement across them is a real positive signal, not
circular. Each keeps its own bibliographic fields on its own `sources` row
(no cross-adapter merge at write time -- see that model's docstring); this
module is what checks whether they actually agree, by comparing the
raw_facts each source contributed independently.

Section 16 also warns: "Do not count mirrored NCBI and ENA records as
fully independent evidence." This pipeline doesn't currently have two
sources describing the *same* record (ncbi_biosample gives sample
attributes, ena gives run metadata -- overlapping scope, not duplicate
records), so no mirror-collapsing logic is needed yet. If a future source
mirrors an existing one (e.g. a dedicated NCBI SRA adapter duplicating
ena's read_run data), collapse them via `sources.mirror_group` before
counting independent sources here, rather than double-counting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.enums import ValidationStatus
from fair_ocean_agent.database.models import RawFact, Source


def _normalize(value: str) -> str:
    """Lowercase, collapse whitespace, and strip trailing sentence
    punctuation. Found via live validation: Europe PMC's "title" field
    always ends in a period ("...thermal challenge."), while Crossref's and
    OpenAlex's don't -- with a naive normalize, EVERY title comparison
    across these three sources registered as "conflicting" (61/98 real
    studies), when 47 of those were this exact same non-content artifact.
    A real disagreement (different words) still won't match after this."""
    return " ".join(value.strip().lower().split()).rstrip(".")


@dataclass
class CrossSourceComparison:
    fact_type_candidate: str
    values_by_source: dict[str, str] = field(default_factory=dict)
    status: str = ValidationStatus.NOT_ASSESSED.value
    message: str = ""


def compare_field_across_sources(session: Session, study_id: str, fact_type_candidate: str) -> CrossSourceComparison:
    """Looks at every raw_fact for this study with this fact_type_candidate,
    takes the first value reported per distinct source_name (a source
    reporting the same field twice for one study would be unexpected), and
    compares. Fewer than 2 independent sources means there's nothing to
    cross-check -- NOT_ASSESSED, not a failure."""
    facts = session.scalars(
        select(RawFact).where(RawFact.study_id == study_id, RawFact.fact_type_candidate == fact_type_candidate)
    ).all()

    values_by_source: dict[str, str] = {}
    for fact in facts:
        if fact.source_id is None or not fact.raw_value:
            continue
        source = session.get(Source, fact.source_id)
        if source is None:
            continue
        values_by_source.setdefault(source.source_name, fact.raw_value)

    if len(values_by_source) < 2:
        return CrossSourceComparison(
            fact_type_candidate,
            values_by_source,
            ValidationStatus.NOT_ASSESSED.value,
            f"Only {len(values_by_source)} independent source(s) reported {fact_type_candidate!r}; nothing to cross-check.",
        )

    distinct_normalized = {_normalize(v) for v in values_by_source.values()}
    if len(distinct_normalized) == 1:
        return CrossSourceComparison(
            fact_type_candidate,
            values_by_source,
            ValidationStatus.CONFIRMED.value,
            f"{len(values_by_source)} independent sources agree on {fact_type_candidate!r}.",
        )
    return CrossSourceComparison(
        fact_type_candidate,
        values_by_source,
        ValidationStatus.CONFLICTING.value,
        f"Sources disagree on {fact_type_candidate!r}: {values_by_source}",
    )
