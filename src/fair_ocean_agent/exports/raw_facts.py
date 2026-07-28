"""Raw long-form fact table export (section 19, item 13). The only export
that's meaningful in Milestone 1 -- FAIRe/BeBOP exports need the mapping
layer (Milestone 6)."""
from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.models import RawFact

FIELDS = (
    "fact_id",
    "study_id",
    "entity_id",
    "source_id",
    "source_locator",
    "raw_field_name",
    "raw_value",
    "normalized_text_value",
    "evidence_quote",
    "fact_type_candidate",
    "entity_level",
    "support_type",
    "extraction_method",
    "extractor_version",
    "model_name",
    "prompt_version",
    "review_status",
    "created_at",
)


def export_raw_facts(session: Session, output_path: str | Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    facts = session.scalars(select(RawFact).order_by(RawFact.study_id, RawFact.created_at)).all()

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for fact in facts:
            writer.writerow({field: getattr(fact, field) for field in FIELDS})

    return len(facts)
