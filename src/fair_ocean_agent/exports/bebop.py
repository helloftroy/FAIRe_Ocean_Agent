"""BeBOP/MIOP export -- depends on mapping/bebop.py's map_study_to_bebop,
which is out of scope for this pipeline's current (paper-derived) input.
See that module's docstring."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from fair_ocean_agent.mapping.bebop import BebopMappingNotApplicable


def export_bebop(session: Session, output_dir: str | Path) -> dict[str, int]:
    raise BebopMappingNotApplicable(
        "BeBOP/MIOP export depends on mapping/bebop.py's map_study_to_bebop, "
        "which is out of scope for this pipeline's current input types -- "
        "see that module's docstring."
    )
