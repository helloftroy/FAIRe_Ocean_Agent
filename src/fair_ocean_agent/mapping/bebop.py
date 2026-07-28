"""BeBOP/MIOP raw_fact mapping -- decided out of scope for this pipeline's
current input types, not blocked or undecided.

The schema itself is available and compiled: `schemas/miop/` +
`schemas/bebop/` -> `standards/compiled/bebop_miop_registry.json` (see
Milestone 6b). What's resolved here is the separate question of whether
this pipeline should map its own raw_facts onto those fields the way
`mapping/faire.py` does for FAIRe.

MIOP/BeBOP's ~21 fields describe metadata *about a protocol document*
(who wrote it, its license, version, maturity level -- see
schemas/miop/README.md) -- a citable, authored, versioned SOP artifact.
Every study this pipeline currently ingests comes from a published paper
or a repository record (Crossref/Europe PMC/OpenAlex/NCBI/ENA), never from
an actual BeBOP protocol submission. A paper's free-text methods section
(what Milestone 4's LLM extraction pulls out as `DNA_extraction_method`,
`PCR_amplification_conditions`, etc.) is not the same kind of thing as an
authored, licensed, versioned protocol document, and forcing it into
MIOP's fields would fabricate authorship/license/version metadata papers
don't carry. Confirmed with the user: for paper-derived input, this is
correctly out of scope, not a gap to close.

This is deliberately **not** a permanent decision against BeBOP/MIOP
mapping in general -- if this pipeline ever ingests an actual protocol
submission (e.g. a BeBOP-formatted document, or a lab's own SOP with real
authorship/license/version metadata), `map_study_to_bebop` is exactly
where that would be built, following `mapping/faire.py`'s
rules-table pattern against `standards/compiled/bebop_miop_registry.json`.
Revisit this module if/when that input type exists; don't build it
speculatively against input this pipeline doesn't have.
"""
from __future__ import annotations

from sqlalchemy.orm import Session


class BebopMappingNotApplicable(NotImplementedError):
    pass


def map_study_to_bebop(session: Session, study_id: str) -> int:
    raise BebopMappingNotApplicable(
        "BeBOP/MIOP raw_fact mapping is out of scope for this pipeline's "
        "current input types (published papers and repository records), not "
        "unbuilt or blocked -- MIOP/BeBOP describes protocol-document "
        "metadata (author, license, version) that paper-derived facts don't "
        "carry. See mapping/bebop.py's module docstring for the reasoning "
        "and what would justify revisiting this."
    )
