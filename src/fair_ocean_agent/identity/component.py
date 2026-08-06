"""Computes the connected component of Studies a given Study belongs to,
for workflow/settle_handlers.py's settle-check and identity/
root_determination.py's root-determination pass.

A component is the transitive closure over TWO distinct edge types, and
walking only one of them misses a real case each:

- **Shared-entity edges** (EntityStudy): two studies are connected if they
  both link to the same real SAMPLE/EXPERIMENT_RUN/SEQUENCING_RUN entity.
  This is the only edge type that connects two independently-seeded
  studies with no citation relationship between them (both
  discovery_depth=0, both discovery_parent_study_id=None) -- confirmed via
  identity/resolution.py::_linked_via_discovery_lineage, which only ever
  checks discovery_parent_study_id/discovery_root_study_id and would see
  no relationship between them at all.
- **Discovery-lineage edges** (Study.discovery_parent_study_id/
  discovery_root_study_id): a citing study freshly created by
  workflow/handlers.py::handle_discover_citing_studies has ZERO EntityStudy
  rows until its own (not-yet-run) DISCOVER_IDENTIFIERS task completes --
  shared-entity edges alone would miss it entirely, understating the
  component and letting a settle-check declare victory while a task that
  could still grow the component is sitting in the queue.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fair_ocean_agent.database.models import EntityStudy, Study


def compute_study_component(session: Session, anchor_study_id: str) -> set[str]:
    visited: set[str] = set()
    frontier: set[str] = {anchor_study_id}

    while frontier:
        study_id = frontier.pop()
        if study_id in visited:
            continue
        visited.add(study_id)

        entity_ids = session.scalars(
            select(EntityStudy.entity_id).where(EntityStudy.study_id == study_id)
        ).all()
        sibling_study_ids: set[str] = set()
        if entity_ids:
            sibling_study_ids = set(
                session.scalars(
                    select(EntityStudy.study_id).where(EntityStudy.entity_id.in_(entity_ids))
                ).all()
            )

        lineage_ids: set[str] = set()
        study = session.get(Study, study_id)
        if study is not None:
            if study.discovery_parent_study_id:
                lineage_ids.add(study.discovery_parent_study_id)
            if study.discovery_root_study_id:
                lineage_ids.add(study.discovery_root_study_id)
            lineage_ids |= set(
                session.scalars(
                    select(Study.study_id).where(Study.discovery_parent_study_id == study_id)
                ).all()
            )
            root_for_children = study.discovery_root_study_id or study_id
            lineage_ids |= set(
                session.scalars(
                    select(Study.study_id).where(Study.discovery_root_study_id == root_for_children)
                ).all()
            )

        frontier |= (sibling_study_ids | lineage_ids) - visited

    return visited
