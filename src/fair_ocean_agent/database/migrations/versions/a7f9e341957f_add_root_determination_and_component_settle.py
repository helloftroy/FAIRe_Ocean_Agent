"""add root determination and component-settle bookkeeping

Revision ID: a7f9e341957f
Revises: 63fd262d6dc7
Create Date: 2026-08-06

Follow-up to 63fd262d6dc7's entity_studies table (which made a real
BioSample/experiment_run/sequencing_run Entity linkable to more than one
Study). That migration made SHARING representable; this one makes it
possible to say WHICH linked Study is the authoritative ("root") source
for a shared entity's broadcast-style (study-wide LLM/text) facts --
entities.study_id ("home") is just whichever study's discovery happened to
reach the accession first, an accident of task-queue processing order for
two independently-seeded studies with no citation relationship between
them, not a deliberate answer about which paper actually collected the
data.

Root determination (identity/root_determination.py) only runs once every
Study sharing an entity has finished discovering -- otherwise an early
answer could be invalidated by a not-yet-processed sibling. Whether that
point has been reached is tracked per connected component (the transitive
closure of studies linked by shared entities and/or citation-discovery
lineage, identity/component.py) via the new studies.entity_component_*
columns, checked by a new self-rescheduling CHECK_COMPONENT_SETTLED task
(workflow/settle_handlers.py) -- TaskType is a plain string column with no
DB-level enum/CheckConstraint anywhere in this schema, so that new task
type itself needs no migration.

Backfill: every entity that today has exactly one entity_studies row is
unambiguously its own root (no algorithm needed, matches
identity/entity_linking.py::create_entity's own eager-set-at-creation
behavior going forward); an entity with more than one is left `pending`
for the first real settle-check to resolve, since the actual algorithm
needs live publication-date reads that don't belong in a schema migration.
Every study's entity_component_status starts `not_applicable` unless it
has at least one shareable-level entity, in which case it starts
`pending` (the very next settle-check, triggered organically by ordinary
task processing, will pick it up).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a7f9e341957f"
down_revision: Union[str, Sequence[str], None] = "63fd262d6dc7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SHAREABLE_ENTITY_LEVELS = ("sample", "experiment_run", "sequencing_run")


def upgrade() -> None:
    op.add_column("studies", sa.Column("entity_component_id", sa.String(), nullable=True))
    op.add_column(
        "studies",
        sa.Column("entity_component_status", sa.String(), nullable=False, server_default="not_applicable"),
    )
    op.add_column("studies", sa.Column("entity_component_settled_at", sa.DateTime(timezone=True), nullable=True))

    # No explicit FK constraint on root_study_id, matching 63fd262d6dc7's
    # own precedent for discovery_parent_study_id/discovery_root_study_id
    # (self-referential nullable string columns added the same way there --
    # SQLite doesn't support adding a FK constraint via a simple ALTER TABLE
    # ADD COLUMN, and this codebase doesn't bother with a batch table
    # rebuild for it; the ORM model's own ForeignKey() declaration still
    # applies to any freshly created database via create_all()).
    op.add_column("entities", sa.Column("root_study_id", sa.String(), nullable=True))
    op.add_column("entities", sa.Column("root_status", sa.String(), nullable=False, server_default="not_shared"))
    op.add_column("entities", sa.Column("root_determined_at", sa.DateTime(timezone=True), nullable=True))

    conn = op.get_bind()

    entities = sa.table(
        "entities",
        sa.column("entity_id", sa.String), sa.column("study_id", sa.String), sa.column("entity_level", sa.String),
        sa.column("external_identifier", sa.String),
        sa.column("root_study_id", sa.String), sa.column("root_status", sa.String),
        sa.column("root_determined_at", sa.DateTime(timezone=True)),
    )
    entity_studies = sa.table(
        "entity_studies", sa.column("entity_id", sa.String), sa.column("study_id", sa.String),
    )
    studies = sa.table(
        "studies", sa.column("study_id", sa.String), sa.column("entity_component_status", sa.String),
    )

    now = sa.func.now()

    # Every existing entity with exactly one entity_studies row is
    # unambiguously its own root.
    link_counts = dict(
        conn.execute(
            sa.select(entity_studies.c.entity_id, sa.func.count())
            .group_by(entity_studies.c.entity_id)
        ).all()
    )
    for row in conn.execute(sa.select(entities.c.entity_id, entities.c.study_id)).fetchall():
        if link_counts.get(row.entity_id, 0) == 1:
            conn.execute(
                entities.update()
                .where(entities.c.entity_id == row.entity_id)
                .values(root_study_id=row.study_id, root_status="determined", root_determined_at=now)
            )
        elif link_counts.get(row.entity_id, 0) > 1:
            conn.execute(
                entities.update().where(entities.c.entity_id == row.entity_id).values(root_status="pending")
            )

    placeholders = ", ".join(f"'{level}'" for level in _SHAREABLE_ENTITY_LEVELS)
    shareable_study_ids = {
        row.study_id
        for row in conn.execute(
            sa.text(f"SELECT DISTINCT study_id FROM entities WHERE entity_level IN ({placeholders})")
        ).fetchall()
    }
    for study_id in shareable_study_ids:
        conn.execute(
            studies.update().where(studies.c.study_id == study_id).values(entity_component_status="pending")
        )


def downgrade() -> None:
    """Structural reversal only -- root/component determinations made under
    this schema are not preserved anywhere else once these columns are
    dropped (same convention as 63fd262d6dc7's own downgrade note)."""
    op.drop_column("entities", "root_determined_at")
    op.drop_column("entities", "root_status")
    op.drop_column("entities", "root_study_id")
    op.drop_column("studies", "entity_component_settled_at")
    op.drop_column("studies", "entity_component_status")
    op.drop_column("studies", "entity_component_id")
