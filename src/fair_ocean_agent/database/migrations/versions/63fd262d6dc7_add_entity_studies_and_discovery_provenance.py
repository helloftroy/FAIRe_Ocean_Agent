"""add entity_studies and study discovery provenance

Revision ID: 63fd262d6dc7
Revises: e4b7c1d2a930
Create Date: 2026-08-06

New `entity_studies` join table: which Study(ies) an Entity belongs to,
for the shareable entity levels (sample/experiment_run/sequencing_run --
see database/enums.py's SHAREABLE_ENTITY_LEVELS). Added *alongside*, not
instead of, `entities.study_id` -- that plain FK stays the fast "home"
read path everywhere in the codebase, unchanged (exact same rationale as
87d37c7b8fe7's study_sources migration, mirrored here for entities
instead of sources). Root cause: two different studies citing the same
real BioSample/experiment/sequencing run previously got two fully
independent Entity rows with no linkage -- confirmed live against a real
paper (10.1038/s42003-024-06136-2, BioProject PRJNA529480/PRJEB73262) that
this silently lost the ability to recognize shared data.

A second, partial unique index on `entities` (entity_level +
external_identifier, shareable levels only) makes a *global* (not
per-study) get-or-create lookup safe going forward. Confirmed directly
against this database before writing this migration: zero existing
(entity_level, external_identifier) collisions across different studies
today (every one of 7787 existing sample/experiment_run/sequencing_run
entities across 101 studies is currently unique by that pair) -- so this
index can be added directly with no merge/backfill needed for existing
duplicates.

Data migration note: every pre-existing `entities` row gets exactly one
`entity_studies` row backfilled here: relationship_type='is_home_of',
confidence='structured_source' -- the same reasoning as 87d37c7b8fe7's own
backfill note (every existing Entity row was created from a directly-
resolved adapter fetch, never a probabilistic guess).

Also adds Study.discovery_depth/discovery_parent_study_id/
discovery_root_study_id/discovery_trigger for citation-expansion discovery
provenance (see workflow/handlers.py's handle_discover_citing_studies) --
0/None/None/None for every pre-existing study, since none of them were
discovered via citation expansion (the mechanism didn't exist yet).
"""
from datetime import datetime
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fair_ocean_agent.database.ids import new_id

# revision identifiers, used by Alembic.
revision: str = "63fd262d6dc7"
down_revision: Union[str, Sequence[str], None] = "e4b7c1d2a930"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SHAREABLE_ENTITY_LEVELS = ("sample", "experiment_run", "sequencing_run")


def upgrade() -> None:
    op.add_column("studies", sa.Column("discovery_depth", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("studies", sa.Column("discovery_parent_study_id", sa.String(), nullable=True))
    op.add_column("studies", sa.Column("discovery_root_study_id", sa.String(), nullable=True))
    op.add_column("studies", sa.Column("discovery_trigger", sa.String(), nullable=True))

    op.create_table(
        "entity_studies",
        sa.Column("entity_study_id", sa.String(), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=False),
        sa.Column("study_id", sa.String(), nullable=False),
        sa.Column("relationship_type", sa.String(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.entity_id"]),
        sa.ForeignKeyConstraint(["study_id"], ["studies.study_id"]),
        sa.PrimaryKeyConstraint("entity_study_id"),
        sa.UniqueConstraint("study_id", "entity_id", name="uq_entity_study"),
    )
    op.create_index(op.f("ix_entity_studies_entity_id"), "entity_studies", ["entity_id"], unique=False)
    op.create_index(op.f("ix_entity_studies_study_id"), "entity_studies", ["study_id"], unique=False)

    placeholders = ", ".join(f"'{level}'" for level in _SHAREABLE_ENTITY_LEVELS)
    op.create_index(
        "uq_entity_shareable_level_external_id",
        "entities",
        ["entity_level", "external_identifier"],
        unique=True,
        sqlite_where=sa.text(f"entity_level IN ({placeholders}) AND external_identifier IS NOT NULL"),
        postgresql_where=sa.text(f"entity_level IN ({placeholders}) AND external_identifier IS NOT NULL"),
    )

    conn = op.get_bind()

    entities = sa.table(
        "entities",
        sa.column("entity_id", sa.String), sa.column("study_id", sa.String),
        sa.column("created_at", sa.DateTime(timezone=True)), sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    entity_studies = sa.table(
        "entity_studies",
        sa.column("entity_study_id", sa.String), sa.column("study_id", sa.String),
        sa.column("entity_id", sa.String), sa.column("relationship_type", sa.String),
        sa.column("confidence", sa.String),
        sa.column("created_at", sa.DateTime(timezone=True)), sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    now = datetime.utcnow()
    for row in conn.execute(sa.select(entities)).fetchall():
        conn.execute(
            entity_studies.insert().values(
                entity_study_id=new_id("ENTSTUDY"),
                study_id=row.study_id,
                entity_id=row.entity_id,
                relationship_type="is_home_of",
                confidence="structured_source",
                created_at=row.created_at or now,
                updated_at=row.updated_at or now,
            )
        )


def downgrade() -> None:
    """Structural reversal only -- the fact that an entity was "home" to a
    study before this migration is not preserved anywhere else once the
    table is dropped (same as 87d37c7b8fe7's own downgrade note)."""
    op.drop_index("uq_entity_shareable_level_external_id", table_name="entities")
    op.drop_index(op.f("ix_entity_studies_study_id"), table_name="entity_studies")
    op.drop_index(op.f("ix_entity_studies_entity_id"), table_name="entity_studies")
    op.drop_table("entity_studies")
    op.drop_column("studies", "discovery_trigger")
    op.drop_column("studies", "discovery_root_study_id")
    op.drop_column("studies", "discovery_parent_study_id")
    op.drop_column("studies", "discovery_depth")
