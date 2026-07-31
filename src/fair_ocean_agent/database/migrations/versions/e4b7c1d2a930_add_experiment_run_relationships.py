"""add experiment-run entity relationships

Revision ID: e4b7c1d2a930
Revises: c31f0d8a62b4
Create Date: 2026-07-31

FAIRe experimentRunMetadata rows represent sample/assay-specific library
instances. They are not physical sequencing runs: multiple libraries may
share one sequencing run. Entity levels are stored as strings, so adding
the `experiment_run` enum value requires no entities-table alteration; this
migration adds the normalized links between those entities.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4b7c1d2a930"
down_revision: Union[str, Sequence[str], None] = "c31f0d8a62b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "entity_relationships",
        sa.Column("entity_relationship_id", sa.String(), nullable=False),
        sa.Column("study_id", sa.String(), nullable=False),
        sa.Column("from_entity_id", sa.String(), nullable=False),
        sa.Column("to_entity_id", sa.String(), nullable=False),
        sa.Column("relationship_type", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["from_entity_id"], ["entities.entity_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["study_id"], ["studies.study_id"]),
        sa.ForeignKeyConstraint(["to_entity_id"], ["entities.entity_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("entity_relationship_id"),
        sa.UniqueConstraint(
            "from_entity_id",
            "to_entity_id",
            "relationship_type",
            name="uq_entity_relationship",
        ),
    )
    op.create_index(
        "ix_entity_relationships_from_entity_id",
        "entity_relationships",
        ["from_entity_id"],
        unique=False,
    )
    op.create_index(
        "ix_entity_relationships_study_id",
        "entity_relationships",
        ["study_id"],
        unique=False,
    )
    op.create_index(
        "ix_entity_relationships_study_type",
        "entity_relationships",
        ["study_id", "relationship_type"],
        unique=False,
    )
    op.create_index(
        "ix_entity_relationships_to_entity_id",
        "entity_relationships",
        ["to_entity_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_entity_relationships_to_entity_id", table_name="entity_relationships")
    op.drop_index("ix_entity_relationships_study_type", table_name="entity_relationships")
    op.drop_index("ix_entity_relationships_study_id", table_name="entity_relationships")
    op.drop_index("ix_entity_relationships_from_entity_id", table_name="entity_relationships")
    op.drop_table("entity_relationships")
