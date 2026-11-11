"""add curation exclusions

Revision ID: b5d1c7e2f9a4
Revises: 9c2e4f7b1a3d
Create Date: 2026-11-11

New `curation_exclusions` table: one row per curation decision made by
scripts/apply_curation.py (sample deleted / unlinked, BioProject dropped,
study flagged not marine), kept after the rows it describes are deleted.
See data_integrity_audit.md. No foreign keys on purpose.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b5d1c7e2f9a4"
down_revision: Union[str, Sequence[str], None] = "9c2e4f7b1a3d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "curation_exclusions",
        sa.Column("curation_id", sa.String(), primary_key=True),
        sa.Column("level", sa.String(), nullable=False),
        sa.Column("accession", sa.String(), nullable=True),
        sa.Column("entity_id", sa.String(), nullable=True),
        sa.Column("study_id", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("rule", sa.String(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("run_label", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_curation_exclusions_level", "curation_exclusions", ["level"])
    op.create_index("ix_curation_exclusions_accession", "curation_exclusions", ["accession"])
    op.create_index("ix_curation_exclusions_entity_id", "curation_exclusions", ["entity_id"])
    op.create_index("ix_curation_exclusions_study_id", "curation_exclusions", ["study_id"])


def downgrade() -> None:
    op.drop_table("curation_exclusions")
