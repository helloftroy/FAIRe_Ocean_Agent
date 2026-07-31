"""add prepared source texts

Revision ID: c31f0d8a62b4
Revises: 87d37c7b8fe7
Create Date: 2026-07-30

Persists normalized text prepared from retrieved source assets so
supplement retrieval and optional LLM extraction can run as distinct,
resumable stages.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c31f0d8a62b4"
down_revision: Union[str, Sequence[str], None] = "87d37c7b8fe7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prepared_source_texts",
        sa.Column("prepared_source_text_id", sa.String(), nullable=False),
        sa.Column("study_id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("data_asset_id", sa.String(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("preparation_method", sa.String(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("llm_model_name", sa.String(), nullable=True),
        sa.Column("llm_prompt_version", sa.String(), nullable=True),
        sa.Column("llm_extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["data_asset_id"], ["data_assets.asset_id"]),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"]),
        sa.ForeignKeyConstraint(["study_id"], ["studies.study_id"]),
        sa.PrimaryKeyConstraint("prepared_source_text_id"),
        sa.UniqueConstraint(
            "data_asset_id",
            "content_hash",
            "title",
            name="uq_prepared_source_text_asset_hash_title",
        ),
    )
    op.create_index(
        "ix_prepared_source_texts_content_hash",
        "prepared_source_texts",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        "ix_prepared_source_texts_data_asset_id",
        "prepared_source_texts",
        ["data_asset_id"],
        unique=False,
    )
    op.create_index(
        "ix_prepared_source_texts_source_id",
        "prepared_source_texts",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_prepared_source_texts_study_id",
        "prepared_source_texts",
        ["study_id"],
        unique=False,
    )
    op.create_index(
        "ix_prepared_source_texts_study_source",
        "prepared_source_texts",
        ["study_id", "source_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_prepared_source_texts_study_source", table_name="prepared_source_texts")
    op.drop_index("ix_prepared_source_texts_study_id", table_name="prepared_source_texts")
    op.drop_index("ix_prepared_source_texts_source_id", table_name="prepared_source_texts")
    op.drop_index("ix_prepared_source_texts_data_asset_id", table_name="prepared_source_texts")
    op.drop_index("ix_prepared_source_texts_content_hash", table_name="prepared_source_texts")
    op.drop_table("prepared_source_texts")
