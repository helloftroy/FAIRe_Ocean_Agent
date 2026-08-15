"""add api paper corrections

Revision ID: 415058e96f35
Revises: a7f9e341957f
Create Date: 2026-08-12 18:29:20.052302

New `api_paper_corrections` table: a durable, code-populated log of every
case where a structured API value (BioSample/ENA/...) was found to
contradict the paper's own text and was corrected -- per an explicit user
request to stop losing track of these. Only ever written by the LLM
verification mechanism that found the mismatch; never hand-edited.

Note: autogenerate also proposed 3 unrelated foreign-key-constraint diffs
on `entities`/`studies` (root_study_id, discovery_root_study_id,
discovery_parent_study_id) -- a known SQLite-reflection artifact from
earlier migrations, not a real schema change caused by this table. Left
out of this migration since they're not this change's concern.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '415058e96f35'
down_revision: Union[str, Sequence[str], None] = 'a7f9e341957f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'api_paper_corrections',
        sa.Column('correction_id', sa.String(), nullable=False),
        sa.Column('study_id', sa.String(), nullable=False),
        sa.Column('entity_id', sa.String(), nullable=True),
        sa.Column('api_faire_term', sa.String(), nullable=False),
        sa.Column('api_value', sa.Text(), nullable=False),
        sa.Column('corrected_faire_term', sa.String(), nullable=False),
        sa.Column('corrected_value', sa.Text(), nullable=False),
        sa.Column('supporting_quote', sa.Text(), nullable=False),
        sa.Column('detector', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['entity_id'], ['entities.entity_id']),
        sa.ForeignKeyConstraint(['study_id'], ['studies.study_id']),
        sa.PrimaryKeyConstraint('correction_id'),
    )
    op.create_index(op.f('ix_api_paper_corrections_entity_id'), 'api_paper_corrections', ['entity_id'], unique=False)
    op.create_index(op.f('ix_api_paper_corrections_study_id'), 'api_paper_corrections', ['study_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_api_paper_corrections_study_id'), table_name='api_paper_corrections')
    op.drop_index(op.f('ix_api_paper_corrections_entity_id'), table_name='api_paper_corrections')
    op.drop_table('api_paper_corrections')
