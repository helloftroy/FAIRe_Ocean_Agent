"""add entity external_identifier

Revision ID: 6d8324744576
Revises: dd906b116ed2
Create Date: 2026-07-27 14:08:05.870525

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d8324744576'
down_revision: Union[str, Sequence[str], None] = 'dd906b116ed2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # batch_alter_table: SQLite can't ALTER a table to add a constraint
    # in-place (no ALTER TABLE ADD CONSTRAINT support), so this does the
    # standard copy-and-move under the hood on SQLite; on PostgreSQL it
    # compiles to plain ALTER TABLE statements, same end result either way.
    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.add_column(sa.Column('external_identifier', sa.String(), nullable=True))
        batch_op.create_index(
            batch_op.f('ix_entities_external_identifier'), ['external_identifier'], unique=False
        )
        batch_op.create_unique_constraint(
            'uq_entity_study_level_external_id', ['study_id', 'entity_level', 'external_identifier']
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.drop_constraint('uq_entity_study_level_external_id', type_='unique')
        batch_op.drop_index(batch_op.f('ix_entities_external_identifier'))
        batch_op.drop_column('external_identifier')
