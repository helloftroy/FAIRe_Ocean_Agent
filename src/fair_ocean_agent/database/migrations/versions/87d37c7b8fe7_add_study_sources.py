"""add study sources

Revision ID: 87d37c7b8fe7
Revises: 5b9e1d3c7a20
Create Date: 2026-07-30 16:23:22.728605

New `study_sources` join table: which Study(ies) a Source belongs to, with
per-link `relationship_type` and `confidence`. Added *alongside*, not
instead of, `sources.study_id` -- that plain FK stays the fast "home" read
path everywhere in the codebase, unchanged; this table is what makes a
Source belonging to more than one Study representable at all (a paper can
be tied to many projects/samples and vice versa -- see
identity/resolution.py's resolve_or_create_study()).

Data migration note: every pre-existing `sources` row gets exactly one
`study_sources` row backfilled here: relationship_type='is_home_of',
confidence='structured_source'. This is a genuine, not an approximate,
description of that data -- every existing Source row was created from
either a directly-resolved adapter fetch_record() call or a structurally-
parsed supplement reference (never a probabilistic guess), which is
exactly what 'structured_source' (tier 1 in the new confidence hierarchy)
means. Tagging it 'inferred' would be actively wrong. created_at/updated_at
are copied from each source row's own timestamps (not the migration's own
run time) since this link has existed since the Source was created.
"""
from datetime import datetime
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fair_ocean_agent.database.ids import new_id

# revision identifiers, used by Alembic.
revision: str = '87d37c7b8fe7'
down_revision: Union[str, Sequence[str], None] = '5b9e1d3c7a20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'study_sources',
        sa.Column('study_source_id', sa.String(), nullable=False),
        sa.Column('study_id', sa.String(), nullable=False),
        sa.Column('source_id', sa.String(), nullable=False),
        sa.Column('relationship_type', sa.String(), nullable=False),
        sa.Column('confidence', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['sources.source_id']),
        sa.ForeignKeyConstraint(['study_id'], ['studies.study_id']),
        sa.PrimaryKeyConstraint('study_source_id'),
        sa.UniqueConstraint('study_id', 'source_id', name='uq_study_source'),
    )
    op.create_index(op.f('ix_study_sources_source_id'), 'study_sources', ['source_id'], unique=False)
    op.create_index(op.f('ix_study_sources_study_id'), 'study_sources', ['study_id'], unique=False)

    conn = op.get_bind()

    sources = sa.table(
        'sources',
        sa.column('source_id', sa.String), sa.column('study_id', sa.String),
        sa.column('created_at', sa.DateTime(timezone=True)), sa.column('updated_at', sa.DateTime(timezone=True)),
    )
    study_sources = sa.table(
        'study_sources',
        sa.column('study_source_id', sa.String), sa.column('study_id', sa.String),
        sa.column('source_id', sa.String), sa.column('relationship_type', sa.String),
        sa.column('confidence', sa.String),
        sa.column('created_at', sa.DateTime(timezone=True)), sa.column('updated_at', sa.DateTime(timezone=True)),
    )

    now = datetime.utcnow()
    for row in conn.execute(sa.select(sources)).fetchall():
        conn.execute(
            study_sources.insert().values(
                study_source_id=new_id('STUDYSRC'),
                study_id=row.study_id,
                source_id=row.source_id,
                relationship_type='is_home_of',
                confidence='structured_source',
                created_at=row.created_at or now,
                updated_at=row.updated_at or now,
            )
        )


def downgrade() -> None:
    """Structural reversal only -- the fact that a source was "home" to a
    study before this migration is not preserved anywhere else once the
    table is dropped (same as 0fc0bf2bf46a's own downgrade note)."""
    op.drop_index(op.f('ix_study_sources_study_id'), table_name='study_sources')
    op.drop_index(op.f('ix_study_sources_source_id'), table_name='study_sources')
    op.drop_table('study_sources')
