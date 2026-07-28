"""fold missingness into standardized_values and publications into sources

Revision ID: 0fc0bf2bf46a
Revises: 6d8324744576
Create Date: 2026-07-28 14:41:09

Schema simplification (two tables that both had to stay in sync for the
same information, folded into their natural home instead):

- `missingness` -> new `missingness_status`/`sources_inspected`/`reason`
  columns on `standardized_values`. Every (study, entity, target_field)
  this pipeline checks now gets exactly one row there, whether or not a
  value was found -- `standardized_value` is null and `missingness_status`
  carries why when nothing was found.
- `publications` -> new `authors`/`journal`/`publication_year`/
  `fulltext_available` columns on `sources` (a publication is just one
  flavor of source, `source_type='publication_api'`). `open_access_status`
  folds onto `sources.access_status`, which already used the same enum.
  `doi`/`pmid`/`pmcid`/`openalex_id`/`title`/`relationship_to_study` are
  NOT recreated -- `external_identifier` already holds the DOI for every
  publication-type source row, `ExternalIdentifier` already tracks
  cross-references, and `Study.title` already holds the canonical merged
  title; duplicating them here would just relocate the sync problem this
  migration removes.

Data migration note: `publications` held one row per study, merged
"first non-null wins" across whichever of crossref/europe_pmc/openalex
answered -- it never recorded which adapter contributed which field. This
migration attaches that merged snapshot to one real Source row per study
(preferring crossref, then europe_pmc, then openalex, matching
workflow/handlers.py's own adapter-priority order) as a one-time best-
effort backfill; it is not a lossless per-adapter reconstruction, since
that attribution was already discarded by the old design. Re-running
`fair-ocean weekly-update` + `worker` after this migration will naturally
repopulate these fields per-adapter going forward (Milestone 7's refresh
path). A study with no publication-type Source row at all (shouldn't
happen in practice -- a Publication row implies at least one adapter
answered) has nothing to attach to and that historical row is dropped.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from fair_ocean_agent.database.ids import new_id

# revision identifiers, used by Alembic.
revision: str = '0fc0bf2bf46a'
down_revision: Union[str, Sequence[str], None] = '6d8324744576'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PUBLICATION_ADAPTER_PRIORITY = ('crossref', 'europe_pmc', 'openalex')


def upgrade() -> None:
    with op.batch_alter_table('standardized_values', schema=None) as batch_op:
        batch_op.add_column(sa.Column('missingness_status', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('sources_inspected', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('reason', sa.Text(), nullable=True))

    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.add_column(sa.Column('authors', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('journal', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('publication_year', sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column('fulltext_available', sa.Boolean(), nullable=False, server_default=sa.false())
        )

    conn = op.get_bind()

    missingness = sa.table(
        'missingness',
        sa.column('study_id', sa.String), sa.column('entity_id', sa.String),
        sa.column('target_schema', sa.String), sa.column('target_field', sa.String),
        sa.column('missingness_status', sa.String), sa.column('sources_inspected', sa.JSON),
        sa.column('reason', sa.Text),
        sa.column('created_at', sa.DateTime(timezone=True)), sa.column('updated_at', sa.DateTime(timezone=True)),
    )
    standardized_values = sa.table(
        'standardized_values',
        sa.column('standardized_value_id', sa.String), sa.column('study_id', sa.String),
        sa.column('entity_id', sa.String), sa.column('target_schema', sa.String),
        sa.column('target_schema_version', sa.String), sa.column('target_field', sa.String),
        sa.column('standardized_value', sa.Text), sa.column('mapping_method', sa.String),
        sa.column('validation_status', sa.String), sa.column('review_required', sa.Boolean),
        sa.column('missingness_status', sa.String), sa.column('sources_inspected', sa.JSON),
        sa.column('reason', sa.Text),
        sa.column('created_at', sa.DateTime(timezone=True)), sa.column('updated_at', sa.DateTime(timezone=True)),
    )

    for row in conn.execute(sa.select(missingness)).fetchall():
        entity_clause = (
            standardized_values.c.entity_id == row.entity_id
            if row.entity_id is not None else standardized_values.c.entity_id.is_(None)
        )
        existing = conn.execute(
            sa.select(standardized_values.c.standardized_value_id).where(
                standardized_values.c.study_id == row.study_id,
                standardized_values.c.target_schema == row.target_schema,
                standardized_values.c.target_field == row.target_field,
                entity_clause,
            )
        ).first()
        if existing is not None:
            conn.execute(
                standardized_values.update()
                .where(standardized_values.c.standardized_value_id == existing.standardized_value_id)
                .values(
                    missingness_status=row.missingness_status,
                    sources_inspected=row.sources_inspected,
                    reason=row.reason,
                )
            )
        else:
            conn.execute(
                standardized_values.insert().values(
                    standardized_value_id=new_id('STDVAL'),
                    study_id=row.study_id,
                    entity_id=row.entity_id,
                    target_schema=row.target_schema,
                    target_schema_version='1.0.2' if row.target_schema == 'faire' else '',
                    target_field=row.target_field,
                    standardized_value=None,
                    mapping_method='unresolved',
                    validation_status='not_assessed',
                    review_required=False,
                    missingness_status=row.missingness_status,
                    sources_inspected=row.sources_inspected,
                    reason=row.reason,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )

    publications = sa.table(
        'publications',
        sa.column('study_id', sa.String), sa.column('authors', sa.JSON),
        sa.column('journal', sa.String), sa.column('publication_year', sa.Integer),
        sa.column('open_access_status', sa.String), sa.column('fulltext_available', sa.Boolean),
    )
    sources = sa.table(
        'sources',
        sa.column('source_id', sa.String), sa.column('study_id', sa.String),
        sa.column('source_type', sa.String), sa.column('source_name', sa.String),
        sa.column('access_status', sa.String), sa.column('authors', sa.JSON),
        sa.column('journal', sa.String), sa.column('publication_year', sa.Integer),
        sa.column('fulltext_available', sa.Boolean),
    )

    for pub in conn.execute(sa.select(publications)).fetchall():
        if pub.study_id is None:
            continue
        candidates = conn.execute(
            sa.select(sources).where(
                sources.c.study_id == pub.study_id, sources.c.source_type == 'publication_api'
            )
        ).fetchall()
        if not candidates:
            continue
        by_name = {c.source_name: c for c in candidates}
        target = next((by_name[name] for name in _PUBLICATION_ADAPTER_PRIORITY if name in by_name), candidates[0])

        new_access_status = target.access_status
        if pub.open_access_status == 'open':
            new_access_status = 'open'

        conn.execute(
            sources.update().where(sources.c.source_id == target.source_id).values(
                authors=pub.authors,
                journal=pub.journal,
                publication_year=pub.publication_year,
                fulltext_available=bool(pub.fulltext_available),
                access_status=new_access_status,
            )
        )

    op.drop_table('missingness')
    op.drop_table('publications')


def downgrade() -> None:
    """Structural reversal only -- NOT a lossless data restore. Once
    missingness rows have been merged into pre-existing standardized_values
    rows (upgrade()'s "existing is not None" branch), and once
    publications' per-study merged view has been distributed across
    per-adapter source rows, neither can be cleanly un-merged. This
    recreates the empty table shapes and drops the new columns so the
    schema itself is reversible; the data is not.
    """
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.drop_column('fulltext_available')
        batch_op.drop_column('publication_year')
        batch_op.drop_column('journal')
        batch_op.drop_column('authors')

    with op.batch_alter_table('standardized_values', schema=None) as batch_op:
        batch_op.drop_column('reason')
        batch_op.drop_column('sources_inspected')
        batch_op.drop_column('missingness_status')

    op.create_table(
        'publications',
        sa.Column('publication_id', sa.String(), nullable=False),
        sa.Column('study_id', sa.String(), nullable=True),
        sa.Column('doi', sa.String(), nullable=True),
        sa.Column('pmid', sa.String(), nullable=True),
        sa.Column('pmcid', sa.String(), nullable=True),
        sa.Column('openalex_id', sa.String(), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('authors', sa.JSON(), nullable=True),
        sa.Column('publication_year', sa.Integer(), nullable=True),
        sa.Column('journal', sa.String(), nullable=True),
        sa.Column('open_access_status', sa.String(), nullable=False),
        sa.Column('fulltext_available', sa.Boolean(), nullable=False),
        sa.Column('relationship_to_study', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['study_id'], ['studies.study_id']),
        sa.PrimaryKeyConstraint('publication_id'),
    )
    op.create_index(op.f('ix_publications_doi'), 'publications', ['doi'], unique=False)
    op.create_index(op.f('ix_publications_openalex_id'), 'publications', ['openalex_id'], unique=False)
    op.create_index(op.f('ix_publications_pmcid'), 'publications', ['pmcid'], unique=False)
    op.create_index(op.f('ix_publications_pmid'), 'publications', ['pmid'], unique=False)
    op.create_index(op.f('ix_publications_study_id'), 'publications', ['study_id'], unique=False)

    op.create_table(
        'missingness',
        sa.Column('missingness_id', sa.String(), nullable=False),
        sa.Column('study_id', sa.String(), nullable=False),
        sa.Column('entity_id', sa.String(), nullable=True),
        sa.Column('target_schema', sa.String(), nullable=False),
        sa.Column('target_field', sa.String(), nullable=False),
        sa.Column('missingness_status', sa.String(), nullable=False),
        sa.Column('sources_inspected', sa.JSON(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('review_status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['entity_id'], ['entities.entity_id']),
        sa.ForeignKeyConstraint(['study_id'], ['studies.study_id']),
        sa.PrimaryKeyConstraint('missingness_id'),
    )
    op.create_index(op.f('ix_missingness_entity_id'), 'missingness', ['entity_id'], unique=False)
    op.create_index(op.f('ix_missingness_study_id'), 'missingness', ['study_id'], unique=False)
