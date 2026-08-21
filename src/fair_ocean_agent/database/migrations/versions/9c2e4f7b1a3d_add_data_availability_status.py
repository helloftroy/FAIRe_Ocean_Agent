"""add data availability status

Revision ID: 9c2e4f7b1a3d
Revises: 415058e96f35
Create Date: 2026-08-21

New `studies.data_availability_status` column (DataAvailabilityStatus enum:
unknown/accessible/not_accessible). Set once, at the end of
handle_discover_identifiers, after the staged repository search
(discovery/text_identifiers.py's Pass 1: BioProject/SRA/ENA accessions;
Pass 2: Zenodo/Dryad/Figshare/OSF; Pass 3: DataCite relatedIdentifiers on
the article's own DOI) has run. not_accessible is the "give up" signal
enqueue_seed_backfill/enqueue_full_rediscovery/
enqueue_citation_rediscovery_backfill now check before re-queueing a study
for more discovery work -- per an explicit user request to stop retrying
papers where nothing accessible was ever found, except when the study's
discovery_trigger is "primer_reference_citation" (its own goal was a primer
sequence or the next reference in the chain, not full sample data).

Not reusing the existing-but-dead CanonicalStatus.REJECTED (confirmed via
scheduling/rediscovery.py's own comment and a full-codebase grep that it's
never actually set anywhere) -- its meaning ("this isn't a real distinct
study") is a different question from "this study has no accessible
sequence data," and this codebase is consistently careful about not
overloading field meanings. Mirrors marine_relevance_status/
molecular_relevance_status's existing `*_status` string-enum shape on
Study exactly, including their same "unknown" default -- no backfill
needed, every existing row correctly starts unscanned.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c2e4f7b1a3d"
down_revision: Union[str, Sequence[str], None] = "415058e96f35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "studies",
        sa.Column("data_availability_status", sa.String(), nullable=False, server_default="unknown"),
    )


def downgrade() -> None:
    op.drop_column("studies", "data_availability_status")
