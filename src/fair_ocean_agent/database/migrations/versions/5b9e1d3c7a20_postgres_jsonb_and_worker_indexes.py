"""postgres jsonb and worker indexes

Revision ID: 5b9e1d3c7a20
Revises: 0fc0bf2bf46a
Create Date: 2026-07-28 19:10:00

PostgreSQL hardening:

- convert document-shaped JSON columns to JSONB on PostgreSQL only;
- add composite indexes for multi-worker task claiming and common
  study/schema/fact lookups.

SQLite keeps its generic JSON affinity and receives only the compatible
index additions.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "5b9e1d3c7a20"
down_revision: Union[str, Sequence[str], None] = "0fc0bf2bf46a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_JSONB_COLUMNS = (
    ("candidate_matches", "supporting_features"),
    ("raw_facts", "confidence_metadata"),
    ("sources", "authors"),
    ("standardized_values", "sources_inspected"),
    ("tasks", "payload"),
    ("validation_results", "compared_values"),
    ("workflow_runs", "model_config_snapshot"),
    ("workflow_runs", "schema_versions"),
    ("workflow_runs", "sources_queried"),
)


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if _is_postgresql():
        for table_name, column_name in _JSONB_COLUMNS:
            op.alter_column(
                table_name,
                column_name,
                type_=postgresql.JSONB(),
                postgresql_using=f"{column_name}::jsonb",
                existing_type=sa.JSON(),
                existing_nullable=True,
            )

    op.create_index(
        "ix_tasks_claimable",
        "tasks",
        ["status", "available_after", "priority", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_tasks_type_status_available",
        "tasks",
        ["task_type", "status", "available_after"],
        unique=False,
    )
    op.create_index(
        "ix_raw_facts_study_fact_type_entity",
        "raw_facts",
        ["study_id", "fact_type_candidate", "entity_id"],
        unique=False,
    )
    op.create_index(
        "ix_standardized_values_schema_study_field",
        "standardized_values",
        ["target_schema", "study_id", "target_field"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_standardized_values_schema_study_field", table_name="standardized_values")
    op.drop_index("ix_raw_facts_study_fact_type_entity", table_name="raw_facts")
    op.drop_index("ix_tasks_type_status_available", table_name="tasks")
    op.drop_index("ix_tasks_claimable", table_name="tasks")

    if _is_postgresql():
        for table_name, column_name in _JSONB_COLUMNS:
            op.alter_column(
                table_name,
                column_name,
                type_=sa.JSON(),
                postgresql_using=f"{column_name}::json",
                existing_type=postgresql.JSONB(),
                existing_nullable=True,
            )
