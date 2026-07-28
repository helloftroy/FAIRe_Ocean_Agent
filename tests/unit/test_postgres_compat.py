from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql, sqlite

from fair_ocean_agent.database.models import RawFact, StandardizedValue, Task


def test_json_document_columns_compile_as_jsonb_for_postgres():
    task_sql = str(CreateTable(Task.__table__).compile(dialect=postgresql.dialect()))
    raw_fact_sql = str(CreateTable(RawFact.__table__).compile(dialect=postgresql.dialect()))
    standardized_sql = str(CreateTable(StandardizedValue.__table__).compile(dialect=postgresql.dialect()))

    assert "payload JSONB" in task_sql
    assert "confidence_metadata JSONB" in raw_fact_sql
    assert "sources_inspected JSONB" in standardized_sql


def test_json_document_columns_stay_sqlite_compatible():
    task_sql = str(CreateTable(Task.__table__).compile(dialect=sqlite.dialect()))

    assert "payload JSON" in task_sql
    assert "JSONB" not in task_sql
