"""Re-mapping a study with more FAIRe values than SQLite allows bound
parameters in one statement (found live: ~323,000 values in one study
crashed merge_extraction_shards.py's re-mapping step)."""
from sqlalchemy import func, insert, select

from fair_ocean_agent.database.models import StandardizedValue, Study
from fair_ocean_agent.mapping.faire import TARGET_SCHEMA, TARGET_SCHEMA_VERSION, _clear_existing_faire_mappings

# Over SQLite's default 32,766-parameter limit.
VALUE_COUNT = 40_000


def _values(study_id: str, count: int, prefix: str) -> list[dict]:
    return [
        {
            "standardized_value_id": f"{prefix}-{i}",
            "study_id": study_id,
            "target_schema": TARGET_SCHEMA,
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "target_field": "samp_name",
            "standardized_value": str(i),
        }
        for i in range(count)
    ]


def test_clear_existing_faire_mappings_handles_more_values_than_sqlite_parameters(db_session):
    db_session.add_all([Study(study_id="STUDY-big"), Study(study_id="STUDY-other")])
    db_session.flush()
    db_session.execute(insert(StandardizedValue), _values("STUDY-big", VALUE_COUNT, "STDVAL-big"))
    db_session.execute(insert(StandardizedValue), _values("STUDY-other", 3, "STDVAL-other"))
    db_session.flush()

    _clear_existing_faire_mappings(db_session, "STUDY-big")

    remaining = dict(
        db_session.execute(
            select(StandardizedValue.study_id, func.count()).group_by(StandardizedValue.study_id)
        ).all()
    )
    assert remaining == {"STUDY-other": 3}
