import pytest

from fair_ocean_agent.exports.bebop import export_bebop
from fair_ocean_agent.mapping.bebop import BebopMappingNotApplicable, map_study_to_bebop


def test_map_study_to_bebop_raises_clear_not_applicable_error(db_session):
    with pytest.raises(BebopMappingNotApplicable, match="out of scope"):
        map_study_to_bebop(db_session, "STUDY-1")


def test_export_bebop_raises_clear_not_applicable_error(db_session, tmp_path):
    with pytest.raises(BebopMappingNotApplicable, match="out of scope"):
        export_bebop(db_session, tmp_path)
