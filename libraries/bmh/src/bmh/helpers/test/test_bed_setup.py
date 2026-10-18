import json

from ..bed_setup import BED_SIZE_X, BED_SIZE_Z, STACKING_TIME, TOTAL_VOLUME, X_MAX, X_MIN, get_core_limits


def test_the_core_ends_half_the_bed_depth_from_both_ends_of_the_bed():
    assert get_core_limits(60.0, 20.0) == (10.0, 50.0)
    assert get_core_limits(59, 20) == (10.0, 49.0)
    assert get_core_limits(BED_SIZE_X, BED_SIZE_Z) == (X_MIN, X_MAX)


def test_the_setup_keeps_the_values_recorded_with_existing_results():
    # recorded in dataset settings and hashed into cache keys as JSON, so integers must stay integers
    assert json.dumps([BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, X_MIN, X_MAX]) == "[59, 20, 2500, 10.0, 49.0]"
    assert STACKING_TIME == 86400
