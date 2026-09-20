import math

import pytest

from bmh_ml.metrics import get_accuracy


def test_a_perfect_prediction():
    assert get_accuracy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == {"r2": 1.0, "mean_absolute_error": 0.0}


def test_predicting_the_mean_has_no_explanatory_power():
    accuracy = get_accuracy([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])

    assert accuracy["r2"] == pytest.approx(0.0)
    assert accuracy["mean_absolute_error"] == pytest.approx(2.0 / 3.0)


def test_a_bad_prediction_can_be_worse_than_the_mean():
    assert get_accuracy([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])["r2"] == pytest.approx(-3.0)


def test_constant_expected_values_have_no_r2():
    assert math.isnan(get_accuracy([2.0, 2.0], [1.0, 3.0])["r2"])
