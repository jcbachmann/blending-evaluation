import numpy as np
import pytest
from bmh.helpers.chevron import stops_path

from bmh_ml.encoding import StopEncoding, equal_time_paths, get_stop_limits
from bmh_ml.problems import SimulatorProblem, StopSimulatorProblem
from bmh_ml.settings import TOTAL_VOLUME, X_MAX, X_MIN, SimulatorSettings, simulator_settings
from bmh_ml.tests.test_simulation import skip_unless_built_with

DAY = 86400.0


def test_the_stop_limits_leave_one_portion_cone_at_each_end():
    low, high = get_stop_limits(59.0, 2500.0, 50)
    assert low == pytest.approx((3 * 50 / np.pi) ** (1 / 3))
    assert high == pytest.approx(59.0 - low)


def test_without_stops_the_path_is_that_of_the_classic_variables():
    rng = np.random.default_rng(1)
    deposition = rng.uniform(X_MIN, X_MAX, (2, 20))
    encoding = StopEncoding()
    paths, fractions = encoding.decode(np.hstack([deposition, np.zeros((2, 20))]))
    for (times, x), (classic_times, classic_x) in zip(paths, equal_time_paths(deposition), strict=True):
        np.testing.assert_allclose(times, classic_times)
        np.testing.assert_allclose(x, classic_x)
    np.testing.assert_array_equal(fractions, 1.0)


def test_stops_add_knots_and_take_their_share_of_the_time():
    encoding = StopEncoding(knots=3)
    ((times, x),) = encoding.decode([[10.0, 40.0, 20.0, 1.0, 0.0, 2.0]])[0]
    # stop 1, travel 1, travel 1, stop 2: 5 units
    np.testing.assert_allclose(times, np.array([0, 1, 2, 3, 5]) * DAY / 5)
    np.testing.assert_allclose(x, [10, 10, 40, 20, 20])


def test_travel_times_and_the_stop_fraction():
    encoding = StopEncoding(knots=3, travel=True, stop_fraction=(0.5, 1.0))
    assert encoding.n_var == 3 + 3 + 2 + 1
    low, high = encoding.bounds
    assert low[-1] == 0.5
    assert high[-1] == 1.0
    ((times, x),) = encoding.decode([[10.0, 40.0, 20.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.8]])[0]
    np.testing.assert_allclose(times, np.array([0, 1, 1.5]) / 1.5 * 0.8 * DAY)
    np.testing.assert_allclose(x, [10, 40, 20])


@pytest.mark.parametrize("shrinking", [False, True])
def test_a_chevron_with_stops_encodes_with_its_turning_points_and_stops(shrinking):
    times, x, _ = stops_path(TOTAL_VOLUME, DAY, X_MIN, X_MAX, 19, shrinking=shrinking)
    encoding = StopEncoding(travel=True)
    variables = encoding.encode(times, x)
    low, high = encoding.bounds
    assert np.all(variables >= low)
    assert np.all(variables <= high)
    ((decoded_times, decoded_x),) = encoding.decode(variables)[0]
    # the same time at every turning point; with shrinking limits only the short moves between two turning points are merged
    turning = np.array([t for t, a, b in zip(times[1:], x[1:], x[:-1], strict=True) if a != b])
    tolerance = 2e-3 * DAY if shrinking else 1e-6 * DAY
    for t in turning:
        assert np.min(np.abs(decoded_times - t)) < tolerance
    assert decoded_x.min() == pytest.approx(x.min(), abs=1e-9 if not shrinking else 1.0)


def test_a_constant_speed_chevron_encodes_without_stops():
    deposition = np.where(np.arange(20) % 2 == 0, X_MIN, X_MAX).astype(float)
    variables = StopEncoding().encode(np.linspace(0, DAY, 20), deposition)
    np.testing.assert_allclose(variables[:20], deposition)
    np.testing.assert_allclose(variables[20:], 0.0)


def test_too_many_turning_points_do_not_encode():
    with pytest.raises(ValueError, match="turning points"):
        StopEncoding(knots=3).encode(np.linspace(0, DAY, 5), [10.0, 40.0, 10.0, 40.0, 10.0])


@pytest.mark.parametrize("simulator", ["cone", "formula"])
def test_the_stop_problem_without_stops_is_the_classic_problem(simulator):
    skip_unless_built_with(simulator)
    rng = np.random.default_rng(4)
    material = rng.uniform(5, 10, 50)
    deposition = rng.uniform(X_MIN, X_MAX, (3, 20))
    with simulator_settings(SimulatorSettings(simulator=simulator)):
        classic = SimulatorProblem(material).evaluate(deposition)
        stops = StopSimulatorProblem(material).evaluate(np.hstack([deposition, np.zeros((3, 20))]))
    np.testing.assert_allclose(stops, classic, rtol=1e-12)


@pytest.mark.parametrize("simulator", ["cone", "formula"])
def test_a_stop_fraction_stacks_part_of_the_material(simulator):
    skip_unless_built_with(simulator)
    from bmh_ml.simulation import reclaim_paths

    encoding = StopEncoding(stop_fraction=(0.5, 1.0))
    rng = np.random.default_rng(5)
    variables = np.hstack([rng.uniform(X_MIN, X_MAX, (1, 20)), np.zeros((1, 20)), [[0.6]]])
    paths, _ = encoding.decode(variables)
    with simulator_settings(SimulatorSettings(simulator=simulator)):
        reclaimed = reclaim_paths(rng.uniform(5, 10, 50), paths, 59.0, 20.0, 2500.0)
    assert reclaimed["volume"].sum() == pytest.approx(0.6 * 2500.0)
