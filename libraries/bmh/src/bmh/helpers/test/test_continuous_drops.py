import blending_simulator_lib
import numpy as np
import pytest

from .. import continuous_drops as module
from ..continuous_drops import continuous_drops, drop_grid


@pytest.fixture(params=["numpy", "native"], autouse=True)
def implementation(request, monkeypatch):
    """Every test runs with the numpy code and with the C++ code of the bindings (where they have it)."""
    if request.param == "native" and not hasattr(blending_simulator_lib, "continuous_drops"):
        pytest.skip("blending_simulator_lib built without continuous_drops")
    monkeypatch.setattr(module, "_NUMPY_ONLY", request.param == "numpy")
    return request.param


def dwell(volumes, portion_times, path_times, path_x, cell, samples=400_000):
    """Volume per x cell from a fine sampling of the flow: the stacker position at evenly spaced times, weighted by the flow rate."""
    t = portion_times[0] + (np.arange(samples) + 0.5) / samples * (portion_times[-1] - portion_times[0])
    portion = np.clip(np.searchsorted(portion_times, t, side="right") - 1, 0, len(volumes) - 1)
    rate = volumes / np.diff(portion_times)
    weight = rate[portion] * (portion_times[-1] - portion_times[0]) / samples
    return np.bincount(np.floor(np.interp(t, path_times, path_x) / cell).astype(int), weight, minlength=100)


def test_every_cell_gets_the_volume_of_the_time_the_stacker_spends_over_it():
    volumes = np.array([30.0, 0.0, 50.0, 20.0])  # the second portion is a gap without flow
    portion_times = np.array([0.0, 1.0, 1.5, 3.0, 4.0])
    path_times = np.array([0.0, 1.3, 2.0, 4.0])
    path_x = np.array([3.3, 17.8, 17.8, 1.2])  # out, a stop, and back past the start
    rows = continuous_drops(volumes, np.arange(4.0), portion_times, path_times, path_x, None, 0.7, 0.7, z=5.0)
    np.testing.assert_allclose(rows[:, 2].sum(), 100.0)
    got = np.bincount(np.floor(rows[:, 0] / 0.7).astype(int), rows[:, 2], minlength=100)
    np.testing.assert_allclose(got, dwell(volumes, portion_times, path_times, path_x, 0.7), atol=0.01)
    assert np.all(rows[:, 1] == 5.0)
    assert set(rows[:, 3]) == {0.0, 2.0, 3.0}  # nothing of the gap


def test_partial_cells_at_the_ends_of_a_pass_get_their_overlap():
    # 1 m cells, a pass from 2.5 to 5.5 m at constant speed: half cells at both ends
    rows = continuous_drops(np.array([3.0]), np.array([7.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([2.5, 5.5]), None, 1.0, 1.0, z=1.0)
    np.testing.assert_allclose(rows[:, 2], [0.5, 1.0, 1.0, 0.5])
    np.testing.assert_allclose(rows[:, 0], [2.75, 3.5, 4.5, 5.25])
    np.testing.assert_allclose(rows[:, 3], 7.0)


def test_a_standing_stacker_drops_each_portion_once_at_its_position():
    rows = continuous_drops(
        np.array([10.0, 20.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0]),
        np.array([10.0, 10.0]),
        None,
        1.0,
        1.0,
        z=10.0,
    )
    np.testing.assert_array_equal(rows, [[10.0, 10.0, 10.0, 1.0, 2.0], [10.0, 10.0, 20.0, 3.0, 4.0]])


def test_lateral_motion_is_cut_at_the_cells_in_z():
    rows = continuous_drops(
        np.array([4.0]), np.array([1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([5.2, 5.2]), np.array([1.5, 3.5]), 1.0, 0.5, z=None
    )
    np.testing.assert_allclose(rows[:, 2], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(rows[:, 1], [1.75, 2.25, 2.75, 3.25])


@pytest.mark.parametrize("simulator", ["fast", "cone", "formula", "lattice"])
def test_the_drop_grid_of_each_simulator(simulator):
    cx, cz = drop_grid(simulator, 8.0, 59.0, 20.0)
    if simulator == "fast":
        assert cx == cz == pytest.approx(0.5)
    elif simulator == "cone":
        assert (cx, cz) == (pytest.approx(59.0 / 118), pytest.approx(0.5))
    elif simulator == "formula":
        assert (cx, cz) == (pytest.approx(59.0 / 118), 20.0)
    else:
        assert cz == pytest.approx(cx * np.sqrt(3) / 2)
    with pytest.raises(ValueError, match="unknown simulator"):
        drop_grid("bullet", 1.0, 59.0, 20.0)


def test_the_native_code_gives_the_drops_of_the_numpy_code(implementation, monkeypatch):
    if implementation != "native":
        pytest.skip("compares the two implementations once")
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(1, 60))
        volumes = rng.uniform(0, 10, n) * (rng.random(n) > 0.1)
        bounds = np.concatenate([[0.0], np.cumsum(rng.uniform(0.1, 2.0, n))])
        knots = np.sort(rng.uniform(bounds[0], bounds[-1], int(rng.integers(2, 30))))
        path_x, path_z = rng.uniform(0, 50, len(knots)), rng.uniform(5, 15, len(knots))
        parameters = rng.uniform(0, 1, (n, 2))
        native = continuous_drops(volumes, parameters, bounds, knots, path_x, path_z, 0.63, 0.55)
        monkeypatch.setattr(module, "_NUMPY_ONLY", True)
        reference = continuous_drops(volumes, parameters, bounds, knots, path_x, path_z, 0.63, 0.55)
        monkeypatch.setattr(module, "_NUMPY_ONLY", False)
        assert native.shape == reference.shape
        np.testing.assert_allclose(native, reference, rtol=1e-12, atol=1e-12)


def test_evenly_spaced_portions_flow_in_equal_intervals():
    from ..continuous_drops import portion_bounds

    np.testing.assert_allclose(portion_bounds(np.linspace(0.0, 10.0, 6)), np.linspace(0.0, 10.0, 7))
    bounds = portion_bounds(np.array([0.0, 1.0, 3.0, 4.0]))
    assert bounds[0] == 0.0
    assert bounds[-1] == 4.0
    assert np.all(np.diff(bounds) > 0)


def test_material_deposition_drops_keep_the_volume_and_follow_the_path():
    from ..continuous_drops import material_deposition_drops

    rows = material_deposition_drops(
        np.linspace(0.0, 1.0, 5), np.full(5, 20.0), np.arange(5.0), np.array([0.0, 1.0]), np.array([5.0, 45.0]), np.full(2, 10.0), 1.0, 1.0, with_times=True
    )
    np.testing.assert_allclose(rows[:, 3].sum(), 100.0)
    assert np.all(np.diff(rows[:, 0]) > 0)  # times increase along the path
    np.testing.assert_allclose(rows[:, 1], 5.0 + 40.0 * rows[:, 0], atol=0.5)  # a drop lies where the stacker is at its time
