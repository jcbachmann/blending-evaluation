import itertools
import math

import numpy as np
import pytest
from scipy import integrate

from ..chevron import front_volume, ideal_radius, schedule, stops_path
from ..scenarios import DURATION, PATTERNS, X_MAX, X_MIN, get_layer_end_times, get_pattern_volume, make_material_deposition

VOLUME = get_pattern_volume("chevron-5")
CORE = X_MAX - X_MIN
RADIUS = ideal_radius(VOLUME, CORE)


def front_volume_numeric(r, s):
    def height(z, x):
        rho = math.hypot(x, z)
        return max(0.0, (r - rho) - max(s - abs(z), 0.0)) if rho < r else 0.0

    # Split at the ridge strip edges, where the integrand has kinks
    edges = sorted({-r, -s, s, r})
    return sum(integrate.dblquad(height, 0.0, r, a, b, epsabs=1e-9)[0] for a, b in itertools.pairwise(edges))


@pytest.mark.parametrize(("r", "s"), [(10.0, 0.0), (10.0, 5.0), (4.738, 2.185), (3.0, 2.9), (1.0, 0.3)])
def test_front_volume(r, s):
    assert front_volume(r, s) == pytest.approx(front_volume_numeric(r, s), rel=1e-5, abs=1e-9)


def test_front_volume_limits():
    assert front_volume(7.0, 0.0) == pytest.approx(math.pi / 6.0 * 7.0**3)
    assert front_volume(7.0, 7.0) == 0.0
    assert front_volume(7.0, 6.999999) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("shrinking", [False, True])
@pytest.mark.parametrize("layers", [1, 5, 25, 500])
def test_schedule_volumes(layers, shrinking):
    layer_list = schedule(VOLUME, X_MIN, X_MAX, layers, shrinking)
    stops = sum(layer["start"] + layer["end"] for layer in layer_list)
    travel = sum(layer["per_meter"] * (layer["high"] - layer["low"]) for layer in layer_list)
    assert stops + travel == pytest.approx(VOLUME)
    assert all(layer["end"] >= -1e-9 for layer in layer_list)
    # The last layer completes the final stockpile on the final core
    assert layer_list[-1]["radius"] == pytest.approx(RADIUS)
    assert layer_list[-1]["low"] == pytest.approx(X_MIN)
    assert layer_list[-1]["high"] == pytest.approx(X_MAX)
    if not shrinking:
        # With fixed turning points the stops build both final end cones, the travel builds the prism
        assert stops == pytest.approx(math.pi / 3.0 * RADIUS**3)


@pytest.mark.parametrize("layers", [5, 25])
def test_shrinking_keeps_toe(layers):
    for layer in schedule(VOLUME, X_MIN, X_MAX, layers, shrinking=True):
        assert layer["low"] - layer["radius"] == pytest.approx(X_MIN - RADIUS)
        assert layer["high"] + layer["radius"] == pytest.approx(X_MAX + RADIUS)


@pytest.mark.parametrize("shrinking", [False, True])
def test_single_layer_is_start_cone(shrinking):
    (layer,) = schedule(VOLUME, X_MIN, X_MAX, 1, shrinking)
    assert layer["start"] == pytest.approx(math.pi / 3.0 * RADIUS**3)
    assert layer["end"] == pytest.approx(0.0, abs=1e-9)


def test_schedule_angle_scales_volumes():
    flat = schedule(VOLUME, X_MIN, X_MAX, 5, tan_theta=1.0)
    steep = schedule(VOLUME * 2.0, X_MIN, X_MAX, 5, tan_theta=2.0)
    # Doubling tan(theta) and the volume keeps the radii and doubles every volume
    assert [layer["radius"] for layer in steep] == pytest.approx([layer["radius"] for layer in flat])
    assert [layer["start"] for layer in steep] == pytest.approx([2.0 * layer["start"] for layer in flat])


@pytest.mark.parametrize("shrinking", [False, True])
def test_stops_path(shrinking):
    timestamps, x, layer_ends = stops_path(VOLUME, DURATION, X_MIN, X_MAX, 5, shrinking)
    assert np.all(np.diff(timestamps) >= 0.0)
    assert timestamps[-1] == pytest.approx(DURATION)
    assert len(layer_ends) == 5
    assert layer_ends[-1] == pytest.approx(DURATION)
    assert x.min() >= X_MIN - RADIUS - 1e-9
    assert x.max() <= X_MAX + RADIUS + 1e-9


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_patterns(pattern):
    data = make_material_deposition(pattern).data
    assert data["volume"].sum() == pytest.approx(get_pattern_volume(pattern))
    if pattern != "cone":
        assert len(get_layer_end_times(pattern)) == int(pattern.split("-")[1])
