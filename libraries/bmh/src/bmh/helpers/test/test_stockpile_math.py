#!/usr/bin/env python
import unittest
from typing import ClassVar

import numpy as np
import pytest

from ..stockpile_math import get_stockpile_height, get_stockpile_slice_volume, get_stockpile_volume


def get_stockpile_volume_from_slices(core_length: float, height: float):
    bed_size_x = core_length + 2.0 * height
    margins = 10.0
    slices = 10000
    x_min = height
    x_diff = (bed_size_x + 2 * margins) / (slices - 1)  # -1 required, linspace includes start and end!
    xs = np.linspace(0.0 - margins, bed_size_x + margins, slices)
    volumes = [get_stockpile_slice_volume(x, core_length, height, x_min, x_diff) for x in xs]
    return sum(volumes)


class TestStockpileMath(unittest.TestCase):
    TEST_SET: ClassVar[list[dict[str, float]]] = [
        {"height": 12.3, "length": 321.0, "volume": 50512.78536550256, "delta": 1.0},
        {"height": 111.1, "length": 3.2, "volume": 1475552.3506640848, "delta": 30.0},
        {"height": 12.3, "length": 131231.0, "volume": 19855886.685365506, "delta": 400.0},
        {"height": 1231234.3, "length": 1.0, "volume": 1.9545692940755512e18, "delta": 2.0e13},
        {"height": 12.3, "length": 0.0, "volume": 1948.6953655025595, "delta": 0.04},
        {"height": 0.0, "length": 321.0, "volume": 0.0, "delta": 0.0},
    ]

    def test_get_stockpile_volume(self):
        for entry in TestStockpileMath.TEST_SET:
            assert get_stockpile_volume(
                height=entry["height"],
                core_length=entry["length"],
            ) == pytest.approx(entry["volume"])

    def test_get_stockpile_height(self):
        for entry in TestStockpileMath.TEST_SET:
            assert get_stockpile_height(
                volume=entry["volume"],
                core_length=entry["length"],
            ) == pytest.approx(entry["height"])

    def test_array_get_stockpile_volume(self):
        height = np.array([12.3, 111.1])
        length = np.array([321.0, 3.2])
        volume = np.array([50512.78536550256, 1475552.3506640848])
        np.testing.assert_allclose(get_stockpile_volume(height, length), volume)

    def test_array_get_stockpile_height(self):
        height = np.array([12.3, 111.1])
        length = np.array([321.0, 3.2])
        volume = np.array([50512.78536550256, 1475552.3506640848])
        np.testing.assert_allclose(get_stockpile_height(volume, length), height)

    def test_get_stockpile_slice_volume(self):
        for entry in TestStockpileMath.TEST_SET:
            assert get_stockpile_volume_from_slices(
                core_length=entry["length"],
                height=entry["height"],
            ) == pytest.approx(entry["volume"], abs=entry["delta"])


def test_integrated_ideal_volumes_sum_to_the_volume_and_match_a_numerical_integral():
    from bmh.helpers.stockpile_math import get_ideal_reclaimed_volume, get_ideal_stockpile_volumes, get_stockpile_height

    x = np.arange(0.0, 60.0)
    integrated = get_ideal_stockpile_volumes(x, 2500.0, 10.0, 49.0)
    point = get_ideal_stockpile_volumes(x, 2500.0, 10.0, 49.0, rule="point")
    assert integrated.sum() == pytest.approx(2500.0, rel=1e-12)
    assert point.sum() == pytest.approx(2503.48, abs=0.01)  # the point rule overshoots
    assert integrated[30] == pytest.approx(point[30])  # inside the core both rules agree

    # the volume reclaimed up to cut position p, from the stockpile's height function (cut plane x = p + y at 45 degrees)
    core, r, d = 39.0, get_stockpile_height(2500.0, 39.0), 0.05
    gx, gz = np.meshgrid(np.arange(-r, core + r, d) + d / 2, np.arange(-r, r, d) + d / 2, indexing="ij")
    height = np.maximum(np.where(gx < 0, r - np.hypot(gx, gz), np.where(gx > core, r - np.hypot(gx - core, gz), r - np.abs(gz))), 0.0)
    for p in (-5.0, 0.0, 20.0, core, core + 4.0):
        numeric = np.clip(height - np.maximum(gx - p, 0.0), 0.0, None).sum() * d * d
        assert get_ideal_reclaimed_volume(p, core, r) == pytest.approx(numeric, rel=2e-3)
