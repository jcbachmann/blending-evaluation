#!/usr/bin/env python
import unittest

import numpy as np

from ..stockpile_math import get_stockpile_height, get_stockpile_volume, get_stockpile_slice_volume


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
    TEST_SET = [
        {'height': 12.3, 'length': 321.0, 'volume': 50512.78536550256, 'delta': 1.0},
        {'height': 111.1, 'length': 3.2, 'volume': 1475552.3506640848, 'delta': 30.0},
        {'height': 12.3, 'length': 131231.0, 'volume': 19855886.685365506, 'delta': 400.0},
        {'height': 1231234.3, 'length': 1.0, 'volume': 1.9545692940755512e+18, 'delta': 2.0e+13},
        {'height': 12.3, 'length': 0.0, 'volume': 1948.6953655025595, 'delta': 0.04},
        {'height': 0.0, 'length': 321.0, 'volume': 0.0, 'delta': 0.0},
    ]

    def test_get_stockpile_volume(self):
        for entry in TestStockpileMath.TEST_SET:
            self.assertAlmostEqual(
                get_stockpile_volume(
                    height=entry['height'],
                    core_length=entry['length']
                ),
                entry['volume']
            )

    def test_get_stockpile_height(self):
        for entry in TestStockpileMath.TEST_SET:
            self.assertAlmostEqual(
                get_stockpile_height(
                    volume=entry['volume'],
                    core_length=entry['length']
                ),
                entry['height']
            )

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
            self.assertAlmostEqual(
                get_stockpile_volume_from_slices(
                    core_length=entry['length'],
                    height=entry['height']
                ),
                entry['volume'],
                delta=entry['delta']
            )
