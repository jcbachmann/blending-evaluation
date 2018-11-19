#!/usr/bin/env python
import unittest

from bmh_apps.helpers.stockpile_math import get_stockpile_height, get_stockpile_volume


class TestStockpileMath(unittest.TestCase):
    TEST_SET = [
        {'height': 12.3, 'length': 321.0, 'volume': 50512.78536550256},
        {'height': 111.1, 'length': 3.2, 'volume': 1475552.3506640848},
        {'height': 12.3, 'length': 131231.0, 'volume': 19855886.685365506},
        {'height': 1231234.3, 'length': 1.0, 'volume': 1.9545692940755512e+18},
        {'height': 12.3, 'length': 0.0, 'volume': 1948.6953655025595},
        {'height': 0.0, 'length': 321.0, 'volume': 0.0},
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
