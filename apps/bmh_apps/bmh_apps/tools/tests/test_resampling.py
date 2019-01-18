#!/usr/bin/env python
import unittest

import pandas as pd
from bmh.benchmark.material_deposition import Material

from ..resampling import resample


class TestResampling(unittest.TestCase):
    def test_downsampling_t(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'volume': [1, 1, 1, 1, 1, 1]
        })

        r = resample(Material.from_data(df), '3s').data.to_dict(orient='list')
        self.assertEqual([3, 6], r['timestamp'])

    def test_downsampling_q(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'volume': [1, 1, 1, 1, 1, 1]
        })

        r = resample(Material.from_data(df), '3s').data.to_dict(orient='list')
        self.assertEqual([4 / 3, 5 / 3], r['q'])

    def test_downsampling_v(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'volume': [1, 1, 1, 1, 1, 1]
        })

        r = resample(Material.from_data(df), '3s').data.to_dict(orient='list')
        self.assertEqual([3, 3], r['volume'])

    def test_upsampling_t(self):
        df = pd.DataFrame({
            'timestamp': [10, 20],
            'q': [1, 2],
            'volume': [2, 4]
        })

        r = resample(Material.from_data(df), '5s').data.to_dict(orient='list')
        self.assertEqual([5, 10, 15, 20], r['timestamp'])

    def test_upsampling_q(self):
        df = pd.DataFrame({
            'timestamp': [10, 20],
            'q': [1, 2],
            'volume': [2, 4]
        })

        r = resample(Material.from_data(df), '5s').data.to_dict(orient='list')
        self.assertEqual([1, 1, 2, 2], r['q'])

    def test_upsampling_v(self):
        df = pd.DataFrame({
            'timestamp': [10, 20],
            'q': [1, 2],
            'volume': [2, 4]
        })

        r = resample(Material.from_data(df), '5s').data.to_dict(orient='list')
        self.assertEqual([1, 1, 2, 2], r['volume'])

    def test_combined_t(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
            'q': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            'volume': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        })

        r = resample(Material.from_data(df), '5s').data.to_dict(orient='list')
        self.assertEqual([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                         r['timestamp'])

    def test_combined_q(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
            'q': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            'volume': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        })

        r = resample(Material.from_data(df), '5s').data.to_dict(orient='list')
        self.assertEqual(
            [1.5714285714285714, 1.6666666666666667, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            r['q']
        )

    def test_combined_v(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
            'q': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            'volume': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        })

        r = resample(Material.from_data(df), '5s').data.to_dict(orient='list')
        self.assertEqual(
            [7, 6, 0.5, 0.5, 0.5, 0.5, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142,
             0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142,
             0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142,
             0.07142857142857142],
            r['volume']
        )

    def test_volume_weighting_t(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'volume': [1, 2, 1, 2, 1, 2]
        })

        r = resample(Material.from_data(df), '3s').data.to_dict(orient='list')
        self.assertEqual([3, 6], r['timestamp'])

    def test_volume_weighting_q(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'volume': [1, 2, 1, 2, 1, 2]
        })

        r = resample(Material.from_data(df), '3s').data.to_dict(orient='list')
        self.assertEqual([1.5, 1.8], r['q'])

    def test_volume_weighting_v(self):
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'volume': [1, 2, 1, 2, 1, 2]
        })

        r = resample(Material.from_data(df), '3s').data.to_dict(orient='list')
        self.assertEqual([4, 5], r['volume'])
