#!/usr/bin/env python
import unittest

import pandas as pd
from bmh.benchmark.material_deposition import Material
from numpy.testing import assert_almost_equal

from bmh_apps.tools.resampling import resample


class TestResampling(unittest.TestCase):
    def test_downsampling_t(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "q": [1, 2, 1, 2, 1, 2],
                "volume": [1, 1, 1, 1, 1, 1],
            }
        )

        r = resample(Material.from_data(df), "3s").data.to_dict(orient="list")
        assert_almost_equal(r["timestamp"], [3, 6])

    def test_downsampling_q(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "q": [1, 2, 1, 2, 1, 2],
                "volume": [1, 1, 1, 1, 1, 1],
            }
        )

        r = resample(Material.from_data(df), "3s").data.to_dict(orient="list")
        assert_almost_equal(r["q"], [4 / 3, 5 / 3])

    def test_downsampling_v(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "q": [1, 2, 1, 2, 1, 2],
                "volume": [1, 1, 1, 1, 1, 1],
            }
        )

        r = resample(Material.from_data(df), "3s").data.to_dict(orient="list")
        assert_almost_equal(r["volume"], [3, 3])

    def test_upsampling_t(self):
        df = pd.DataFrame(
            {
                "timestamp": [10, 20],
                "q": [1, 2],
                "volume": [2, 4],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["timestamp"], [5, 10, 15, 20])

    def test_upsampling_q(self):
        df = pd.DataFrame(
            {
                "timestamp": [10, 20],
                "q": [1, 2],
                "volume": [2, 4],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["q"], [1, 1, 2, 2])

    def test_upsampling_v(self):
        df = pd.DataFrame(
            {
                "timestamp": [10, 20],
                "q": [1, 2],
                "volume": [2, 4],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["volume"], [1, 1, 2, 2])

    def test_combined_t(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
                "q": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                "volume": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(
            r["timestamp"],
            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        )

    def test_combined_q(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
                "q": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                "volume": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["q"], [1.5714285714285714, 1.6666666666666667, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_combined_v(self):
        df = pd.DataFrame(
            {"timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100], "q": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], "volume": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]}
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(
            r["volume"],
            [
                7,
                6,
                0.5,
                0.5,
                0.5,
                0.5,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
                0.07142857142857142,
            ],
        )

    def test_volume_weighting_t(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "q": [1, 2, 1, 2, 1, 2],
                "volume": [1, 2, 1, 2, 1, 2],
            }
        )

        r = resample(Material.from_data(df), "3s").data.to_dict(orient="list")
        assert_almost_equal([3, 6], r["timestamp"])

    def test_volume_weighting_q(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "q": [1, 2, 1, 2, 1, 2],
                "volume": [1, 2, 1, 2, 1, 2],
            }
        )

        r = resample(Material.from_data(df), "3s").data.to_dict(orient="list")
        assert_almost_equal(r["q"], [1.5, 1.8])

    def test_volume_weighting_v(self):
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "q": [1, 2, 1, 2, 1, 2],
                "volume": [1, 2, 1, 2, 1, 2],
            }
        )

        r = resample(Material.from_data(df), "3s").data.to_dict(orient="list")
        assert_almost_equal(r["volume"], [4, 5])

    def test_misaligned_t(self):
        df = pd.DataFrame(
            {
                "timestamp": [3, 6, 9],
                "q": [2, 5, 1],
                "volume": [3, 3, 1],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["timestamp"], [5.0, 10.0])

    def test_misaligned_q(self):
        df = pd.DataFrame(
            {
                "timestamp": [3, 6, 9],
                "q": [2, 5, 1],
                "volume": [3, 3, 1],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["q"], [3.2, 3.0])

    def test_misaligned_v(self):
        df = pd.DataFrame(
            {
                "timestamp": [3, 6, 9],
                "q": [2, 5, 1],
                "volume": [3, 3, 1],
            }
        )

        r = resample(Material.from_data(df), "5s").data.to_dict(orient="list")
        assert_almost_equal(r["volume"], [5.0, 2.0])

    def test_premature_t(self):
        df = pd.DataFrame(
            {
                "timestamp": [5, 10, 15],
                "q": [2, 5, 1],
                "volume": [3, 3, 1],
            }
        )

        r = resample(Material.from_data(df), "10s").data.to_dict(orient="list")
        assert_almost_equal(r["timestamp"], [10.0, 20.0])

    def test_premature_q(self):
        df = pd.DataFrame(
            {
                "timestamp": [5, 10, 15],
                "q": [2, 5, 1],
                "volume": [3, 3, 1],
            }
        )

        r = resample(Material.from_data(df), "10s").data.to_dict(orient="list")
        assert_almost_equal(r["q"], [3.5, 1.0])

    def test_premature_v(self):
        df = pd.DataFrame(
            {
                "timestamp": [5, 10, 15],
                "q": [2, 5, 1],
                "volume": [3, 3, 1],
            }
        )

        r = resample(Material.from_data(df), "10s").data.to_dict(orient="list")
        assert_almost_equal(r["volume"], [6.0, 1.0])
