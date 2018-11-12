#!/usr/bin/env python
import unittest

from bmh_apps.bulk_density.evaluate_bulk_density import get_height_map_volume


class TestScaling(unittest.TestCase):
    def test_simple(self):
        height_map = [[1, 1], [1, 1]]
        size = 1.0
        volume = get_height_map_volume(height_map, size)
        self.assertAlmostEqual(volume, 1)

    def test_scaling_2(self):
        height_map = [[1, 1], [1, 1]]
        size = 2.0
        volume = get_height_map_volume(height_map, size)
        self.assertAlmostEqual(volume, 4)

    def test_height_2(self):
        height_map = [[2, 2], [2, 2]]
        size = 1.0
        volume = get_height_map_volume(height_map, size)
        self.assertAlmostEqual(volume, 2)

    def test_combined(self):
        height_map = [[2, 2], [2, 2]]
        size = 2.0
        volume = get_height_map_volume(height_map, size)
        self.assertAlmostEqual(volume, 8)


class TestBulkDensity(unittest.TestCase):
    def test_empty(self):
        height_map = []
        size = 1.0
        volume = get_height_map_volume(height_map, size)
        self.assertEqual(volume, 0)

    def test_zero(self):
        size = 1.0
        self.assertEqual(get_height_map_volume([[0]], size), 0)
        self.assertEqual(get_height_map_volume([[0, 0], [0, 0]], size), 0)

    def test_corners_up(self):
        size = 1.0
        self.assertAlmostEqual(get_height_map_volume([[1, 0], [0, 0]], size), 0.25)
        self.assertAlmostEqual(get_height_map_volume([[0, 1], [0, 0]], size), 0.25)
        self.assertAlmostEqual(get_height_map_volume([[0, 0], [1, 0]], size), 0.25)
        self.assertAlmostEqual(get_height_map_volume([[0, 0], [0, 1]], size), 0.25)

    def test_corners_down(self):
        size = 1.0
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [1, 0]], size), 0.75)
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [0, 1]], size), 0.75)
        self.assertAlmostEqual(get_height_map_volume([[1, 0], [1, 1]], size), 0.75)
        self.assertAlmostEqual(get_height_map_volume([[0, 1], [1, 1]], size), 0.75)

    def test_half(self):
        size = 1.0
        self.assertAlmostEqual(get_height_map_volume([[0, 0], [1, 1]], size), 0.5)
        self.assertAlmostEqual(get_height_map_volume([[0, 1], [0, 1]], size), 0.5)
        self.assertAlmostEqual(get_height_map_volume([[0, 1], [1, 0]], size), 0.5)
        self.assertAlmostEqual(get_height_map_volume([[1, 0], [0, 1]], size), 0.5)
        self.assertAlmostEqual(get_height_map_volume([[1, 0], [1, 0]], size), 0.5)
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [0, 0]], size), 0.5)

    def test_corners_high(self):
        size = 1.0
        self.assertAlmostEqual(get_height_map_volume([[2, 1], [1, 1]], size), 1.25)
        self.assertAlmostEqual(get_height_map_volume([[1, 2], [1, 1]], size), 1.25)
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [2, 1]], size), 1.25)
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [1, 2]], size), 1.25)

    def test_corners_scaling(self):
        size = 2.0
        self.assertAlmostEqual(get_height_map_volume([[2, 1], [1, 1]], size), 5)
        self.assertAlmostEqual(get_height_map_volume([[1, 2], [1, 1]], size), 5)
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [2, 1]], size), 5)
        self.assertAlmostEqual(get_height_map_volume([[1, 1], [1, 2]], size), 5)

    def test_spike(self):
        height_map = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        size = 2.0
        volume = get_height_map_volume(height_map, size)
        self.assertAlmostEqual(volume, 1)
