#!/usr/bin/env python
import unittest

import pytest

from bmh_apps.bulk_density.evaluate_bulk_density import get_height_map_volume


class TestScaling(unittest.TestCase):
    def test_simple(self):
        height_map = [[1, 1], [1, 1]]
        size = 1.0
        volume = get_height_map_volume(height_map, size)
        assert volume == pytest.approx(1)

    def test_scaling_2(self):
        height_map = [[1, 1], [1, 1]]
        size = 2.0
        volume = get_height_map_volume(height_map, size)
        assert volume == pytest.approx(4)

    def test_height_2(self):
        height_map = [[2, 2], [2, 2]]
        size = 1.0
        volume = get_height_map_volume(height_map, size)
        assert volume == pytest.approx(2)

    def test_combined(self):
        height_map = [[2, 2], [2, 2]]
        size = 2.0
        volume = get_height_map_volume(height_map, size)
        assert volume == pytest.approx(8)


class TestBulkDensity(unittest.TestCase):
    def test_empty(self):
        height_map = []
        size = 1.0
        volume = get_height_map_volume(height_map, size)
        assert volume == 0

    def test_zero(self):
        size = 1.0
        assert get_height_map_volume([[0]], size) == 0
        assert get_height_map_volume([[0, 0], [0, 0]], size) == 0

    def test_corners_up(self):
        size = 1.0
        assert get_height_map_volume([[1, 0], [0, 0]], size) == pytest.approx(0.25)
        assert get_height_map_volume([[0, 1], [0, 0]], size) == pytest.approx(0.25)
        assert get_height_map_volume([[0, 0], [1, 0]], size) == pytest.approx(0.25)
        assert get_height_map_volume([[0, 0], [0, 1]], size) == pytest.approx(0.25)

    def test_corners_down(self):
        size = 1.0
        assert get_height_map_volume([[1, 1], [1, 0]], size) == pytest.approx(0.75)
        assert get_height_map_volume([[1, 1], [0, 1]], size) == pytest.approx(0.75)
        assert get_height_map_volume([[1, 0], [1, 1]], size) == pytest.approx(0.75)
        assert get_height_map_volume([[0, 1], [1, 1]], size) == pytest.approx(0.75)

    def test_half(self):
        size = 1.0
        assert get_height_map_volume([[0, 0], [1, 1]], size) == pytest.approx(0.5)
        assert get_height_map_volume([[0, 1], [0, 1]], size) == pytest.approx(0.5)
        assert get_height_map_volume([[0, 1], [1, 0]], size) == pytest.approx(0.5)
        assert get_height_map_volume([[1, 0], [0, 1]], size) == pytest.approx(0.5)
        assert get_height_map_volume([[1, 0], [1, 0]], size) == pytest.approx(0.5)
        assert get_height_map_volume([[1, 1], [0, 0]], size) == pytest.approx(0.5)

    def test_corners_high(self):
        size = 1.0
        assert get_height_map_volume([[2, 1], [1, 1]], size) == pytest.approx(1.25)
        assert get_height_map_volume([[1, 2], [1, 1]], size) == pytest.approx(1.25)
        assert get_height_map_volume([[1, 1], [2, 1]], size) == pytest.approx(1.25)
        assert get_height_map_volume([[1, 1], [1, 2]], size) == pytest.approx(1.25)

    def test_corners_scaling(self):
        size = 2.0
        assert get_height_map_volume([[2, 1], [1, 1]], size) == pytest.approx(5)
        assert get_height_map_volume([[1, 2], [1, 1]], size) == pytest.approx(5)
        assert get_height_map_volume([[1, 1], [2, 1]], size) == pytest.approx(5)
        assert get_height_map_volume([[1, 1], [1, 2]], size) == pytest.approx(5)

    def test_spike(self):
        height_map = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        size = 2.0
        volume = get_height_map_volume(height_map, size)
        assert volume == pytest.approx(1)
