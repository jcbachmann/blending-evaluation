#!/usr/bin/env python
import unittest

import bulk_density


class TestScaling(unittest.TestCase):
	def test_simple(self):
		height_map = [[1, 1], [1, 1]]
		xz_scaling = 1.0
		volume = bulk_density.get_height_map_volume(height_map, xz_scaling)
		self.assertAlmostEqual(volume, 1)

	def test_scaling_2(self):
		height_map = [[1, 1], [1, 1]]
		xz_scaling = 2.0
		volume = bulk_density.get_height_map_volume(height_map, xz_scaling)
		self.assertAlmostEqual(volume, 4)

	def test_height_2(self):
		height_map = [[2, 2], [2, 2]]
		xz_scaling = 1.0
		volume = bulk_density.get_height_map_volume(height_map, xz_scaling)
		self.assertAlmostEqual(volume, 2)

	def test_combined(self):
		height_map = [[2, 2], [2, 2]]
		xz_scaling = 2.0
		volume = bulk_density.get_height_map_volume(height_map, xz_scaling)
		self.assertAlmostEqual(volume, 8)


class TestBulkDensity(unittest.TestCase):
	def test_empty(self):
		height_map = []
		xz_scaling = 1.0
		volume = bulk_density.get_height_map_volume(height_map, xz_scaling)
		self.assertEqual(volume, 0)

	def test_zero(self):
		xz_scaling = 1.0
		self.assertEqual(bulk_density.get_height_map_volume([[0]], xz_scaling), 0)
		self.assertEqual(bulk_density.get_height_map_volume([[0, 0], [0, 0]], xz_scaling), 0)

	def test_corners_up(self):
		xz_scaling = 1.0
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 0], [0, 0]], xz_scaling), 0.25)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 1], [0, 0]], xz_scaling), 0.25)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 0], [1, 0]], xz_scaling), 0.25)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 0], [0, 1]], xz_scaling), 0.25)

	def test_corners_down(self):
		xz_scaling = 1.0
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [1, 0]], xz_scaling), 0.75)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [0, 1]], xz_scaling), 0.75)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 0], [1, 1]], xz_scaling), 0.75)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 1], [1, 1]], xz_scaling), 0.75)

	def test_half(self):
		xz_scaling = 1.0
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 0], [1, 1]], xz_scaling), 0.5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 1], [0, 1]], xz_scaling), 0.5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[0, 1], [1, 0]], xz_scaling), 0.5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 0], [0, 1]], xz_scaling), 0.5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 0], [1, 0]], xz_scaling), 0.5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [0, 0]], xz_scaling), 0.5)

	def test_corners_high(self):
		xz_scaling = 1.0
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[2, 1], [1, 1]], xz_scaling), 1.25)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 2], [1, 1]], xz_scaling), 1.25)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [2, 1]], xz_scaling), 1.25)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [1, 2]], xz_scaling), 1.25)

	def test_corners_scaling(self):
		xz_scaling = 2.0
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[2, 1], [1, 1]], xz_scaling), 5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 2], [1, 1]], xz_scaling), 5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [2, 1]], xz_scaling), 5)
		self.assertAlmostEqual(bulk_density.get_height_map_volume([[1, 1], [1, 2]], xz_scaling), 5)

	def test_spike(self):
		height_map = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
		xz_scaling = 1.0
		volume = bulk_density.get_height_map_volume(height_map, xz_scaling)
		self.assertAlmostEqual(volume, 1)
