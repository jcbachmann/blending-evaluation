import unittest

from .mathematical_blending import calculate_output


class TestMathematicalBlending(unittest.TestCase):
    def test_empty(self):
        v, p = calculate_output(0, [], [], [], 1)
        self.assertEqual(v, [0])
        self.assertEqual(p, [[0]])

    def test_size(self):
        v, p = calculate_output(0, [], [], [], 4)
        self.assertEqual(v, [0, 0, 0, 0])
        self.assertEqual(p, [0, 0, 0, 0])

    def test_simple(self):
        v, p = calculate_output(1, [1], [2], [1], 4)
        self.assertEqual(v, [0, 1, 0, 0])
        self.assertEqual(p, [0, 2, 0, 0])

    def test_averaging(self):
        v, p = calculate_output(2, [1, 1], [2, 4], [1, 1], 4)
        self.assertEqual(v, [0, 2, 0, 0])
        self.assertEqual(p, [0, 3, 0, 0])
