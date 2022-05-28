from unittest import TestCase

from jmetal.core.solution import Solution
from jmetal.util.comparator import DominanceComparator

from jmetalpy_extensions.util.comparator import FastDominanceComparator


def perform_compartor_test(self, t):
    self.assertEqual(t(), 0)
    self.assertEqual(t(), -1)
    self.assertEqual(t(), -1)
    self.assertEqual(t(), -1)
    self.assertEqual(t(), -1)
    self.assertEqual(t(), -1)
    self.assertEqual(t(), -1)
    self.assertEqual(t(), 1)
    self.assertEqual(t(), 1)
    self.assertEqual(t(), 1)
    self.assertEqual(t(), 1)
    self.assertEqual(t(), 1)
    self.assertEqual(t(), 1)
    self.assertEqual(t(), 0)
    self.assertEqual(t(), 0)
    self.assertEqual(t(), 0)


class TestFastDominanceComparator(TestCase):
    def test_match_dominance_comparator(self):
        test_vectors = [
            ([1, 2, 3], [1, 2, 3]),
            ([0, 2, 3], [1, 2, 3]),
            ([1, 0, 3], [1, 2, 3]),
            ([1, 2, 0], [1, 2, 3]),
            ([1, 2, 3], [1, 2, 4]),
            ([1, 2, 3], [1, 3, 3]),
            ([1, 2, 3], [2, 2, 3]),
            ([2, 2, 3], [1, 2, 3]),
            ([1, 3, 3], [1, 2, 3]),
            ([1, 2, 4], [1, 2, 3]),
            ([1, 2, 3], [1, 2, 2]),
            ([1, 2, 3], [1, 1, 3]),
            ([1, 2, 3], [0, 2, 3]),
            ([0, 3, 3], [1, 2, 3]),
            ([2, 1, 3], [1, 2, 3]),
            ([1, 2, 3], [1, 1, 4]),
            ([1, 2, 3], [1, 3, 2]),
        ]
        dc = DominanceComparator()
        fdc = FastDominanceComparator()
        for v1, v2 in test_vectors:
            s1 = Solution(number_of_variables=0, number_of_objectives=len(v1))
            s1.objectives = v1
            s2 = Solution(number_of_variables=0, number_of_objectives=len(v2))
            s2.objectives = v2
            self.assertEqual(
                dc.compare(s1, s2),
                fdc.compare(s1, s2)
            )
