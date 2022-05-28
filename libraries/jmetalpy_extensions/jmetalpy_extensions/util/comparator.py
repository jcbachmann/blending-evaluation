from typing import TypeVar

from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator

S = TypeVar('S')


# This is a fast version of the standard DominanceComparator which ignores constraint evaluation
class FastDominanceComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        result = 0
        for v1, v2 in zip(solution1.objectives, solution2.objectives):
            if v1 > v2:
                if result == -1:
                    return 0
                result = 1
            elif v1 < v2:
                if result == 1:
                    return 0
                result = -1

        return result
