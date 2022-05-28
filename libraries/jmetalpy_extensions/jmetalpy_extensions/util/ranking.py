import itertools
from functools import lru_cache
from typing import TypeVar, List

from jmetal.util.comparator import Comparator, SolutionAttributeComparator
from jmetal.util.ranking import Ranking

S = TypeVar('S')


# This is a caching implementation of the compare method of DominanceComparator
@lru_cache(maxsize=1000000)
def compare(objectives1: tuple, objectives2: tuple) -> int:
    result = 0
    for v1, v2 in zip(objectives1, objectives2):
        if v1 > v2:
            if result == -1:
                return 0
            result = 1
        elif v1 < v2:
            if result == 1:
                return 0
            result = -1

    return result


# Warning: this class completely ignores the dominance comparator and does not evaluate constraints
# It is simply meant to be the fastest version of
class FastestNonDominatedRanking(Ranking[List[S]]):
    """ Class implementing the non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002]_ """

    def __init__(self):
        super(FastestNonDominatedRanking, self).__init__()

    def compute_ranking(self, solutions: List[S], k: int = None):
        """ Compute ranking of solutions.

        :param solutions: Solution list.
        :param k: Number of individuals.
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]

        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]

        # Convert objectives to tuples to enable use of lru_cache
        tupelized_objectives = [tuple(solution.objectives) for solution in solutions]

        number_of_comparisons = 0

        for p, q in itertools.combinations(range(len(tupelized_objectives)), 2):
            dominance_test_result = compare(tupelized_objectives[p], tupelized_objectives[q])
            number_of_comparisons += 1
            if dominance_test_result == -1:
                ith_dominated[p].append(q)
                dominating_ith[q] += 1
            elif dominance_test_result == 1:
                ith_dominated[q].append(p)
                dominating_ith[p] += 1

        self.number_of_comparisons += number_of_comparisons
        # Only supported from Python 3.8 onwards
        # self.number_of_comparisons += math.comb(len(solutions), 2)

        for i in range(len(solutions)):
            if dominating_ith[i] == 0:
                front[0].append(i)
                solutions[i].attributes['dominance_ranking'] = 0

        i = 0
        while len(front[i]) != 0:
            i += 1
            for p in front[i - 1]:
                if p <= len(ith_dominated):
                    for q in ith_dominated[p]:
                        dominating_ith[q] -= 1
                        if dominating_ith[q] == 0:
                            front[i].append(q)
                            solutions[q].attributes['dominance_ranking'] = i

        self.ranked_sublists = [[]] * i
        for j in range(i):
            q = [0] * len(front[j])
            for m in range(len(front[j])):
                q[m] = solutions[front[j][m]]
            self.ranked_sublists[j] = q

        if k:
            count = 0
            for i, front in enumerate(self.ranked_sublists):
                count += len(front)
                if count >= k:
                    self.ranked_sublists = self.ranked_sublists[:i + 1]
                    break

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator('dominance_ranking')
