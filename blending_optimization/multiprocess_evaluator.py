import functools
from multiprocessing.pool import Pool
from typing import List

from jmetal.component.evaluator import Evaluator, S
from jmetal.core.problem import Problem


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes=None):
        self.pool = Pool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        solution_list = self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)
        return solution_list
