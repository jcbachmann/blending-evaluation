import functools
from multiprocessing.pool import Pool
from typing import List

from jmetal.component.evaluator import Evaluator, S
from jmetal.core.problem import Problem

from .evaluator_observer import EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes=None, observer: EvaluatorObserver = None):
        self.pool = Pool(processes)
        self.observer = observer

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        solution_list = self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)
        if self.observer is not None:
            self.observer.notify(solution_list)
        return solution_list
