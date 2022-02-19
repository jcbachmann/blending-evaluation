import functools
from multiprocessing.pool import Pool
from typing import List, Optional

from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator, S

from .observable_evaluator import ObservableEvaluator, EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class MultiprocessEvaluator(ObservableEvaluator[S]):
    def __init__(self, processes=None, observer: Optional[EvaluatorObserver] = None):
        super().__init__(observer)
        self.pool = Pool(processes)

    def observed_evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)

    def stop(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
