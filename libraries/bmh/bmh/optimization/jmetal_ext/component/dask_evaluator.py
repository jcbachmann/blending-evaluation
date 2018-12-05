from typing import List, Optional

import dask
from jmetal.component.evaluator import Evaluator, S
from jmetal.core.problem import Problem

from .evaluator_observer import EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, observer: Optional[EvaluatorObserver] = None, scheduler='processes'):
        self.observer = observer
        self.scheduler = scheduler

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        with dask.config.set(scheduler=self.scheduler):
            calculations = [
                dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list
            ]
            solution_list = list(dask.compute(*calculations))

        if self.observer is not None:
            self.observer.notify(solution_list)

        return solution_list
