from typing import List, Optional

import dask
from jmetal.component.evaluator import Evaluator, S
from jmetal.core.problem import Problem

from .observable_evaluator import ObservableEvaluator, EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(ObservableEvaluator[S]):
    def __init__(self, observer: Optional[EvaluatorObserver] = None, scheduler='processes'):
        super().__init__(observer)
        self.scheduler = scheduler

    def observed_evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        with dask.config.set(scheduler=self.scheduler):
            return list(dask.compute(*[
                dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list
            ]))
