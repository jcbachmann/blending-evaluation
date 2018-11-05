import functools
from typing import List, Optional

from dask.distributed import Client, LocalCluster
from jmetal.component.evaluator import Evaluator, S
from jmetal.core.problem import Problem

from .evaluator_observer import EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, observer: Optional[EvaluatorObserver] = None, scheduler_address: Optional[str] = None):
        self.observer = observer
        if scheduler_address is None:
            self.client = Client(LocalCluster())
        else:
            self.client = Client(address=scheduler_address)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        calcs = self.client.map(functools.partial(evaluate_solution, problem=problem), solution_list)
        solution_list = list(self.client.gather(calcs))
        if self.observer is not None:
            self.observer.notify(solution_list)
        return solution_list
