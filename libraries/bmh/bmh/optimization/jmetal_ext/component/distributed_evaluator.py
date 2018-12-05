import functools
from typing import List, Optional

from dask.distributed import Client, LocalCluster
from jmetal.component.evaluator import Evaluator, S
from jmetal.core.problem import Problem

from .evaluator_observer import EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DistributedEvaluator(Evaluator[S]):
    def __init__(self, observer: Optional[EvaluatorObserver] = None, scheduler: Optional[str] = None):
        self.observer = observer

        if scheduler is None:
            self.local_cluster = LocalCluster()
            self.client = Client(self.local_cluster)
        else:
            self.local_cluster = None
            self.client = Client(address=scheduler)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        calculations = self.client.map(functools.partial(evaluate_solution, problem=problem), solution_list)
        solution_list = list(self.client.gather(calculations))

        if self.observer is not None:
            self.observer.notify(solution_list)

        return solution_list

    def stop(self):
        if self.client:
            self.client.close()

        if self.local_cluster:
            self.local_cluster.close()
