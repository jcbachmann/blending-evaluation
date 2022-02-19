import functools
from typing import List, Optional

from dask.distributed import Client, LocalCluster
from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator, S

from .observable_evaluator import ObservableEvaluator, EvaluatorObserver


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DistributedEvaluator(ObservableEvaluator[S]):
    def __init__(self, observer: Optional[EvaluatorObserver] = None, scheduler: Optional[str] = None):
        super().__init__(observer)

        if scheduler is None:
            self.local_cluster = LocalCluster()
            self.client = Client(self.local_cluster)
        else:
            self.local_cluster = None
            self.client = Client(address=scheduler)

    def observed_evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        calculations = self.client.map(functools.partial(evaluate_solution, problem=problem), solution_list)
        return list(self.client.gather(calculations))

    def stop(self):
        if self.client:
            self.client.close()

        if self.local_cluster:
            self.local_cluster.close()
