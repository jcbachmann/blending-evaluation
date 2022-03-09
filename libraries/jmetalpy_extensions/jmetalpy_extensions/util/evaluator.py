import functools
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List, Optional

try:
    import dask
except ImportError:
    pass

try:
    from dask.distributed import Client, LocalCluster
except ImportError:
    pass

from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator, S


class EvaluatorObserver:
    def notify(self, solution_list: List[S]):
        pass


class ObservableEvaluator(Evaluator[S], ABC):
    def __init__(self, observer: Optional[EvaluatorObserver] = None):
        self.observer = observer

    @abstractmethod
    def observed_evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        pass

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        solution_list = self.observed_evaluate(solution_list, problem)

        if self.observer is not None:
            self.observer.notify(solution_list)

        return solution_list

    def set_observer(self, observer: Optional[EvaluatorObserver] = None):
        self.observer = observer


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


class DaskEvaluator(ObservableEvaluator[S]):
    def __init__(self, observer: Optional[EvaluatorObserver] = None, scheduler='processes'):
        super().__init__(observer)
        self.scheduler = scheduler

    def observed_evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        with dask.config.set(scheduler=self.scheduler):
            return list(dask.compute(*[
                dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list
            ]))


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
