import contextlib
import functools
from abc import ABC, abstractmethod
from multiprocessing import Pool

with contextlib.suppress(ImportError):
    import dask

with contextlib.suppress(ImportError):
    from dask.distributed import Client, LocalCluster

from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator, S


class EvaluatorObserver:
    def notify(self, solution_list: list[S]):
        pass


class ObservableEvaluator(Evaluator[S], ABC):
    def __init__(self, observer: EvaluatorObserver | None = None):
        self.observer = observer

    @abstractmethod
    def observed_evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        pass

    def evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        solution_list = self.observed_evaluate(solution_list, problem)

        if self.observer is not None:
            self.observer.notify(solution_list)

        return solution_list

    def set_observer(self, observer: EvaluatorObserver | None = None):
        self.observer = observer


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class MultiprocessEvaluator(ObservableEvaluator[S]):
    def __init__(self, processes=None, observer: EvaluatorObserver | None = None):
        super().__init__(observer)
        self.pool = Pool(processes)

    def observed_evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)

    def stop(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None


class DaskEvaluator(ObservableEvaluator[S]):
    def __init__(self, observer: EvaluatorObserver | None = None, scheduler="processes"):
        super().__init__(observer)
        self.scheduler = scheduler

    def observed_evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        with dask.config.set(scheduler=self.scheduler):
            return list(dask.compute(*[dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list]))


class DistributedEvaluator(ObservableEvaluator[S]):
    def __init__(self, observer: EvaluatorObserver | None = None, scheduler: str | None = None):
        super().__init__(observer)

        if scheduler is None:
            self.local_cluster = LocalCluster()
            self.client = Client(self.local_cluster)
        else:
            self.local_cluster = None
            self.client = Client(address=scheduler)

    def observed_evaluate(self, solution_list: list[S], problem: Problem) -> list[S]:
        calculations = self.client.map(functools.partial(evaluate_solution, problem=problem), solution_list)
        return list(self.client.gather(calculations))

    def stop(self):
        if self.client:
            self.client.close()

        if self.local_cluster:
            self.local_cluster.close()
