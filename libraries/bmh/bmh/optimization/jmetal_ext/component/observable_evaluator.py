from abc import ABC, abstractmethod
from typing import List, Optional

from jmetal.component.evaluator import S, Evaluator
from jmetal.core.problem import Problem


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
