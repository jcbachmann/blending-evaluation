from typing import List

from jmetal.component.evaluator import S


class EvaluatorObserver:
    def notify(self, solution_list: List[S]):
        pass
