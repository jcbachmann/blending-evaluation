from typing import List

from ..benchmark.material_deposition import Deposition


class OptimizationResult:
    def __init__(self, deposition: Deposition, variables: List[float], objectives: List[float],
                 objective_labels: List[str]):
        self.deposition = deposition
        self.variables = variables
        self.objectives = objectives
        self.objective_labels = objective_labels
