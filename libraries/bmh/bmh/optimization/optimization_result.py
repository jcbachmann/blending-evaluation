from typing import List

from ..benchmark.material_deposition import Deposition, Material


class OptimizationResult:
    def __init__(self, deposition: Deposition, variables: List[float], objectives: List[float],
                 objective_labels: List[str], reclaimed_material: Material = None):
        self.deposition = deposition
        self.variables = variables
        self.objectives = objectives
        self.objective_labels = objective_labels
        self.reclaimed_material = reclaimed_material
