import random
from typing import List

import numpy as np
import pandas as pd
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from blending_simulator.bsl_blending_simulator import BslBlendingSimulator
from blending_simulator.material_deposition import MaterialDeposition, Material, Deposition
from ciglobal.cimath import weighted_avg_and_std, stdev


def evaluate_solution(
        length: float,
        depth: float,
        material: pd.DataFrame,
        deposition: pd.DataFrame,
        material_quality_stdev: float,
        total_material_volume: float
) -> [float, float]:
    sim = BslBlendingSimulator(bed_size_x=length, bed_size_z=depth, ppm3=0.125)
    reclaimed_material = sim.stack_reclaim(material_deposition=MaterialDeposition(
        material=Material(data=material),
        deposition=Deposition(None, data=deposition)
    ), x_per_s=0.5)
    reclaimed_data = reclaimed_material.data

    # TODO make work with more parameters
    quality_average, quality_stdev = weighted_avg_and_std(reclaimed_data['p_1'], reclaimed_data['volume'])

    # TODO make work with xz_scaling != 1
    offset = 0.1 * depth
    central_reclaim_volumes = reclaimed_data['volume'].values[int(offset):int(length - depth - offset)]
    central_reclaim_volumes[0] += reclaimed_data['volume'].values[:int(offset)].sum()
    central_reclaim_volumes[-1] += reclaimed_data['volume'].values[int(length - depth - offset):].sum()
    volume_stdev = stdev(central_reclaim_volumes)
    worst_case_volume_stdev = stdev(np.array([total_material_volume / len(central_reclaim_volumes), 0]))

    return [quality_stdev / material_quality_stdev, volume_stdev / worst_case_volume_stdev]


class HomogenizationProblem(FloatProblem):
    def __init__(self, length: float, depth: float, material: pd.DataFrame, number_of_variables: int = 2):
        super().__init__()

        self.length = length
        self.depth = depth
        self.material = material

        # TODO make work with more parameters
        _, self.material_quality_stdev = weighted_avg_and_std(self.material['p_1'], self.material['volume'])
        self.total_material_volume = material['volume'].sum()

        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.lower_bound = [0.0 for _ in range(number_of_variables)]
        self.upper_bound = [1.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> None:
        min_pos = self.depth / 2
        max_pos = self.length - self.depth / 2
        deposition = pd.DataFrame(data={
            'x': [elem * (max_pos - min_pos) + min_pos for elem in solution.variables],
            'z': [self.depth / 2] * self.number_of_variables,
            'timestamp': np.linspace(0, self.material['timestamp'].values[-1], self.number_of_variables)
        })

        solution.objectives = evaluate_solution(
            length=self.length,
            depth=self.depth,
            material=self.material,
            deposition=deposition,
            material_quality_stdev=self.material_quality_stdev,
            total_material_volume=self.total_material_volume
        )

    @staticmethod
    def get_objective_labels() -> List[str]:
        return ['Quality Stdev', 'Volume Stdev']

    def get_variable_labels(self) -> List[str]:
        return [f'v{(i + 1)}' for i in range(self.number_of_variables)]

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
            self.lower_bound,
            self.upper_bound
        )

        new_solution.variables = [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in
                                  range(self.number_of_variables)] if random.random() < 0.8 else [i % 2 for i in range(
            self.number_of_variables)]

        return new_solution
