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
        parameter_columns: List[str],
        deposition: pd.DataFrame,
        material_parameter_standard_deviations: List[float],
        total_material_volume: float
) -> [float, float]:
    sim = BslBlendingSimulator(bed_size_x=length, bed_size_z=depth, ppm3=0.125)
    reclaimed_material = sim.stack_reclaim(material_deposition=MaterialDeposition(
        material=Material(data=material),
        deposition=Deposition(None, data=deposition)
    ), x_per_s=0.5)
    reclaimed_data = reclaimed_material.data

    standard_deviations = [weighted_avg_and_std(reclaimed_data[parameter_column], reclaimed_data['volume'])[1] for
                           parameter_column in parameter_columns]

    volume_stdev = stdev(reclaimed_data['volume'].values.copy())
    worst_case_volume_stdev = stdev(np.array([total_material_volume / reclaimed_data['volume'].shape[0], 0]))

    return [s / d for s, d in zip(standard_deviations, material_parameter_standard_deviations)] + [
        volume_stdev / worst_case_volume_stdev]


class HomogenizationProblem(FloatProblem):
    def __init__(self, length: float, depth: float, material: pd.DataFrame, parameter_columns: List[str],
                 number_of_variables: int = 2):
        super().__init__()

        self.length = length
        self.depth = depth
        self.material = material
        self.parameter_columns = parameter_columns

        self.material_parameter_standard_deviations = [
            weighted_avg_and_std(self.material[parameter_column], self.material['volume'])[1] for parameter_column in
            parameter_columns
        ]
        self.total_material_volume = material['volume'].sum()

        self.number_of_objectives = len(parameter_columns)
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
            parameter_columns=self.parameter_columns,
            deposition=deposition,
            material_parameter_standard_deviations=self.material_parameter_standard_deviations,
            total_material_volume=self.total_material_volume
        )

    def get_objective_labels(self) -> List[str]:
        return [col + ' Stdev' for col in self.parameter_columns] + ['Volume Stdev']

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

        def solution_random(v):
            return [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in range(v)]

        def solution_full_speed(v):
            return [i % 2 for i in range(v)]

        def solution_fixed_random_speed(v):
            speed = random.randint(1, 10)

            def pos(i):
                nonlocal speed

                p = (i % speed) / speed
                return p if int(i / speed) % 2 == 0 else 1 - p

            return [pos(i) for i in range(v)]

        def solution_random_speed(v):
            offset = 0
            speed = random.randint(1, 10)
            start_dir = True

            def pos(i):
                nonlocal offset
                nonlocal speed
                nonlocal start_dir

                i_rel = i - offset

                if i_rel > 0 and i_rel % speed == 0:
                    offset = i
                    speed = random.randint(1, 10)
                    start_dir = not start_dir
                    i_rel = 0

                p = (i_rel % speed) / speed
                return p if start_dir else 1 - p

            return [pos(i) for i in range(v)]

        weighted_choices = [
            (solution_random, 8),
            (solution_full_speed, 2),
            (solution_fixed_random_speed, 5),
            (solution_random_speed, 5)
        ]

        new_solution.variables = random.choices(
            [c[0] for c in weighted_choices], weights=[c[1] for c in weighted_choices]
        )[0](self.number_of_variables)

        return new_solution
