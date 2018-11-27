import random
from typing import List, Optional

import numpy as np
from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition
from bmh.helpers.math import weighted_avg_and_std, stdev
from bmh.helpers.stockpile_math import get_stockpile_height
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from pandas import DataFrame


def process_material_deposition(material: Material, deposition: Deposition, ppm3: float = 0.125):
    sim = BslBlendingSimulator(
        bed_size_x=deposition.meta.bed_size_x,
        bed_size_z=deposition.meta.bed_size_z,
        ppm3=ppm3
    )
    material_deposition = MaterialDeposition(
        material=material,
        deposition=deposition
    )
    return sim.stack_reclaim(material_deposition)


def variables_to_deposition_generic(
        variables: List[float], x_min: float, x_max: float, bed_size_x: float, bed_size_z: float, max_timestamp: float
):
    return Deposition.from_data(DataFrame(
        data={
            'x': [elem * (x_max - x_min) + x_min for elem in variables],
            'z': [bed_size_z / 2] * len(variables),
            'timestamp': np.linspace(0, max_timestamp, len(variables))
        }),
        bed_size_x=bed_size_x, bed_size_z=bed_size_z, reclaim_x_per_s=0.5
    )


def calculate_reference_objectives_generic(
        bed_size_x: float, bed_size_z: float,
        x_min: float, x_max: float,
        raw_material: Material,
        number_of_variables: int,
        variables_prefix: List[float]
):
    max_timestamp = raw_material.data['timestamp'].values[-1]

    # Generate a Chevron deposition with maximum speed
    chevron = [i % 2 for i in range(number_of_variables)]
    chevron_deposition = variables_to_deposition_generic(
        variables=variables_prefix + chevron, x_min=x_min, x_max=x_max, bed_size_x=bed_size_x, bed_size_z=bed_size_z,
        max_timestamp=max_timestamp
    )

    # Stack and reclaim material to acquire reference reclaimed material
    reclaimed_material = process_material_deposition(material=raw_material, deposition=chevron_deposition)

    # Set the parameter reference values to full speed Chevron stacked and reclaimed material data
    reclaimed_evaluator = MaterialEvaluator(reclaimed_material)
    chevron_parameter_stdev = reclaimed_evaluator.get_parameter_stdev()

    # Set the maximum acceptable volume standard deviation to a a factor of two for all slices
    volume_per_slice = reclaimed_material.get_volume() / reclaimed_evaluator.get_slice_count()
    worst_acceptable_volume_stdev = stdev(np.array([4.0 / 3.0 * volume_per_slice, 2.0 / 3.0 * volume_per_slice]))

    return chevron_parameter_stdev + [worst_acceptable_volume_stdev]


class MaterialEvaluator:
    def __init__(self, reclaimed: Material, x_min: Optional[float] = None, x_max: Optional[float] = None):
        self.reclaimed = reclaimed
        self.x_min = x_min
        self.x_max = x_max

        # Caches
        self._core_data = None
        self._parameter_stdev = None
        self._volume_stdev = None

    def get_core_data(self) -> DataFrame:
        if self._core_data is None:
            rdf = self.reclaimed.data
            if self.x_min is None or self.x_max is None:
                return rdf
            height = get_stockpile_height(volume=self.reclaimed.get_volume(), core_length=self.x_max - self.x_min)
            self._core_data = rdf[(rdf['x'] >= self.x_min) & (rdf['x'] <= self.x_max - height)]
        return self._core_data

    def get_core_volume_stdev(self) -> float:
        if self._volume_stdev is None:
            core_data = self.get_core_data()
            self._volume_stdev = stdev(core_data['volume'].values)
        return self._volume_stdev

    def get_parameter_stdev(self):
        if self._parameter_stdev is None:
            reclaimed_df = self.reclaimed.data
            cols = self.reclaimed.get_parameter_columns()
            self._parameter_stdev = [weighted_avg_and_std(reclaimed_df[col], reclaimed_df['volume'])[1] for col in cols]
        return self._parameter_stdev

    def get_all_stdev(self):
        return self.get_parameter_stdev() + [self.get_core_volume_stdev()]

    def get_all_stdev_relative(self, reference: List[float]):
        return [s / r for s, r in zip(self.get_all_stdev(), reference)]

    def get_slice_count(self):
        return self.reclaimed.data['volume'].shape[0]


class HomogenizationProblem(FloatProblem):
    def __init__(self, bed_size_x: float, bed_size_z: float, material: Material, number_of_variables: int = 2):
        super().__init__()

        # Copy parameters
        self.bed_size_x = bed_size_x
        self.bed_size_z = bed_size_z
        self.material = material
        self.number_of_variables = number_of_variables

        # Calculate stacker travel range
        self.x_min = 0.5 * self.bed_size_z
        self.x_max = self.bed_size_x - 0.5 * self.bed_size_z

        # Buffer values
        self.max_timestamp = material.data['timestamp'].values[-1]
        self.variables_prefix = []
        self.solution_pool = None

        # Evaluate reference data
        self.reference = self.calculate_reference_objectives()

        # Setup problem base variables
        self.number_of_objectives = len(material.get_parameter_columns())
        self.number_of_constraints = 0

        self.lower_bound = [0.0 for _ in range(number_of_variables)]
        self.upper_bound = [1.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> None:
        deposition = self.variables_to_deposition(variables=solution.variables)
        reclaimed_material = process_material_deposition(material=self.material, deposition=deposition)
        reclaimed_evaluator = MaterialEvaluator(reclaimed=reclaimed_material, x_min=self.x_min, x_max=self.x_max)
        solution.objectives = reclaimed_evaluator.get_all_stdev_relative(self.reference)

    def get_objective_labels(self) -> List[str]:
        return [col + ' Stdev' for col in self.material.get_parameter_columns()] + ['Volume Stdev']

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
            starting_side = random.choice([0, 1])
            return [(i + starting_side) % 2 for i in range(v)]

        def solution_random_end(v):
            return [random.choice([0, 1]) for _ in range(v)]

        def solution_fixed_random_speed(v):
            starting_side = random.choice([0, 1])
            speed = random.randint(1, 10)

            def pos(i):
                nonlocal speed

                p = (i % speed) / speed
                return p if (int(i / speed) + starting_side) % 2 == 0 else 1 - p

            return [pos(i) for i in range(v)]

        def solution_random_speed(v):
            offset = 0
            speed = random.randint(1, 10)
            start_dir = random.choice([False, True])

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

        def solution_from_pool(_):
            return random.choice(self.solution_pool)

        weighted_choices = [
            (solution_random, 5),
            (solution_full_speed, 2),
            (solution_random_end, 5),
            (solution_fixed_random_speed, 5),
            (solution_random_speed, 5)
        ]

        if self.solution_pool is not None:
            weighted_choices.append((solution_from_pool, 15))

        new_solution.variables = random.choices(
            [c[0] for c in weighted_choices], weights=[c[1] for c in weighted_choices]
        )[0](self.number_of_variables)

        return new_solution

    def variables_to_deposition(self, variables: List[float]):
        return variables_to_deposition_generic(
            self.variables_prefix + variables, self.x_min, self.x_max, self.bed_size_x, self.bed_size_z,
            self.max_timestamp
        )

    def calculate_reference_objectives(self):
        return calculate_reference_objectives_generic(
            bed_size_x=self.bed_size_x, bed_size_z=self.bed_size_z,
            x_min=self.x_min, x_max=self.x_max,
            raw_material=self.material,
            number_of_variables=self.number_of_variables,
            variables_prefix=self.variables_prefix
        )

    def set_variables_prefix(self, variables_prefix):
        self.variables_prefix = variables_prefix
        self.reference = self.calculate_reference_objectives()

    def set_solution_pool(self, solution_pool):
        self.solution_pool = solution_pool
