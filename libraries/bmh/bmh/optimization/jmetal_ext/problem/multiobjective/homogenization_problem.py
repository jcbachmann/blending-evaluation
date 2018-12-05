import random
from typing import List, Optional, Callable, Tuple

import numpy as np
from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition, DepositionMeta
from bmh.helpers.math import weighted_avg_and_std, stdev
from bmh.helpers.stockpile_math import get_stockpile_height
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from pandas import DataFrame


def process_material_deposition(material: Material, deposition: Deposition, ppm3: float = 0.125) -> Material:
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
        variables: List[float], *, x_min: float, x_max: float, max_timestamp: float,
        deposition_meta: DepositionMeta, deposition_prefix: Optional[Deposition] = None
) -> Deposition:
    if deposition_prefix and deposition_prefix.data.shape[0] > 0:
        start_timestamp = deposition_prefix.data['timestamp'].values[-1]
        min_timestamp = start_timestamp + (max_timestamp - start_timestamp) / len(variables)
    else:
        min_timestamp = 0.0

    deposition = Deposition(
        meta=deposition_meta.copy(),
        data=DataFrame({
            'x': [elem * (x_max - x_min) + x_min for elem in variables],
            'z': [deposition_meta.bed_size_z / 2] * len(variables),
            'timestamp': np.linspace(min_timestamp, max_timestamp, len(variables))
        })
    )

    if deposition_prefix and deposition_prefix.data.shape[0] > 0:
        deposition.data = deposition_prefix.data.append(deposition.data, ignore_index=True, sort=False)

    deposition.meta.time = deposition.data['timestamp'].values[-1]

    return deposition


def calculate_reference_objectives_generic(
        x_min: float, x_max: float, raw_material: Material, number_of_variables: int,
        deposition_meta: DepositionMeta, deposition_prefix: Optional[Deposition]
) -> List[float]:
    max_timestamp = raw_material.data['timestamp'].values[-1]

    # Generate a Chevron deposition with maximum speed
    chevron = [float(i % 2) for i in range(number_of_variables)]
    chevron_deposition = variables_to_deposition_generic(
        variables=chevron, x_min=x_min, x_max=x_max, max_timestamp=max_timestamp,
        deposition_meta=deposition_meta, deposition_prefix=deposition_prefix
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
        self._core_data: DataFrame = None
        self._parameter_stdev: Optional[List[float]] = None
        self._volume_stdev: Optional[float] = None

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

    def get_parameter_stdev(self) -> List[float]:
        if self._parameter_stdev is None:
            reclaimed_df = self.reclaimed.data
            cols = self.reclaimed.get_parameter_columns()
            self._parameter_stdev = [weighted_avg_and_std(reclaimed_df[col], reclaimed_df['volume'])[1] for col in cols]
        return self._parameter_stdev

    def get_all_stdev(self) -> List[float]:
        return self.get_parameter_stdev() + [self.get_core_volume_stdev()]

    def get_all_stdev_relative(self, reference: List[float]) -> List[float]:
        return [s / r for s, r in zip(self.get_all_stdev(), reference)]

    def get_slice_count(self) -> int:
        return self.reclaimed.data['volume'].shape[0]


class HomogenizationProblem(FloatProblem):
    def __init__(self, *, deposition_meta: DepositionMeta, x_min: float, x_max: float, material: Material,
                 number_of_variables: int = 2):
        super().__init__()

        # Copy parameters
        self.deposition_meta = deposition_meta
        self.x_min = x_min
        self.x_max = x_max
        self.material = material
        self.number_of_variables = number_of_variables

        # Buffer values
        self.max_timestamp = material.data['timestamp'].values[-1]
        self.deposition_prefix: Optional[Deposition] = None
        self.solution_pool: Optional[List[List[float]]] = None

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

        def solution_random(v: int) -> List[float]:
            return [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in range(v)]

        def solution_full_speed(v: int) -> List[float]:
            starting_side = random.choice([0, 1])
            return [(i + starting_side) % 2 for i in range(v)]

        def solution_random_end(v: int) -> List[float]:
            return [random.choice([0, 1]) for _ in range(v)]

        def solution_fixed_random_speed(v: int) -> List[float]:
            starting_side = random.choice([0, 1])
            speed = random.randint(1, 10)

            def pos(i):
                nonlocal speed

                p = (i % speed) / speed
                return p if (int(i / speed) + starting_side) % 2 == 0 else 1 - p

            return [pos(i) for i in range(v)]

        def solution_random_speed(v: int) -> List[float]:
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

        def solution_from_pool(_: int) -> List[float]:
            return random.choice(self.solution_pool) if self.solution_pool else []

        weighted_choices: List[Tuple[Callable[[int], List[float]], int]] = [
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

    def variables_to_deposition(self, variables: List[float]) -> Deposition:
        return variables_to_deposition_generic(
            variables, x_min=self.x_min, x_max=self.x_max, max_timestamp=self.max_timestamp,
            deposition_meta=self.deposition_meta, deposition_prefix=self.deposition_prefix
        )

    def calculate_reference_objectives(self) -> List[float]:
        return calculate_reference_objectives_generic(
            x_min=self.x_min, x_max=self.x_max,
            raw_material=self.material,
            number_of_variables=self.number_of_variables,
            deposition_meta=self.deposition_meta, deposition_prefix=self.deposition_prefix
        )

    def set_deposition_prefix(self, deposition_prefix: Deposition) -> None:
        self.deposition_prefix = deposition_prefix
        self.reference = self.calculate_reference_objectives()

    def set_solution_pool(self, solution_pool: List[List[float]]) -> None:
        self.solution_pool = solution_pool
