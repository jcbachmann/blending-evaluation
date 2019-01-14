import math
from typing import List, Optional, Tuple

import numpy as np
from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition, DepositionMeta
from bmh.helpers.math import weighted_avg_and_std, stdev
from bmh.helpers.stockpile_math import get_stockpile_height
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from pandas import DataFrame

from .solution_generator import SolutionGenerator, RandomSolutionGenerator


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


def verify_timestamps(timestamps: List[float], *, number_of_variables: int, max_timestamp: float,
                      deposition_prefix: Deposition = None):
    if len(timestamps) != number_of_variables:
        raise ValueError(f'Length of timestamps {len(timestamps)}'
                         f' does not match length of variables {number_of_variables}')

    if deposition_prefix and deposition_prefix.data.shape[0] > 0:
        if timestamps[0] <= deposition_prefix.data['timestamp'].values[-1]:
            raise ValueError(f'First timestamp {timestamps[0]} must be greater than '
                             f"last deposition prefix timestamp {deposition_prefix.data['timestamp'].values[-1]}")
    else:
        if timestamps[0] != 0.0:
            raise ValueError(f'First timestamp {timestamps[0]} != 0.0')

    if timestamps[-1] != max_timestamp:
        raise ValueError(f'Last timestamp {timestamps[-1]} does not match max timestamp {max_timestamp}')


def variables_to_deposition_generic(
        variables: List[float], *, x_min: float, x_max: float, max_timestamp: float, v_max: float,
        deposition_meta: DepositionMeta, deposition_prefix: Optional[Deposition] = None,
        timestamps: Optional[List[float]] = None
) -> Deposition:
    if deposition_prefix and deposition_prefix.data.shape[0] > 0:
        start_timestamp = deposition_prefix.data['timestamp'].values[-1]
        min_timestamp = start_timestamp + (max_timestamp - start_timestamp) / len(variables)
    else:
        min_timestamp = 0.0

    deposition = Deposition(
        meta=deposition_meta.copy(),
        data=DataFrame({
            'timestamp': timestamps if timestamps else np.linspace(min_timestamp, max_timestamp, len(variables)),
            'x': [elem * (x_max - x_min) + x_min for elem in variables],
            'z': [deposition_meta.bed_size_z / 2] * len(variables),
        })
    )

    # Check and fix speed always below v_max
    if deposition_prefix and deposition_prefix.data.shape[0] > 0:
        t_last = deposition_prefix.data['timestamp'].values[-1]
        x_last = deposition_prefix.data['x'].values[-1]
    else:
        t_last = deposition.data['timestamp'].values[0]
        x_last = deposition.data['x'].values[0]
    for i in deposition.data.index:
        t = deposition.data.at[i, 'timestamp']
        x = deposition.data.at[i, 'x']
        x_diff_max = v_max * (t - t_last)
        x_diff = x - x_last
        if abs(x_diff) > x_diff_max:
            x = x_last + math.copysign(x_diff_max, x_diff)
        deposition.data.at[i, 'x'] = x
        t_last = t
        x_last = x

    if deposition_prefix and deposition_prefix.data.shape[0] > 0:
        deposition.data = deposition_prefix.data.append(deposition.data, ignore_index=True, sort=False)

    deposition.meta.time = deposition.data['timestamp'].values[-1]

    return deposition


def get_full_speed_deposition(
        x_min: float, x_max: float, deposition_meta: DepositionMeta, t_max: float, v_max: float
) -> Deposition:
    # Generate a Chevron deposition with maximum speed
    x_center = 0.5 * (x_min + x_max)
    z_center = deposition_meta.bed_size_z / 2
    time_per_section = (x_max - x_min) / v_max

    t = 0.0
    x = x_min
    data = DataFrame({'timestamp': [0.0], 'x': [x_min], 'z': [z_center], })
    while t < t_max:
        t_last = t
        t = min(t + time_per_section, t_max)
        x_diff = (t - t_last) * v_max
        x = x_min + x_diff if x < x_center else x_max - x_diff
        data = data.append(
            DataFrame({'timestamp': [t], 'x': [x], 'z': [z_center]}),
            ignore_index=True, sort=False
        )

    deposition = Deposition(meta=deposition_meta.copy(), data=data)
    deposition.meta.time = deposition.data['timestamp'].values[-1]
    return deposition


def calculate_reference_objectives(reclaimed_material: Material) -> List[float]:
    reclaimed_evaluator = MaterialEvaluator(reclaimed_material)
    chevron_parameter_stdev = reclaimed_evaluator.get_parameter_stdev()

    # Set the maximum acceptable volume standard deviation to a a factor of two for all slices
    volume_per_slice = reclaimed_material.get_volume() / reclaimed_evaluator.get_slice_count()
    worst_acceptable_volume_stdev = stdev(np.array([3.5 / 3.0 * volume_per_slice, 2.5 / 3.0 * volume_per_slice]))

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

    @staticmethod
    def get_relative(objectives: List[float], reference: List[float]) -> List[float]:
        return [s / r for s, r in zip(objectives, reference)]

    def get_slice_count(self) -> int:
        return self.reclaimed.data['volume'].shape[0]


class HomogenizationProblem(FloatProblem):
    def __init__(self, *, deposition_meta: DepositionMeta, x_min: float, x_max: float, material: Material,
                 number_of_variables: int = 2, deposition_prefix: Deposition = None, v_max: float,
                 timestamps: Optional[List[float]] = None,
                 solution_generator: SolutionGenerator = RandomSolutionGenerator()):
        super().__init__()

        # Copy parameters
        self.deposition_meta = deposition_meta
        self.x_min = x_min
        self.x_max = x_max
        self.material = material
        self.number_of_variables = number_of_variables
        self.deposition_prefix: Optional[Deposition] = deposition_prefix
        self.v_max = v_max
        self.timestamps = timestamps
        self.solution_generator = solution_generator

        # Buffer values
        self.max_timestamp = material.data['timestamp'].values[-1]

        # Check timestamps
        if timestamps:
            verify_timestamps(
                self.timestamps, number_of_variables=self.number_of_variables,
                max_timestamp=self.max_timestamp, deposition_prefix=self.deposition_prefix
            )

        # Reference deposition (full speed Chevron deposition)
        self.reference_deposition = get_full_speed_deposition(
            x_min=self.x_min, x_max=self.x_max, deposition_meta=self.deposition_meta,
            t_max=self.max_timestamp, v_max=self.v_max
        )
        reference_reclaimed_material = process_material_deposition(self.material, self.reference_deposition)
        # Biased absolute reference objectives
        self.reference_objectives = calculate_reference_objectives(reference_reclaimed_material)
        # Objectives of the reference deposition relative to the reference objectives
        self.reference_deposition_objectives = self.evaluate_reclaimed_material(reference_reclaimed_material)

        # Setup problem base variables
        self.number_of_objectives = len(material.get_parameter_columns()) + 1
        self.number_of_constraints = 0

        self.lower_bound = [0.0 for _ in range(number_of_variables)]
        self.upper_bound = [1.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> None:
        deposition = self.variables_to_deposition(variables=solution.variables)
        reclaimed_material = process_material_deposition(material=self.material, deposition=deposition)
        solution.objectives = self.evaluate_reclaimed_material(reclaimed_material)

    def evaluate_reclaimed_material(self, reclaimed_material: Material) -> List[float]:
        return MaterialEvaluator.get_relative(
            MaterialEvaluator(reclaimed=reclaimed_material, x_min=self.x_min, x_max=self.x_max).get_all_stdev(),
            self.reference_objectives
        )

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        # if constraint violation should be used set the number of constraints accordingly
        # don't forget that constraints are disregarded in HPSEA
        pass

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
        new_solution.variables = self.solution_generator.gen(self.number_of_variables)
        return new_solution

    def variables_to_deposition(self, variables: List[float]) -> Deposition:
        return variables_to_deposition_generic(
            variables, x_min=self.x_min, x_max=self.x_max, max_timestamp=self.max_timestamp, v_max=self.v_max,
            deposition_meta=self.deposition_meta, deposition_prefix=self.deposition_prefix,
            timestamps=self.timestamps
        )

    def get_reference_relative(self) -> Tuple[Deposition, List[float]]:
        return self.reference_deposition, self.reference_deposition_objectives
