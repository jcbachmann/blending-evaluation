import math
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from pandas import DataFrame

from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition, DepositionMeta
from bmh.helpers.reclaimed_material_evaluator import ReclaimedMaterialEvaluator
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator


def process_material_deposition(material: Material, deposition: Deposition, ppm3: float) -> Material:
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


def get_chevron_deposition(x_min: float, x_max: float, max_timestamp: float, v: float, deposition_meta: DepositionMeta):
    x_diff = x_max - x_min
    time_per_layer = x_diff / v
    layers = max_timestamp / time_per_layer
    steps = int(math.ceil(layers)) + 1

    def get_chevron():
        last_at_min = False
        positions = []
        for i in range(steps):
            if last_at_min:
                if i == steps - 1:
                    positions.append(x_min + x_diff * (layers % 1))
                else:
                    positions.append(x_max)
                last_at_min = False
            else:
                if i == steps - 1:
                    positions.append(x_max - x_diff * (layers % 1))
                else:
                    positions.append(x_min)
                last_at_min = True

        return positions

    deposition = Deposition(
        meta=deposition_meta.copy(),
        data=DataFrame({
            'timestamp': np.append(np.arange(0, max_timestamp, time_per_layer), max_timestamp),
            'x': get_chevron(),
            'z': [deposition_meta.bed_size_z / 2] * steps,
        })
    )

    deposition.meta.time = deposition.data['timestamp'].values[-1]

    return deposition


def get_chevron_ideal_deposition(variables: List[float], *, x_min: float, x_max: float, max_timestamp: float,
                                 v_max: float, deposition_meta: DepositionMeta):
    layers = len(variables) - 1
    move_time_v_max_per_layer = (x_max - x_min) / v_max
    move_time_v_max_total = move_time_v_max_per_layer * layers
    stand_time_count = len(variables)
    stand_time_max_total = max_timestamp - move_time_v_max_total
    stand_time_max = stand_time_max_total / stand_time_count
    stand_time_total = sum(variables) * stand_time_max
    move_time_total = max_timestamp - stand_time_total
    move_time_per_layer = move_time_total / layers

    def get_timestamps():
        timestamps: List[float] = [0.0, stand_time_max * variables[0]]

        for i in range(1, len(variables)):
            timestamps.append(timestamps[-1] + move_time_per_layer)
            timestamps.append(timestamps[-1] + stand_time_max * variables[i])

        return timestamps

    def get_positions():
        last_at_min = False
        positions = []
        for i in range(len(variables)):
            if last_at_min:
                positions.extend([x_max, x_max])
                last_at_min = False
            else:
                positions.extend([x_min, x_min])
                last_at_min = True

        return positions

    deposition = Deposition(
        meta=deposition_meta.copy(),
        data=DataFrame({
            'timestamp': get_timestamps(),
            'x': get_positions(),
            'z': [deposition_meta.bed_size_z / 2] * (2 * len(variables)),
        })
    )

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
        data = pd.concat(
            [data, DataFrame({'timestamp': [t], 'x': [x], 'z': [z_center]})],
            ignore_index=True,
            sort=False
        )

    deposition = Deposition(meta=deposition_meta.copy(), data=data)
    deposition.meta.data = deposition
    deposition.meta.time = deposition.data['timestamp'].values[-1]
    return deposition


def calculate_reference_objectives(reclaimed_material: Material) -> Dict[str, float]:
    reclaimed_evaluator = ReclaimedMaterialEvaluator(reclaimed_material)
    chevron_parameter_stdev = reclaimed_evaluator.get_parameter_stdev()

    # Set the maximum acceptable volume standard deviation to a factor of two for all slices
    volume_per_slice = reclaimed_material.get_volume() / reclaimed_evaluator.get_slice_count()
    worst_acceptable_volume_stdev = 1.0 / 6.0 * volume_per_slice

    return {
        **chevron_parameter_stdev,
        'F2': worst_acceptable_volume_stdev,
    }


class HomogenizationProblem(FloatProblem):
    def __init__(self, *, deposition_meta: DepositionMeta, x_min: float, x_max: float, material: Material,
                 number_of_variables: int = 2, deposition_prefix: Deposition = None, v_max: float, ppm3: float,
                 timestamps: Optional[List[float]] = None, objectives: List[str] = None):
        super(HomogenizationProblem, self).__init__()

        # Copy parameters
        self.deposition_meta = deposition_meta
        self.x_min = x_min
        self.x_max = x_max
        self.material = material
        self.number_of_variables = number_of_variables
        self.deposition_prefix: Optional[Deposition] = deposition_prefix
        self.v_max = v_max
        self.ppm3 = ppm3
        self.timestamps = timestamps
        self.objectives = objectives

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
        self.reference_reclaimed_material = process_material_deposition(
            self.material, self.reference_deposition, ppm3=self.ppm3
        )
        # Biased absolute reference objectives
        self.reference_objectives = calculate_reference_objectives(self.reference_reclaimed_material)
        # Objectives of the reference deposition relative to the reference objectives
        self.reference_deposition_objectives = self.evaluate_reclaimed_material(self.reference_reclaimed_material)

        # Setup problem base variables
        self.number_of_objectives = len(objectives)
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
        self.obj_labels = self.get_objective_labels()

        self.lower_bound = [0.0 for _ in range(number_of_variables)]
        self.upper_bound = [1.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def get_name(self) -> str:
        return "Homogenization Problem"

    def evaluate_objective(self, deposition: Deposition, reclaimed_material: Material, objective: str) -> float:
        objective_type = objective.split('/')[0]
        if objective_type == 'F1':
            evaluator = ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min=self.x_min, x_max=self.x_max)
            return evaluator.get_single_parameter_stdev(objective.split('/')[1]) / self.reference_objectives[objective]
        elif objective_type == 'F2':
            evaluator = ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min=self.x_min, x_max=self.x_max)
            return evaluator.get_volume_stdev() / self.reference_objectives[objective]
        elif objective_type == 'F3':
            return self.evaluate_distance_travelled(deposition)
        elif objective_type == 'F4':
            return self.evaluate_max_speed(deposition)

        raise ValueError(f"Unknown objective: {objective_type}")

    def evaluate(self, solution: FloatSolution) -> None:
        deposition = self.variables_to_deposition(variables=solution.variables)
        reclaimed_material = process_material_deposition(material=self.material, deposition=deposition, ppm3=self.ppm3)
        solution.objectives = [
            self.evaluate_objective(deposition, reclaimed_material, objective)
            for objective in self.objectives
        ]

    def evaluate_reclaimed_material(self, reclaimed_material: Material) -> Dict[str, float]:
        return ReclaimedMaterialEvaluator.get_relative(
            ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min=self.x_min,
                                       x_max=self.x_max).get_all_stdev(),
            self.reference_objectives
        )

    def evaluate_distance_travelled(self, deposition: Deposition) -> float:
        return deposition.data['x'].diff().abs().sum()  # FIXME normalize correctly with self.v_max

    def evaluate_max_speed(self, deposition: Deposition) -> float:
        return deposition.data['x'].diff().abs().max() / self.v_max  # FIXME normalize correctly with self.v_max

    def get_objective_labels(self) -> List[str]:
        return self.objectives

    def variables_to_deposition(self, variables: List[float]) -> Deposition:
        return variables_to_deposition_generic(
            variables, x_min=self.x_min, x_max=self.x_max, max_timestamp=self.max_timestamp, v_max=self.v_max,
            deposition_meta=self.deposition_meta, deposition_prefix=self.deposition_prefix,
            timestamps=self.timestamps
        )
        # return get_chevron_deposition(
        #     x_min=self.x_min, x_max=self.x_max, max_timestamp=self.max_timestamp, v=self.v_max * variables[0],
        #     deposition_meta=self.deposition_meta
        # )
        # return get_chevron_ideal_deposition(
        #     variables, x_min=self.x_min, x_max=self.x_max, max_timestamp=self.max_timestamp, v_max=self.v_max,
        #     deposition_meta=self.deposition_meta
        # )

    def get_reference_relative(self) -> Tuple[Deposition, Material, Dict[str, float]]:
        return self.reference_deposition, self.reference_reclaimed_material, self.reference_deposition_objectives
