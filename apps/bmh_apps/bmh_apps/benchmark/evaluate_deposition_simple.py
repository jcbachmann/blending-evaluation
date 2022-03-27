#!/usr/bin/env python
import argparse
import logging
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from bmh.benchmark import core
from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import MaterialMeta, DepositionMeta, MaterialDeposition, Deposition
from bmh.benchmark.simulator_meta import SimulatorMeta
from bmh.helpers.identifiers import get_identifier
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from ..helpers.configure_logging import configure_logging


def compute_deposition1(identifier: str, material_meta: MaterialMeta, bed_size_x: float, bed_size_z: float,
                        x_min: float,
                        x_max: float, layers: int) -> DepositionMeta:
    z_center = bed_size_z / 2
    time_per_layer = material_meta.time / layers
    # volume_per_layer = material_meta.volume / layers
    # core_length = x_max - x_min
    # height_first_layer = get_stockpile_height(volume_per_layer, core_length)
    # volume_cone = math.pi / 3.0 * height_first_layer ** 3.0
    # time_before_start = time_per_layer * (volume_cone / volume_per_layer)
    # time_before_end = time_per_layer * (volume_cone / volume_per_layer) * 0.18

    data = DataFrame({'timestamp': [0.0], 'x': [x_min], 'z': [z_center], })
    for layer in range(layers):
        # height_before_this_layer = get_stockpile_height(layer * volume_per_layer, core_length)
        # height_including_this_layer = get_stockpile_height((layer + 1) * volume_per_layer, core_length)
        # height_only_this_layer = height_including_this_layer - height_before_this_layer
        t_start = layer * time_per_layer
        x = x_min if layer % 2 == 0 else x_max
        data = data.append(
            DataFrame({'timestamp': [t_start], 'x': [x], 'z': [z_center]}),
            ignore_index=True, sort=False
        )
        # f = pow(height_including_this_layer / height_only_this_layer, 0.288)
        # data = data.append(
        #     DataFrame({'timestamp': [t_start + time_before_start * f], 'x': [x], 'z': [z_center]}),
        #     ignore_index=True, sort=False
        # )
        x = x_max if layer % 2 == 0 else x_min
        # f = pow(height_including_this_layer / height_only_this_layer, 0.7)
        # data = data.append(
        #     DataFrame({'timestamp': [t_start + time_per_layer - time_before_end * f], 'x': [x],
        #                'z': [z_center]}),
        #     ignore_index=True, sort=False
        # )
        data = data.append(
            DataFrame({'timestamp': [t_start + time_per_layer], 'x': [x], 'z': [z_center]}),
            ignore_index=True, sort=False
        )

    deposition_meta = DepositionMeta(identifier, path='', meta_dict={
        'label': f'Computed {identifier}',
        'description': f'Computed deposition for {identifier}',
        'category': 'computed',
        'time': material_meta.time,
        'data': BenchmarkData.DATA_CSV,
        'bed_size_x': bed_size_x,
        'bed_size_z': bed_size_z,
        'reclaim_x_per_s': bed_size_x / material_meta.time
    })
    deposition = Deposition(meta=deposition_meta, data=data)
    deposition.meta.data = deposition
    return deposition.meta


def compute_deposition2(identifier: str, material_meta: MaterialMeta, bed_size_x: float, bed_size_z: float,
                        x_min: float,
                        x_max: float, layers: int) -> DepositionMeta:
    z_center = bed_size_z / 2
    time_per_layer = material_meta.time / layers
    volume_per_layer = material_meta.volume / layers
    core_length = x_max - x_min
    height_first_layer = get_stockpile_height(volume_per_layer, core_length)
    volume_cone = math.pi / 3.0 * height_first_layer ** 3.0
    time_before_start = time_per_layer * (volume_cone / volume_per_layer)
    total_height = get_stockpile_height(material_meta.volume, core_length)

    data = DataFrame({'timestamp': [0.0], 'x': [x_min], 'z': [z_center], })
    for layer in range(layers):
        height_before_this_layer = get_stockpile_height(layer * volume_per_layer, core_length)
        height_including_this_layer = get_stockpile_height((layer + 1) * volume_per_layer, core_length)
        height_only_this_layer = height_including_this_layer - height_before_this_layer
        offset = total_height - height_including_this_layer
        t_start = layer * time_per_layer
        f = 1.0
        x = x_min - offset if layer % 2 == 0 else x_max + offset
        data = data.append(
            DataFrame({'timestamp': [t_start + time_before_start * f], 'x': [x], 'z': [z_center]}),
            ignore_index=True, sort=False
        )
        x = x_min - offset if layer % 2 == 1 else x_max + offset
        data = data.append(
            DataFrame({'timestamp': [t_start + time_per_layer], 'x': [x], 'z': [z_center]}),
            ignore_index=True, sort=False
        )

    deposition_meta = DepositionMeta(identifier, path='', meta_dict={
        'label': f'Computed {identifier}',
        'description': f'Computed deposition for {identifier}',
        'category': 'computed',
        'time': material_meta.time,
        'data': BenchmarkData.DATA_CSV,
        'bed_size_x': bed_size_x,
        'bed_size_z': bed_size_z,
        'reclaim_x_per_s': bed_size_x / material_meta.time
    })
    deposition = Deposition(meta=deposition_meta, data=data)
    deposition.meta.data = deposition
    return deposition.meta


def process_material(identifier: str, material_meta: MaterialMeta, deposition_meta: DepositionMeta,
                     simulator_meta: SimulatorMeta) -> MaterialMeta:
    logger = logging.getLogger(__name__)
    logger.info(f'Processing "{identifier}" with material "{material_meta}" and deposition "{deposition_meta}" using '
                f'simulator type {simulator_meta.type}')

    logger.debug('Creating simulator')
    sim_params = simulator_meta.get_params().copy()
    sim_params['bed_size_x'] = deposition_meta.bed_size_x
    sim_params['bed_size_z'] = deposition_meta.bed_size_z
    sim_params['ppm3'] = 27.0
    sim = simulator_meta.get_type()(**sim_params)

    logger.debug('Combining material and deposition')
    material_deposition = MaterialDeposition(material_meta.get_material(), deposition_meta.get_deposition())
    logger.debug(f'Material and deposition combined:\n{material_deposition.data.describe()}')

    logger.debug('Stacking and reclaiming material')
    reclaimed_material = sim.stack_reclaim(material_deposition)
    logger.debug(f'Reclaimed material:\n{reclaimed_material.data.describe()}')

    logger.info('Processing finished')

    return reclaimed_material.meta


def get_ideal_reclaimed_material(reclaimed_meta: MaterialMeta, x_min: float, x_max: float) -> MaterialMeta:
    ideal = reclaimed_meta.data.copy()
    for p in ideal.get_parameter_columns():
        avg = np.average(ideal.data[p], weights=ideal.data['volume'])
        ideal.data[p] = avg
    height = get_stockpile_height(ideal.data['volume'].sum(), x_max - x_min)
    ideal.data['x_diff'] = (ideal.data['x'] - ideal.data['x'].shift(1)).fillna(0.0)
    ideal.data['volume'] = ideal.data.apply(
        lambda row: get_stockpile_slice_volume(
            row['x'], x_max - x_min, height, x_min, row['x_diff']
        ), axis=1
    )
    return ideal.meta


def main(args: argparse.Namespace):
    # Setup logging
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Setup timestamp identifier
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.info(f'Starting evaluation with timestamp {timestamp_str}')

    # Load benchmark base data
    benchmark = BenchmarkData(args.path)
    benchmark.read_base()

    # Acquire material
    material_identifier = get_identifier(args.material)
    material_meta = benchmark.get_material_meta(material_identifier)

    # Acquire and test simulator
    sim_identifier = get_identifier(args.sim)
    simulator_meta = benchmark.get_simulator_meta(sim_identifier)
    core.test_simulator(simulator_meta)

    # Set identifier
    identifier = f'{timestamp_str} {material_identifier} {sim_identifier}'

    # Assumptions:
    # - maximum stockpile height 20m
    # - angle of repose: 45 degrees
    max_stockpile_height = 20

    # To allow flexibility for optimization the required space is overestimated by 25%
    max_volume = material_meta.volume * 1.25

    # Compute the core length of the stockpile by matching the maximum volume with the cone + core volume
    core_length = (max_volume - math.pi * math.pow(max_stockpile_height, 3) / 3) / math.pow(max_stockpile_height, 2)
    bed_size_x = core_length + 2 * max_stockpile_height
    bed_size_z = 2 * max_stockpile_height
    x_min = 0.5 * bed_size_z
    x_max = bed_size_x - 0.5 * bed_size_z

    layers = 15

    # Compute and evaluate deposition
    deposition_meta = compute_deposition2(
        identifier=identifier,
        material_meta=material_meta,
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        x_min=x_min,
        x_max=x_max,
        layers=layers
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('position')
    ax.set_ylabel('volume')

    for layer in range(layers):
        t_max = material_meta.time * (layer + 1) / layers
        material_copy = material_meta.get_material().copy()
        material_copy.meta.time = t_max
        material_copy.data = material_copy.data[material_copy.data['timestamp'] < t_max]
        reclaimed_meta = process_material(
            identifier=identifier,
            material_meta=material_copy.meta,
            deposition_meta=deposition_meta,
            simulator_meta=simulator_meta
        )
        ideal_meta = get_ideal_reclaimed_material(reclaimed_meta, x_min, x_max)

        ax.plot(reclaimed_meta.data.data['x'], reclaimed_meta.data.data['volume'], color='red', marker='',
                linestyle='-')
        ax.plot(ideal_meta.data.data['x'], ideal_meta.data.data['volume'], color='green', marker='', linestyle='-')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Determine deposition and evaluate data for a given material curve'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Benchmark path')
    parser.add_argument('--sim', type=str, default='bsl_mid', help='Simulator identifier')
    parser.add_argument('--material', type=str, default='generated_2Y45', help='Material curve identifier')

    main(parser.parse_args())
