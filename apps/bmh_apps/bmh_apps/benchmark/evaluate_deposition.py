#!/usr/bin/env python

import argparse
import logging
import math
from datetime import datetime
from typing import List

import numpy as np
from pandas import DataFrame

from bmh.benchmark import core
from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import MaterialMeta, DepositionMeta, Deposition
from bmh.helpers.identifiers import get_identifier
from bmh.optimization.optimization import DepositionOptimizer
from ..helpers.configure_logging import configure_logging


def set_chevron_deposition(identifier: str, material_meta: MaterialMeta, deposition_meta: DepositionMeta,
                           chevron_layers: int, starting_side: int = 0) -> None:
    """
    Set the deposition to traditional Chevron stacking in layers
    :param identifier: identifier of this deposition computation
    :param material_meta: material description to current knowledge which will be stacked
    :param deposition_meta: deposition meta to which the deposition data will be added
    :param chevron_layers: amount of layers for Chevron stacking
    :param starting_side: set 0 for same side as reclaimer or 1 for opposite side
    """
    core_length = deposition_meta.bed_size_x - deposition_meta.bed_size_z

    deposition_data = DataFrame({
        'timestamp': [material_meta.time * layer / chevron_layers for layer in range(0, chevron_layers + 1)],
        'x': [0.5 * deposition_meta.bed_size_z + core_length * float((layer + starting_side) % 2) for layer in
              range(0, chevron_layers + 1)],
        'z': [0.5 * deposition_meta.bed_size_z] * (chevron_layers + 1),
    })
    deposition_meta.data = Deposition(data=deposition_data, meta=deposition_meta)
    deposition_meta.label = f'{identifier} - Chevron {chevron_layers} layers'


def set_optimized_deposition(identifier: str, material_meta: MaterialMeta, deposition_meta: DepositionMeta,
                             chevron_layers: int, objectives: List[str]) -> None:
    """
    Optimize the deposition regarding the material information provided
    :param identifier: identifier of this deposition computation
    :param material_meta: material description to current knowledge which will be stacked
    :param deposition_meta: deposition meta to which the deposition data will be added
    :param chevron_layers: amount of layers for Chevron stacking for speed determination
    :param objectives: list of objectives which will be optimized
    """
    material = material_meta.get_material()

    x_min = 0.5 * deposition_meta.bed_size_z
    x_max = deposition_meta.bed_size_x - x_min

    optimizer = DepositionOptimizer(
        deposition_meta=deposition_meta,
        x_min=x_min,
        x_max=x_max,
        population_size=250,
        max_evaluations=25000,
        v_max=0.1,
        parameter_labels=material.get_parameter_columns(),
        objectives=objectives,
    )
    optimizer.run(
        material=material,
        variables=chevron_layers + 1
    )

    results = optimizer.get_final_results()
    selection = min(results, key=lambda r: np.sum(np.square(r.objectives)))

    selection.deposition.meta = deposition_meta
    deposition_meta.data = selection.deposition
    deposition_meta.label = f'{identifier} - Optimized {chevron_layers + 1} variables'


def compute_deposition(identifier: str, material_meta: MaterialMeta, objectives: List[str]) -> DepositionMeta:
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

    # Use as much time for reclaiming, as for stacking
    reclaim_x_per_s = bed_size_x / material_meta.time

    deposition_meta = DepositionMeta(identifier, path='', meta_dict={
        'label': f'Computed {identifier}',
        'description': f'Computed deposition for {identifier}',
        'category': 'computed',
        'time': material_meta.time,
        'data': BenchmarkData.DATA_CSV,
        'bed_size_x': bed_size_x,
        'bed_size_z': bed_size_z,
        'reclaim_x_per_s': reclaim_x_per_s
    })

    # Currently fixed amount of layers for Chevron stacking
    set_optimized_deposition(identifier, material_meta, deposition_meta, chevron_layers=60, objectives=objectives)

    return deposition_meta


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

    # Compute and evaluate deposition
    deposition_meta = compute_deposition(
        identifier=identifier,
        material_meta=material_meta,
        objectives=args.objectives
    )

    core.process(
        identifier=identifier,
        material_meta=material_meta,
        deposition_meta=deposition_meta,
        simulator_meta=simulator_meta,
        path=args.path,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        benchmark.write_deposition(deposition_meta)

    logger.info('Processing finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Determine deposition and evaluate data for a given material curve'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Benchmark path')
    parser.add_argument('--dry_run', action='store_true', help='Do not write files')
    parser.add_argument('--sim', type=str, default='bsl_low', help='Simulator identifier')
    parser.add_argument('--material', type=str, default='generated_2Y45', help='Material curve identifier')
    parser.add_argument('--objectives', type=str, nargs='+', help='Objectives to optimize')

    main(parser.parse_args())
