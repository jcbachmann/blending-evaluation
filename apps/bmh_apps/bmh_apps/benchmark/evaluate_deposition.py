#!/usr/bin/env python

import argparse
import json
import logging
import math
import os
from datetime import datetime

from bmh.benchmark import core
from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import MaterialMeta, DepositionMeta, Deposition
from bmh.helpers.identifiers import get_identifier
from bmh.optimization.optimization import optimize
from pandas import DataFrame

from bmh_apps.helpers.configure_logging import configure_logging


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
        'timestamp': [material_meta.time * l / chevron_layers for l in range(0, chevron_layers + 1)],
        'x': [0.5 * deposition_meta.bed_size_z + core_length * float((l + starting_side) % 2) for l in
              range(0, chevron_layers + 1)],
        'z': [0.5 * deposition_meta.bed_size_z] * (chevron_layers + 1),
    })
    deposition_meta.data = Deposition(data=deposition_data, meta=deposition_meta)
    deposition_meta.label = f'{identifier} - Chevron {chevron_layers} layers'


def set_optimized_deposition(identifier: str, material_meta: MaterialMeta, deposition_meta: DepositionMeta,
                             chevron_layers: int, starting_side: int = 0) -> None:
    """
    Optimize the deposition regarding the material information provided
    :param identifier: identifier of this deposition computation
    :param material_meta: material description to current knowledge which will be stacked
    :param deposition_meta: deposition meta to which the deposition data will be added
    :param chevron_layers: amount of layers for Chevron stacking for speed determination
    :param starting_side: set 0 for same side as reclaimer or 1 for opposite side
    """
    material = material_meta.get_material()

    # TODO respect starting side
    # TODO use same simulator?
    optimization_result = optimize(
        length=deposition_meta.bed_size_x,
        depth=deposition_meta.bed_size_z,
        variables=chevron_layers + 1,
        material=material.data,
        parameter_columns=material.get_parameter_columns(),
        population_size=250,
        max_evaluations=25000
    )

    result_population = optimization_result.result_population
    result_population.sort(key=lambda s: s.objectives[0])
    chosen_solution = result_population[0]

    core_length = deposition_meta.bed_size_x - deposition_meta.bed_size_z

    deposition_data = DataFrame({
        'timestamp': [material_meta.time * l / chevron_layers for l in range(0, chevron_layers + 1)],
        'x': [0.5 * deposition_meta.bed_size_z + core_length * p for p in chosen_solution.variables],
        'z': [0.5 * deposition_meta.bed_size_z] * (chevron_layers + 1),
    })
    deposition_meta.data = Deposition(data=deposition_data, meta=deposition_meta)
    deposition_meta.label = f'{identifier} - Optimized {chevron_layers + 1} variables'


def compute_deposition(identifier: str, material_meta: MaterialMeta) -> DepositionMeta:
    # TODO v3 Determine prediction from material curve
    # TODO v3 Store prediction

    # TODO v3 Optimized deposition based on prediction - only one optimization before stacking
    # TODO v4 Optimized deposition based on prediction - optimize every 5 simulation minutes

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

    deposition = DepositionMeta(identifier, path='', meta_dict={
        'label': f'Computed {identifier}',
        'description': f'Computed deposition for {identifier}',
        'category': 'computed',
        'time': material_meta.time,
        'data': core.DATA_CSV,
        'bed_size_x': bed_size_x,
        'bed_size_z': bed_size_z,
        'reclaim_x_per_s': reclaim_x_per_s
    })

    # Currently fixed amount of layers for Chevron stacking
    set_optimized_deposition(identifier, material_meta, deposition, chevron_layers=60)

    return deposition


def write_deposition(identifier: str, deposition_meta: DepositionMeta, dst: str, dry_run: bool):
    logger = logging.getLogger(__name__)

    deposition_directory = os.path.join(dst, identifier, core.COMPUTED_DEPOSITION_DIR)

    logger.debug(f'Creating directory "{deposition_directory}"')
    if not dry_run:
        os.mkdir(deposition_directory)

    deposition_meta_file = os.path.join(deposition_directory, BenchmarkData.META_JSON)
    logger.debug(f'Writing reclaimed material meta to "{deposition_meta_file}"')
    if not dry_run:
        json.dump(
            deposition_meta.to_dict(),
            open(deposition_meta_file, 'w'),
            indent=4
        )

    deposition = deposition_meta.get_deposition()
    data_file = os.path.join(deposition_directory, core.DATA_CSV)
    logger.debug(f'Writing material data to "{data_file}"')
    if not dry_run:
        deposition.data.to_csv(data_file, sep='\t', index=False)


def main(args: argparse.Namespace):
    # Setup logging
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Setup timestamp identifier
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.info(f'Starting evaluation with timestamp {timestamp_str}')

    # Load benchmark base data
    benchmark = BenchmarkData()
    benchmark.read_base(args.path)

    # Acquire material
    material_identifier = get_identifier(args.material)
    material_meta = benchmark.get_material_meta(material_identifier)

    # Acquire and test simulator
    sim_identifier = get_identifier(args.sim)
    simulator_meta = benchmark.get_simulator_meta(sim_identifier)
    core.test_simulator(simulator_meta)

    # Prepare output directory
    identifier = f'{timestamp_str} {material_identifier} {sim_identifier}'
    dst = os.path.join(args.dst, identifier)
    core.prepare_dst(dst, args.dry_run)

    # Compute and evaluate deposition
    deposition_meta = compute_deposition(
        identifier=identifier,
        material_meta=material_meta
    )

    core.process(
        identifier=identifier,
        material_meta=material_meta,
        deposition_meta=deposition_meta,
        simulator_meta=simulator_meta,
        dst=dst,
        dry_run=args.dry_run,
        computed_deposition=True
    )

    write_deposition(
        identifier=identifier,
        deposition_meta=deposition_meta,
        dst=dst,
        dry_run=args.dry_run
    )

    logger.info('Processing finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Determine deposition and evaluate data for a given material curve'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--dst', default='.', help='Path where results will be stored')
    parser.add_argument('--dry_run', action='store_true', help='Do not write files')
    parser.add_argument('--sim', type=str, default='bsl_low', help='Simulator identifier')
    parser.add_argument('--material', type=str, default='generated_2Y45', help='Material curve identifier')

    main(parser.parse_args())
