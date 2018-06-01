#!/usr/bin/env python

import argparse
import json
import logging
import math
import os
from datetime import datetime

import pandas as pd

from benchmark import core, helpers
from benchmark.data import BenchmarkData
from benchmark.simulator_meta import SimulatorMeta
from blending_simulator.material_deposition import MaterialMeta, DepositionMeta, Deposition


def set_chevron_deposition(identifier: str, material_meta: MaterialMeta, deposition: DepositionMeta,
                           chevron_layers: int, starting_side: int = 0) -> None:
    """
    Set the deposition to traditional Chevron stacking in layers
    :param identifier: identifier of this deposition computation
    :param material_meta: material description to current knowledge which will be stacked
    :param deposition: deposition meta to which the deposition data will be added
    :param chevron_layers: amount of layers for Chevron stacking
    :param starting_side: set 0 for same side as reclaimer or 1 for opposite side
    """
    core_length = deposition.bed_size_x - deposition.bed_size_z

    deposition_data = pd.DataFrame({
        'timestamp': [material_meta.time * l / chevron_layers for l in range(0, chevron_layers + 1)],
        'x': [0.5 * deposition.bed_size_z + core_length * float((l + starting_side) % 2) for l in
              range(0, chevron_layers + 1)],
        'z': [0.5 * deposition.bed_size_z] * (chevron_layers + 1),
    })
    deposition.data = Deposition(data=deposition_data, meta=deposition)
    deposition.label = f'{identifier} - Chevron {chevron_layers} layers'


def compute_deposition(identifier: str, material_meta: MaterialMeta) -> DepositionMeta:
    # TODO v2 Optimized deposition based on full knowledge - only one optimization before stacking
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
    set_chevron_deposition(identifier, material_meta, deposition, chevron_layers=60)

    return deposition


def process_data(identifier: str, material_meta: MaterialMeta, simulator_meta: SimulatorMeta, dst: str, dry_run: bool):
    logging.info(f'Processing data with simulator "{simulator_meta.type}"')

    logging.debug('Writing simulator type and parameters to destination directory')
    if not dry_run:
        json.dump({'simulator': str(simulator_meta)}, open(os.path.join(dst, core.SIMULATOR_JSON), 'w'), indent=4)

    # Determine prediction from material curve
    # TODO v3 Determine prediction from material curve
    # TODO v3 Store prediction

    # Compute deposition
    deposition_meta = compute_deposition(identifier, material_meta)

    # Process material with computed deposition and selected simulator
    core.process(identifier, material_meta, deposition_meta, simulator_meta, dst, dry_run, computed_deposition=True)

    # Write material deposition
    deposition_directory = os.path.join(dst, identifier, core.COMPUTED_DEPOSITION_DIR)

    logging.debug(f'Creating directory "{deposition_directory}"')
    if not dry_run:
        os.mkdir(deposition_directory)

    deposition_meta_file = os.path.join(deposition_directory, BenchmarkData.META_JSON)
    logging.debug(f'Writing reclaimed material meta to "{deposition_meta_file}"')
    if not dry_run:
        json.dump(
            deposition_meta.to_dict(),
            open(deposition_meta_file, 'w'),
            indent=4
        )

    deposition = deposition_meta.get_deposition()
    data_file = os.path.join(deposition_directory, core.DATA_CSV)
    logging.debug(f'Writing material data to "{data_file}"')
    if not dry_run:
        deposition.data.to_csv(data_file, sep='\t', index=False)


def main(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )

    timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logging.info(f'Starting evaluation with timestamp {timestamp_str}')

    # Initialization
    benchmark = BenchmarkData()
    benchmark.read_base(args.path)

    # Parse and validate material
    material_identifier = helpers.get_identifier(args.material)
    benchmark.validate_material(material_identifier)
    material_meta = benchmark.materials[material_identifier]

    # Parse and validate simulator
    sim_identifier = helpers.get_identifier(args.sim)
    benchmark.validate_simulator(sim_identifier)
    simulator_meta = benchmark.simulators[sim_identifier]
    core.prepare_simulator(simulator_meta)

    # Prepare output directory
    identifier = f'{timestamp_str} {material_identifier} {sim_identifier}'
    dst = os.path.join(args.dst, identifier)
    core.prepare_dst(dst, args.dry_run)

    # Processing
    process_data(
        identifier=identifier,
        material_meta=material_meta,
        simulator_meta=simulator_meta,
        dst=dst,
        dry_run=args.dry_run
    )

    logging.info('Processing finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Determine deposition and evaluate data for a given material curve'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--dst', default='.', help='Path where results will be stored')
    parser.add_argument('--dry_run', action='store_true', help='Do not write files')
    parser.add_argument('--sim', type=str, required=True, help='Simulator identifier')
    parser.add_argument('--material', type=str, required=True, help='Material curve identifier')

    main(parser.parse_args())
