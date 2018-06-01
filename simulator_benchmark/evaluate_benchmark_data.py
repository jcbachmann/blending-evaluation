#!/usr/bin/env python

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict

from blending_simulator.material_deposition import MaterialMeta, DepositionMeta, MaterialDeposition
from simulator_benchmark import helpers
from simulator_benchmark.benchmark import Benchmark
from simulator_benchmark.reference_meta import ReferenceMeta
from simulator_benchmark.simulator_meta import SimulatorMeta

DATA_CSV = 'data.csv'
RECLAIMED_MATERIAL_DIR = 'material'
SIMULATOR_JSON = 'simulator.json'
SIMULATOR_PARAMS_JSON = 'simulator_params.json'


def prepare_dst(dst: str, dry_run: bool):
    logging.debug(f'Preparing destination directory "{dst}"')
    if os.path.exists(dst):
        if os.path.isdir(dst):
            if len(os.listdir(dst)) > 0:
                raise IOError(f'Destination path "{dst}" is not empty')
            else:
                logging.debug(f'Destination path "{dst}" already exists and is empty')
        else:
            raise IOError(f'Destination path "{dst}" is not a directory')
    else:
        logging.debug(f'Creating destination path "{dst}"')
        if not dry_run:
            os.makedirs(dst)


def process_reference(reference: ReferenceMeta, material: MaterialMeta, deposition: DepositionMeta, dst: str,
                      simulator_meta: SimulatorMeta, dry_run: bool):
    logging.info(f'Processing reference "{reference}" with material "{material}" and deposition "{deposition}"')

    logging.debug('Creating simulator')
    sim_params = simulator_meta.get_params().copy()
    sim_params['bed_size_x'] = deposition.bed_size_x
    sim_params['bed_size_z'] = deposition.bed_size_z
    sim = simulator_meta.get_type()(**sim_params)

    logging.debug('Combining material and deposition')
    material_deposition = MaterialDeposition(material.get_material(), deposition.get_deposition())
    logging.debug(f'Material and deposition combined:\n{material_deposition.data.describe()}')

    logging.debug('Stacking and reclaiming material')
    reclaimed_material = sim.stack_reclaim(material_deposition, x_per_s=deposition.reclaim_x_per_s)
    logging.debug(f'Reclaimed material:\n{reclaimed_material.data.describe()}')

    directory = os.path.join(dst, reference.identifier)
    logging.debug(f'Creating directory "{directory}"')
    if not dry_run:
        os.mkdir(directory)

    reclaimed_reference = ReferenceMeta(reference.identifier, directory, reference.meta_dict)
    reclaimed_reference.reclaimed_path = RECLAIMED_MATERIAL_DIR

    meta_file = os.path.join(directory, Benchmark.META_JSON)
    logging.debug(f'Writing reference meta to "{meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_reference.to_dict(),
            open(meta_file, 'w'),
            indent=4
        )

    reclaimed_material_path = os.path.join(directory, reclaimed_reference.reclaimed_path)
    reclaimed_material_meta = MaterialMeta(reference.identifier, reclaimed_material_path, {
        'label': 'Reclaimed ' + reference.identifier,
        'description': 'Reclaimed material from reference ' + reference.identifier,
        'category': 'reclaimed',
        'time': reclaimed_material.data.timestamp.max(),
        'volume': reclaimed_material.data.volume.sum(),
        'data': DATA_CSV
    })

    logging.debug(f'Creating directory "{reclaimed_material_path}"')
    if not dry_run:
        os.mkdir(reclaimed_material_path)

    reclaimed_material_meta_file = os.path.join(reclaimed_material_path, Benchmark.META_JSON)
    logging.debug(f'Writing reclaimed material meta to "{reclaimed_material_meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_material_meta.to_dict(),
            open(reclaimed_material_meta_file, 'w'),
            indent=4
        )

    data_file = os.path.join(reclaimed_material_path, DATA_CSV)
    logging.debug(f'Writing material data to "{data_file}"')
    if not dry_run:
        reclaimed_material.data.to_csv(data_file, sep='\t', index=False)


def process_data(benchmark: Benchmark, references: Dict[str, ReferenceMeta], dst: str, sim_identifier: str,
                 dry_run: bool):
    simulator_meta = benchmark.simulators[sim_identifier]
    logging.info(f'Processing data with simulator "{simulator_meta.type}"')

    logging.debug('Writing simulator type and parameters to destination directory')
    if not dry_run:
        json.dump({'simulator': simulator_meta.identifier}, open(os.path.join(dst, SIMULATOR_JSON), 'w'), indent=4)

    for _, reference in references.items():
        process_reference(
            reference,
            benchmark.materials[reference.material],
            benchmark.depositions[reference.deposition],
            dst,
            simulator_meta,
            dry_run
        )

    logging.info('Processing finished')


def prepare_simulator(simulator_meta: SimulatorMeta):
    logging.debug(f'Acquiring simulator type for "{simulator_meta.type}"')

    # Acquire simulator type
    sim_type = simulator_meta.get_type()

    # Read simulator parameters
    sim_params = simulator_meta.get_params()

    sim_params_copy = sim_params.copy()
    sim_params_copy['bed_size_x'] = 10
    sim_params_copy['bed_size_z'] = 10

    # Create demo simulator to test if params are all accepted
    # If no exception occurs everything seems to be fine
    logging.debug('Testing creation of simulator')
    sim = sim_type(**sim_params_copy)

    logging.debug('Testing deletion of simulator')
    del sim


def main(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )

    timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logging.info(f'Starting evaluation with timestamp {timestamp_str}')

    # Initialization
    benchmark = Benchmark()
    benchmark.read_base(args.path)
    references = benchmark.read_references(args.src)

    # Make sure everything will work out
    benchmark.validate_references(references)

    # Parse simulator identifiers (strip away everything but the part after the last slash)
    sim_identifiers = helpers.get_identifiers(args.sim)

    # Make sure simulation will work properly
    benchmark.validate_simulators(sim_identifiers)
    for sim_identifier in sim_identifiers:
        prepare_simulator(benchmark.simulators[sim_identifier])

    logging.info(f'Evaluating {len(references)} references with {len(sim_identifiers)} simulator(s)')
    for sim_identifier in sim_identifiers:
        # Prepare output directory
        dst = os.path.join(args.dst, timestamp_str + ' ' + sim_identifier)
        prepare_dst(dst, args.dry_run)

        # Processing
        process_data(benchmark, references, dst, sim_identifier, args.dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate benchmark data for a given set of material deposition combinations'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--src', default='./benchmark', help='Path with reference configuration files')
    parser.add_argument('--dst', default='.', help='Path where results will be stored')
    parser.add_argument('--dry_run', action='store_true', help='Do not write files')
    parser.add_argument('--sim', nargs='+', help='Which simulator is used to calculate results')

    main(parser.parse_args())
