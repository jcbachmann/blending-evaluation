#!/usr/bin/env python

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict

from benchmark import helpers
from benchmark.core import Benchmark
from benchmark.processing import prepare_simulator, prepare_dst, SIMULATOR_JSON, process
from benchmark.reference_meta import ReferenceMeta


def process_data(benchmark: Benchmark, references: Dict[str, ReferenceMeta], dst: str, sim_identifier: str,
                 dry_run: bool):
    simulator_meta = benchmark.simulators[sim_identifier]
    logging.info(f'Processing data with simulator "{simulator_meta.type}"')

    logging.debug('Writing simulator type and parameters to destination directory')
    if not dry_run:
        json.dump({'simulator': simulator_meta.identifier}, open(os.path.join(dst, SIMULATOR_JSON), 'w'), indent=4)

    for _, reference in references.items():
        process(
            reference.identifier,
            benchmark.materials[reference.material],
            benchmark.depositions[reference.deposition],
            simulator_meta,
            dst,
            dry_run,
            computed_deposition=False
        )

    logging.info('Processing finished')


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
    references = Benchmark.read_references(args.src)

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
