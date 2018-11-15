#!/usr/bin/env python

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict

from bmh.benchmark import core
from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.reference_meta import ReferenceMeta
from bmh.helpers.identifiers import get_identifiers

from bmh_apps.helpers.configure_logging import configure_logging


def process_data(benchmark_data: BenchmarkData, references: Dict[str, ReferenceMeta], dst: str, sim_identifier: str,
                 dry_run: bool):
    logger = logging.getLogger(__name__)
    simulator_meta = benchmark_data.simulators[sim_identifier]
    logger.info(f'Processing data with simulator "{simulator_meta.type}"')

    logger.debug('Writing simulator type and parameters to destination directory')
    if not dry_run:
        json.dump({'simulator': str(simulator_meta)}, open(os.path.join(dst, core.SIMULATOR_JSON), 'w'), indent=4)

    for _, reference in references.items():
        core.process(
            identifier=str(reference),
            material_meta=benchmark_data.materials[reference.material],
            deposition_meta=benchmark_data.depositions[reference.deposition],
            simulator_meta=simulator_meta,
            dst=dst,
            dry_run=dry_run,
            computed_deposition=False
        )

    logger.info('Processing finished')


def main(args: argparse.Namespace):
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.info(f'Starting evaluation with timestamp {timestamp_str}')

    # Initialization
    benchmark_data = BenchmarkData()
    benchmark_data.read_base(args.path)
    references = benchmark_data.read_references(args.src)

    # Make sure everything will work out
    benchmark_data.validate_references(references)

    # Parse simulator identifiers (strip away everything but the part after the last slash)
    sim_identifiers = get_identifiers(args.sim)

    # Make sure simulation will work properly
    benchmark_data.validate_simulators(sim_identifiers)
    for sim_identifier in sim_identifiers:
        core.test_simulator(benchmark_data.simulators[sim_identifier])

    logger.info(f'Evaluating {len(references)} references with {len(sim_identifiers)} simulator(s)')
    for sim_identifier in sim_identifiers:
        # Prepare output directory
        dst = os.path.join(args.dst, timestamp_str + ' ' + sim_identifier)
        core.prepare_dst(dst, args.dry_run)

        # Processing
        process_data(benchmark_data, references, dst, sim_identifier, args.dry_run)


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
