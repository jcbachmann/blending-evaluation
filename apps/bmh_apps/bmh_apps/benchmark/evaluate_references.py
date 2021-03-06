#!/usr/bin/env python

import argparse
import logging
import uuid
from datetime import datetime

from bmh.benchmark import core
from bmh.benchmark.data import BenchmarkData
from bmh.helpers.identifiers import get_identifiers

from ..helpers.configure_logging import configure_logging


def main(args: argparse.Namespace):
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.info(f'Starting evaluation with timestamp {timestamp_str}')

    # Initialization
    benchmark_data = BenchmarkData(args.path)
    benchmark_data.read_base()
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
        simulator_meta = benchmark_data.simulators[sim_identifier]

        # Processing
        for _, reference in references.items():
            core.process(
                identifier=f'{reference.identifier}-{uuid.uuid4()[:4]}',
                material_meta=benchmark_data.materials[reference.material],
                deposition_meta=benchmark_data.depositions[reference.deposition],
                simulator_meta=simulator_meta,
                path=args.path,
                dry_run=args.dry_run,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate benchmark data for a given set of material deposition combinations'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--src', default='./benchmark', help='Path with reference configuration files')
    parser.add_argument('--dry_run', action='store_true', help='Do not write files')
    parser.add_argument('--sim', nargs='+', help='Which simulator is used to calculate results')

    main(parser.parse_args())
