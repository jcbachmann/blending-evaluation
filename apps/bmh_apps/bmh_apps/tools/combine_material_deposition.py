#!/usr/bin/env python

import argparse
import json
import logging
import os

from bmh.benchmark.data import BenchmarkData
from bmh.helpers.identifiers import get_identifier

from ..helpers.configure_logging import configure_logging

META_JSON = 'meta.json'


def add_combination(material: str, deposition: str, path: str):
    logger = logging.getLogger(__name__)
    dir_path = os.path.join(path, BenchmarkData.BENCHMARK_DIR, f'{material} x {deposition}')
    logger.info(f'Destination: {dir_path}')
    os.makedirs(dir_path)
    json.dump(
        {
            'material': material,
            'deposition': deposition
        },
        open(os.path.join(dir_path, META_JSON), 'w'),
        indent=4
    )


def main(args):
    configure_logging(args.verbose)

    for material in args.material:
        for deposition in args.deposition:
            add_combination(
                material=get_identifier(material),
                deposition=get_identifier(deposition),
                path=args.path
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a reference of material and deposition'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--material', '-m', type=str, nargs='+', help='Material')
    parser.add_argument('--deposition', '-d', type=str, nargs='+', help='Deposition')

    main(parser.parse_args())
