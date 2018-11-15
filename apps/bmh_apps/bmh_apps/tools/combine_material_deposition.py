#!/usr/bin/env python

import argparse
import json
import logging
import os

from bmh.helpers.identifiers import get_identifier

from bmh_apps.helpers.configure_logging import configure_logging

META_JSON = 'meta.json'


def add_combination(material: str, deposition: str, dst: str):
    logger = logging.getLogger(__name__)
    dst_dir = os.path.join(dst, f'{material} x {deposition}')
    logger.info(f'Destination: {dst_dir}')
    os.makedirs(dst_dir)
    json.dump(
        {
            'material': material,
            'deposition': deposition
        },
        open(os.path.join(dst_dir, META_JSON), 'w'),
        indent=4
    )


def main(args):
    configure_logging(args.verbose)

    for material in args.material:
        for deposition in args.deposition:
            add_combination(get_identifier(material), get_identifier(deposition), args.dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a reference of material and deposition'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dst', default='./benchmark', help='Simulator benchmark path')
    parser.add_argument('--material', '-m', type=str, nargs='+', help='Material')
    parser.add_argument('--deposition', '-d', type=str, nargs='+', help='Deposition')

    main(parser.parse_args())
