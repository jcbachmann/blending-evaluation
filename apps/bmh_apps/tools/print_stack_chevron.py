#!/usr/bin/env python
import argparse

from ..helpers import chevron_path
from ..helpers.stack_with_printer import stack_with_printer


def main(args):
    stack_with_printer(
        length=args.length,
        depth=args.depth,
        material=args.material,
        stacker_path=chevron_path(args.layers)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chevron stacker')
    parser.add_argument('--length', type=float, required=True, help='Blending bed length')
    parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
    parser.add_argument('--material', type=str, required=True, help='Material input file')
    parser.add_argument('--layers', type=float, required=True, help='Speed')

    main(parser.parse_args())
