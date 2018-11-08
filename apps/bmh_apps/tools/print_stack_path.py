#!/usr/bin/env python
import argparse

from ..helpers.stack_with_printer import stack_with_printer


def main(args):
    stack_with_printer(
        length=args.length,
        depth=args.depth,
        material=args.material,
        stacker_path=args.stacker_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='height map evaluator')
    parser.add_argument('--length', type=float, required=True, help='Blending bed length')
    parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
    parser.add_argument('--material', type=str, required=True, help='Material input file')
    parser.add_argument('--stacker_path', type=str, required=True, help='Stacker traverse path file')

    main(parser.parse_args())
