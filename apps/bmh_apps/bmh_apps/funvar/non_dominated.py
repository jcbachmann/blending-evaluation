import argparse
import functools
import logging
import operator
import os.path

from jmetal.util.solution import get_non_dominated_solutions, read_solutions, print_function_values_to_file


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    fronts = [read_solutions(filename) for filename in args.filename]
    all_solutions = functools.reduce(operator.concat, fronts)

    non_dominated = get_non_dominated_solutions(all_solutions)
    print_function_values_to_file(
        solutions=non_dominated,
        filename=f'{os.path.commonpath(args.filename)}/non_dominated.FUN'
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
