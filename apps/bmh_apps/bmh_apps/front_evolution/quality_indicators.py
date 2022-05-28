import argparse
import os

from jmetal.core.quality_indicator import HyperVolume, GenerationalDistance, InvertedGenerationalDistance
from jmetal.util.solution import read_solutions
from tqdm import tqdm


def main(args: argparse.Namespace):
    reference_front = read_solutions(args.reference)
    reference_front_objectives = [solution.objectives for solution in reference_front]
    if len(reference_front) == 0:
        raise Exception('Reference front is empty')

    quality_indicators = [
        HyperVolume(reference_point=[1.0] * len(reference_front[0].objectives)),
        GenerationalDistance(reference_front=reference_front_objectives),
        InvertedGenerationalDistance(reference_front=reference_front_objectives)
    ]

    with open(args.output, 'w+') as of:
        of.write(f'{",".join([indicator.get_short_name() for indicator in quality_indicators])}\n')

        for i in tqdm(range(args.generations)):
            solutions = read_solutions(os.path.join(args.fronts, f'FUN.{i}'))
            objectives = [solution.objectives for solution in solutions]
            results = [str(indicator.compute(objectives)) for indicator in quality_indicators]

            of.write(','.join(results) + '\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fronts', type=str,
                        help='Path to directory with fronts')
    parser.add_argument('--reference', type=str,
                        help='File containing the reference front')
    parser.add_argument('--output', type=str, default='quality_indicators.csv',
                        help='File where the results will be saved')
    parser.add_argument('--generations', type=int,
                        help='Number of generations')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
