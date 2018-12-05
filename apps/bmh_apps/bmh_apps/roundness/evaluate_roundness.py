#!/usr/bin/env python
import argparse
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from .roundness_evaluator import RoundnessEvaluator
from ..helpers.pretty_plot import pretty_line_plot


def evaluate_likelihoods(
        *,
        dist_seg_size: float,
        angle_seg_count: int,
        pos: int,
        start: float,
        stop: float,
        steps: int,
        runs: int,
        volume: float
):
    likelihoods = np.linspace(start, stop, steps)

    results: List[List[Union[float, int]]] = []

    for run in range(runs):
        for likelihood in likelihoods:
            evaluator = RoundnessEvaluator(dist_seg_size, angle_seg_count)
            evaluator.simulate(likelihood, pos, volume, run)
            result = evaluator.evaluate()
            results.append([likelihood, volume, run, result])

    return results


def evaluate_volumes(
        *,
        dist_seg_size: float,
        angle_seg_count: int,
        pos: int,
        start: float,
        stop: float,
        steps: int,
        runs: int,
        volumes: List[float]
) -> DataFrame:
    results: List[List[Union[float, int]]] = []

    for volume in volumes:
        v_results = evaluate_likelihoods(
            dist_seg_size=dist_seg_size,
            angle_seg_count=angle_seg_count,
            pos=pos,
            start=start,
            stop=stop,
            steps=steps,
            runs=runs,
            volume=volume
        )
        results.extend(v_results)

    return DataFrame(
        results,
        columns=['likelihood', 'volume', 'run', 'results']
    )


def calculate_linear(args):
    # Compute graph
    df = evaluate_volumes(
        dist_seg_size=args.dist_seg_size,
        angle_seg_count=args.angle_seg_count,
        pos=args.pos,
        start=args.start,
        stop=args.stop,
        steps=args.steps,
        runs=args.runs,
        volumes=args.volumes
    )

    # Visualize results
    ax = pretty_line_plot(
        data=df,
        x_col='likelihood',
        split_col='volume',
        y_col='results'
    )
    ax.set_ylim(0, )
    plt.show()


def main(args):
    calculate_linear(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find best cone shape')
    parser.add_argument('--pos', type=int, default=150, help='Pile position')
    parser.add_argument('--dist_seg_size', type=float, default=2.0, help='Size of a single distance segment')
    parser.add_argument('--angle_seg_count', type=int, default=9, help='Amount of angle segments in range 0-45°')
    parser.add_argument('--start', type=float, default=0.0, help='Likelihood range start')
    parser.add_argument('--stop', type=float, default=1.0, help='Likelihood range stop')
    parser.add_argument('--steps', type=int, default=5, help='Likelihood range step count')
    parser.add_argument('--runs', type=int, default=5, help='Amount of runs to evaluate statistical variation')
    parser.add_argument('--volumes', type=float, nargs='+', default=range(1000, 10000, 1000),
                        help='Volumes to be evaluated')

    main(parser.parse_args())
