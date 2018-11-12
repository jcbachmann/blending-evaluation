#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from bmh_apps.helpers.ciplot import ciplot
from bmh_apps.roundness.roundness_evaluator import RoundnessEvaluator


def evaluate_likelihoods(dist_seg_size, angle_seg_count, pos, start, stop, steps, runs, volume):
    likelihoods = np.linspace(start, stop, steps)

    results = []

    for run in range(runs):
        for likelihood in likelihoods:
            evaluator = RoundnessEvaluator(dist_seg_size, angle_seg_count)
            evaluator.simulate(likelihood, pos, volume, run)
            result = evaluator.evaluate()
            results.append([likelihood, volume, run, result])

    return results


def evaluate_volumes(dist_seg_size, angle_seg_count, pos, start, stop, steps, runs, volumes) -> DataFrame:
    results = []

    for volume in volumes:
        v_results = evaluate_likelihoods(dist_seg_size, angle_seg_count, pos, start, stop, steps, runs, volume)
        results.extend(v_results)

    return DataFrame(
        results,
        columns=['likelihood', 'volume', 'run', 'results']
    )


def visualize(df):
    plt.cla()
    ax = ciplot(
        data=df,
        x_col='likelihood',
        split_col='volume',
        y_col='results'
    )
    ax.set_ylim(0, )
    plt.gcf().canvas.draw()


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
    visualize(df)
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
