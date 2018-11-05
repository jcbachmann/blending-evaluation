#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np
from dask import delayed
from dask.distributed import Client
from pandas import DataFrame

from blending_simulator.evaluation.roundness.roundness import execute_for_roundness
from ciglobal.ciplot import ciplot


def evaluate_likelihoods(dist_seg_size, angle_seg_count, pos, start, stop, steps, runs, volume):
    likelihoods = np.linspace(start, stop, steps)

    results = []

    for run in range(runs):
        for likelihood in likelihoods:
            result = delayed(execute_for_roundness)(
                likelihood, dist_seg_size, angle_seg_count, pos, volume, run
            )
            results.append([likelihood, volume, run, result])

    return results


def evaluate_volumes(dist_seg_size, angle_seg_count, pos, start, stop, steps, runs, volumes):
    results = []

    for volume in volumes:
        v_results = evaluate_likelihoods(dist_seg_size, angle_seg_count, pos, start, stop, steps, runs, volume)
        results.extend(v_results)

    return results


def to_df(results):
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
    # Setup Dask
    if args.client:
        client = Client('127.0.0.1:8786')
        client.upload_file('roundness.py')
        client.upload_file('pile.py')

    # Compute graph
    results = evaluate_volumes(
        dist_seg_size=args.dist_seg_size,
        angle_seg_count=args.angle_seg_count,
        pos=args.pos,
        start=args.start,
        stop=args.stop,
        steps=args.steps,
        runs=args.runs,
        volumes=args.volumes
    )
    rdf = delayed(to_df)(results)
    # rdf.visualize(filename='dask-graph')

    # Compute results
    df = rdf.compute()

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
    parser.add_argument('volumes', type=int, nargs='+', help='Volumes to be evaluated')

    parser.add_argument('--client', action='store_true', help='Use external dask workers')

    main(parser.parse_args())
