#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fminbound

from blending_simulator.evaluation.roundness import execute_for_roundness
from ciglobal.ciplot import ciplot


class OptEvaluator:
    df = pd.DataFrame()
    evaluations = 0

    def evaluate(self, likelihood, dist_seg_size, angle_seg_count, pos, volume, runs):
        results = []

        for run in range(runs):
            result = execute_for_roundness(likelihood, dist_seg_size, angle_seg_count, pos, volume, run)
            results.append([likelihood, volume, OptEvaluator.evaluations, run, result])

        df_i = pd.DataFrame(
            results,
            columns=['likelihood', 'volume', 'run_group', 'run', 'results']
        )
        self.df = self.df.append(df_i, ignore_index=True)

        OptEvaluator.evaluations += 1

        return df_i['results'].mean()


def visualize(df):
    plt.cla()
    ax = ciplot(
        data=df,
        x_col='likelihood',
        unique_col='run_group',
        split_col='volume',
        y_col='results'
    )
    ax.set_ylim(0, )
    plt.gcf().canvas.draw()


def calculate_optimized_scipy(args):
    volume = args.volumes[0]
    e = OptEvaluator()
    x_opt = fminbound(
        func=e.evaluate,
        x1=args.start,
        x2=args.stop,
        args=(args.dist_seg_size, args.angle_seg_count, args.pos, volume, args.runs),
        xtol=0.001,
        maxfun=200,
        full_output=True,
        disp=3
    )
    print(x_opt)

    # Visualize results
    visualize(e.df)
    plt.show()


def minimize_brute_force(func, x_start, x_stop, args, x_tol, f_tol):
    x_opt = 0.5 * (x_start + x_stop)
    f_opt = func(x_opt, *args)

    f_last = None
    last_run = False

    while f_last is None or x_stop - x_start > x_tol and not last_run:
        last_run = f_last is not None and f_last - f_opt < f_tol
        f_last = f_opt

        for x in np.linspace(x_start, x_stop, 10):
            f = func(x, *args)
            if f < f_opt:
                x_opt = x
                f_opt = f

        x_range = 0.33 * (x_stop - x_start)
        x_start = x_opt - 0.5 * x_range
        x_stop = x_start + x_range

    print(f'Optimum found: f({x_opt}) = {f_opt}')

    return x_opt


def optimize_single(volume, args):
    e = OptEvaluator()
    x_opt = minimize_brute_force(
        func=e.evaluate,
        x_start=args.start,
        x_stop=args.stop,
        args=(args.dist_seg_size, args.angle_seg_count, args.pos, volume, args.runs),
        x_tol=0.01,
        f_tol=0.01
    )
    return x_opt, e.df


def calculate_optimized(args):
    all_df = pd.DataFrame()

    for volume in args.volumes:
        x_opt, df = optimize_single(volume, args)
        all_df = all_df.append(df, ignore_index=True)

    # Visualize results
    visualize(all_df)
    plt.show()


def main(args):
    calculate_optimized(args)


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
