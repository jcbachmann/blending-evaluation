#!/usr/bin/env python
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from dask import delayed
from dask.distributed import Client
from pandas import DataFrame

from blending_simulator.external_blending_simulator import ExternalBlendingSimulatorInterface
from ciglobal.ciplot import ciplot_scatter


def get_volume(a1, a2, c, xz_scaling):
    """
    Calculate the volume under the half square triangle a1, c, a2 with side length xz_scaling
    :param a1: height at corner connected to hypotenuse 1
    :param a2: height at corner connected to hypotenuse 2
    :param c: height at corner with right angle
    :param xz_scaling: square side length
    :return: volume under triangle
    """
    return 0.25 * (0.5 * (a1 + a2) + c) * xz_scaling * xz_scaling


def get_height_map_volume(height_map, size):
    length = len(height_map)
    if length <= 1:
        return 0
    width = len(height_map[0])
    volume = 0
    xz_scaling = size / (length - 1)
    for z in range(length - 1):
        for x in range(width - 1):
            volume += get_volume(height_map[z][x + 1], height_map[z + 1][x], height_map[z][x], xz_scaling)
            volume += get_volume(height_map[z][x + 1], height_map[z + 1][x], height_map[z + 1][x + 1], xz_scaling)
    return volume


def get_height_map_volume_df(height_map_df, size):
    if height_map_df is None:
        return None
    return get_height_map_volume(height_map_df.values, size)


def load_results_from_path(path, size):
    results = []

    r1 = re.compile('heights-(\d+)-(\d+)\.txt')
    r2 = re.compile('heights-vol(\d+)-res([\d.]+)-run(\d+)\.txt')
    for file in os.listdir(path):
        in_volume = 0.0
        ppm3 = 1.0
        run = 0

        g = r1.match(file)
        if g:
            in_volume = int(g.group(1))
            run = int(g.group(2))
        else:
            g = r2.match(file)
            if g:
                in_volume = int(g.group(1))
                ppm3 = float(g.group(2))
                run = int(g.group(3))
        if g:
            df = delayed(load_for_bulk_density)(file=file, path=path)
            out_volume = delayed(get_height_map_volume_df)(df, size)
            results.append([in_volume, ppm3, run, out_volume])
    return results


def execute_for_bulk_density(
        pos: float,
        size: float,
        volume: float,
        ppm3: float,
        run: int,
        dropheight: float,
        detailed: bool,
        visualize: bool,
        bulkdensity: float,
        path: str,
        executable: str = './BlendingSimulator'
):
    print(f'processing volume {volume} with ppm3 {ppm3:.1f} (run {run})')
    path += f'/heights-vol{volume}-res{ppm3:.1f}-run{run}.txt'

    ExternalBlendingSimulatorInterface(
        executable=executable,
        length=size,
        depth=size,
        heights=path,
        ppm3=ppm3,
        dropheight=dropheight,
        detailed=detailed,
        visualize=visualize,
        bulkdensity=bulkdensity
    ).run(
        lambda sim: sim.communicate(f'0 {pos} {pos} {volume}'.encode())
    )

    return pd.read_csv(path, header=None, delimiter='\t', index_col=None)


def load_for_bulk_density(file: str, path: str = '/tmp'):
    if path is None:
        path = '/tmp'
    path += f'/{file}'
    if os.path.isfile(path):
        return pd.read_csv(path, header=None, delimiter='\t', index_col=None)
    else:
        return None


def calculate_results(args):
    results = []
    for in_volume in args.volumes:
        for ppm3 in args.ppm3s:
            for run in range(args.runs):
                df = delayed(execute_for_bulk_density)(
                    pos=args.size / 2,
                    size=args.size,
                    volume=in_volume,
                    ppm3=ppm3,
                    run=run,
                    dropheight=args.dropheight,
                    detailed=args.detailed,
                    visualize=args.visualize,
                    bulkdensity=args.bulkdensity,
                    path=args.path
                )
                out_volume = delayed(get_height_map_volume_df)(df, args.size)
                results.append([in_volume, ppm3, run, out_volume])
    return results


def acquire_results(args):
    if args.reuse:
        return load_results_from_path(args.path, args.size)
    else:
        if not os.path.exists(args.path):
            os.makedirs(args.path)

        return calculate_results(args)


def to_df(results):
    results = list(filter(lambda entry: entry[2] is not None, results))
    return DataFrame(results, columns=['in_volume', 'ppm3', 'run', 'out_volume'])


def main(args):
    if args.client:
        client = Client('127.0.0.1:8786')
        client.upload_file('../pile.py')
        client.upload_file('../roundness.py')
        client.upload_file('../execute.py')

    results = acquire_results(args)
    results_df_delayed = delayed(to_df)(results)
    results_df = results_df_delayed.compute()

    if len(args.ppm3s) == 1:
        ciplot_scatter(
            data=results_df,
            x_col='in_volume',
            y_col='out_volume',
            log_scale=False
        )
    else:
        results_df['out/in volume ratio'] = results_df['out_volume'] / results_df['in_volume']
        ciplot_scatter(
            data=results_df,
            x_col='ppm3',
            y_col='out/in volume ratio',
            split_col='in_volume',
            log_scale=False,
            equal=False
        )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='height map evaluator')
    parser.add_argument('--volumes', type=int, default=[1000], nargs='*', help='Volumes')
    parser.add_argument('--ppm3s', type=float, default=[1.0], nargs='*', help='Particles per cubic meter')
    parser.add_argument('--size', type=int, default=100, help='Blending bed length and depth')
    parser.add_argument('--dropheight', type=float, default=25, help='Stacker drop height')
    parser.add_argument('--runs', type=int, default=1, help='Runs')
    parser.add_argument('--detailed', action='store_true', help='Use detailed simulation')
    parser.add_argument('--visualize', action='store_true', help='Visualize simulation')
    parser.add_argument('--reuse', action='store_true', help='Reuse old calculation data')
    parser.add_argument('--client', action='store_true', help='Use external dask workers')
    parser.add_argument('--path', type=str, default='/tmp', help='Output path for intermediate files')
    parser.add_argument('--bulkdensity', type=float, default=1.0, help='Bulk density')

    main(parser.parse_args())
