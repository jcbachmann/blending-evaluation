#!/usr/bin/env python
import argparse
import os
import re
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from seaborn.palettes import color_palette

from blending_simulator.external_blending_simulator import ExternalBlendingSimulatorInterface
from blending_simulator.stacker import stack_with_printer
from blending_simulator.stacker.chevron_stacker import chevron_path
from ciglobal.cimath import weighted_avg_and_std


def simulate(args, reclaim, layers):
    ExternalBlendingSimulatorInterface(
        length=args.length,
        depth=args.depth,
        dropheight=(args.depth / 2),
        reclaim=reclaim,
        ppm3=10
    ).run(lambda sim: stack_with_printer(
        length=args.length,
        depth=args.depth,
        material=args.material,
        stacker_path=chevron_path(layers),
        header=False,
        out_buffer=sim.stdin
    ))


def get_results(meta, data: pd.DataFrame, c_meta='layers', c_weights: str = 'volume', c_values: str = 'p_1'):
    minvol = 0.75 * data[c_weights].sum() / len(data.index)
    larger = data.query(f'{c_weights}>={minvol}')
    lbound = larger[c_values].min()
    mean, std = weighted_avg_and_std(data[c_values], weights=data[c_weights])
    lstd = mean - std
    ustd = mean + std
    ubound = larger[c_values].max()

    smaller = data.query(f'{c_weights}<{minvol}').query(f'{c_weights}>0')
    if len(smaller.index) > 0:
        lbound = min(lbound, np.average(smaller[c_values], weights=smaller[c_weights]))
        ubound = max(ubound, np.average(smaller[c_values], weights=smaller[c_weights]))

    return pd.DataFrame(
        data=[(meta, lbound, lstd, mean, ustd, ubound)],
        columns=[c_meta, 'lbound', 'lstd', 'mean', 'ustd', 'ubound']
    )


def get_results_for_file(layers, file, path):
    df = pd.read_csv(path + '/' + file, delimiter='\t', index_col=None)
    return get_results(meta=layers, data=df, c_values='p_1')


def get_reference(file):
    df = pd.read_csv(file, delimiter='\t', index_col=None)
    return get_results(meta=0, data=df, c_values='p_1')


def load_results_from_path(path):
    results = pd.DataFrame()

    r1 = re.compile('reclaim-layers-([\d.]+)\.csv')
    for file in os.listdir(path):
        g = r1.match(file)
        if g:
            layers = float(g.group(1))
            results = results.append(get_results_for_file(layers, file, path))

    return results.sort_values('layers')


def plot_results(df_reference, df_layers):
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle('Homogenization Sensitivity to Speed Change')
    fig.subplots_adjust(top=0.94)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 15], wspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    colors = color_palette(n_colors=1)

    y_range_raw = df_reference['ubound'][0] - df_reference['lbound'][0]
    y_range_margin = 0.02 * y_range_raw
    y_lim = [df_reference['lbound'][0] - y_range_margin, df_reference['ubound'][0] + y_range_margin]

    ax0.fill_between(
        [0, 1],
        [df_reference['lbound'][0], df_reference['lbound'][0]],
        [df_reference['ubound'][0], df_reference['ubound'][0]],
        facecolor=colors[0], alpha=0.3
    )
    ax0.fill_between(
        [0, 1],
        [df_reference['lstd'][0], df_reference['lstd'][0]],
        [df_reference['ustd'][0], df_reference['ustd'][0]],
        facecolor=colors[0], alpha=0.5
    )
    ax0.plot(
        [0, 1],
        [df_reference['mean'][0], df_reference['mean'][0]],
        color=colors[0], linestyle='-'
    )

    ax0.set_xlim([0, 1])
    ax0.set_ylim(y_lim)
    ax0.set_xlabel('Raw Input')
    ax0.set_ylabel('Material Output Quality')
    ax0.legend(loc=0)
    ax0.tick_params(
        axis='x',
        bottom='off',
        labelbottom='off'
    )

    df_layers = df_layers.sort_values('layers')
    x = df_layers['layers'].values

    ax1.fill_between(x, df_layers['lbound'], df_layers['ubound'], facecolor=colors[0], alpha=0.3, label='Min/Max')
    ax1.fill_between(x, df_layers['lstd'], df_layers['ustd'], facecolor=colors[0], alpha=0.5, label='Std')
    ax1.plot(x, df_layers['mean'].values, color=colors[0], linestyle='-', label='Mean')

    ax1.set_xlim([np.min(x), np.max(x)])
    ax1.set_ylim(y_lim)
    ax1.set_xlabel('Speed in Amount of Layers')
    ax1.set_xscale('log')
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(loc=0)
    ax1.tick_params(
        axis='y',
        left='off',
        labelleft='off'
    )


def main(args):
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if not args.reuse:
        p = Pool(8)
        p.starmap(simulate, [
            (args, f'{args.path}/reclaim-layers-{layers:.4f}.csv', layers) for layers in np.linspace(1, 100, 10)
        ])

    reference = get_reference(args.material)
    results = load_results_from_path(args.path)
    plot_results(reference, results)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='height map evaluator')
    parser.add_argument('--length', type=float, required=True, help='Blending bed length')
    parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
    parser.add_argument('--material', type=str, required=True, help='Material input file')
    parser.add_argument('--path', type=str, default='/tmp', help='Output path')
    parser.add_argument('--reuse', action='store_true', help='Use material available in path')

    main(parser.parse_args())
