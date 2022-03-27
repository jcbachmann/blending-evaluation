#!/usr/bin/env python
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from pandas import DataFrame
from seaborn.palettes import color_palette

from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import MaterialDeposition, Deposition, Material
from bmh.helpers.identifiers import get_identifier
from bmh.helpers.math import weighted_avg_and_std
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from bmh_apps.chevron.chevron_path import chevron_path
from bmh_apps.helpers.bed_size import get_bed_size
from bmh_apps.helpers.configure_logging import configure_logging


def simulate(layers, material: Material, bed_size_x: float, bed_size_z: float, ppm3: float) -> DataFrame:
    x_min = 0.5 * bed_size_z
    x_max = bed_size_x - x_min

    path = chevron_path(layers)
    max_part = path['part'].values[-1]

    deposition_data = DataFrame()
    deposition_data['timestamp'] = material.meta.time * path['part'] / max_part
    deposition_data['x'] = path['path'] * (x_max - x_min) + x_min
    deposition_data['z'] = bed_size_z / 2

    material_deposition = MaterialDeposition(
        material=material,
        deposition=Deposition.from_data(
            deposition_data, bed_size_x=bed_size_x, bed_size_z=bed_size_z, reclaim_x_per_s=1.0
        )
    )

    simulator = BslBlendingSimulator(
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        ppm3=ppm3,
    )
    reclaim = simulator.stack_reclaim(material_deposition)

    return reclaim.data


def get_results(meta, data: DataFrame, c_meta='layers', c_weights: str = 'volume', c_values: str = 'p_1') -> DataFrame:
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

    return DataFrame(
        data=[(meta, lbound, lstd, mean, ustd, ubound)],
        columns=[c_meta, 'lbound', 'lstd', 'mean', 'ustd', 'ubound']
    )


def plot_results(df_reference, df_layers):
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle('Homogenization Sensitivity to Speed Change')
    fig.subplots_adjust(top=0.94)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 15], wspace=0.05)
    ax0: Axes = plt.subplot(gs[0])
    ax1: Axes = plt.subplot(gs[1])

    colors = color_palette(n_colors=1)

    y_range_raw = df_reference['ubound'][0] - df_reference['lbound'][0]
    y_range_margin = 0.02 * y_range_raw
    # y_min, y_max = (df_reference['lbound'][0] - y_range_margin, df_reference['ubound'][0] + y_range_margin)
    y_min, y_max = (df_reference['lstd'][0] - y_range_margin, df_reference['ustd'][0] + y_range_margin)

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

    ax0.set_xlim(0, 1)
    ax0.set_ylim(y_min, y_max)
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

    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Speed in Amount of Layers')
    ax1.set_xscale('log')
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.legend(loc=0)
    ax1.tick_params(
        axis='y',
        left='off',
        labelleft='off'
    )


def compute(layers, *, material: Material, bed_size_x: float, bed_size_z: float):
    logging.debug(f'Computing for {layers} layers')
    df = simulate(layers, material, bed_size_x, bed_size_z, ppm3=10)
    return get_results(meta=layers, data=df, c_values=material.get_parameter_columns()[0])


def main(path: str, material_identifier: str, verbose: bool):
    configure_logging(verbose=verbose)
    benchmark = BenchmarkData(path)
    benchmark.read_base()
    material_meta = benchmark.get_material_meta(material_identifier)
    material = material_meta.get_material()
    bed_size_x, bed_size_z = get_bed_size(volume=material_meta.volume, max_stockpile_height=10)

    # with Pool() as p:
    #     results_list = p.map(
    #         functools.partial(compute, material=material, bed_size_x=bed_size_x, bed_size_z=bed_size_z),
    #         range(1, 400)
    #     )
    results_list = [
        compute(layers, material=material, bed_size_x=bed_size_x, bed_size_z=bed_size_z)
        for layers in range(1, 400)
    ]

    results_df = DataFrame()
    for results in results_list:
        results_df = results_df.append(results)

    reference = get_results(meta=0, data=material.data, c_values=material.get_parameter_columns()[0])
    plot_results(reference, results_df)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate homogenization efficiency for various amount of layers'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--material', type=str, default='generated_2Y45', help='Material curve identifier')
    args = parser.parse_args()

    main(path=args.path, material_identifier=get_identifier(args.material), verbose=args.verbose)
