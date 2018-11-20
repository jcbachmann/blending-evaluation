#!/usr/bin/env python
import argparse
import configparser
import logging
import os
import time
from typing import List

import matplotlib.pyplot as plt
from bmh.benchmark.material_deposition import Material
from bmh.optimization.optimization import OptimizationResult, optimize
from pandas import DataFrame

from bmh_apps.helpers.configure_logging import configure_logging
from bmh_apps.helpers.material_path_io import read_material


def write_optimization_result_to_file(optimization_result: OptimizationResult, directory: str):
    variables = DataFrame(data=[solution.variables for solution in optimization_result.result_population])
    variables.columns = optimization_result.variable_labels
    variables.to_csv(f'{directory}/variables.csv', sep='\t', index=False)

    objectives = DataFrame(
        data=[solution.objectives for solution in optimization_result.result_population],
        columns=optimization_result.objective_labels
    )
    objectives.to_csv(f'{directory}/objectives.csv', sep='\t', index=False)

    all_variables_df = DataFrame(data=optimization_result.all_variables, columns=optimization_result.variable_labels)
    all_variables_df.to_csv(f'{directory}/all_variables.csv', sep='\t', index=False)

    all_objectives_df = DataFrame(data=optimization_result.all_objectives, columns=optimization_result.objective_labels)
    all_objectives_df.to_csv(f'{directory}/all_objectives.csv', sep='\t', index=False)


def plot_optimization_result(optimization_result: OptimizationResult, directory: str):
    df = DataFrame(
        data=[solution.objectives for solution in optimization_result.result_population],
        columns=optimization_result.objective_labels
    )
    for c0 in range(len(df.columns) - 1):
        for c1 in range(c0 + 1, len(df.columns)):
            df.plot(x=df.columns[c0], y=df.columns[c1], kind='scatter')

    plt.savefig(f'{directory}/optimization_result.png')
    plt.show()


def get_parameter_columns(material: DataFrame) -> List[str]:
    logger = logging.getLogger(__name__)
    non_parameter_columns = ['timestamp', 'volume']
    parameter_columns = list(set(material.columns) - set(non_parameter_columns))
    logger.info(f'Found the following parameter columns: {", ".join(parameter_columns)}')
    return parameter_columns


def write_arguments_to_file(args, directory: str):
    with open(f'{directory}/args.ini', 'w') as args_file:
        c = configparser.ConfigParser()
        c.read_dict({'Optimization': {str(k): str(v) for k, v in vars(args).items() if v is not None}})
        c.write(args_file)


def main(args) -> None:
    configure_logging(args.verbose)

    # Prepare directory
    datetime = time.strftime('%Y-%m-%d %H-%M-%S')
    directory = f'{datetime} {args.length}x{args.depth} v{args.variables} ' \
                f'{args.population_size}of{args.max_evaluations} {args.material}'
    if os.path.exists(directory):
        raise Exception(f'directory "{directory}" already exists')
    os.makedirs(directory)
    write_arguments_to_file(args, directory)

    # Execute optimization
    material = read_material(args.material)
    optimization_result = optimize(
        bed_size_x=args.length,
        bed_size_z=args.depth,
        variables=args.variables,
        material=Material(data=material),
        population_size=args.population_size,
        max_evaluations=args.max_evaluations,
    )

    # Write results
    write_optimization_result_to_file(optimization_result, directory)
    plot_optimization_result(optimization_result, directory)


if __name__ == '__main__':
    conf_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                          add_help=False)
    conf_parser.add_argument('--args', dest='args_file', help='Specify config file', metavar='FILE')
    conf_args, remaining_argv = conf_parser.parse_known_args()

    extended_argv = []
    if conf_args.args_file:
        config = configparser.ConfigParser()
        config.read(conf_args.args_file)
        extended_argv = [e for (k, v) in config.items('Optimization') for e in [f'--{k}', v]]
    extended_argv.extend(remaining_argv)

    parser = argparse.ArgumentParser(parents=[conf_parser], description='Stacker Path Optimization')
    parser.add_argument('--length', type=float, default=300, help='Blending bed length')
    parser.add_argument('--depth', type=float, default=50, help='Blending bed depth')
    parser.add_argument('--material', type=str, default='quality_input_curve.txt', help='Material input file')
    parser.add_argument('--variables', type=int, default=20, help='Amount of variables')
    parser.add_argument('--population_size', type=int, default=100, help='Amount of individuals in population')
    parser.add_argument('--max_evaluations', type=int, default=5000, help='Maximum amount of function evaluations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    main(parser.parse_args(extended_argv))
