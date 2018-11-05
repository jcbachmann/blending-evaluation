#!/usr/bin/env python
import argparse
import configparser
import logging
import os
import time

import matplotlib.pyplot as plt
from pandas import DataFrame

from blending_simulator.stacker.stacker import read_material
from .jmetal_ext.problem.multiobjective.homogenization_problem import HomogenizationProblem
from .optimization import OptimizationResult, optimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_optimization_result_to_file(optimization_result: OptimizationResult, problem: HomogenizationProblem,
                                      directory: str):
    variables = DataFrame(data=[solution.variables for solution in optimization_result.result_population])
    variables.columns = problem.get_variable_labels()
    variables.to_csv(f'{directory}/variables.csv', sep='\t', index=False)

    objectives = DataFrame(
        data=[solution.objectives for solution in optimization_result.result_population],
        columns=problem.get_objective_labels()
    )
    objectives.to_csv(f'{directory}/objectives.csv', sep='\t', index=False)

    all_variables_df = DataFrame(data=optimization_result.all_variables, columns=problem.get_variable_labels())
    all_variables_df.to_csv(f'{directory}/all_variables.csv', sep='\t', index=False)

    all_objectives_df = DataFrame(data=optimization_result.all_objectives, columns=problem.get_objective_labels())
    all_objectives_df.to_csv(f'{directory}/all_objectives.csv', sep='\t', index=False)


def plot_optimization_result(optimization_result: OptimizationResult, problem, directory: str):
    df = DataFrame(
        data=[solution.objectives for solution in optimization_result.result_population],
        columns=problem.get_objective_labels()
    )
    for c0 in range(len(df.columns) - 1):
        for c1 in range(c0 + 1, len(df.columns)):
            df.plot(x=df.columns[c0], y=df.columns[c1], kind='scatter')

    plt.savefig(f'{directory}/optimization_result.png')
    plt.show()


def write_arguments_to_file(args, directory: str):
    with open(f'{directory}/args.ini', 'w') as args_file:
        c = configparser.ConfigParser()
        c.read_dict({'Optimization': {str(k): str(v) for k, v in vars(args).items() if v is not None}})
        c.write(args_file)


def main(args) -> None:
    datetime = time.strftime('%Y-%m-%d %H-%M-%S')
    directory = f'{datetime} {args.length}x{args.depth} v{args.variables} {args.population_size}of{args.max_evaluations} {args.material}'
    if os.path.exists(directory):
        raise Exception(f'directory "{directory}" already exists')

    os.makedirs(directory)
    write_arguments_to_file(args, directory)

    optimization_result, problem = optimize(
        length=args.length,
        depth=args.depth,
        variables=args.variables,
        material=read_material(args.material),
        population_size=args.population_size,
        max_evaluations=args.max_evaluations,
        # TODO: parameter_columns=
    )

    write_optimization_result_to_file(optimization_result, problem, directory)
    plot_optimization_result(optimization_result, problem, directory)


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
    parser.add_argument('--length', type=float, required=True, help='Blending bed length')
    parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
    parser.add_argument('--material', type=str, required=True, help='Material input file')
    parser.add_argument('--variables', type=int, required=True, help='Amount of variables')
    parser.add_argument('--population_size', type=int, required=True, help='Amount of individuals in population')
    parser.add_argument('--max_evaluations', type=int, required=True, help='Maximum amount of function evaluations')
    main(parser.parse_args(extended_argv))
