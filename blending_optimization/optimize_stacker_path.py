#!/usr/bin/env python
import argparse
import configparser
import logging
import os
import time
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from jmetal.component import RankingAndCrowdingDistanceComparator
from jmetal.component.evaluator import MapEvaluator
from jmetal.component.observer import Observer
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection

from blending_optimization.homogenization_problem import HomogenizationProblem
from blending_optimization.hpsea import HPSEA
from blending_optimization.plot_server import PlotServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationResult:
    def __init__(self, result_population, all_variables, all_objectives):
        self.result_population = result_population
        self.all_variables = all_variables
        self.all_objectives = all_objectives


class MyObserver(Observer):
    def update(self, *args, **kwargs):
        evaluations = kwargs['evaluations']
        population = kwargs['population']
        computing_time = kwargs['computing time']
        cps = evaluations / computing_time if computing_time > 0 else '-'
        logging.info(
            f'{evaluations} evaluations after {computing_time:.1f}s @{cps:.2f}cps, best fitness: {str(population[0].objectives)}')


def optimize(length: float, depth: float, variables: int, material: Union[str, pd.DataFrame], population_size: int,
             max_evaluations: int):
    problem = HomogenizationProblem(
        length=length,
        depth=depth,
        number_of_variables=variables,
        material=material
    )

    algorithm = HPSEA[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=population_size,
        max_evaluations=max_evaluations,
        mutation=Polynomial(1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
        evaluator=MapEvaluator(processes=8)
    )

    algorithm.observable.register(MyObserver())

    plot_server = PlotServer(problem.get_new_solutions)
    plot_server.serve_background()

    algorithm.run()
    all_variables, all_objectives = problem.get_all_solutions()

    return OptimizationResult(
        result_population=algorithm.get_result(),
        all_variables=all_variables,
        all_objectives=all_objectives
    ), problem


def write_optimization_result_to_file(optimization_result: OptimizationResult, problem: HomogenizationProblem,
                                      directory: str):
    variables = pd.DataFrame(data=[solution.variables for solution in optimization_result.result_population])
    variables.columns = problem.get_variable_labels()
    variables.to_csv(f'{directory}/variables.csv', sep='\t', index=False)

    objectives = pd.DataFrame(
        data=[solution.objectives for solution in optimization_result.result_population],
        columns=problem.get_objective_labels()
    )
    objectives.to_csv(f'{directory}/objectives.csv', sep='\t', index=False)

    all_variables_df = pd.DataFrame(data=optimization_result.all_variables, columns=problem.get_variable_labels())
    all_variables_df.to_csv(f'{directory}/all_variables.csv', sep='\t', index=False)

    all_objectives_df = pd.DataFrame(data=optimization_result.all_objectives, columns=problem.get_objective_labels())
    all_objectives_df.to_csv(f'{directory}/all_objectives.csv', sep='\t', index=False)


def plot_optimization_result(optimization_result: OptimizationResult, problem, directory: str):
    df = pd.DataFrame(
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
        material=args.material,
        population_size=args.population_size,
        max_evaluations=args.max_evaluations
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
