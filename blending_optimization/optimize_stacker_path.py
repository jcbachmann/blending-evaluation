#!/usr/bin/env python
import argparse
import logging
import os
import time
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from blending_optimization.homogenization_problem import HomogenizationProblem
from jmetalpy.jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetalpy.jmetal.component.evaluator import ParallelEvaluator
from jmetalpy.jmetal.core.solution import FloatSolution
from jmetalpy.jmetal.operator.crossover import SBX
from jmetalpy.jmetal.operator.mutation import Polynomial
from jmetalpy.jmetal.operator.selection import BinaryTournamentSelection
from jmetalpy.jmetal.util.comparator import RankingAndCrowdingDistanceComparator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationResult:
	def __init__(self, result_population, all_variables, all_objectives):
		self.result_population = result_population
		self.all_variables = all_variables
		self.all_objectives = all_objectives


def optimize(length: float, depth: float, variables, material, population_size: int, max_evaluations: int):
	problem = HomogenizationProblem(
		length=length,
		depth=depth,
		number_of_variables=variables,
		material=material
	)

	algorithm = NSGAII[FloatSolution, List[FloatSolution]](
		problem=problem,
		population_size=population_size,
		max_evaluations=max_evaluations,
		mutation=Polynomial(1.0 / problem.number_of_variables, distribution_index=20),
		crossover=SBX(1.0, distribution_index=20),
		selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()),
		evaluator=ParallelEvaluator(processes=8)
	)

	algorithm.run()
	all_variables, all_objectives = problem.get_all_solutions()

	return OptimizationResult(
		result_population=algorithm.get_result(),
		all_variables=all_variables,
		all_objectives=all_objectives
	), problem


def write_optimization_result_to_file(optimization_result: OptimizationResult, problem, directory: str = ''):
	variables = pd.DataFrame(data=[solution.variables for solution in optimization_result.result_population])
	variables.columns = problem.get_variable_labels()
	variables.to_csv(directory + '/variables.csv', sep='\t', index=False)

	objectives = pd.DataFrame(
		data=[solution.objectives for solution in optimization_result.result_population],
		columns=problem.get_objective_labels()
	)
	objectives.to_csv(directory + '/objectives.csv', sep='\t', index=False)

	all_variables_df = pd.DataFrame(data=optimization_result.all_variables, columns=problem.get_variable_labels())
	all_variables_df.to_csv(directory + '/all_variables.csv', sep='\t', index=False)

	all_objectives_df = pd.DataFrame(data=optimization_result.all_objectives, columns=problem.get_objective_labels())
	all_objectives_df.to_csv(directory + '/all_objectives.csv', sep='\t', index=False)


def plot_optimization_result(optimization_result: OptimizationResult, problem):
	df = pd.DataFrame(
		data=[solution.objectives for solution in optimization_result.result_population],
		columns=problem.get_objective_labels()
	)
	for c0 in range(len(df.columns) - 1):
		for c1 in range(c0 + 1, len(df.columns)):
			df.plot(x=df.columns[c0], y=df.columns[c1], kind='scatter')

	plt.show()


def main(args) -> None:
	directory = time.strftime('%Y-%m-%d %H-%M-%S')
	if os.path.exists(directory):
		raise Exception('directory "%s" already exists' % directory)

	os.makedirs(directory)

	optimization_result, problem = optimize(
		length=args.length,
		depth=args.depth,
		variables=args.variables,
		material=args.material,
		population_size=args.population_size,
		max_evaluations=args.max_evaluations
	)

	write_optimization_result_to_file(optimization_result, problem, directory=directory)

	plot_optimization_result(optimization_result, problem)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='height map evaluator')
	parser.add_argument('--length', type=float, required=True, help='Blending bed length')
	parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
	parser.add_argument('--material', type=str, required=True, help='Material input file')
	parser.add_argument('--variables', type=int, required=True, help='Amount of variables')
	parser.add_argument('--population_size', type=int, required=True, help='Amount of individuals in population')
	parser.add_argument('--max_evaluations', type=int, required=True, help='Maximum amount of function evaluations')

	main(parser.parse_args())
