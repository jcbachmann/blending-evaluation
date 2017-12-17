from io import StringIO
from typing import List

import numpy as np
import pandas as pd
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from blending_simulator.blending_simulator import BlendingSimulator
from blending_simulator.stacker import read_material, stack_with_printer
from ciglobal.cimath import weighted_avg_and_std, stdev


def evaluate_solution(
		length: float,
		depth: float,
		material: pd.DataFrame,
		raw_quality_stdev: float,
		stacker_path_list: [float]
) -> [float, float]:
	out = BlendingSimulator(
		length=length,
		depth=depth,
		dropheight=(depth / 2),
		reclaim='stdout',
		ppm3=1
	).run(lambda sim: stack_with_printer(
		length=length,
		depth=depth,
		material=material,
		stacker_path=pd.DataFrame(data=stacker_path_list, columns=['path']),
		header=False,
		out_buffer=sim.stdin
	))

	df = pd.read_csv(StringIO(out.decode('utf-8')), delimiter='\t', index_col=None)

	# TODO make work with more parameters
	quality_average, quality_stdev = weighted_avg_and_std(df['p_1'], df['volume'])

	# TODO make work with xz_scaling != 1
	central_reclaim_volumes = df['volume'].values[int(depth):int(length - depth)]
	central_reclaim_volumes[0] += df['volume'].values[:int(depth)].sum()
	central_reclaim_volumes[-1] += df['volume'].values[int(length - depth):].sum()
	l = len(central_reclaim_volumes)
	split_volume = df['volume'].sum() / (l / 2)
	bad_case_volumes = np.array([split_volume if i < l / 2 else 0 for i in range(l)])
	worst_case_volume_stdev = stdev(bad_case_volumes)
	volume_stdev = stdev(central_reclaim_volumes) / worst_case_volume_stdev

	return [quality_stdev / raw_quality_stdev, volume_stdev]


class HomogenizationProblem(FloatProblem):
	def __init__(self, length: float, depth: float, material: str, number_of_variables: int = 2):
		super().__init__()

		self.length = length
		self.depth = depth
		self.material = material

		if isinstance(self.material, str):
			self.material = read_material(self.material)

		_, self.material_quality_stdev = weighted_avg_and_std(self.material['p_1'], self.material['volume'])
		self.number_of_objectives = 2
		self.number_of_variables = number_of_variables
		self.number_of_constraints = 0

		self.lower_bound = [0.0 for _ in range(number_of_variables)]
		self.upper_bound = [1.0 for _ in range(number_of_variables)]

		FloatSolution.lower_bound = self.lower_bound
		FloatSolution.upper_bound = self.upper_bound

		self.evaluated_variables = []
		self.evaluated_objectives = []

	def evaluate(self, solution: FloatSolution) -> None:
		solution.objectives = evaluate_solution(
			self.length,
			self.depth,
			self.material,
			self.material_quality_stdev,
			solution.variables
		)

		self.evaluated_variables.append(solution.variables)
		self.evaluated_objectives.append(solution.objectives)

	@staticmethod
	def get_objective_labels() -> List[str]:
		return ['Quality Stdev', 'Volume Stdev']

	def get_variable_labels(self) -> List[str]:
		return ['v%d' % (i + 1) for i in range(self.number_of_variables)]

	def get_all_solutions(self):
		return self.evaluated_variables, self.evaluated_objectives