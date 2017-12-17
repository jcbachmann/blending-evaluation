import matplotlib.pyplot as plt
import numpy as np

from data_explorer.rating import RaterColorScale, Rater
from data_explorer.testlet import Testlet
from solution_explorer.solution import Solution


def white_to_red(p):
	f = min(1, 1.7 - p)
	return [1 * f, (1 - p) * f, (1 - p) * f]


class ObjectiveTestlet(Testlet):
	def __init__(self, index: float, maximum: float, label: str = None):
		self.index = index
		self.maximum = maximum
		self.label = label

	def __str__(self):
		if self.label is not None:
			return self.label

		return f'Objective {self.index}'

	def evaluate(self, solution: Solution):
		value = solution.objectives[self.index]
		return value, f'{value:.3f}'

	def get_result_rater(self) -> Rater:
		return RaterColorScale(minimum=0, maximum=self.maximum, color_func=white_to_red)


class PathDistanceTestlet(Testlet):
	def __init__(self, variables_count: int):
		self.variables_count = variables_count

	def __str__(self):
		return 'Distance'

	def evaluate(self, solution: Solution):
		value = sum(abs(solution.variables[:-1] - solution.variables[1:]))
		return value, f'{value:.2f}'

	def get_result_rater(self) -> Rater:
		return RaterColorScale(minimum=0, maximum=self.variables_count - 1, color_func=white_to_red)


class PathGraphTestlet(Testlet):
	def __str__(self):
		return 'Path'

	def evaluate(self, solution: Solution):
		fig = plt.figure(figsize=(2, 1), dpi=75)
		ax = fig.add_subplot(111)
		ax.plot(np.linspace(0, 1, len(solution.variables)), solution.variables, ls='-')
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		fig.tight_layout(pad=0)

		return fig, fig
