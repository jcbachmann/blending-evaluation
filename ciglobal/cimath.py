import math

import numpy as np


def stdev(values: np.array):
	variance = np.average((values - np.average(values)) ** 2)
	return math.sqrt(variance)


def weighted_avg_and_std(values, weights):
	average = np.average(values, weights=weights)
	variance = np.average((values - average) ** 2, weights=weights)
	return average, math.sqrt(variance)
