#!/usr/bin/env python
import argparse
import math
import os
import re
import subprocess
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn.palettes import color_palette


def execute(args, reclaim, layers):
	with subprocess.Popen(
			[
				'./BlendingSimulator', '--length', str(args.length), '--depth', str(args.depth), '--dropheight',
				str(args.depth / 2), '--reclaim', reclaim, '--parameters', '1', '--ppm3', '10'
			],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE
	) as sim:
		with subprocess.Popen(
				[
					'../chevron_stacker.py', '--length', str(args.length), '--depth', str(args.depth), '--material',
					args.material, '--layers', str(layers)
				],
				stdin=subprocess.PIPE,
				stdout=sim.stdin
		) as generator:
			generator.wait()
		sim.stdin.close()
		sim.wait()


def weighted_avg_and_std(values, weights):
	average = np.average(values, weights=weights)
	variance = np.average((values - average) ** 2, weights=weights)
	return average, math.sqrt(variance)


def get_results_for_file(layers, file, path):
	df = pd.read_csv(path + '/' + file, delimiter='\t', index_col=None)

	minvol = df['volume'].sum() / len(df.index)
	larger = df.query('volume>=%f' % minvol)
	smaller = df.query('volume<%f' % minvol)
	lbound = min(larger['p1'].min(), np.average(smaller['p1'], weights=smaller['volume']))
	mean, std = weighted_avg_and_std(df['p1'], weights=df['volume'])
	lstd = mean - std
	ustd = mean + std
	ubound = max(larger['p1'].max(), np.average(smaller['p1'], weights=smaller['volume']))

	return pd.DataFrame(
		data=[(layers, lbound, lstd, mean, ustd, ubound)],
		columns=['layers', 'lbound', 'lstd', 'mean', 'ustd', 'ubound']
	)


def load_results_from_path(path):
	results = pd.DataFrame()

	r1 = re.compile('reclaim-layers-([\d.]+)\.csv')
	for file in os.listdir(path):
		g = r1.match(file)
		if g:
			layers = float(g.group(1))
			results = results.append(get_results_for_file(layers, file, path))

	return results.sort_values('layers')


def plot_results(df):
	ax = plt.gca()
	colors = color_palette(n_colors=1)

	df = df.sort_values('layers')
	x = df['layers'].values

	ax.fill_between(x, df['lbound'], df['ubound'], facecolor=colors[0], alpha=0.2)
	ax.fill_between(x, df['lstd'], df['ustd'], facecolor=colors[0], alpha=0.2)
	ax.plot(x, df['mean'].values, color=colors[0], marker='', linestyle='-')

	ax.set_xlabel('layers')
	ax.set_ylabel('quality')
	ax.legend(loc=0)

	return ax


def main(args):
	if not os.path.exists(args.path):
		os.makedirs(args.path)

	if not args.reuse:
		p = Pool(8)
		p.starmap(execute, [
			(args, '%s/reclaim-layers-%.4f.csv' % (args.path, layers), layers) for layers in np.linspace(1, 100, 397)
		])

	results = load_results_from_path(args.path)
	plot_results(results)
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='height map evaluator')
	parser.add_argument('--length', type=float, required=True, help='Blending bed length')
	parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
	parser.add_argument('--material', type=str, required=True, help='Material input file')
	parser.add_argument('--path', type=str, default='/tmp', help='Output path')
	parser.add_argument('--reuse', action='store_true', help='Use material available in path')

	main(parser.parse_args())
