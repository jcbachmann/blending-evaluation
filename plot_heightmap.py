#!/usr/bin/env python
import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import *


def process_file(filename):
	length = 0
	depth = 0
	z = []
	with open(filename) as f:
		for line in csv.reader(f, delimiter='\t'):
			length += 1
			depth = len(line)
			z.append([float(i) for i in line])

	m = max(depth, length)
	x, y = np.meshgrid(np.arange(depth), np.arange(length))
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot_surface(x, y, np.array(z), cmap=cm.jet, rstride=5, cstride=5)
	ax.auto_scale_xyz([0, m], [0, m], [0, m])
	plt.show()


def main(args):
	process_file(args.filename)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="script converting height map to png visualization".strip())
	parser.add_argument('filename', type=str, help='heightmap file to process')
	args = parser.parse_args()

	main(args)
