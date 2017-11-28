#!/usr/bin/env python
import argparse
import signal
import sys

import pandas as pd


class Chevron:
	finish = False

	def __init__(self, args):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)

		# Blending bed parameters
		self.length = args.length
		self.depth = args.depth

		self.material = pd.read_csv(args.material, delimiter='\t', index_col=None)
		self.layers = args.layers

	def run_linear(self):
		# Stacker path parameters
		min_pos = self.depth / 2
		max_pos = self.length - self.depth / 2

		# Total volume in cubic meters
		t_total = self.material['timestamp'].max()
		t_per_layer = t_total / self.layers
		z = self.depth / 2

		for _, row in self.material.iterrows():
			if self.finish:
				break

			t = float(row['timestamp'])

			# Position
			p = (t % t_per_layer) / t_per_layer
			if int(t / t_per_layer) % 2:
				p = 1 - p
			x = p * (max_pos - min_pos) + min_pos

			sys.stdout.write('%f %f %f %f %f\n' % (t, x, z, row['volume'], row['p1']))

		sys.stdout.close()

	def run(self):
		self.status('Starting generator')
		try:
			self.run_linear()
		except IOError:
			self.status('Stopping generator due to IOError')
			self.finish = True
		self.status('Generator stopped')

	@staticmethod
	def status(msg):
		print("[generator] " + msg, file=sys.stderr)

	def signal_handler(self, _signum, _frame):
		self.status('Stopping generator')
		self.finish = True


def main(args):
	Chevron(args).run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='height map evaluator')
	parser.add_argument('--length', type=float, default=300, help='Blending bed length')
	parser.add_argument('--depth', type=float, default=50, help='Blending bed depth')
	parser.add_argument('--material', type=str, required=True, help='Material input file')
	parser.add_argument('--layers', type=float, default=1, help='Speed')

	main(parser.parse_args())
