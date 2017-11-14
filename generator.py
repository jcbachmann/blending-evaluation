#!/usr/bin/env python
import argparse
import math
import signal
import sys


class Generator:
	finish = False

	def __init__(self, args):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)

		# Blending bed parameters
		self.length = args.length
		self.depth = args.length

		# Material flow
		self.m3_per_second = 1.5

		# Material parameter output period
		self.t_diff = 1

		# Only 85% of the available depth should be used in average
		self.depth_fill_factor = 0.85

	def run(self):
		self.status('Starting generator')

		# Stacker path parameters
		min_pos = self.depth / 2
		max_pos = self.length - self.depth / 2

		# Total volume in cubic meters
		m3_total = (max_pos - min_pos) * pow(self.depth_fill_factor * self.depth / 2, 2)

		pos_period = 2111
		t = 0
		try:
			while not self.finish and m3_total > 0:
				# Position
				distance = (t % pos_period) / pos_period
				if int(t / pos_period) % 2:
					distance = 1 - distance
				pos = distance * (max_pos - min_pos) + min_pos

				# Mixture
				red_part = 0.5 + 0.5 * math.sin((t * 0.001) * 2 * math.pi)  # 1/(1+exp(-x))
				blue_part = 0.5 + 0.5 * math.sin((t * 0.001 + 0.333) * 2 * math.pi)
				yellow_part = 0.5 + 0.5 * math.sin((t * 0.001 + 0.666) * 2 * math.pi)
				m3_this_time = min(self.m3_per_second * self.t_diff, m3_total)

				if red_part > blue_part:
					if red_part > yellow_part:
						sys.stdout.write('%f %f %f %f %f %f\n' % (t, pos, m3_this_time, 1, 0, 0))
					else:
						sys.stdout.write('%f %f %f %f %f %f\n' % (t, pos, m3_this_time, 0, 0, 1))
				else:
					if blue_part > yellow_part:
						sys.stdout.write('%f %f %f %f %f %f\n' % (t, pos, m3_this_time, 0, 1, 0))
					else:
						sys.stdout.write('%f %f %f %f %f %f\n' % (t, pos, m3_this_time, 0, 0, 1))

				t += self.t_diff
				m3_total -= m3_this_time

			sys.stdout.close()
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
	Generator(args).run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='height map evaluator')
	parser.add_argument('--length', type=float, default=300, help='Blending bed length')
	parser.add_argument('--depth', type=float, default=50, help='Blending bed depth')

	main(parser.parse_args())
