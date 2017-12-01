#!/usr/bin/env python
import argparse
import signal
import sys

import numpy as np
import pandas as pd


def read_material(filepath: str, col_timestamp: str = 'timestamp', col_volume: str = 'volume', cols_p: [str] = None) -> pd.DataFrame:
	df = pd.read_csv(filepath, delimiter='\t', index_col=None)
	if cols_p is None:
		# Use all columns except for timestamp and volume
		cols_p = list(df.columns.drop(col_timestamp, col_volume))
	required_cols = [col_timestamp, col_volume] + cols_p
	if not set(required_cols).issubset(df.columns):
		raise Exception('required columns (%s) not found in material file' % required_cols)
	material = pd.DataFrame()
	material['timestamp'] = df[col_timestamp]
	material['volume'] = df[col_volume]
	for i, col_p in enumerate(cols_p):
		material[col_p] = df[col_p]
	return material


def read_path(filepath: str, col_path: str = 'path', col_part: str = None, col_timestamp: str = None) -> pd.DataFrame:
	required_cols = [col_path]
	if col_part is not None:
		required_cols += [col_part]
	if col_timestamp is not None:
		required_cols += [col_timestamp]
	path = pd.read_csv(filepath, delimiter='\t', index_col=None)
	if not set(required_cols).issubset(path.columns):
		raise Exception('required columns (%s) not found in path file', required_cols)
	path['path'] = path[col_path]
	if col_part is not None:
		path['part'] = path[col_part]
	if col_timestamp is not None:
		path['timestamp'] = path[col_timestamp]
	return path


class Stacker:
	finish = False

	def __init__(self, length: float, depth: float, status=None):
		signal.signal(signal.SIGINT, self.signal_handler)
		signal.signal(signal.SIGTERM, self.signal_handler)

		# Blending bed parameters
		self.length = length
		self.depth = depth

		# Status message callback
		self.status = lambda msg: None if status is None else status

	def run(self, material: pd.DataFrame, path: pd.DataFrame, callback) -> None:
		self.status('Starting stacker')
		try:
			# Stacker path parameters
			min_pos = self.depth / 2
			max_pos = self.length - self.depth / 2

			# Total volume in cubic meters
			t_total = material['timestamp'].max()
			if 'timestamp' not in path.columns:
				# No timestamps provided - generate time stamps
				if 'part' in path.columns:
					# Position relative to time is known
					path['timestamp'] = path['part'] / path['part'].max() * t_total
				else:
					n = len(path.index)
					if n > 1:
						path['timestamp'] = [t_total * i / (n - 1) for i in range(n)]
					else:
						path['timestamp'] = [0]

			z = self.depth / 2

			# noinspection PyTypeChecker
			for _, row in material.iterrows():
				if self.finish:
					break

				t = float(row['timestamp'])

				# Position
				p = np.interp([t], path['timestamp'], path['path'])[0]
				x = p * (max_pos - min_pos) + min_pos

				callback(t, x, z, row['volume'], row.drop(['timestamp', 'volume']))

		except IOError:
			self.status('Stopping stacker due to IOError')
			self.finish = True
		self.status('Stacker stopped')

	def signal_handler(self, _signum, _frame) -> None:
		self.status('Stopping stacker')
		self.finish = True


class Printer:
	def __init__(self, header=True):
		self.header = header

	@staticmethod
	def status(msg):
		print("[stacker] %s" % msg, file=sys.stderr)

	def out(self, timestamp, x, z, volume, parameters):
		if self.header:
			self.header = False
			sys.stdout.write('%s %s %s %s %s\n' % ('timestamp', 'x', 'z', 'volume', ' '.join(parameters.index)))
		sys.stdout.write('%f %f %f %f %s\n' % (timestamp, x, z, volume, ' '.join([str(i) for i in parameters])))

	@staticmethod
	def close():
		sys.stdout.close()


def main(args):
	printer = Printer()
	Stacker(
		args.length,
		args.depth,
		status=printer.status
	).run(
		read_material(args.material),
		read_path(args.stacker_path),
		callback=printer.out
	)
	printer.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='height map evaluator')
	parser.add_argument('--length', type=float, required=True, help='Blending bed length')
	parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
	parser.add_argument('--material', type=str, required=True, help='Material input file')
	parser.add_argument('--stacker_path', type=str, required=True, help='Stacker traverse path file')

	main(parser.parse_args())