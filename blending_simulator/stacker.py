#!/usr/bin/env python
import argparse
import sys
from typing import Union

import numpy as np
import pandas as pd


def read_material(filepath: str, col_timestamp: str = 'timestamp', col_volume: str = 'volume', cols_p: [str] = None) -> pd.DataFrame:
	df = pd.read_csv(filepath, delimiter='\t', index_col=None)
	if cols_p is None:
		# Use all columns except for timestamp and volume
		cols_p = list(df.columns.drop(col_timestamp, col_volume))
	required_cols = [col_timestamp, col_volume] + cols_p
	if not set(required_cols).issubset(df.columns):
		raise Exception(f'required columns ({required_cols}) not found in material file')
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
	def __init__(self, length: float, depth: float, status=None):
		# Blending bed parameters
		self.length = length
		self.depth = depth

		# Status message callback
		self.status = lambda msg: None if status is None else status

	def run(self, material: pd.DataFrame, path: pd.DataFrame, callback) -> None:
		self.status('Starting stacker')

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

		material['z'] = self.depth / 2
		material['x'] = np.interp(material['timestamp'], path['timestamp'], path['path']) * (
				max_pos - min_pos) + min_pos

		try:
			callback(material)
		except IOError:
			self.status('Stopping stacker due to IOError')

		self.status('Stacker stopped')


class StrToBytesWrapper:
	def __init__(self, bytes_buffer):
		self.bytes_buffer = bytes_buffer

	def write(self, s):
		self.bytes_buffer.write(s.encode('utf-8'))


class StackerPrinter:
	def __init__(self, header=True, out_buffer=sys.stdout.buffer):
		self.header = header
		self.out_buffer = out_buffer

	@staticmethod
	def status(msg):
		print("[stacker] %s" % msg, file=sys.stderr)

	def out(self, material):
		first_cols = ['timestamp', 'x', 'z', 'volume']
		col_order = first_cols
		col_order.extend(list(set(material.columns) - set(first_cols)))
		material = material.reindex(columns=col_order)
		material.to_csv(StrToBytesWrapper(self.out_buffer), index=False, header=self.header, sep=' ')

	def flush(self):
		self.out_buffer.flush()


def stack_with_printer(
		length: float,
		depth: float,
		material: Union[str, pd.DataFrame],
		stacker_path: Union[str, pd.DataFrame],
		header: bool = True,
		out_buffer=sys.stdout.buffer
):
	printer = StackerPrinter(header=header, out_buffer=out_buffer)

	if isinstance(material, str):
		material = read_material(material)

	if isinstance(stacker_path, str):
		stacker_path = read_path(stacker_path)

	Stacker(
		length,
		depth,
		status=printer.status
	).run(
		material,
		stacker_path,
		callback=printer.out
	)
	printer.flush()


def main(args):
	stack_with_printer(
		length=args.length,
		depth=args.depth,
		material=args.material,
		stacker_path=args.stacker_path
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='height map evaluator')
	parser.add_argument('--length', type=float, required=True, help='Blending bed length')
	parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
	parser.add_argument('--material', type=str, required=True, help='Material input file')
	parser.add_argument('--stacker_path', type=str, required=True, help='Stacker traverse path file')

	main(parser.parse_args())
