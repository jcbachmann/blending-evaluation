#!/usr/bin/env python
import os
import subprocess

import pandas as pd

from roundness import RoundnessEvaluator


def execute_for_roundness(likelihood, dist_seg_size, angle_seg_count, pos, volume, run):
	print('processing volume %d with likelihood %f (run %d)' % (volume, likelihood, run))
	path = '/tmp/heights-%d-%.4f-%d.txt' % (volume, likelihood, run)

	with subprocess.Popen(
			['./BlendingSimulator', '--config', 'pile.conf', '--heights', path, '--eight', str(likelihood)],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE
	) as sim:
		sim.communicate(('0 %f %f' % (pos, volume)).encode())

	e = RoundnessEvaluator(dist_seg_size, angle_seg_count)
	e.add_from_file(path)
	return e.evaluate()


def execute_for_bulk_density(
		pos: float,
		size: float,
		volume: float,
		ppm3: float,
		run: int,
		params: list,
		path: str,
		executable: str = './BlendingSimulator'
):
	print('processing volume %d with ppm3 %.1f (run %d)' % (volume, ppm3, run))
	path += '/heights-vol%d-res%.1f-run%d.txt' % (volume, ppm3, run)

	with subprocess.Popen([executable, '--length', str(size), '--depth', str(size), '--heights', path, '--ppm3',
						   str(ppm3)] + params,
						  stdin=subprocess.PIPE,
						  stdout=subprocess.PIPE
						  ) as sim:
		sim.communicate(('0 %f %f' % (pos, volume)).encode())

	return pd.read_csv(path, header=None, delimiter='\t', index_col=None)


def load_for_bulk_density(file: str, path: str = '/tmp'):
	if path is None:
		path = '/tmp'
	path += '/%s' % file
	if os.path.isfile(path):
		return pd.read_csv(path, header=None, delimiter='\t', index_col=None)
	else:
		return None
