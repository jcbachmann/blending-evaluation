#!/usr/bin/env python
import os

import pandas as pd

from blendingsimulator import BlendingSimulator
from roundness import RoundnessEvaluator


def execute_for_roundness(likelihood, dist_seg_size, angle_seg_count, pos, volume, run):
	print('processing volume %d with likelihood %f (run %d)' % (volume, likelihood, run))
	path = '/tmp/heights-%d-%.4f-%d.txt' % (volume, likelihood, run)

	BlendingSimulator(config='pile.conf', heights=path, eight=likelihood).run(
		lambda sim: sim.communicate(('0 %f %f' % (pos, volume)).encode())
	)

	e = RoundnessEvaluator(dist_seg_size, angle_seg_count)
	e.add_from_file(path)
	return e.evaluate()


def execute_for_bulk_density(
		pos: float,
		size: float,
		volume: float,
		ppm3: float,
		run: int,
		dropheight: float,
		detailed: bool,
		visualize: bool,
		bulkdensity: float,
		path: str,
		executable: str = './BlendingSimulator'
):
	print('processing volume %d with ppm3 %.1f (run %d)' % (volume, ppm3, run))
	path += '/heights-vol%d-res%.1f-run%d.txt' % (volume, ppm3, run)

	BlendingSimulator(
		executable=executable,
		length=size,
		depth=size,
		heights=path,
		ppm3=ppm3,
		dropheight=dropheight,
		detailed=detailed,
		visualize=visualize,
		bulkdensity=bulkdensity
	).run(
		lambda sim: sim.communicate(('0 %f %f %f' % (pos, pos, volume)).encode())
	)

	return pd.read_csv(path, header=None, delimiter='\t', index_col=None)


def load_for_bulk_density(file: str, path: str = '/tmp'):
	if path is None:
		path = '/tmp'
	path += '/%s' % file
	if os.path.isfile(path):
		return pd.read_csv(path, header=None, delimiter='\t', index_col=None)
	else:
		return None
