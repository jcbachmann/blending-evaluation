#!/usr/bin/env python

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
