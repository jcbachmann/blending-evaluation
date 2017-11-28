#!/usr/bin/env python
import subprocess


class BlendingSimulator:
	def __init__(
			self,
			executable='./BlendingSimulator',
			config: str = None,
			verbose: bool = False,
			detailed: bool = False,
			circular: bool = False,
			length: float = None,
			depth: float = None,
			reclaimangle: float = None,
			eight: float = None,
			bulkdensity: float = None,
			ppm3: float = None,
			dropheight: float = None,
			reclaimincrement: float = None,
			visualize: bool = False,
			pretty: bool = False,
			heights: str = None,
			reclaim: str = None,
	):
		self.executable = executable
		self.config = config
		self.verbose = verbose
		self.detailed = detailed
		self.circular = circular
		self.length = length
		self.depth = depth
		self.reclaimangle = reclaimangle
		self.eight = eight
		self.bulkdensity = bulkdensity
		self.ppm3 = ppm3
		self.dropheight = dropheight
		self.reclaimincrement = reclaimincrement
		self.visualize = visualize
		self.pretty = pretty
		self.heights = heights
		self.reclaim = reclaim

	def run(self, observer):
		p = [self.executable]

		if self.config is not None:
			p.extend(['--config', self.config])
		if self.verbose:
			p.append('--verbose')
		if self.detailed:
			p.append('--detailed')
		if self.circular:
			p.append('--circular')
		if self.length is not None:
			p.extend(['--length', str(self.length)])
		if self.depth is not None:
			p.extend(['--depth', str(self.depth)])
		if self.reclaimangle is not None:
			p.extend(['--reclaimangle', str(self.reclaimangle)])
		if self.eight is not None:
			p.extend(['--eight', str(self.eight)])
		if self.bulkdensity is not None:
			p.extend(['--bulkdensity', str(self.bulkdensity)])
		if self.ppm3 is not None:
			p.extend(['--ppm3', str(self.ppm3)])
		if self.dropheight is not None:
			p.extend(['--dropheight', str(self.dropheight)])
		if self.reclaimincrement is not None:
			p.extend(['--reclaimincrement', str(self.reclaimincrement)])
		if self.visualize:
			p.append('--visualize')
		if self.pretty:
			p.append('--pretty')
		if self.heights is not None:
			p.extend(['--heights', self.heights])
		if self.reclaim is not None:
			p.extend(['--reclaim', self.reclaim])

		with subprocess.Popen(
				p,
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE
		) as sim:
			observer(sim)
			sim.stdin.close()
			sim.wait()
