#!/usr/bin/env python
import logging
import subprocess
from typing import List, Union

from blending_simulator.blending_simulator import BlendingSimulator


class ExternalBlendingSimulatorInterface:
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
            reclaim: str = None
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

    def get_process_arguments(self):
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

        return p

    def run(self, observer):
        with self.start() as sim_popen:
            observer(sim_popen)
            return self.stop(sim_popen)

    def start(self) -> subprocess.Popen:
        sim_popen = subprocess.Popen(
            self.get_process_arguments(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
            # stderr=subprocess.PIPE,
            # bufsize=1000000
        )
        # t = threading.Thread(target=partial(self.log_errors, sim_popen))
        # t.start()
        return sim_popen

    @staticmethod
    def log_errors(sim_popen: subprocess.Popen):
        logging.info('Starting popen reader')
        for line in sim_popen.stderr:
            logging.info(f'Simulator: {line.decode().strip()}')
        logging.info('Popen reader stopped')

    @staticmethod
    def stop(sim_popen: subprocess.Popen):
        logging.info('Closing stdin')
        sim_popen.stdin.close()
        logging.info('Waiting for simulator')
        sim_popen.wait()
        logging.info('Reading simulator output')
        return sim_popen.stdout.read()


class ExternalBlendingSimulator(BlendingSimulator):
    def __init__(self, bed_size_x: float, bed_size_z: float, **kwargs):
        super().__init__(bed_size_x, bed_size_z)
        sim = ExternalBlendingSimulatorInterface(
            length=bed_size_x,
            depth=bed_size_z,
            dropheight=0.5 * bed_size_z,
            reclaim='stdout',
            **kwargs
        )
        self.sim_popen = sim.start()
        self.stopped = False
        self.reclaimed = None

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        if self.stopped:
            raise Exception('Can not call stack on blending simulator where reclaiming has started')

        self.sim_popen.stdin.write(
            (' '.join([str(timestamp), str(x), str(z), str(volume)] + [str(p) for p in parameter]) + '\n').encode(
                'utf-8'))

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        if not self.stopped:
            logging.info('Stopping blending simulator')
            self.stopped = True
            out = ExternalBlendingSimulatorInterface.stop(self.sim_popen).decode()
            data = [[float(element) for element in line.split('\t')] for line in out.split('\n')[1:-1]]
            self.reclaimed = [[d[0], d[1], d[2:]] for d in data]

        return self.reclaimed
