import logging

from plant_simulator.material_handler import MaterialHandler
from plant_simulator.plant import Plant
from .reclaimer import Reclaimer
from .stacker import Stacker
from .stockpile import Stockpile


class BlendingSystem(MaterialHandler):
    def __init__(self, label, plant, src, length: float = 600, depth: float = 50, max_stacker_speed: float = 0.5,
                 max_reclaimer_speed: float = 0.005, strategy: str = 'A', strategy_params=None,
                 simulator: str = 'fast'):
        super().__init__(label, plant, 'cylinder')

        self.src_gen = self.unpack_src_gen(src)
        self._sample = (0, 0)
        self.length = length
        self.depth = depth
        self.stacker = Stacker(max_stacker_speed, strategy, strategy_params)
        self.reclaimer = Reclaimer(max_reclaimer_speed)
        self.simulator = simulator

    def step_blending_system(self, tph, q):
        if self.stacker.stacking_finished():
            if self.reclaimer.reclaiming_finished():
                # Move stockpile from stacker to reclaimer
                self.reclaimer.stockpile = self.stacker.stockpile
                self.stacker.stockpile = None
            else:
                logging.warning('Illegal situation occurred: stacker is full while reclaimer is not yet finished')
                logging.warning(f'Material illegally dropped: {tph} tph, quality: {q}')
                tph, q = 0, 0

        if self.stacker.stockpile is None:
            # Create new stockpile for stacker
            self.stacker.stockpile = Stockpile(length=0.5 * self.length, depth=self.depth, simulator=self.simulator)
            self.stacker.last_pos = None

        self.stacker.stack(self.plant.time, tph * Plant.TIME_INCREMENT / 3600, q)
        tons, q = self.reclaimer.reclaim()
        return tons * 3600 / Plant.TIME_INCREMENT, q

    def gen(self, i: int = 0):
        while True:
            tph, q = self.step_blending_system(*next(self.src_gen))
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]
