import logging

from blending_simulator.external_blending_simulator import ExternalBlendingSimulator
from blending_simulator.mathematical_blending_simulator import MathematicalBlendingSimulator
from blending_simulator.smooth_blending_simulator import SmoothBlendingSimulator
from .material_handler import MaterialHandler
from .plant import Plant


class Stockpile:
    def __init__(self, length: float, depth: float, simulator: str = 'fast'):
        self.length = length
        self.depth = depth
        self.max_tons = (length - depth) * 0.25 * depth * depth
        self.stacked_tons = 0
        self.stack_time_start = None
        self.stack_time_end = None
        self._stacking_finished = False
        self.reclaimed_buffer = None
        self.reclaimer_position = 0

        self.simulator = {
            'fast': lambda: ExternalBlendingSimulator(
                bed_size_x=length,
                bed_size_z=depth,
                dropheight=(depth / 2),
                reclaim='stdout',
                ppm3=1
            ),
            'smooth': lambda: SmoothBlendingSimulator(bed_size_x=length, sigma=10),
            'mathematical': lambda: MathematicalBlendingSimulator(bed_size_x=length),
        }[simulator]()

    def stack(self, timestamp, pos, tons, q):
        if tons > 0:
            if self.stacked_tons == 0:
                self.stack_time_start = timestamp - Plant.TIME_INCREMENT
            self.stack_time_end = timestamp

        volume = tons
        self.simulator.stack(timestamp, pos, self.depth / 2, volume, [q])
        self.stacked_tons += tons

        if self.stacked_tons >= self.max_tons:
            self._stacking_finished = True

    def stacking_finished(self):
        return self._stacking_finished

    def reclaim(self, speed):
        self._stacking_finished = True

        if self.reclaimed_buffer is None:
            self.reclaimed_buffer = self.simulator.reclaim()

        if self.reclaimer_position < self.length:
            old_pos = self.reclaimer_position
            new_pos = min(self.reclaimer_position + speed * Plant.TIME_INCREMENT, self.length)
            new_pos_index = new_pos / self.length * len(self.reclaimed_buffer)
            old_pos_index = old_pos / self.length * len(self.reclaimed_buffer)
            first_index = int(old_pos_index)
            last_index = min(int(new_pos_index), len(self.reclaimed_buffer) - 1)

            volume, q = 0, 0
            for i in range(first_index, last_index + 1):
                factor = min(float(i) + 1, new_pos_index) - max(float(i), old_pos_index)
                d = self.reclaimed_buffer[i]
                volume += factor * d[1]
                q += factor * d[1] * d[2][0]

            if volume > 0:
                q /= volume

            self.reclaimer_position = new_pos
            return volume, q
        else:
            return 0, 0

    def reclaiming_finished(self):
        return self.reclaimed_buffer is not None and self.reclaimer_position >= self.length


class Stacker:
    def __init__(self, max_speed: float, strategy: str):
        self.stockpile = None
        self.max_speed = max_speed
        self.strategy = strategy

    def stack(self, tons, q):
        timestamp = 0  # TODO
        pos = {
            'A': (self.stockpile.stacked_tons / self.stockpile.max_tons) * (
                    self.stockpile.length - self.stockpile.depth) + 0.5 * self.stockpile.depth,
            'B': (1 - self.stockpile.stacked_tons / self.stockpile.max_tons) * (
                    self.stockpile.length - self.stockpile.depth) + 0.5 * self.stockpile.depth,
        }[self.strategy]  # TODO
        self.stockpile.stack(timestamp, pos, tons, q)

    def stacking_finished(self):
        return self.stockpile is not None and self.stockpile.stacking_finished()


class Reclaimer:
    def __init__(self, max_speed: float):
        self.stockpile = None
        self.max_speed = max_speed

    def reclaim(self):
        if self.reclaiming_finished():
            return 0, 0
        else:
            return self.stockpile.reclaim(self.max_speed)

    def reclaiming_finished(self):
        return self.stockpile is None or self.stockpile.reclaiming_finished()


class BlendingSystem(MaterialHandler):
    def __init__(self, label, src, length: float = 600, depth: float = 50, max_stacker_speed: float = 0.5,
                 max_reclaimer_speed: float = 0.005, strategy: str = 'A', simulator: str = 'fast'):
        super().__init__(label, 'cylinder')
        self.src_gen = self.unpack_src_gen(src)
        self._sample = (0, 0)
        self.length = length
        self.depth = depth
        self.stacker = Stacker(max_stacker_speed, strategy)
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

        self.stacker.stack(tph * Plant.TIME_INCREMENT / 3600, q)
        tons, q = self.reclaimer.reclaim()
        return tons * 3600 / Plant.TIME_INCREMENT, q

    def gen(self, i: int = 0):
        while True:
            tph, q = self.step_blending_system(*next(self.src_gen))
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]
