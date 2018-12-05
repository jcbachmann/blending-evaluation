from typing import Optional

from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from bmh.simulation.mathematical_blending_simulator import MathematicalBlendingSimulator
from bmh.simulation.smooth_blending_simulator import SmoothBlendingSimulator

from ..plant import Plant


class Stockpile:
    def __init__(self, length: float, depth: float, simulator: str = 'fast'):
        self.length = length
        self.depth = depth
        self.max_tons = (length - depth) * 0.25 * depth * depth
        self.stacked_tons = 0.0
        self.stack_time_start: Optional[float] = None
        self.stack_time_end: Optional[float] = None
        self._stacking_finished = False
        self.reclaimed_buffer = None
        self.reclaimer_position = 0.0

        self.simulator = {
            'fast': lambda: BslBlendingSimulator(
                bed_size_x=length,
                bed_size_z=depth,
                ppm3=1
            ),
            'smooth': lambda: SmoothBlendingSimulator(
                bed_size_x=length,
                buffer_size=80,
                sigma_x=10
            ),
            'mathematical': lambda: MathematicalBlendingSimulator(
                bed_size_x=length,
                buffer_size=80
            ),
        }[simulator]()

    def stack(self, timestamp: float, x: float, z: float, tons: float, q):
        if tons > 0.0:
            if self.stacked_tons == 0.0:
                self.stack_time_start = timestamp - Plant.TIME_INCREMENT
            self.stack_time_end = timestamp

        volume = tons
        self.simulator.stack(timestamp, x, z, volume, [q])
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
