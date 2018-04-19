from typing import List, Union

import numpy as np

from blending_simulator.blending_simulator import BlendingSimulator


def gaussian(x, sigma):
    return np.exp(-np.power(x / sigma, 2.) / 2.)


class SmoothBlendingSimulator(BlendingSimulator):
    def __init__(self, bed_size_x: float, buffer_size: int, sigma_x: float, **kwargs):
        super().__init__(bed_size_x, 0)
        self.buffer_size = buffer_size
        self.sigma_x = sigma_x
        self.buffer = [[(i + 1) / self.buffer_size * bed_size_x, 0, 0] for i in range(self.buffer_size)]

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        first = max(0, min(int((x - 2 * self.sigma_x) / self.bed_size_x * self.buffer_size), self.buffer_size - 1))
        last = max(0, min(int((x + 2 * self.sigma_x) / self.bed_size_x * self.buffer_size), self.buffer_size - 1))

        norm_sum = 0
        for i in range(first, last + 1):
            norm_sum += gaussian(i * self.bed_size_x / self.buffer_size - x, self.sigma_x)

        for i in range(first, last + 1):
            elem = self.buffer[i]
            g = gaussian(i * self.bed_size_x / self.buffer_size - x, self.sigma_x)
            v = volume * g / norm_sum
            elem[1] += v
            elem[2] += v * parameter[0]

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        return [[b[0], b[1], [b[2] / b[1] if b[1] > 0 else 0]] for b in self.buffer]
