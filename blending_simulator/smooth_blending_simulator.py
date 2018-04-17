from typing import List

import numpy as np


def gaussian(x, sigma):
    return np.exp(-np.power(x / sigma, 2.) / 2.)


class SmoothBlendingSimulator:
    def __init__(self, length: float, buffer_size: int = 80, sigma: float = None):
        self.length = length
        self.buffer_size = buffer_size
        self.sigma = sigma
        self.buffer = [[i / self.buffer_size * length, 0, 0] for i in range(self.buffer_size)]

    def stack(self, timestamp: float, x: float, z: float, volume: float, quality: List[float]):
        first = max(int((x - 2 * self.sigma) / self.length * self.buffer_size), 0)
        last = min(int((x + 2 * self.sigma) / self.length * self.buffer_size), self.buffer_size - 1)

        norm_sum = 0
        for i in range(first, last + 1):
            norm_sum += gaussian(i * self.length / self.buffer_size - x, self.sigma)

        for i in range(first, last + 1):
            elem = self.buffer[i]
            g = gaussian(i * self.length / self.buffer_size - x, self.sigma)
            v = volume * g / norm_sum
            elem[1] += v
            elem[2] += v * quality[0]

    def reclaim(self):
        return [[b[0], b[1], [b[2] / b[1] if b[1] > 0 else 0]] for b in self.buffer]
