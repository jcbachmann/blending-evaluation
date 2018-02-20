import logging
import random

from .material_handler import MaterialSource


class SimulatedMine(MaterialSource):
    def __init__(self, label, max_tph, availability, q_min, q_exp, q_max):
        super().__init__(label)

        self.label = label
        self.max_tph = max_tph
        self.availability = availability
        self.currently_unavailable = False

        self.q_min = q_min
        self.q_exp = q_exp
        self.q_max = q_max

        self.tph_current = self.max_tph * random.triangular(0, self.max_tph, self.max_tph)
        self.q_current = self.q_exp

        self.failure_probability = 0.0005
        if availability >= 1.0:
            self.repair_probability = 1.0
        else:
            self.repair_probability = min(self.failure_probability / (1 - availability), 1.0)
        self._sample = (0, 0)

    def gen(self, i: int = 0):
        while True:
            if self.currently_unavailable:
                if random.random() < self.repair_probability:
                    self.currently_unavailable = False
            else:
                if random.random() < self.failure_probability:
                    self.currently_unavailable = True

            if self.currently_unavailable:
                self.tph_current = 0
            else:
                self.tph_current = random.triangular(0, self.max_tph, self.max_tph)

            max_change = 0.003
            limit = self.q_max if self.q_current > self.q_exp else self.q_min
            comp = 0.5 + 0.5 * (self.q_current - self.q_exp) / abs(limit - self.q_exp)
            change = (max_change if random.random() > comp else -max_change) * random.random()
            self.q_current = self.q_current + change

            tph = self.tph_current
            q = 0 if self.currently_unavailable else self.q_current

            logging.debug(f'Source {self.label}: ({tph:.1f}, {q:.1f})')
            self._sample = (tph, q)
            yield tph, q

    def sample(self):
        return [self._sample]
