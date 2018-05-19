import math
from typing import Dict

from plant_simulator.plant import Plant
from .deposition_strategies import DEPOSITION_STRATEGIES


class Stacker:
    def __init__(self, max_speed: float, strategy: str, strategy_params: Dict):
        self.stockpile = None
        self.max_speed = max_speed
        if strategy_params is None:
            strategy_params = {}
        self.strategy = DEPOSITION_STRATEGIES[strategy](**strategy_params)
        self.last_pos = None

    def stack(self, tons, q):
        timestamp = 0  # TODO
        new_pos = self.strategy.get_pos(timestamp, self.stockpile)
        if self.last_pos is not None:
            dx = new_pos[0] - self.last_pos[0]
            dz = new_pos[1] - self.last_pos[1]
            speed = math.sqrt(dx * dx + dz * dz) / Plant.TIME_INCREMENT
            if speed > self.max_speed:
                raise ValueError(f'Stacker speed too high: {speed:.2f} > {self.max_speed:.2f}')
        self.stockpile.stack(timestamp, new_pos[0], new_pos[1], tons, q)
        self.last_pos = new_pos

    def stacking_finished(self):
        return self.stockpile is not None and self.stockpile.stacking_finished()
