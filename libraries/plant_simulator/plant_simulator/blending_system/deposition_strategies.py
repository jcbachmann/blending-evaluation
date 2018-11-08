from typing import Tuple

from .stockpile import Stockpile


class DepositionStrategy:
    def __init__(self):
        pass

    def get_pos(self, timestamp: float, stockpile: Stockpile) -> Tuple[float, float]:
        pass


class DepositionStrategyTimeChevron(DepositionStrategy):
    def __init__(self, layers: int = 1, by: str = 'mass'):
        super().__init__()

        self.layers = layers
        self.by = by

    def get_pos(self, timestamp: float, stockpile: Stockpile) -> Tuple[float, float]:
        if self.by == 'mass':
            p = stockpile.stacked_tons / stockpile.max_tons
            p_layers = p * self.layers
            forward = int(p_layers) % 2 == 0
            p_in_layer = (0 if forward else 1) + (p_layers - int(p_layers)) * (1 if forward else -1)
            return p_in_layer * (stockpile.length - stockpile.depth) + 0.5 * stockpile.depth, 0.5 * stockpile.depth
        else:
            return 0, 0


class DepositionStrategyPile(DepositionStrategy):
    def __init__(self):
        super().__init__()

    def get_pos(self, timestamp: float, stockpile: Stockpile) -> Tuple[float, float]:
        return 0.5 * stockpile.length, 0.5 * stockpile.depth


DEPOSITION_STRATEGIES = {
    'Chevron': DepositionStrategyTimeChevron,
    'Pile': DepositionStrategyPile
}
