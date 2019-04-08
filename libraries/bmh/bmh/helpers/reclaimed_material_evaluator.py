from typing import Optional, List

from .math import stdev, weighted_avg_and_std
from .stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from ..benchmark.material_deposition import Material


class ReclaimedMaterialEvaluator:
    def __init__(self, reclaimed: Material, x_min: Optional[float] = None, x_max: Optional[float] = None):
        self.reclaimed = reclaimed
        self.x_min = x_min
        self.x_max = x_max

        # Caches
        self._parameter_stdev: Optional[List[float]] = None
        self._volume_stdev: Optional[float] = None

    def get_volume_stdev(self) -> float:
        if self._volume_stdev is None:
            ideal_df = self.reclaimed.data.copy()
            ideal_height = get_stockpile_height(ideal_df['volume'].sum(), self.x_max - self.x_min)
            ideal_df['x_diff'] = (ideal_df['x'] - ideal_df['x'].shift(1)).fillna(0.0)
            ideal_df['volume'] = ideal_df.apply(
                lambda row: get_stockpile_slice_volume(
                    row['x'], self.x_max - self.x_min, ideal_height, self.x_min, row['x_diff']
                ), axis=1
            )

            self._volume_stdev = stdev((ideal_df['volume'] - self.reclaimed.data['volume']).values)

        return self._volume_stdev

    def get_parameter_stdev(self) -> List[float]:
        if self._parameter_stdev is None:
            reclaimed_df = self.reclaimed.data
            cols = self.reclaimed.get_parameter_columns()
            self._parameter_stdev = [weighted_avg_and_std(reclaimed_df[col], reclaimed_df['volume'])[1] for col in cols]
        return self._parameter_stdev

    def get_all_stdev(self) -> List[float]:
        return self.get_parameter_stdev() + [self.get_volume_stdev()]

    @staticmethod
    def get_relative(objectives: List[float], reference: List[float]) -> List[float]:
        return [s / r for s, r in zip(objectives, reference)]

    def get_slice_count(self) -> int:
        return self.reclaimed.data['volume'].shape[0]
