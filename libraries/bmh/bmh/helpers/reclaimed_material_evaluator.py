from typing import Optional, Dict

from .math import stdev, weighted_avg_and_std
from .stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from ..benchmark.material_deposition import Material


class ReclaimedMaterialEvaluator:
    def __init__(self, reclaimed: Material, x_min: Optional[float] = None, x_max: Optional[float] = None):
        self.reclaimed = reclaimed
        self.x_min = x_min
        self.x_max = x_max

        # Caches
        self._parameter_stdev: Optional[Dict[str, float]] = None
        self._volume_stdev: Optional[float] = None

    def get_volume_stdev(self) -> float:
        if self._volume_stdev is None:
            ideal_df = self.reclaimed.data.copy()
            ideal_height = get_stockpile_height(volume=ideal_df['volume'].sum(), core_length=self.x_max - self.x_min)
            ideal_df['x_diff'] = (ideal_df['x'] - ideal_df['x'].shift(1)).fillna(0.0)
            ideal_df['volume'] = ideal_df.apply(
                lambda row: get_stockpile_slice_volume(
                    x=row['x'],
                    core_length=self.x_max - self.x_min,
                    height=ideal_height,
                    x_min=self.x_min,
                    x_diff=row['x_diff']
                ), axis=1
            )

            self._volume_stdev = stdev((ideal_df['volume'] - self.reclaimed.data['volume']).values)

        return self._volume_stdev

    def get_parameter_stdev(self) -> Dict[str, float]:
        if self._parameter_stdev is None:
            cols = self.reclaimed.get_parameter_columns()
            self._parameter_stdev = {f'F1/{col}': self.get_single_parameter_stdev(col) for col in cols}
        return self._parameter_stdev

    def get_single_parameter_stdev(self, parameter: str) -> float:
        return weighted_avg_and_std(self.reclaimed.data[parameter], self.reclaimed.data['volume'])[1]

    def get_all_stdev(self) -> Dict[str, float]:
        return {
            **self.get_parameter_stdev(),
            'F2': self.get_volume_stdev()
        }

    @staticmethod
    def get_relative(objectives: Dict[str, float], reference: Dict[str, float]) -> Dict[str, float]:
        return {k: v / reference[k] for k, v in objectives.items()}

    def get_slice_count(self) -> int:
        return self.reclaimed.data['volume'].shape[0]
