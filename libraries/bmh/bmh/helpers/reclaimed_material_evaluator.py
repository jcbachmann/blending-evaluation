from typing import Optional, List

from pandas import DataFrame

from .math import stdev, weighted_avg_and_std
from .stockpile_math import get_stockpile_height
from ..benchmark.material_deposition import Material


class ReclaimedMaterialEvaluator:
    def __init__(self, reclaimed: Material, x_min: Optional[float] = None, x_max: Optional[float] = None):
        self.reclaimed = reclaimed
        self.x_min = x_min
        self.x_max = x_max

        # Caches
        self._core_data: DataFrame = None
        self._parameter_stdev: Optional[List[float]] = None
        self._volume_stdev: Optional[float] = None

    def get_core_data(self) -> DataFrame:
        if self._core_data is None:
            rdf = self.reclaimed.data
            if self.x_min is None or self.x_max is None:
                return rdf
            height = get_stockpile_height(volume=self.reclaimed.get_volume(), core_length=self.x_max - self.x_min)
            self._core_data = rdf[(rdf['x'] >= self.x_min) & (rdf['x'] <= self.x_max - height)]
        return self._core_data

    def get_core_volume_stdev(self) -> float:
        if self._volume_stdev is None:
            core_data = self.get_core_data()
            self._volume_stdev = stdev(core_data['volume'].values)
        return self._volume_stdev

    def get_parameter_stdev(self) -> List[float]:
        if self._parameter_stdev is None:
            reclaimed_df = self.reclaimed.data
            cols = self.reclaimed.get_parameter_columns()
            self._parameter_stdev = [weighted_avg_and_std(reclaimed_df[col], reclaimed_df['volume'])[1] for col in cols]
        return self._parameter_stdev

    def get_all_stdev(self) -> List[float]:
        return self.get_parameter_stdev() + [self.get_core_volume_stdev()]

    @staticmethod
    def get_relative(objectives: List[float], reference: List[float]) -> List[float]:
        return [s / r for s, r in zip(objectives, reference)]

    def get_slice_count(self) -> int:
        return self.reclaimed.data['volume'].shape[0]
