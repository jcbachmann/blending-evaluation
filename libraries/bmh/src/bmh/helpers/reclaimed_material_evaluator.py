import numpy as np

from ..benchmark.material_deposition import Material
from .math import stdev, weighted_avg_and_std
from .stockpile_math import get_ideal_stockpile_volumes


def get_ideal_reclaimed_material(reclaimed: Material, x_min: float, x_max: float) -> Material:
    """
    Ideal version of a reclaimed material: all parameters at their volume weighted average and the volume curve of an ideal stockpile
    :param reclaimed: reclaimed material with an x column
    :param x_min: first reclaim position of the stockpile core
    :param x_max: last reclaim position of the stockpile core
    :return: copy of the reclaimed material, the reclaimed material itself is not modified
    """
    ideal = reclaimed.copy()
    for parameter in reclaimed.get_parameter_columns():
        ideal.data[parameter] = np.average(reclaimed.data[parameter], weights=reclaimed.data["volume"])
    ideal.data["volume"] = get_ideal_stockpile_volumes(reclaimed.data["x"].to_numpy(), reclaimed.data["volume"].sum(), x_min, x_max)
    return ideal


class ReclaimedMaterialEvaluator:
    def __init__(self, reclaimed: Material, x_min: float | None = None, x_max: float | None = None):
        self.reclaimed = reclaimed
        self.x_min = x_min
        self.x_max = x_max

        # Caches
        self._parameter_stdev: dict[str, float] | None = None
        self._volume_stdev: float | None = None

    def get_volume_stdev(self) -> float:
        if self._volume_stdev is None:
            data = self.reclaimed.data
            ideal_volumes = get_ideal_stockpile_volumes(data["x"].to_numpy(), data["volume"].sum(), self.x_min, self.x_max)
            self._volume_stdev = stdev(ideal_volumes - data["volume"].to_numpy())

        return self._volume_stdev

    def get_parameter_stdev(self) -> dict[str, float]:
        if self._parameter_stdev is None:
            cols = self.reclaimed.get_parameter_columns()
            self._parameter_stdev = {f"F1/{col}": self.get_single_parameter_stdev(col) for col in cols}
        return self._parameter_stdev

    def get_single_parameter_stdev(self, parameter: str) -> float:
        return weighted_avg_and_std(self.reclaimed.data[parameter], self.reclaimed.data["volume"])[1]

    def get_all_stdev(self) -> dict[str, float]:
        return {
            **self.get_parameter_stdev(),
            "F2": self.get_volume_stdev(),
        }

    @staticmethod
    def get_relative(objectives: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
        return {k: v / reference[k] for k, v in objectives.items()}

    def get_slice_count(self) -> int:
        return self.reclaimed.data["volume"].shape[0]
