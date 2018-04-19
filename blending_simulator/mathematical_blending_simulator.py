from typing import List, Union

from blending_simulator.blending_simulator import BlendingSimulator
from blending_simulator.mathematical_blending import mathematical_blending


class MathematicalBlendingSimulator(BlendingSimulator):
    def __init__(self, bed_size_x: float, buffer_size: int, **kwargs):
        super().__init__(bed_size_x, 0)
        self.buffer_size = buffer_size
        self.positions = []
        self.volumes = []
        self.qualities = []

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        self.positions.append(max(0, min(int(x / self.bed_size_x * self.buffer_size), self.buffer_size - 1)))
        self.volumes.append(volume)
        self.qualities.append(parameter)

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        volumes, qualities = mathematical_blending.calculate_blended_output(
            time_slots=len(self.positions),
            in_volumes=self.volumes,
            in_qualities=self.qualities,
            positions=self.positions,
            bed_width=self.buffer_size
        )
        return [[p, v, q] for p, v, q in zip(
            [(i + 1) / self.buffer_size * self.bed_size_x for i in range(self.buffer_size)],
            volumes,
            qualities
        )]
