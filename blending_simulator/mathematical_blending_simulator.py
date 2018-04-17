from typing import List, Union

from blending_simulator.mathematical_blending import mathematical_blending


class MathematicalBlendingSimulator:
    def __init__(self, length: float, buffer_size: int = 15):
        self.length = length
        self.buffer_size = buffer_size
        self.positions = []
        self.volumes = []
        self.qualities = []

        self.positions.append(max(0, min(int(x / self.length * self.buffer_size), self.buffer_size - 1)))
    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        self.volumes.append(volume)
        self.qualities.append(parameter[0])  # TODO

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        volumes, qualities = mathematical_blending.calculate_blended_output(
            time_slots=len(self.positions),
            in_volumes=self.volumes,
            in_qualities=self.qualities,
            positions=self.positions,
            bed_width=self.buffer_size
        )
        return [[p, v, q] for p, v, q in zip(
            [i / self.buffer_size * self.length for i in range(self.buffer_size)],
            volumes,
            [[q] for q in qualities]
        )]
