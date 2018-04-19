from typing import List, Union


class BlendingSimulator:
    def __init__(self, bed_size_x: float, bed_size_z: float, **kwargs):
        self.bed_size_x = bed_size_x
        self.bed_size_z = bed_size_z

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        raise NotImplemented('BlendingSimulator.stack not implemented')

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        raise NotImplemented('BlendingSimulator.reclaim not implemented')
