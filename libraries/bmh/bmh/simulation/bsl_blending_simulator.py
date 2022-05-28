import math
from typing import Optional, List, Union

from blending_simulator_lib import BlendingSimulatorLib
from pandas import DataFrame

from .blending_simulator import BlendingSimulator, MaterialDeposition, Material


class BslBlendingSimulator(BlendingSimulator):
    def __init__(self, bed_size_x: float, bed_size_z: float, reclaimangle: Optional[float] = None,
                 ppm3: Optional[float] = None, circular: Optional[bool] = None, eight: Optional[float] = None,
                 bulkdensity: Optional[float] = None, dropheight: Optional[float] = None,
                 detailed: Optional[bool] = None, reclaimincrement: Optional[float] = None):
        super().__init__(bed_size_x, bed_size_z)
        if reclaimangle is None:
            reclaimangle = 45.0
        if ppm3 is None:
            ppm3 = 1.0
        if circular is None:
            circular = False
        if eight is None:
            eight = 0.87
        if bulkdensity is None:
            bulkdensity = 1.0
        if dropheight is None:
            dropheight = 0.5 * bed_size_z
        if detailed is None:
            detailed = False
        if reclaimincrement is None:
            reclaimincrement = 1.0 / math.sqrt(ppm3)

        self.bsl = BlendingSimulatorLib(
            bed_size_x,
            bed_size_z,
            reclaimangle,
            ppm3,
            circular,
            eight,
            bulkdensity,
            dropheight,
            detailed,
            reclaimincrement
        )

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        self.bsl.stack(timestamp, x, z, volume, parameter)

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        raise NotImplementedError()

    def stack_reclaim(self, material_deposition: MaterialDeposition) -> Material:
        """
        Stack material according to material deposition and reclaim into new blended material.
        :param material_deposition: material and deposition data
        :return: reclaimed material
        """

        # stack all data
        self.bsl.stack_list(
            material_deposition.data.to_numpy(),
            material_deposition.data.columns.values.tolist()
        )

        # reclaim stacked material
        data_dict = self.bsl.reclaim()

        # Extract reclaimer speed from deposition meta
        reclaim_x_per_s = material_deposition.deposition.meta.reclaim_x_per_s

        # calculate timestamp column from x positions
        data_dict['timestamp'] = [v / reclaim_x_per_s for v in data_dict['x']]

        # reorganize reclaimed material into pandas DataFrame
        data = DataFrame(data_dict)

        return Material.from_data(data, category='reclaimed')

    def get_heights(self):
        return self.bsl.get_heights()
