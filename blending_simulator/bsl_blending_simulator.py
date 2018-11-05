#!/usr/bin/env python
from blending_simulator_lib import BlendingSimulatorLib
from typing import Optional

import pandas as pd

from blending_simulator.blending_simulator import BlendingSimulator
from blending_simulator.material_deposition import MaterialDeposition, Material


class BslBlendingSimulator(BlendingSimulator):
    def __init__(self, bed_size_x: float, bed_size_z: float, reclaimangle: Optional[float] = None,
                 ppm3: Optional[float] = None, circular: Optional[bool] = None, eight: Optional[float] = None,
                 bulkdensity: Optional[float] = None, dropheight: Optional[float] = None,
                 detailed: Optional[bool] = None, reclaimincrement: Optional[float] = None):
        super().__init__(bed_size_x, bed_size_z)
        if reclaimangle is None: reclaimangle = 45.0
        if ppm3 is None: ppm3 = 1.0
        if circular is None: circular = False
        if eight is None: eight = 0.87
        if bulkdensity is None: bulkdensity = 1.0
        if dropheight is None: dropheight = 0.5 * bed_size_z
        if detailed is None: detailed = False
        if reclaimincrement is None: reclaimincrement = 1.0

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

    def stack_reclaim(self, material_deposition: MaterialDeposition, x_per_s: float) -> Material:
        """
        Stack material according to material deposition and reclaim into new blended material.
        :param material_deposition: material and deposition data
        :param x_per_s: speed in units of x per second
        :return: reclaimed material
        """

        # stack all data
        # TODO use ctypes array / bytes or ndarray to avoid conversion from list on c++ side
        stack_data = material_deposition.data.values.tolist()

        self.bsl.stack_list(
            stack_data,
            material_deposition.data.columns.values.tolist(),
            material_deposition.data.values.shape[0],
            material_deposition.data.values.shape[1]
        )

        # reclaim stacked material
        data_dict = self.bsl.reclaim()

        # calculate timestamp column from x positions
        data_dict['timestamp'] = [v / x_per_s for v in data_dict['x']]
        del data_dict['x']

        # reorganize reclaimed material into pandas DataFrame
        data = pd.DataFrame(data_dict)

        return Material(data=data)
