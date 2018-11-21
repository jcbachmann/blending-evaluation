from typing import List, Union

from pandas import DataFrame

from ..benchmark.material_deposition import MaterialDeposition, Material


class BlendingSimulator:
    def __init__(self, bed_size_x: float, bed_size_z: float, **kwargs):
        self.bed_size_x = bed_size_x
        self.bed_size_z = bed_size_z

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: List[float]) -> None:
        raise NotImplemented('BlendingSimulator.stack not implemented')

    def reclaim(self) -> List[List[Union[float, List[float]]]]:
        raise NotImplemented('BlendingSimulator.reclaim not implemented')

    def stack_reclaim(self, material_deposition: MaterialDeposition) -> Material:
        """
        Stack material according to material deposition and reclaim into new blended material.
        :param material_deposition: material and deposition data
        :return: reclaimed material
        """

        # call self.stack for every material-deposition row
        param_cols = material_deposition.material.get_parameter_columns()
        material_deposition_data = material_deposition.data.filter(
            ['timestamp', 'x', 'z', 'volume']
        ).to_dict(orient='records')
        material_deposition_parameters = material_deposition.data.filter(param_cols).to_dict(orient='split')['data']
        for i, p in enumerate(material_deposition_parameters):
            material_deposition_data[i]['parameter'] = p

        for row in material_deposition_data:
            self.stack(**row)

        # reclaim stacked material
        data_list = self.reclaim()

        # reorganize reclaimed material into pandas DataFrame
        data_dict = DataFrame(
            [row[2] for row in data_list], columns=material_deposition.material.get_parameter_columns()
        ).fillna(0).to_dict(orient='list')
        data_dict['x'] = [row[0] for row in data_list]
        data_dict['volume'] = [row[1] for row in data_list]
        data = DataFrame(data_dict)

        # Extract reclaimer speed from deposition meta
        reclaim_x_per_s = material_deposition.deposition.meta.reclaim_x_per_s

        # calculate timestamp column from x positions
        data['timestamp'] = data['x'] / reclaim_x_per_s

        # reorder columns and remove x position
        data = data[['timestamp', 'volume'] + material_deposition.material.get_parameter_columns()]

        return Material.from_data(data)
