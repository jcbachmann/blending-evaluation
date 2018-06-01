import logging
import os
from typing import List

import numpy as np
import pandas as pd


def read_data_file(data_file: str) -> pd.DataFrame:
    """
    Read data file provided in the arguments from tab separated file
    :param data_file: file which is read into pandas DataFrame
    :return: pandas DataFrame containing data contained in data_file
    """
    logging.debug(f'Reading data file "{data_file}"')
    if not os.path.isfile(data_file):
        raise IOError(f'Data file "{data_file}" does not exist')

    data = pd.read_csv(data_file, sep='\t')
    logging.debug(f'"{data_file}" data:\n{data.describe()}')

    return data


def check_required_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Data is check whether all required columns are provided. A ValueError is raised if the data does not contain all
    required columns.
    :param data: data for which the check is performed
    :param required_columns: list of required columns
    """
    logging.debug(f'Checking required columns')
    required_columns_str = ', '.join(required_columns)
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f'Data does not contain all required columns: {required_columns_str}')


class Material:
    """
    Object managing material data
    """

    REQUIRED_COLUMNS = ['timestamp', 'volume']

    def __init__(self, *, data: pd.DataFrame = None, data_file: str = None, meta=None):
        if data is not None:
            self.data = data
        elif data_file is not None:
            self.data = read_data_file(data_file)
        else:
            raise ValueError('No data provided')
        check_required_columns(self.data, Material.REQUIRED_COLUMNS)

        self.meta = meta

    def get_parameter_columns(self) -> List[str]:
        return list(set(self.data.columns).difference(Material.REQUIRED_COLUMNS))


class MaterialMeta:
    """
    Object managing meta information of one material
    """

    def __init__(self, identifier: str, path: str, meta_dict: dict):
        """
        :param identifier: unique identifier of this material
        :param path: directory where meta.json and data for this material are stored
        :param meta_dict: dict parsed from json file
        """
        self.identifier = identifier
        self.path = path

        # copy data read from json file
        self.label = meta_dict['label']
        self.description = meta_dict['description']
        self.category = meta_dict['category']
        self.time = meta_dict['time']
        self.volume = meta_dict['volume']
        self.data_file = meta_dict['data']

        # original data read from json file and stored in dict
        self.meta_dict = meta_dict

        # data buffer
        self.data = None

    def __str__(self) -> str:
        return self.identifier

    def get_material(self) -> Material:
        """
        Load data file on first call and buffer data to avoid unnecessary loading of data files
        :return: Deposition object containing data for this deposition
        """
        if self.data is None:
            self.data = Material(data_file=os.path.join(self.path, self.data_file), meta=self)

        return self.data

    def to_dict(self):
        return {
            'label': self.label,
            'description': self.description,
            'category': self.category,
            'time': self.time,
            'volume': self.volume,
            'data': self.data_file
        }


class Deposition:
    """
    Object managing deposition data
    """

    REQUIRED_COLUMNS = ['timestamp', 'x', 'z']

    def __init__(self, meta, data_file: str = None, data=None):
        if data_file is not None:
            self.data = read_data_file(data_file)
        elif data is not None:
            self.data = data
        else:
            raise ValueError('Deposition requires data_file or data to be not None')

        check_required_columns(self.data, Deposition.REQUIRED_COLUMNS)
        self.meta = meta


class DepositionMeta:
    """
    Object managing meta information of one deposition
    """

    def __init__(self, identifier: str, path: str, meta_dict: dict):
        """
        :param identifier: unique identifier of this deposition
        :param path: directory where meta.json and data for this deposition are stored
        :param meta_dict: dict parsed from json file
        """
        self.identifier = identifier
        self.path = path

        # copy data read from json file
        self.label = meta_dict['label']
        self.description = meta_dict['description']
        self.category = meta_dict['category']
        self.time = meta_dict['time']
        self.data_file = meta_dict['data']
        self.bed_size_x = meta_dict['bed_size_x']
        self.bed_size_z = meta_dict['bed_size_z']
        self.reclaim_x_per_s = meta_dict['reclaim_x_per_s']

        # original data read from json file and stored in dict
        self.meta_dict = meta_dict

        # data buffer
        self.data = None

    def __str__(self) -> str:
        return self.identifier

    def get_deposition(self) -> Deposition:
        """
        Load data file on first call and buffer data to avoid unnecessary loading of data files
        :return: Deposition object containing data for this deposition
        """
        if self.data is None:
            self.data = Deposition(data_file=os.path.join(self.path, self.data_file), meta=self)

        return self.data


class MaterialDeposition:
    """
    Object managing the combination of material and deposition
    """

    def __init__(self, material: Material, deposition: Deposition):
        """
        :param material: material data which is combined with deposition data
        :param deposition: deposition data which is combined with material data
        """
        self.material = material
        self.deposition = deposition

        self.data = MaterialDeposition.prepare(material.data, deposition.data)

    @staticmethod
    def prepare(material_df: pd.DataFrame, deposition_df: pd.DataFrame, t_diff_max: float = 15):
        # Copy material data
        data = pd.DataFrame()

        for index, row in material_df.iterrows():
            if 'timestamp' in data.columns:
                t_last = data['timestamp'].max()
            else:
                t_last = 0
            t_curr = row['timestamp']
            sub_steps = int((t_curr - t_last) / t_diff_max)
            t_per_step = (t_curr - t_last) / sub_steps
            volume_per_step = row['volume'] / sub_steps
            parameter = row['parameter']
            sub_rows = {
                'timestamp': [],
                'volume': [],
                'parameter': []
            }
            for step in range(sub_steps):
                sub_rows['timestamp'].append(t_last + (step + 1) * t_per_step)
                sub_rows['volume'].append(volume_per_step)
                sub_rows['parameter'].append(parameter)

            data = data.append(pd.DataFrame(sub_rows))

        data['x'] = np.interp(data['timestamp'], deposition_df['timestamp'], deposition_df['x'])
        data['z'] = np.interp(data['timestamp'], deposition_df['timestamp'], deposition_df['z'])

        return data
