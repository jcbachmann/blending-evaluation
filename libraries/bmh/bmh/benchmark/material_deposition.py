import logging
import os
import uuid
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from ..helpers.stockpile_math import get_stockpile_height, get_stockpile_volume


def read_data_file(data_file: str) -> DataFrame:
    """
    Read data file provided in the arguments from tab separated file
    :param data_file: file which is read into pandas DataFrame
    :return: pandas DataFrame containing data contained in data_file
    """
    logger = logging.getLogger(__name__)
    logger.debug(f'Reading data file "{data_file}"')
    if not os.path.isfile(data_file):
        raise IOError(f'Data file "{data_file}" does not exist')

    data = pd.read_csv(data_file, sep='\t')
    logger.debug(f'"{data_file}" data:\n{data.describe()}')

    return data


def check_required_columns(data: DataFrame, required_columns: List[str]) -> None:
    """
    Data is check whether all required columns are provided. A ValueError is raised if the data does not contain all
    required columns.
    :param data: data for which the check is performed
    :param required_columns: list of required columns
    """
    logger = logging.getLogger(__name__)
    logger.debug(f'Checking required columns')
    required_columns_str = ', '.join(required_columns)
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f'Data does not contain all required columns: {required_columns_str}')


class Material:
    """
    Object managing material data
    """

    REQUIRED_COLUMNS = ['timestamp', 'volume']
    OPTIONAL_COLUMNS = ['x']

    def __init__(self, *, meta: 'MaterialMeta', data: DataFrame):
        self.meta = meta
        self.data = data
        check_required_columns(self.data, Material.REQUIRED_COLUMNS)

    def get_parameter_columns(self) -> List[str]:
        return list(set(self.data.columns).difference(Material.REQUIRED_COLUMNS + Material.OPTIONAL_COLUMNS))

    def get_volume(self):
        return self.data['volume'].sum()

    @classmethod
    def from_data(cls, data: DataFrame, *, identifier=str(uuid.uuid4())):
        meta = MaterialMeta(identifier, path='', meta_dict={
            'label': f'Material {identifier}',
            'description': f'Created from data with length ({data.shape[0]})',
            'category': 'reclaimed',
            'time': data['timestamp'].max(),
            'volume': data['volume'].sum(),
            'data': None
        })
        return cls(meta=meta, data=data)

    def copy(self):
        return Material(
            meta=self.meta.copy(),
            data=self.data.copy()
        )


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
        self.data: Optional['Material'] = None

    def __str__(self) -> str:
        return self.identifier

    def get_material(self) -> Material:
        """
        Load data file on first call and buffer data to avoid unnecessary loading of data files
        :return: Deposition object containing data for this deposition
        """
        if self.data is None:
            material_data = read_data_file(os.path.join(self.path, self.data_file))
            self.data = Material(meta=self, data=material_data)

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

    def copy(self, copy_data: bool = False):
        meta = MaterialMeta(
            identifier=self.identifier,
            path=self.path,
            meta_dict=self.to_dict()
        )
        if copy_data and self.data:
            meta.data = self.data.copy()
        return meta


class Deposition:
    """
    Object managing deposition data
    """

    REQUIRED_COLUMNS = ['timestamp', 'x', 'z']

    def __init__(self, *, meta: 'DepositionMeta', data: DataFrame):
        self.meta = meta
        self.data = data
        check_required_columns(self.data, Deposition.REQUIRED_COLUMNS)

    @classmethod
    def from_data(cls, data: DataFrame, *, identifier=str(uuid.uuid4()), bed_size_x: float, bed_size_z: float,
                  reclaim_x_per_s: float):
        meta = DepositionMeta(identifier, path='', meta_dict={
            'label': f'Deposition {identifier}',
            'description': f'Created from data with length ({data.shape[0]})',
            'category': 'from_data',
            'time': data['timestamp'].max(),
            'data': None,
            'bed_size_x': bed_size_x,
            'bed_size_z': bed_size_z,
            'reclaim_x_per_s': reclaim_x_per_s
        })
        return cls(meta=meta, data=data)

    def __str__(self):
        out = f'{self.meta.identifier}\n'
        out += self.data.to_string()
        return out

    def copy(self):
        return Deposition(
            meta=self.meta.copy(),
            data=self.data.copy()
        )

    @staticmethod
    def create_empty(*, identifier=str(uuid.uuid4()), bed_size_x: float, bed_size_z: float,
                     reclaim_x_per_s: float):
        meta = DepositionMeta.create_empty(
            identifier=identifier,
            bed_size_x=bed_size_x,
            bed_size_z=bed_size_z,
            reclaim_x_per_s=reclaim_x_per_s,
        )
        return meta.data

    @staticmethod
    def create_empty_data():
        return DataFrame(dict([(c, pd.Series(dtype=np.dtype('float'))) for c in Deposition.REQUIRED_COLUMNS]))


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
        self.data: Optional[Deposition] = None

    def __str__(self) -> str:
        return self.identifier

    def get_deposition(self) -> Deposition:
        """
        Load data file on first call and buffer data to avoid unnecessary loading of data files
        :return: Deposition object containing data for this deposition
        """
        if self.data is None:
            deposition_data = read_data_file(os.path.join(self.path, self.data_file))
            self.data = Deposition(meta=self, data=deposition_data)

        return self.data

    def to_dict(self):
        return {
            'label': self.label,
            'description': self.description,
            'category': self.category,
            'time': self.time,
            'data': self.data_file,
            'bed_size_x': self.bed_size_x,
            'bed_size_z': self.bed_size_z,
            'reclaim_x_per_s': self.reclaim_x_per_s
        }

    def copy(self, copy_data: bool = False):
        meta = DepositionMeta(
            identifier=self.identifier,
            path=self.path,
            meta_dict=self.to_dict()
        )
        if copy_data and self.data:
            meta.data = self.data.copy()
        return meta

    @classmethod
    def create_empty(cls, *, identifier=str(uuid.uuid4()), bed_size_x: float, bed_size_z: float,
                     reclaim_x_per_s: float):
        meta = cls(
            identifier=identifier,
            path='',
            meta_dict={
                'label': f'Deposition {identifier}',
                'description': f'Created empty',
                'category': 'empty',
                'time': 0.0,
                'data': None,
                'bed_size_x': bed_size_x,
                'bed_size_z': bed_size_z,
                'reclaim_x_per_s': reclaim_x_per_s
            }
        )
        meta.data = Deposition(
            meta=meta,
            data=Deposition.create_empty_data()
        )
        return meta


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
    def prepare(material_df: DataFrame, deposition_df: DataFrame) -> DataFrame:
        # Copy material data
        data = material_df.copy()

        # Add cone deposition to the beginning of every layer
        # deposition_df = MaterialDeposition.add_cone_per_layer(deposition_df, material_df)

        data['x'] = np.interp(data['timestamp'], deposition_df['timestamp'], deposition_df['x'])
        data['z'] = np.interp(data['timestamp'], deposition_df['timestamp'], deposition_df['z'])

        return data

    @staticmethod
    def upsample_material(mat_df: DataFrame, t_diff_max: float = 15) -> DataFrame:
        # Very slow...
        data = DataFrame()
        for index, row in mat_df.iterrows():
            if 'timestamp' in data.columns:
                t_last = data['timestamp'].max()
            else:
                t_last = 0
            t_curr = row['timestamp']
            sub_steps = int((t_curr - t_last) / t_diff_max)
            t_per_step = (t_curr - t_last) / sub_steps
            volume_per_step = row['volume'] / sub_steps
            parameter = row['parameter']  # TODO WTF?
            sub_rows: Dict[str, List[float]] = {
                'timestamp': [],
                'volume': [],
                'parameter': []
            }
            for step in range(sub_steps):
                sub_rows['timestamp'].append(t_last + (step + 1) * t_per_step)
                sub_rows['volume'].append(volume_per_step)
                sub_rows['parameter'].append(parameter)

            data = data.append(DataFrame(sub_rows), ignore_index=True, sort=False)
        return data

    @staticmethod
    def add_cone_per_layer(dep_df: DataFrame, mat_df: DataFrame) -> DataFrame:
        dep_df['t_end'] = dep_df['timestamp'].shift(-1)
        dep_df['v_layer'] = dep_df.apply(lambda row: mat_df[
            (mat_df['timestamp'] >= row['timestamp']) & (mat_df['timestamp'] < row['t_end'])]['volume'].sum(),
                                         axis=1)
        dep_df['t_diff'] = dep_df['timestamp'].shift(-1) - dep_df['timestamp']
        dep_df['core_length'] = abs(dep_df['x'].shift(-1) - dep_df['x'])

        dep_df['height'] = get_stockpile_height(dep_df['v_layer'], dep_df['core_length'])
        dep_df['v_cone'] = get_stockpile_volume(dep_df['height'], 0.0)
        dep_df['t_wait'] = 6.0 * dep_df['t_diff'] * dep_df['v_cone'] / dep_df['v_layer']  # TODO why 6.0? two pi??

        waits = dep_df.copy()
        waits['timestamp'] += dep_df['t_wait']
        dep_df = dep_df.append(waits, ignore_index=True)
        dep_df.sort_values(['timestamp'], inplace=True)

        dep_df = dep_df[['timestamp', 'x', 'z']].copy()
        dep_df.dropna(inplace=True)
        return dep_df
