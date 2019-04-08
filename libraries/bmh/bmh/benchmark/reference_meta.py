import json
import os
from typing import Optional

from .material_deposition import MaterialMeta

META_JSON = 'meta.json'


class ReferenceMeta:
    """
    Object managing a reference evaluation - the result of a simulation of a material-deposition combination
    """

    def __init__(self, identifier: str, path: str, meta_dict: dict):
        """
        :param identifier: unique identifier of this reference
        :param path: directory where meta.json and data for this reference are stored
        :param meta_dict: dict parsed from json file
        """
        self.identifier = identifier
        self.path = path

        # copy data read from json file
        self.material = meta_dict['material']
        self.deposition = meta_dict['deposition']

        # original data read from json file and stored in dict
        self.meta_dict = meta_dict

        # data buffer
        self.reclaimed_material_meta: Optional[MaterialMeta] = None

    def __str__(self) -> str:
        return self.identifier

    def to_dict(self) -> dict:
        """
        Write all relevant meta information about this reference into a dict.
        :return: dict with relevant meta information about this reference
        """
        return {
            'material': self.material,
            'deposition': self.deposition,
        }

    def get_reclaimed_material_meta(self) -> MaterialMeta:
        if self.reclaimed_material_meta is None:
            path = os.path.join(self.path, 'material')
            meta = json.load(open(os.path.join(path, META_JSON)))
            self.reclaimed_material_meta = MaterialMeta(
                'reclaimed material for ' + self.identifier,
                path,
                meta
            )

        return self.reclaimed_material_meta
