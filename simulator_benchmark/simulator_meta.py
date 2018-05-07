import json
import os
from typing import Dict

from simulator_benchmark.evaluate_benchmark_data import SIMULATOR_TYPE


class SimulatorMeta:
    """
    Object managing a simulator
    """

    def __init__(self, identifier: str, path: str, meta_dict: dict):
        """
        :param identifier: unique identifier of this simulator
        :param path: directory where meta.json and parameters for this simulator are stored
        :param meta_dict: dict parsed from json file
        """
        self.identifier = identifier
        self.path = path

        # copy data read from json file
        self.type = meta_dict['type']
        self.params = meta_dict['params']

        # original data read from json file and stored in dict
        self.meta_dict = meta_dict

        # params from params file stored in dict
        self.params_dict = None

    def __str__(self) -> str:
        return self.identifier

    def get_type(self):
        return SIMULATOR_TYPE[self.type]

    def to_dict(self) -> dict:
        """
        Write all relevant meta information about this reference into a dict.
        :return: dict with relevant meta information about this reference
        """
        return {
            'type': self.type,
            'params': self.params
        }

    def get_params(self) -> Dict:
        if self.params_dict is None:
            self.params_dict = json.load(open(os.path.join(self.path, self.params)))

        return self.params_dict
