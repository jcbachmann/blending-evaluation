import json
import logging
import os
from typing import Dict, List

from .material_deposition import MaterialMeta, DepositionMeta
from .reference_meta import ReferenceMeta
from .simulator_meta import SimulatorMeta, SIMULATOR_TYPE


class BenchmarkData:
    META_JSON = 'meta.json'
    MATERIAL_DIR = 'material'
    DEPOSITION_DIR = 'deposition'
    SIMULATOR_DIR = 'simulator'

    def __init__(self):
        self.materials = None
        self.depositions = None
        self.simulators = None

    def read_materials(self, path: str) -> Dict[str, MaterialMeta]:
        logging.debug('Reading materials')
        self.materials = BenchmarkData.create_instance_for_each_managed_dir(path, MaterialMeta)
        logging.info(f'{len(self.materials)} materials read')
        return self.materials

    def read_depositions(self, path: str) -> Dict[str, DepositionMeta]:
        logging.debug('Reading depositions')
        self.depositions = BenchmarkData.create_instance_for_each_managed_dir(path, DepositionMeta)
        logging.info(f'{len(self.depositions)} depositions read')
        return self.depositions

    def read_simulators(self, path: str) -> Dict[str, SimulatorMeta]:
        logging.debug('Reading references')
        self.simulators = BenchmarkData.create_instance_for_each_managed_dir(path, SimulatorMeta)
        logging.info(f'{len(self.simulators)} simulators read')
        return self.simulators

    @staticmethod
    def read_references(path: str) -> Dict[str, ReferenceMeta]:
        logging.debug('Reading references')
        references = BenchmarkData.create_instance_for_each_managed_dir(path, ReferenceMeta)
        logging.info(f'{len(references)} references read')
        return references

    def read_base(self, path: str) -> None:
        self.read_materials(os.path.join(path, BenchmarkData.MATERIAL_DIR))
        self.read_depositions(os.path.join(path, BenchmarkData.DEPOSITION_DIR))
        self.read_simulators(os.path.join(path, BenchmarkData.SIMULATOR_DIR))

    def validate_references(self, references: Dict[str, ReferenceMeta]):
        logging.debug('Validating references')
        for _, reference in references.items():
            logging.debug(f'Validating reference "{reference}"')
            if reference.material not in self.materials:
                raise ValueError(f'Material "{reference.material}" not found in materials')
            if reference.deposition not in self.depositions:
                raise ValueError(f'Deposition "{reference.deposition}" not found in depositions')
        logging.info('References validated')

    def validate_simulators(self, sim_identifiers: List[str]):
        logging.debug('Validating simulators')
        for sim_identifier in sim_identifiers:
            logging.debug(f'Validating simulator identifier "{sim_identifier}"')
            if sim_identifier not in self.simulators:
                raise ValueError(f'Simulator identifier "{sim_identifier}" not found in simulators')
            if self.simulators[sim_identifier].type not in SIMULATOR_TYPE:
                raise ValueError(
                    f'Simulator type "{self.simulators[sim_identifier]}" not found for identifier "{sim_identifier}"')
        logging.info('Simulators validated')

    def validate_simulator(self, sim_identifier: str):
        logging.debug(f'Validating simulator identifier "{sim_identifier}"')
        if sim_identifier not in self.simulators:
            raise ValueError(f'Simulator identifier "{sim_identifier}" not found in simulators')
        if self.simulators[sim_identifier].type not in SIMULATOR_TYPE:
            raise ValueError(
                f'Simulator type "{self.simulators[sim_identifier]}" not found for identifier "{sim_identifier}"')
        logging.info(f'Simulator identifier "{sim_identifier}" validated')

    def get_simulator_meta(self, sim_identifier: str):
        self.validate_simulator(sim_identifier)
        return self.simulators[sim_identifier]

    def validate_material(self, material_identifier: str):
        logging.debug(f'Validating material identifier "{material_identifier}"')
        if material_identifier not in self.materials:
            raise ValueError(f'Material "{material_identifier}" not found in materials')
        logging.info(f'Material identifier "{material_identifier}" validated')

    def get_material_meta(self, material_identifier: str):
        self.validate_material(material_identifier)
        return self.materials[material_identifier]

    @staticmethod
    def list_managed_dirs(path: str) -> List[str]:
        logging.debug(f'Listing managed directories for path "{path}"')
        return [
            entry for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry)) and os.path.isfile(
                os.path.join(path, entry, BenchmarkData.META_JSON))
        ]

    @staticmethod
    def create_instance_for_each_managed_dir(path: str, instance_type: type) -> dict:
        logging.debug(f'Creating class "{instance_type.__name__}" for each managed directory in path "{path}"')
        entries = {}
        for entry in BenchmarkData.list_managed_dirs(path):
            entry_path = os.path.join(path, entry)
            logging.debug(f'Reading "{instance_type.__name__}" from path "{entry_path}"')

            # Errors which occur during JSON parsing and interpretation should interrupt program execution
            meta = json.load(open(os.path.join(entry_path, BenchmarkData.META_JSON)))
            entries[entry] = instance_type(entry, os.path.abspath(entry_path), meta)

        return entries
