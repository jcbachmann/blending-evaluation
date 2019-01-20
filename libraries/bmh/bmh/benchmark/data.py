import json
import logging
import os
from typing import Dict, List

from .material_deposition import MaterialMeta, DepositionMeta, write_data_file
from .reference_meta import ReferenceMeta
from .simulator_meta import SimulatorMeta, SIMULATOR_TYPE


class BenchmarkData:
    META_JSON = 'meta.json'
    MATERIAL_DIR = 'material'
    DEPOSITION_DIR = 'deposition'
    SIMULATOR_DIR = 'simulator'
    DATA_CSV = 'data.csv'
    PREDICTION_CSV = 'prediction.csv'
    RECLAIMED_MATERIAL_DIR = 'material'
    COMPUTED_DEPOSITION_DIR = 'deposition'
    SIMULATOR_JSON = 'simulator.json'

    def __init__(self):
        self.materials: Dict[str, MaterialMeta] = {}
        self.depositions: Dict[str, DepositionMeta] = {}
        self.simulators: Dict[str, SimulatorMeta] = {}
        self.logger = logging.getLogger(__name__)

    def read_materials(self, path: str) -> Dict[str, MaterialMeta]:
        self.logger.debug('Reading materials')
        self.materials = self.create_instance_for_each_managed_dir(path, MaterialMeta)
        self.logger.info(f'{len(self.materials)} materials read')
        return self.materials

    def read_depositions(self, path: str) -> Dict[str, DepositionMeta]:
        self.logger.debug('Reading depositions')
        self.depositions = self.create_instance_for_each_managed_dir(path, DepositionMeta)
        self.logger.info(f'{len(self.depositions)} depositions read')
        return self.depositions

    def read_simulators(self, path: str) -> Dict[str, SimulatorMeta]:
        self.logger.debug('Reading references')
        self.simulators = self.create_instance_for_each_managed_dir(path, SimulatorMeta)
        self.logger.info(f'{len(self.simulators)} simulators read')
        return self.simulators

    def read_references(self, path: str) -> Dict[str, ReferenceMeta]:
        self.logger.debug('Reading references')
        references = self.create_instance_for_each_managed_dir(path, ReferenceMeta)
        self.logger.info(f'{len(references)} references read')
        return references

    def read_base(self, path: str) -> None:
        self.read_materials(os.path.join(path, BenchmarkData.MATERIAL_DIR))
        self.read_depositions(os.path.join(path, BenchmarkData.DEPOSITION_DIR))
        self.read_simulators(os.path.join(path, BenchmarkData.SIMULATOR_DIR))

    def validate_references(self, references: Dict[str, ReferenceMeta]):
        self.logger.debug('Validating references')
        for _, reference in references.items():
            self.logger.debug(f'Validating reference "{reference}"')
            if reference.material not in self.materials:
                raise ValueError(f'Material "{reference.material}" not found in materials')
            if reference.deposition not in self.depositions:
                raise ValueError(f'Deposition "{reference.deposition}" not found in depositions')
        self.logger.info('References validated')

    def validate_simulators(self, sim_identifiers: List[str]):
        self.logger.debug('Validating simulators')
        for sim_identifier in sim_identifiers:
            self.logger.debug(f'Validating simulator identifier "{sim_identifier}"')
            if sim_identifier not in self.simulators:
                raise ValueError(f'Simulator identifier "{sim_identifier}" not found in simulators')
            if self.simulators[sim_identifier].type not in SIMULATOR_TYPE:
                raise ValueError(
                    f'Simulator type "{self.simulators[sim_identifier]}" not found for identifier "{sim_identifier}"')
        self.logger.info('Simulators validated')

    def validate_simulator(self, sim_identifier: str):
        self.logger.debug(f'Validating simulator identifier "{sim_identifier}"')
        if sim_identifier not in self.simulators:
            raise ValueError(f'Simulator identifier "{sim_identifier}" not found in simulators')
        if self.simulators[sim_identifier].type not in SIMULATOR_TYPE:
            raise ValueError(
                f'Simulator type "{self.simulators[sim_identifier]}" not found for identifier "{sim_identifier}"')
        self.logger.info(f'Simulator identifier "{sim_identifier}" validated')

    def get_simulator_meta(self, sim_identifier: str) -> SimulatorMeta:
        self.validate_simulator(sim_identifier)
        return self.simulators[sim_identifier]

    def validate_material(self, material_identifier: str):
        self.logger.debug(f'Validating material identifier "{material_identifier}"')
        if material_identifier not in self.materials:
            raise ValueError(f'Material "{material_identifier}" not found in materials')
        self.logger.info(f'Material identifier "{material_identifier}" validated')

    def get_material_meta(self, material_identifier: str) -> MaterialMeta:
        self.validate_material(material_identifier)
        return self.materials[material_identifier]

    def list_managed_dirs(self, path: str) -> List[str]:
        self.logger.debug(f'Listing managed directories for path "{path}"')
        return [
            entry for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry)) and os.path.isfile(
                os.path.join(path, entry, BenchmarkData.META_JSON))
        ]

    def create_instance_for_each_managed_dir(self, path: str, instance_type: type) -> dict:
        self.logger.debug(f'Creating class "{instance_type.__name__}" for each managed directory in path "{path}"')
        entries = {}
        for entry in self.list_managed_dirs(path):
            entry_path = os.path.join(path, entry)
            self.logger.debug(f'Reading "{instance_type.__name__}" from path "{entry_path}"')

            # Errors which occur during JSON parsing and interpretation should interrupt program execution
            meta = json.load(open(os.path.join(entry_path, BenchmarkData.META_JSON)))
            entries[entry] = instance_type(entry, os.path.abspath(entry_path), meta)

        return entries

    def write_material(self, material_meta: MaterialMeta, path: str):
        if not os.path.exists(path):
            self.logger.debug(f'Creating directory "{path}"')
            os.makedirs(path)

        material_meta_file = os.path.join(path, BenchmarkData.META_JSON)
        self.logger.debug(f'Writing material meta to "{material_meta_file}"')
        json.dump(
            material_meta.to_dict(),
            open(material_meta_file, 'w'),
            indent=4
        )

        write_data_file(material_meta.get_material().data, os.path.join(path, BenchmarkData.DATA_CSV))
        prediction = material_meta.get_prediction()
        if prediction is not None:
            write_data_file(prediction.data, os.path.join(path, BenchmarkData.PREDICTION_CSV))

    def write_deposition(self, deposition_meta: DepositionMeta, path: str):
        if not os.path.exists(path):
            self.logger.debug(f'Creating directory "{path}"')
            os.makedirs(path)

        deposition_meta_file = os.path.join(path, BenchmarkData.META_JSON)
        self.logger.debug(f'Writing deposition meta to "{deposition_meta_file}"')
        json.dump(
            deposition_meta.to_dict(),
            open(deposition_meta_file, 'w'),
            indent=4
        )

        write_data_file(deposition_meta.get_deposition().data, os.path.join(path, BenchmarkData.DATA_CSV))
