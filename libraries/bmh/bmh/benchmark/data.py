import json
import logging
import os
from typing import Dict, List, Optional

from .material_deposition import MaterialMeta, DepositionMeta, write_data_file
from .reference_meta import ReferenceMeta
from .simulator_meta import SimulatorMeta, SIMULATOR_TYPE


def create_instance_for_each_managed_dir(path: str, instance_type: type, *, recursive: bool = False) -> dict:
    logger = logging.getLogger(__name__)
    logger.debug(f'Creating class "{instance_type.__name__}" for each managed directory in path "{path}"')
    entries = {}
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)

        if os.path.isdir(entry_path):
            meta_path = os.path.join(entry_path, BenchmarkData.META_JSON)
            if os.path.isfile(meta_path):
                logger.debug(f'Reading "{instance_type.__name__}" from path "{entry_path}"')

                # Errors which occur during JSON parsing and interpretation should interrupt program execution
                meta = json.load(open(meta_path))
                entries[entry] = instance_type(entry, os.path.abspath(entry_path), meta)
            else:
                if recursive:
                    sub_entries = create_instance_for_each_managed_dir(entry_path, instance_type, recursive=True)
                    duplicate_identifiers = set(entries.keys()).intersection(sub_entries.keys())
                    if len(duplicate_identifiers) > 0:
                        raise ValueError(f'Duplicate identifiers {duplicate_identifiers}')
                    entries.update(sub_entries)
                else:
                    logger.debug(f'Ignoring directory "{entry_path}"')
        else:
            logger.debug(f'Ignoring file "{entry_path}"')

    return entries


def prepare_path(path: str, *, dry_run: bool = False, empty_required: bool = True):
    logger = logging.getLogger(__name__)
    logger.debug(f'Preparing path "{path}"')
    if os.path.exists(path):
        if os.path.isdir(path):
            if empty_required:
                if len(os.listdir(path)) > 0:
                    raise IOError(f'Path "{path}" is not empty')
                else:
                    logger.debug(f'Path "{path}" already exists and is empty')
            else:
                logger.debug(f'Path "{path}" already exists')
        else:
            raise IOError(f'Path "{path}" exists but is not a directory')
    else:
        logger.debug(f'Creating path "{path}"')
        if not dry_run:
            os.makedirs(path)


class BenchmarkData:
    META_JSON = 'meta.json'
    MATERIAL_DIR = 'material'
    DEPOSITION_DIR = 'deposition'
    SIMULATOR_DIR = 'simulator'
    REFERENCE_DIR = 'reference'
    BENCHMARK_DIR = 'benchmark'
    DATA_CSV = 'data.csv'
    PREDICTION_CSV = 'prediction.csv'
    SIMULATOR_JSON = 'simulator.json'

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path

        self.materials: Dict[str, MaterialMeta] = {}
        self.depositions: Dict[str, DepositionMeta] = {}
        self.simulators: Dict[str, SimulatorMeta] = {}

        self.logger = logging.getLogger(__name__)

    def get_benchmark_dir_path(self, path: Optional[str], benchmark_dir: str):
        if path:
            return path
        else:
            if self.base_path:
                return os.path.join(self.base_path, benchmark_dir)
            else:
                raise ValueError('Base path not provided')

    def read_materials(self, path: Optional[str] = None) -> Dict[str, MaterialMeta]:
        path = self.get_benchmark_dir_path(path, BenchmarkData.MATERIAL_DIR)
        self.logger.debug('Reading materials')
        self.materials = create_instance_for_each_managed_dir(path, MaterialMeta, recursive=True)
        self.logger.info(f'{len(self.materials)} materials read')
        return self.materials

    def read_depositions(self, path: Optional[str] = None) -> Dict[str, DepositionMeta]:
        path = self.get_benchmark_dir_path(path, BenchmarkData.DEPOSITION_DIR)
        self.logger.debug('Reading depositions')
        self.depositions = create_instance_for_each_managed_dir(path, DepositionMeta, recursive=True)
        self.logger.info(f'{len(self.depositions)} depositions read')
        return self.depositions

    def read_simulators(self, path: Optional[str] = None) -> Dict[str, SimulatorMeta]:
        path = self.get_benchmark_dir_path(path, BenchmarkData.SIMULATOR_DIR)
        self.logger.debug('Reading simulators')
        self.simulators = create_instance_for_each_managed_dir(path, SimulatorMeta, recursive=True)
        self.logger.info(f'{len(self.simulators)} simulators read')
        return self.simulators

    def read_references(self, path: Optional[str] = None) -> Dict[str, ReferenceMeta]:
        path = self.get_benchmark_dir_path(path, BenchmarkData.REFERENCE_DIR)
        self.logger.debug('Reading references')
        references = create_instance_for_each_managed_dir(path, ReferenceMeta, recursive=True)
        self.logger.info(f'{len(references)} references read')
        return references

    def read_base(self, path: Optional[str] = None) -> None:
        if not path and not self.base_path:
            raise ValueError('Neither path nor base path provided')

        self.read_materials(path)
        self.read_depositions(path)
        self.read_simulators(path)

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

    def validate_deposition(self, deposition_identifiers: str):
        self.logger.debug(f'Validating deposition identifier "{deposition_identifiers}"')
        if deposition_identifiers not in self.depositions:
            raise ValueError(f'Deposition "{deposition_identifiers}" not found in depositions')
        self.logger.info(f'Deposition identifier "{deposition_identifiers}" validated')

    def get_material_meta(self, material_identifier: str) -> MaterialMeta:
        self.validate_material(material_identifier)
        return self.materials[material_identifier]

    def get_deposition_meta(self, deposition_identifiers: str) -> DepositionMeta:
        self.validate_deposition(deposition_identifiers)
        return self.depositions[deposition_identifiers]

    def write_material(self, material_meta: MaterialMeta, path: Optional[str] = None):
        path = self.get_benchmark_dir_path(path, os.path.join(BenchmarkData.MATERIAL_DIR, material_meta.identifier))

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

    def write_deposition(self, deposition_meta: DepositionMeta, path: Optional[str] = None):
        path = self.get_benchmark_dir_path(path, os.path.join(BenchmarkData.DEPOSITION_DIR, deposition_meta.identifier))
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
