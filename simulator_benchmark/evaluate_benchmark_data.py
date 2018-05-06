import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

from blending_simulator.external_blending_simulator import ExternalBlendingSimulator
from blending_simulator.material_deposition import MaterialMeta, DepositionMeta, MaterialDeposition
from blending_simulator.mathematical_blending_simulator import MathematicalBlendingSimulator
from blending_simulator.smooth_blending_simulator import SmoothBlendingSimulator

META_JSON = 'meta.json'
DATA_CSV = 'data.csv'
RECLAIMED_MATERIAL_DIR = 'material'
SIMULATOR_JSON = 'simulator.json'
SIMULATOR_PARAMS_JSON = 'simulator_params.json'
SIMULATOR_TYPE = {
    'mathematical': MathematicalBlendingSimulator,
    'smooth': SmoothBlendingSimulator,
    'external': ExternalBlendingSimulator,
}


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
        if 'reclaimed_path' in meta_dict and meta_dict['reclaimed_path'] is not None:
            self.reclaimed_path = meta_dict['reclaimed_path']
        else:
            self.reclaimed_path = None

        # original data read from json file and stored in dict
        self.meta_dict = meta_dict

        # data buffer
        self.reclaimed_material_meta = None

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
            'reclaimed_path': self.reclaimed_path
        }

    def get_reclaimed_material_meta(self) -> MaterialMeta:
        if self.reclaimed_material_meta is None:
            reclaimed_path = os.path.join(self.path, self.reclaimed_path)
            meta = json.load(open(os.path.join(reclaimed_path, META_JSON)))
            self.reclaimed_material_meta = MaterialMeta(
                'reclaimed material for ' + self.identifier,
                reclaimed_path,
                meta
            )

        return self.reclaimed_material_meta


def list_managed_dirs(path: str) -> List[str]:
    logging.debug(f'Listing managed directories for path "{path}"')
    return [
        entry for entry in os.listdir(path)
        if os.path.isdir(os.path.join(path, entry)) and os.path.isfile(os.path.join(path, entry, META_JSON))
    ]


def create_instance_for_each_managed_dir(path: str, instance_type: type) -> dict:
    logging.debug(f'Creating class "{instance_type.__name__}" for each managed directory in path "{path}"')
    entries = {}
    for entry in list_managed_dirs(path):
        entry_path = os.path.join(path, entry)
        logging.debug(f'Reading "{instance_type.__name__}" from path "{entry_path}"')

        # Errors which occur during JSON parsing and interpretation should interrupt program execution
        meta = json.load(open(os.path.join(entry_path, META_JSON)))
        entries[entry] = instance_type(entry, os.path.abspath(entry_path), meta)

    return entries


def read_materials(path: str) -> Dict[str, MaterialMeta]:
    logging.debug('Reading materials')
    return create_instance_for_each_managed_dir(path, MaterialMeta)


def read_depositions(path: str) -> Dict[str, DepositionMeta]:
    logging.debug('Reading depositions')
    return create_instance_for_each_managed_dir(path, DepositionMeta)


def read_references(path: str) -> Dict[str, ReferenceMeta]:
    logging.debug('Reading references')
    return create_instance_for_each_managed_dir(path, ReferenceMeta)


def validate_references(references: Dict[str, ReferenceMeta], materials: Dict[str, MaterialMeta],
                        depositions: Dict[str, DepositionMeta]):
    logging.debug('Validating references')
    for _, reference in references.items():
        logging.debug(f'Validating reference "{reference}"')
        if reference.material not in materials:
            raise ValueError(f'Material "{reference.material}" not found in materials')
        if reference.deposition not in depositions:
            raise ValueError(f'Deposition "{reference.deposition}" not found in depositions')
    logging.info('References validated')


def prepare_dst(dst: str, dry_run: bool):
    logging.debug('Preparing destination directory')
    if os.path.exists(dst):
        if os.path.isdir(dst):
            if len(os.listdir(dst)) > 0:
                raise IOError('Destination path is not empty')
            else:
                logging.debug('Destination path already exists and is empty')
        else:
            raise IOError('Destination path is not a directory')
    else:
        logging.debug(f'Creating destination path "{dst}"')
        if not dry_run:
            os.makedirs(dst)


def process_reference(reference: ReferenceMeta, material: MaterialMeta, deposition: DepositionMeta, dst: str,
                      sim_type: type, sim_params: dict, dry_run: bool):
    logging.info(f'Processing reference "{reference}" with material "{material}" and deposition "{deposition}"')

    logging.debug('Creating simulator')
    sim_params['bed_size_x'] = deposition.bed_size_x
    sim_params['bed_size_z'] = deposition.bed_size_z
    sim = sim_type(**sim_params)

    logging.debug('Combining material and deposition')
    material_deposition = MaterialDeposition(material.get_material(), deposition.get_deposition())
    logging.debug(f'Material and deposition combined:\n{material_deposition.data.describe()}')

    logging.debug('Stacking and reclaiming material')
    reclaimed_material = sim.stack_reclaim(material_deposition, x_per_s=deposition.reclaim_x_per_s)
    logging.debug(f'Reclaimed material:\n{reclaimed_material.data.describe()}')

    directory = os.path.join(dst, reference.identifier)
    logging.debug(f'Creating directory "{directory}"')
    if not dry_run:
        os.mkdir(directory)

    reclaimed_reference = ReferenceMeta(reference.identifier, directory, reference.meta_dict)
    reclaimed_reference.reclaimed_path = RECLAIMED_MATERIAL_DIR

    meta_file = os.path.join(directory, META_JSON)
    logging.debug(f'Writing reference meta to "{meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_reference.to_dict(),
            open(meta_file, 'w'),
            indent=4
        )

    reclaimed_material_path = os.path.join(directory, reclaimed_reference.reclaimed_path)
    reclaimed_material_meta = MaterialMeta(reference.identifier, reclaimed_material_path, {
        'label': 'Reclaimed ' + reference.identifier,
        'description': 'Reclaimed material from reference ' + reference.identifier,
        'category': 'reclaimed',
        'time': reclaimed_material.data.timestamp.max(),
        'volume': reclaimed_material.data.volume.sum(),
        'data': DATA_CSV
    })

    logging.debug(f'Creating directory "{reclaimed_material_path}"')
    if not dry_run:
        os.mkdir(reclaimed_material_path)

    reclaimed_material_meta_file = os.path.join(reclaimed_material_path, META_JSON)
    logging.debug(f'Writing reclaimed material meta to "{reclaimed_material_meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_material_meta.to_dict(),
            open(reclaimed_material_meta_file, 'w'),
            indent=4
        )

    data_file = os.path.join(reclaimed_material_path, DATA_CSV)
    logging.debug(f'Writing material data to "{data_file}"')
    if not dry_run:
        reclaimed_material.data.to_csv(data_file, sep='\t', index=False)


def process_data(references: Dict[str, ReferenceMeta], materials: Dict[str, MaterialMeta],
                 depositions: Dict[str, DepositionMeta], dst: str, sim_type: type, sim_params: dict, dry_run: bool):
    logging.info(f'Processing data with simulator "{sim_type.__name__}"')

    logging.debug('Writing simulator type and parameters to destination directory')
    if not dry_run:
        json.dump(
            {'simulator_type': sim_type.__name__},
            open(os.path.join(dst, SIMULATOR_JSON), 'w'),
            indent=4
        )
        json.dump(
            sim_params,
            open(os.path.join(dst, SIMULATOR_PARAMS_JSON), 'w'),
            indent=4
        )

    for _, reference in references.items():
        process_reference(reference, materials[reference.material], depositions[reference.deposition],
                          dst, sim_type,
                          sim_params, dry_run)

    logging.info('Processing finished')


def prepare_simulator(sim_type_str: str, sim_params_str: str) -> Tuple[type, dict]:
    logging.debug(f'Acquiring simulator type for "{sim_type_str}"')

    # Acquire simulator type
    sim_type = SIMULATOR_TYPE[sim_type_str]

    # Read simulator parameters
    if sim_params_str is not None:
        logging.debug(f'Loading simulator parameters from "{sim_params_str}"')
        sim_params = json.load(open(sim_params_str))
    else:
        sim_params = {}

    sim_params_copy = sim_params.copy()
    sim_params_copy['bed_size_x'] = 10
    sim_params_copy['bed_size_z'] = 10

    # Create demo simulator to test if params are all accepted
    # If no exception occurs everything seems to be fine
    logging.debug('Testing creation of simulator')
    sim = sim_type(**sim_params_copy)

    logging.debug('Testing deletion of simulator')
    del sim

    return sim_type, sim_params


def main(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )

    # Initialization
    materials = read_materials(os.path.join(args.path, 'material'))
    logging.info(f'{len(materials)} materials read')

    depositions = read_depositions(os.path.join(args.path, 'deposition'))
    logging.info(f'{len(depositions)} depositions read')

    references = read_references(args.src)
    logging.info(f'{len(references)} references read')

    # Make sure everything will work out
    validate_references(references, materials, depositions)

    # Make sure simulation will work properly
    sim_type, sim_params = prepare_simulator(args.sim, args.sim_params)

    # Prepare output directory
    if args.dst is None:
        args.dst = os.path.join('.', datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ' ' + args.sim)
    prepare_dst(args.dst, args.dry_run)

    # Processing
    process_data(references, materials, depositions, args.dst, sim_type, sim_params, args.dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate benchmark data for a given set of material deposition combinations'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--src', default='./benchmark', help='Path with reference configuration files')
    parser.add_argument('--dst', default=None, help='Path where results will be stored')
    parser.add_argument('--dry_run', action='store_true', help='Do not write files')
    parser.add_argument('--sim', help='Which simulator is used to calculate results')
    parser.add_argument('--sim_params', default=None, help='JSON file with parameters passed to the simulator')

    main(parser.parse_args())
