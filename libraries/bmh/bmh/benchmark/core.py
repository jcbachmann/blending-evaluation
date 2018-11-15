import json
import logging
import os

from bmh.helpers import math

from .data import BenchmarkData
from .material_deposition import MaterialMeta, DepositionMeta, MaterialDeposition
from .reference_meta import ReferenceMeta
from .simulator_meta import SimulatorMeta

DATA_CSV = 'data.csv'
RECLAIMED_MATERIAL_DIR = 'material'
COMPUTED_DEPOSITION_DIR = 'deposition'
SIMULATOR_JSON = 'simulator.json'


def test_simulator(simulator_meta: SimulatorMeta):
    logger = logging.getLogger(__name__)
    logger.debug(f'Acquiring simulator type for "{simulator_meta.type}"')

    # Acquire simulator type
    sim_type = simulator_meta.get_type()

    # Read simulator parameters
    sim_params = simulator_meta.get_params()

    sim_params_copy = sim_params.copy()
    sim_params_copy['bed_size_x'] = 10.0
    sim_params_copy['bed_size_z'] = 10.0

    # Create demo simulator to test if params are all accepted
    # If no exception occurs everything seems to be fine
    logger.debug('Testing creation of simulator')
    sim = sim_type(**sim_params_copy)

    logger.debug('Testing deletion of simulator')
    del sim


def prepare_dst(dst: str, dry_run: bool):
    logger = logging.getLogger(__name__)
    logger.debug(f'Preparing destination directory "{dst}"')
    if os.path.exists(dst):
        if os.path.isdir(dst):
            if len(os.listdir(dst)) > 0:
                raise IOError(f'Destination path "{dst}" is not empty')
            else:
                logger.debug(f'Destination path "{dst}" already exists and is empty')
        else:
            raise IOError(f'Destination path "{dst}" is not a directory')
    else:
        logger.debug(f'Creating destination path "{dst}"')
        if not dry_run:
            os.makedirs(dst)


def compute_sigma(material_meta: MaterialMeta):
    logger = logging.getLogger(__name__)
    logger.debug(f'Computing standard deviation sigma for material parameters of "{material_meta}"')

    material = material_meta.get_material()
    parameter_columns = material.get_parameter_columns()
    sigmas = {}

    for parameter_column in parameter_columns:
        mean, sigma = math.weighted_avg_and_std(
            values=material.data[parameter_column].values,
            weights=material.data['volume'].values
        )

        logger.debug(f'{material_meta} - {parameter_column}: sigma = {sigma}')
        sigmas[parameter_column] = sigma

    return sigmas


def compute_sigma_reduction(material_before: MaterialMeta, material_after: MaterialMeta):
    logger = logging.getLogger(__name__)
    sigmas_before = compute_sigma(material_before)
    sigmas_after = compute_sigma(material_after)

    logger.info(f'Sigma reduction ratios for "{material_before}" -> "{material_after}":')
    for parameter, sigma_before in sigmas_before.items():
        reduction = sigma_before / sigmas_after[parameter]
        logger.info(f'{parameter}\t1:{reduction:.2f}')


def process(identifier: str, material_meta: MaterialMeta, deposition_meta: DepositionMeta,
            simulator_meta: SimulatorMeta, dst: str, dry_run: bool, computed_deposition: bool):
    logger = logging.getLogger(__name__)
    logger.info(f'Processing "{identifier}" with material "{material_meta}" and deposition "{deposition_meta}"')

    logger.debug('Creating simulator')
    sim_params = simulator_meta.get_params().copy()
    sim_params['bed_size_x'] = deposition_meta.bed_size_x
    sim_params['bed_size_z'] = deposition_meta.bed_size_z
    sim = simulator_meta.get_type()(**sim_params)

    logger.debug('Combining material and deposition')
    material_deposition = MaterialDeposition(material_meta.get_material(), deposition_meta.get_deposition())
    logger.debug(f'Material and deposition combined:\n{material_deposition.data.describe()}')

    logger.debug('Stacking and reclaiming material')
    reclaimed_material = sim.stack_reclaim(material_deposition, x_per_s=deposition_meta.reclaim_x_per_s)
    logger.debug(f'Reclaimed material:\n{reclaimed_material.data.describe()}')

    directory = os.path.join(dst, identifier)
    logger.debug(f'Creating directory "{directory}"')
    if not dry_run:
        os.mkdir(directory)

    reclaimed_reference = ReferenceMeta(identifier, directory, {
        'material': str(material_meta),
        'deposition': str(deposition_meta),
        'deposition_path': COMPUTED_DEPOSITION_DIR if computed_deposition else None,
        'reclaimed_path': RECLAIMED_MATERIAL_DIR
    })

    meta_file = os.path.join(directory, BenchmarkData.META_JSON)
    logger.debug(f'Writing reference meta to "{meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_reference.to_dict(),
            open(meta_file, 'w'),
            indent=4
        )

    reclaimed_material_path = os.path.join(directory, reclaimed_reference.reclaimed_path)
    reclaimed_material_meta = MaterialMeta(identifier, reclaimed_material_path, {
        'label': 'Reclaimed ' + identifier,
        'description': 'Reclaimed material from ' + identifier,
        'category': 'reclaimed',
        'time': reclaimed_material.data.timestamp.max(),
        'volume': reclaimed_material.data.volume.sum(),
        'data': DATA_CSV
    })
    reclaimed_material_meta.data = reclaimed_material

    logger.debug(f'Creating directory "{reclaimed_material_path}"')
    if not dry_run:
        os.mkdir(reclaimed_material_path)

    reclaimed_material_meta_file = os.path.join(reclaimed_material_path, BenchmarkData.META_JSON)
    logger.debug(f'Writing reclaimed material meta to "{reclaimed_material_meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_material_meta.to_dict(),
            open(reclaimed_material_meta_file, 'w'),
            indent=4
        )

    data_file = os.path.join(reclaimed_material_path, DATA_CSV)
    logger.debug(f'Writing material data to "{data_file}"')
    if not dry_run:
        reclaimed_material.data.to_csv(data_file, sep='\t', index=False)

    compute_sigma_reduction(material_meta, reclaimed_material_meta)
