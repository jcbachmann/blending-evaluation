import json
import logging
import os
from typing import Dict

from .data import BenchmarkData, prepare_path
from .material_deposition import MaterialMeta, DepositionMeta, MaterialDeposition, write_data_file
from .reference_meta import ReferenceMeta
from .simulator_meta import SimulatorMeta
from ..helpers import math


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
            simulator_meta: SimulatorMeta, path: str, dry_run: bool) -> Dict[str, float]:
    logger = logging.getLogger(__name__)
    logger.info(f'Processing "{identifier}" with material "{material_meta}" and deposition "{deposition_meta}" using '
                f'simulator type {simulator_meta.type}')

    logger.debug('Creating simulator')
    sim_params = simulator_meta.get_params().copy()
    sim_params['bed_size_x'] = deposition_meta.bed_size_x
    sim_params['bed_size_z'] = deposition_meta.bed_size_z
    sim = simulator_meta.get_type()(**sim_params)

    logger.debug('Combining material and deposition')
    material_deposition = MaterialDeposition(material_meta.get_material(), deposition_meta.get_deposition())
    logger.debug(f'Material and deposition combined:\n{material_deposition.data.describe()}')

    logger.debug('Stacking and reclaiming material')
    reclaimed_material = sim.stack_reclaim(material_deposition)
    logger.debug(f'Reclaimed material:\n{reclaimed_material.data.describe()}')

    directory = os.path.join(path, BenchmarkData.REFERENCE_DIR, identifier)
    prepare_path(directory, dry_run=dry_run)

    simulator_file = os.path.join(directory, BenchmarkData.SIMULATOR_JSON)
    logger.debug(f'Writing simulator type and parameters to "{simulator_file}"')
    if not dry_run:
        json.dump({'simulator': str(simulator_meta)}, open(simulator_file, 'w'), indent=4)

    reclaimed_reference = ReferenceMeta(identifier, directory, {
        'material': str(material_meta),
        'deposition': str(deposition_meta),
    })

    meta_file = os.path.join(directory, BenchmarkData.META_JSON)
    logger.debug(f'Writing reference meta to "{meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_reference.to_dict(),
            open(meta_file, 'w'),
            indent=4
        )

    reclaimed_material_path = os.path.join(directory, BenchmarkData.MATERIAL_DIR)
    reclaimed_material_meta = MaterialMeta(identifier, path='', meta_dict={
        'label': 'Reclaimed ' + identifier,
        'description': 'Reclaimed material from ' + identifier,
        'category': 'reclaimed',
        'time': reclaimed_material.data.timestamp.max(),
        'volume': reclaimed_material.data.volume.sum(),
        'data': BenchmarkData.DATA_CSV
    })
    reclaimed_material.meta = reclaimed_material_meta
    reclaimed_material_meta.data = reclaimed_material

    logger.debug(f'Creating directory "{reclaimed_material_path}"')
    prepare_path(reclaimed_material_path, dry_run=dry_run)

    reclaimed_material_meta_file = os.path.join(reclaimed_material_path, BenchmarkData.META_JSON)
    logger.debug(f'Writing reclaimed material meta to "{reclaimed_material_meta_file}"')
    if not dry_run:
        json.dump(
            reclaimed_material_meta.to_dict(),
            open(reclaimed_material_meta_file, 'w'),
            indent=4
        )

    if not dry_run:
        write_data_file(
            data=reclaimed_material.data,
            data_file=os.path.join(reclaimed_material_path, BenchmarkData.DATA_CSV)
        )

    compute_sigma_reduction(material_meta, reclaimed_material_meta)
