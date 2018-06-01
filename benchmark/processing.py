import logging
import os

from benchmark.simulator_meta import SimulatorMeta

DATA_CSV = 'data.csv'
RECLAIMED_MATERIAL_DIR = 'material'
COMPUTED_DEPOSITION_DIR = 'deposition'
SIMULATOR_JSON = 'simulator.json'


def prepare_simulator(simulator_meta: SimulatorMeta):
    logging.debug(f'Acquiring simulator type for "{simulator_meta.type}"')

    # Acquire simulator type
    sim_type = simulator_meta.get_type()

    # Read simulator parameters
    sim_params = simulator_meta.get_params()

    sim_params_copy = sim_params.copy()
    sim_params_copy['bed_size_x'] = 10
    sim_params_copy['bed_size_z'] = 10

    # Create demo simulator to test if params are all accepted
    # If no exception occurs everything seems to be fine
    logging.debug('Testing creation of simulator')
    sim = sim_type(**sim_params_copy)

    logging.debug('Testing deletion of simulator')
    del sim


def prepare_dst(dst: str, dry_run: bool):
    logging.debug(f'Preparing destination directory "{dst}"')
    if os.path.exists(dst):
        if os.path.isdir(dst):
            if len(os.listdir(dst)) > 0:
                raise IOError(f'Destination path "{dst}" is not empty')
            else:
                logging.debug(f'Destination path "{dst}" already exists and is empty')
        else:
            raise IOError(f'Destination path "{dst}" is not a directory')
    else:
        logging.debug(f'Creating destination path "{dst}"')
        if not dry_run:
            os.makedirs(dst)
