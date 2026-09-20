import json
import logging
import os

import numpy as np
import pandas as pd

from bmh_ml.settings import MATERIAL_LENGTH, TRAINING_DATA_FILE


def get_params_file(training_data_file: str) -> str:
    # The name of the file generate_training_data writes its parameters to
    return training_data_file.rsplit(".", 1)[0] + "_params.json"


def warn_if_not_complete(data: pd.DataFrame, training_data_file: str):
    params_file = get_params_file(training_data_file)
    if not os.path.exists(params_file):
        return
    with open(params_file) as f:
        params = json.load(f)
    expected_rows = params["materials"] * params["depositions_per_material"]
    if len(data) != expected_rows:
        logging.warning(f"{training_data_file} has {len(data)} rows, but {expected_rows} rows were generated (see {params_file}), the file may be truncated")


def load_training_data(training_data_file: str = TRAINING_DATA_FILE) -> pd.DataFrame:
    """Reads the training data, a cut off last row (interrupted or size limited write) is ignored, other incomplete rows are an error."""
    data = pd.read_csv(training_data_file)

    incomplete = data.isna().any(axis=1)
    if incomplete.any():
        if incomplete.sum() == 1 and incomplete.iloc[-1]:
            logging.warning(f"Ignoring the incomplete last row of {training_data_file}, the file is probably truncated")
            data = data.iloc[:-1]
        else:
            raise ValueError(
                f"{training_data_file} has {int(incomplete.sum())} incomplete rows (first: row {int(incomplete.to_numpy().argmax())}), generate it again"
            )

    warn_if_not_complete(data, training_data_file)
    return data


def load_fixed_material_variables(training_data_file: str = TRAINING_DATA_FILE) -> np.ndarray:
    """The material variables of the first training data row, used as the fixed material of the optimization and its evaluation."""
    first_row = pd.read_csv(training_data_file, nrows=1)
    return first_row.iloc[0, 2 : 2 + MATERIAL_LENGTH].to_numpy(dtype=float)
