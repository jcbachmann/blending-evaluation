"""Settings shared by all scripts, they must match the parameters the training data was generated with (see the defaults of generate_training_data)."""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass

MATERIAL_LENGTH: int = 50  # Length of material variables array
DEPOSITION_LENGTH: int = 20  # Length of deposition variables array
BED_SIZE_X: int = 59  # Bed size in X dimension
BED_SIZE_Z: int = 20  # Bed size in Z dimension
TOTAL_VOLUME: int = 2500  # Total volume for simulation
PROFILE_LENGTH: int = BED_SIZE_X + 1  # Reclaimed slices, one per unit of the bed length (positions 0 to BED_SIZE_X)
MATERIAL_MIN: int = 5
MATERIAL_MAX: int = 10

# Deposition positions are limited to the core of the stockpile
X_MIN: float = 0.5 * BED_SIZE_Z
X_MAX: float = BED_SIZE_X - X_MIN


SIMULATOR_ENVIRONMENT_VARIABLE = "BMH_ML_SIMULATOR"


@dataclass(frozen=True)
class SimulatorSettings:
    """The detail level of the simulator. `ppm3` is the number of particles per cubic meter; the reclaim step is fixed at 1 m, so every
    level reclaims the same 60 slices (the simulator's own default step, 1/sqrt(ppm3), would make the slices thinner and F2 smaller just
    by their width). The defaults are the settings of all data before 2026-10-04."""

    ppm3: float = 1.0
    reclaim_increment: float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict | None) -> "SimulatorSettings":
        return cls(**(values or {}))


def get_simulator_settings() -> SimulatorSettings:
    """The settings every simulation of this process uses: those activated with `use_simulator_settings`, else the defaults."""
    value = os.environ.get(SIMULATOR_ENVIRONMENT_VARIABLE)
    return SimulatorSettings.from_dict(json.loads(value)) if value else SimulatorSettings()


def use_simulator_settings(settings: SimulatorSettings) -> None:
    """Activates the settings for this process and the processes it starts afterwards (they inherit the environment)."""
    os.environ[SIMULATOR_ENVIRONMENT_VARIABLE] = json.dumps(settings.as_dict())


# All scripts read and write relative to the working directory
DATA_DIR = "data"
OUTPUT_DIR = "output"
TRAINING_DATA_FILE = f"{DATA_DIR}/training_data.csv"
MODEL_F1_NAME = "lstm_model_f1"
MODEL_F2_NAME = "lstm_model_f2"

# The surrogate models are stored in model sets: a scaler and the models for F1 and F2, trained together on one dataset. Several sets can be kept
# next to each other, for example one trained on a dataset where every row has its own random material and deposition ("random_training_data") and
# one trained on a dataset with 50 depositions per material. The scripts that use models select the set with --model-set and use this one by
# default. It is the set of the experiments made so far, so it is a deliberate choice, and training always needs the name of the set explicitly.
DEFAULT_MODEL_SET = "random_training_data"

MODEL_SET_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def get_model_set(name: str) -> str:
    if not MODEL_SET_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(f"'{name}' is not a valid model set name, use letters, digits, - and _")
    return name


def get_scaler_file(model_set: str) -> str:
    return f"{DATA_DIR}/scaler_{model_set}.pkl"


def get_model_file(model_name: str, model_set: str) -> str:
    return f"{DATA_DIR}/{model_name}_{model_set}.keras"


def add_model_set_argument(parser: argparse.ArgumentParser, *, required: bool = False):
    parser.add_argument(
        "--model-set",
        type=get_model_set,
        required=required,
        default=None if required else DEFAULT_MODEL_SET,
        help="Name of the model set, files data/scaler_<set>.pkl and data/lstm_model_f{1,2}_<set>.keras"
        + ("" if required else f" (default: {DEFAULT_MODEL_SET})"),
    )
