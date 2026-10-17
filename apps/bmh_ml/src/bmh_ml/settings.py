"""Settings shared by all scripts, they must match the parameters the training data was generated with (see the defaults of generate_training_data)."""

import argparse
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

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

# The simulator release of all results that do not record one (before 2026-10-15): the environments that produced the
# datasets, test sets and Chevron references of the store were built on 2026-10-03 from v2026.1 (checked 2026-10-15: their bindings had no
# seed argument yet). v2026.2 would give the same model with other random numbers.
UNRECORDED_SIMULATOR_VERSION = "2026.1"


def get_installed_simulator_version() -> str:
    """The simulator release the installed blending_simulator_lib is built from; bindings that do not report it count as unrecorded."""
    from bmh.simulation.bsl_blending_simulator import get_simulator_version

    return get_simulator_version() or UNRECORDED_SIMULATOR_VERSION


SIMULATORS = ("fast", "lattice", "cone")
DROPS = ("point", "cells")


@dataclass(frozen=True)
class SimulatorSettings:
    """The simulator that produced a result: its release and its detail level. `ppm3` is the number of particles per cubic meter; the
    reclaim step is fixed at 1 m, so every level reclaims the same 60 slices (the simulator's own default step, one particle size, would
    make the slices thinner and F2 smaller just by their width). New settings are of the installed simulator release; results of
    different releases must not be mixed. Settings read from records without a release, angle or likelihood are those of all data before
    2026-10-15 (`from_dict`), the defaults of ppm3 and the reclaim step those of all data before 2026-10-04.

    `simulator` chooses the simulation ("fast", "lattice" or "cone"; ppm3 sets the particle or cell size of each), `drops` how the stacker's
    path becomes simulator input: "cells" (new results) lets the material flow while the stacker moves and drops it proportionally in
    every grid cell the path crosses (`bmh.helpers.continuous_drops`), "point" stacks every material portion at the one stacker position
    of its time, as all data before 2026-10-17 did. Records without them read as the fast simulator with point drops."""

    ppm3: float = 1.0
    reclaim_increment: float = 1.0
    reclaim_angle: float = 45.0
    eight_likelihood: float = 0.87
    version: str = field(default_factory=get_installed_simulator_version)
    simulator: str = "fast"
    drops: str = "cells"

    def __post_init__(self):
        if self.simulator not in SIMULATORS:
            raise ValueError(f"unknown simulator {self.simulator!r}, one of {SIMULATORS}")
        if self.drops not in DROPS:
            raise ValueError(f"unknown drops {self.drops!r}, one of {DROPS}")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict | None) -> "SimulatorSettings":
        return cls(**{"version": UNRECORDED_SIMULATOR_VERSION, "simulator": "fast", "drops": "point", **(values or {})})


def get_simulator_settings() -> SimulatorSettings:
    """The settings every simulation of this process uses: those activated with `use_simulator_settings`, else the defaults."""
    value = os.environ.get(SIMULATOR_ENVIRONMENT_VARIABLE)
    return SimulatorSettings.from_dict(json.loads(value)) if value else SimulatorSettings()


def use_simulator_settings(settings: SimulatorSettings) -> None:
    """Activates the settings for this process and the processes it starts afterwards (they inherit the environment)."""
    os.environ[SIMULATOR_ENVIRONMENT_VARIABLE] = json.dumps(settings.as_dict())


@contextmanager
def simulator_settings(settings: SimulatorSettings):
    """Simulates at other settings for the duration of the block (e.g. the cheap level inside a model evaluated at a detailed level)."""
    previous = os.environ.get(SIMULATOR_ENVIRONMENT_VARIABLE)
    use_simulator_settings(settings)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SIMULATOR_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[SIMULATOR_ENVIRONMENT_VARIABLE] = previous


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
