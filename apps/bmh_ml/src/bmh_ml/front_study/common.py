"""Shared by the steps of the front study: where the results go and the simulator settings of the study. Pareto dominance on (F1, F2) is
`bmh.helpers.fronts`.

The study reproduces results made before continuous drops and the integrated ideal stockpile: point drops, F2 against the ideal
stockpile at the slice positions, and as the reference the constant-speed Chevron of the 20 deposition variables, which its steps take
directly from `chevron_deposition()` (the Chevron that `chevron_path` gives for settings with point drops), not the scheduled Chevron of
new results."""

import os
from pathlib import Path

from bmh_ml.settings import SimulatorSettings

OUT_ENVIRONMENT_VARIABLE = "BMH_ML_FRONT_STUDY_OUT"


def get_study_settings(ppm3: float = 1.0) -> SimulatorSettings:
    """The fast simulator at `ppm3` with a 1 m reclaim step, point drops and the point rule of the ideal stockpile, at the installed
    simulator release."""
    return SimulatorSettings(ppm3=ppm3, reclaim_increment=1.0, drops="point", ideal_stockpile="point")


def get_out() -> Path:
    """The directory of all results: BMH_ML_FRONT_STUDY_OUT, else workdir/front-interpolation/out relative to the working directory (the
    repository root, from which the steps are run). The results are large and stay out of git (workdir/ is ignored). verify.py, the first
    step, creates it."""
    return Path(os.environ.get(OUT_ENVIRONMENT_VARIABLE, "workdir/front-interpolation/out"))
