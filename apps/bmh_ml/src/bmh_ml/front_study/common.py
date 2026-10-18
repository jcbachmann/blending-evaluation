"""Shared by the steps of the front study: where the results go. Pareto dominance on (F1, F2) is `bmh.helpers.fronts`."""

import os
from pathlib import Path

OUT_ENVIRONMENT_VARIABLE = "BMH_ML_FRONT_STUDY_OUT"


def get_out() -> Path:
    """The directory of all results: BMH_ML_FRONT_STUDY_OUT, else workdir/front-interpolation/out relative to the working directory (the
    repository root, from which the steps are run). The results are large and stay out of git (workdir/ is ignored). verify.py, the first
    step, creates it."""
    return Path(os.environ.get(OUT_ENVIRONMENT_VARIABLE, "workdir/front-interpolation/out"))
