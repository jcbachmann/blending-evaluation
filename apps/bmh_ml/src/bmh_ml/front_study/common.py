"""Shared by the steps of the front study: where the results go, and Pareto dominance on (F1, F2), both minimized."""

import os
from pathlib import Path

import numpy as np

OUT_ENVIRONMENT_VARIABLE = "BMH_ML_FRONT_STUDY_OUT"


def get_out() -> Path:
    """The directory of all results: BMH_ML_FRONT_STUDY_OUT, else workdir/front-interpolation/out relative to the working directory (the
    repository root, from which the steps are run). The results are large and stay out of git (workdir/ is ignored). verify.py, the first
    step, creates it."""
    return Path(os.environ.get(OUT_ENVIRONMENT_VARIABLE, "workdir/front-interpolation/out"))


def dominated_by(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """For each point, whether one of `others` is at least as good in both objectives and better in one."""
    result = np.zeros(len(points), dtype=bool)
    for chunk in range(0, len(points), 5000):
        p = points[chunk : chunk + 5000, None, :]
        result[chunk : chunk + 5000] = np.any(np.all(others[None] <= p, axis=2) & np.any(others[None] < p, axis=2), axis=1)
    return result


def nondominated(objectives: np.ndarray) -> np.ndarray:
    """Rows of the solutions no other solution dominates."""
    return np.flatnonzero(~dominated_by(objectives, objectives))
