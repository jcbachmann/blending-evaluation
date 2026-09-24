"""Chevron stacking as the reference of the relative objectives: F1 and F2 of a solution divided by those of Chevron for the same material.

In the thesis, F1 is the homogenization efficiency ratio against full-speed Chevron stacking, and 1 means "as good as Chevron". Here the
deposition has 20 positions at equal time steps, so the fastest Chevron alternates between the two ends of the bed: 19 passes. F2 is
divided by Chevron's F2 as well (decision of 2026-09-24, PLAN.md section 10), so the point (1, 1) is Chevron in both objectives and a
solution with both relative objectives below 1 is better than Chevron in both.

The reference is simulated `CHEVRON_REPEATS` times and cached in the store per material and setting, so every run uses the same values.
"""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np

from bmh_ml.datasets.simulate import simulate
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, DEPOSITION_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN
from bmh_ml.tracking.environment import get_code_version
from bmh_ml.tracking.store import get_subdirectory

CHEVRON_REPEATS = 64


def chevron_deposition() -> np.ndarray:
    """The 19-pass Chevron in the 20 deposition variables, shape (1, 20): alternating between the two ends of the bed."""
    return np.where(np.arange(DEPOSITION_LENGTH) % 2 == 0, X_MIN, X_MAX).astype(float)[None, :]


def get_reference_key(material: np.ndarray, repeats: int) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(material, dtype=np.float64).tobytes())
    digest.update(json.dumps([BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, DEPOSITION_LENGTH, X_MIN, X_MAX, repeats]).encode())
    return digest.hexdigest()[:16]


def get_chevron_objectives(material: np.ndarray, repeats: int = CHEVRON_REPEATS, n_jobs: int | None = None) -> np.ndarray:
    """F1 and F2 of the 19-pass Chevron for a material, the mean of `repeats` simulations, simulated once and then read from the store."""
    material = np.atleast_2d(material)[:1]
    path = get_subdirectory("references") / f"chevron-{get_reference_key(material, repeats)}.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["objectives"])
    y, _ = simulate(material, np.repeat(chevron_deposition(), repeats, axis=0), 1, n_jobs)  # each row is one simulation
    objectives = y.mean(axis=0)
    record = {
        "objectives": objectives.tolist(),
        "noise_sd": y.std(axis=0, ddof=1).tolist(),
        "repeats": repeats,
        "deposition": chevron_deposition()[0].tolist(),
        "material": material[0].tolist(),
        "settings": {"bed_size_x": BED_SIZE_X, "bed_size_z": BED_SIZE_Z, "total_volume": TOTAL_VOLUME},
        "code_version": get_code_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(record, indent=2))
    return objectives


def get_chevron_hypervolume(relative: np.ndarray) -> float:
    """Hypervolume of solutions in objectives relative to Chevron, with the reference point (1, 1): only what beats Chevron in both counts."""
    from pymoo.indicators.hv import HV

    points = relative[np.all(relative < 1, axis=1)]
    if len(points) == 0:
        return 0.0
    return float(HV(ref_point=np.ones(2))(points))


def get_chevron_metrics(objectives: np.ndarray, chevron: np.ndarray, prefix: str = "chevron_") -> dict[str, float]:
    """The solutions relative to Chevron: hypervolume at (1, 1), the share that beats Chevron in both objectives (`beaten_rate`) and the
    best relative F1 and F2."""
    relative = objectives / chevron
    return {
        f"{prefix}hv": get_chevron_hypervolume(relative),
        f"{prefix}beaten_rate": float(np.mean(np.all(relative < 1, axis=1))),
        f"{prefix}best_F1": float(relative[:, 0].min()),
        f"{prefix}best_F2": float(relative[:, 1].min()),
    }
