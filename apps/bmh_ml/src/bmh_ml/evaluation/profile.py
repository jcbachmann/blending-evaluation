"""F1 and F2 from reclaimed profiles, computed as the simulator's evaluator computes them, for many profiles at once.

A reclaimed profile is the volume and the quality of each reclaimed slice (shape (2, slices), slices at the positions 0 to BED_SIZE_X).
`bmh.helpers.reclaimed_material_evaluator` defines the objectives: F1 is the volume-weighted standard deviation of the quality, F2 the
standard deviation of the difference between the volume of the ideal stockpile and the reclaimed volume. The ideal volumes depend on the
total volume, which is interpolated on a fine grid instead of being computed for every profile.
"""

from functools import cache

import numpy as np

from bmh_ml.settings import PROFILE_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN

TOTALS = np.linspace(0.5 * TOTAL_VOLUME, 1.5 * TOTAL_VOLUME, 2001)


@cache
def get_ideal_table() -> np.ndarray:
    """The ideal slice volumes for each total volume of `TOTALS`, shape (totals, slices)."""
    from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes

    x = np.arange(PROFILE_LENGTH, dtype=float)
    return np.stack([get_ideal_stockpile_volumes(x, float(total), X_MIN, X_MAX) for total in TOTALS])


def get_ideal_volumes(totals: np.ndarray) -> np.ndarray:
    """The ideal slice volumes of stockpiles with the given total volumes, linearly interpolated, shape (n, slices)."""
    table = get_ideal_table()
    totals = np.clip(totals, TOTALS[0], TOTALS[-1])
    upper = np.clip(np.searchsorted(TOTALS, totals), 1, len(TOTALS) - 1)
    weight = ((totals - TOTALS[upper - 1]) / (TOTALS[upper] - TOTALS[upper - 1]))[:, None]
    return table[upper - 1] * (1 - weight) + table[upper] * weight


def get_profile_objectives(profiles: np.ndarray) -> np.ndarray:
    """F1 and F2 of profiles of shape (n, 2, slices), shape (n, 2). Negative volumes (from a model) count as empty slices."""
    volume = np.maximum(np.asarray(profiles[:, 0], dtype=float), 0.0)
    quality = np.asarray(profiles[:, 1], dtype=float)
    difference = get_ideal_volumes(volume.sum(axis=1)) - volume
    f2 = np.sqrt(np.mean((difference - difference.mean(axis=1, keepdims=True)) ** 2, axis=1))
    weight = np.maximum(volume.sum(axis=1), 1e-12)
    mean_quality = (volume * quality).sum(axis=1) / weight
    f1 = np.sqrt((volume * (quality - mean_quality[:, None]) ** 2).sum(axis=1) / weight)
    return np.column_stack([f1, f2])
