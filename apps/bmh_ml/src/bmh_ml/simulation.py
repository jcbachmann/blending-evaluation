"""The fast blending simulator as the pipeline uses it, at a release and detail level (`SimulatorSettings`) that the bundles record."""

import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Deposition, Material
from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes
from bmh.simulation.bsl_blending_simulator import simulate_batch

from bmh_ml.settings import get_installed_simulator_version, get_simulator_settings

MAX_TIMESTAMP = 86400  # Duration of material and deposition; it has no influence on the results


def generate_material(
    material_variables: list[float],
    total_volume: float,
) -> Material:
    return Material.from_data(
        pd.DataFrame(
            {
                "timestamp": np.linspace(0, MAX_TIMESTAMP, len(material_variables)),
                "volume": [total_volume / len(material_variables)] * len(material_variables),
                "quality": material_variables,
            }
        )
    )


def generate_deposition(
    deposition_variables: list[float],
    bed_size_x: float,
    bed_size_z: float,
) -> Deposition:
    return Deposition.from_data(
        data=pd.DataFrame(
            {
                "timestamp": np.linspace(0, MAX_TIMESTAMP, len(deposition_variables)),
                "x": deposition_variables,
                "z": [0.5 * bed_size_z] * len(deposition_variables),
            }
        ),
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        reclaim_x_per_s=bed_size_x / MAX_TIMESTAMP,
    )


def evaluate_sim(
    material_variables,
    deposition_variables,
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
) -> tuple[float, float]:
    f1, f2, _ = evaluate_sim_with_profile(material_variables, deposition_variables, bed_size_x, bed_size_z, total_volume)
    return f1, f2


def evaluate_sim_with_profile(
    material_variables,
    deposition_variables,
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
) -> tuple[float, float, np.ndarray]:
    """F1, F2 and the reclaimed profile they are computed from: the volume and the quality of each reclaimed slice, shape (2, slices)."""
    f, profiles = evaluate_sims_with_profiles([material_variables], [deposition_variables], bed_size_x, bed_size_z, total_volume)
    return float(f[0, 0]), float(f[0, 1]), profiles[0]


def evaluate_sims_with_profiles(
    material_variables,
    deposition_variables,
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """F1 and F2 of many inputs, shape (inputs, 2), and their reclaimed profiles, shape (inputs, 2, slices); one row of material and of
    deposition variables per input.

    Builds the rows that `MaterialDeposition` makes of `generate_material` and `generate_deposition` (each material portion stacked at the
    stacker position interpolated at its timestamp), simulates all inputs in one call and evaluates them like `ReclaimedMaterialEvaluator`,
    with numpy for all inputs at once: at a coarse detail level pandas took several times as long as the simulation itself. `threads`
    simulations run at the same time in this process.
    """
    material = np.atleast_2d(np.asarray(material_variables, dtype=float))
    deposition = np.atleast_2d(np.asarray(deposition_variables, dtype=float))
    inputs, count = material.shape
    x_min = bed_size_z * 0.5
    x_max = bed_size_x - x_min

    material_timestamps = np.linspace(0, MAX_TIMESTAMP, count)
    deposition_timestamps = np.linspace(0, MAX_TIMESTAMP, deposition.shape[1])
    data = np.empty((inputs, count, 4))
    data[:, :, 0] = [np.interp(material_timestamps, deposition_timestamps, row) for row in deposition]
    data[:, :, 1] = 0.5 * bed_size_z
    data[:, :, 2] = total_volume / count
    data[:, :, 3] = material

    settings = get_simulator_settings()
    installed = get_installed_simulator_version()
    if settings.version != installed:
        raise RuntimeError(
            f"The active simulator settings are of simulator release {settings.version} (e.g. the level of a bundle built with it), but"
            f" {installed} is installed: results of different releases must not be mixed, simulate with settings of the installed release"
        )
    reclaimed = simulate_batch(
        data,
        ["x", "z", "volume", "quality"],
        bed_size_x,
        bed_size_z,
        ppm3=settings.ppm3,
        reclaimincrement=settings.reclaim_increment,
        threads=threads,
        reclaimangle=settings.reclaim_angle,
        eight=settings.eight_likelihood,
    )

    volume, quality = reclaimed["volume"], reclaimed["quality"]
    total = volume.sum(axis=1, keepdims=True)
    mean_quality = (quality * volume).sum(axis=1, keepdims=True) / total
    f1 = np.sqrt((volume * (quality - mean_quality) ** 2).sum(axis=1, keepdims=True) / total)[:, 0]
    deviation = get_ideal_stockpile_volumes(reclaimed["x"], total, x_min, x_max) - volume
    f2 = np.sqrt(((deviation - deviation.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
    return np.stack([f1, f2], axis=1), np.stack([volume, quality], axis=1)
