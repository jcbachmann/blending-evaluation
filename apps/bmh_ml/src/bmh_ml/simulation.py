"""The blending simulator as the pipeline uses it, at a release, simulation, detail level and drop mode (`SimulatorSettings`) that the
bundles record."""

import numpy as np
from bmh.helpers.bed_setup import STACKING_TIME, get_core_limits
from bmh.helpers.continuous_drops import continuous_drops, drop_grid
from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes
from bmh.simulation.bsl_blending_simulator import simulate_batch

from bmh_ml.settings import get_installed_simulator_version, get_simulator_settings

MAX_TIMESTAMP = STACKING_TIME  # Duration of material and deposition; it has no influence on the results


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
    deposition_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """F1 and F2 of many inputs, shape (inputs, 2), and their reclaimed profiles, shape (inputs, 2, slices); one row of material and of
    deposition variables per input. The deposition variables are stacker positions at equal time steps over the stacking time, or at
    `deposition_times` (one time per variable, from 0 to `MAX_TIMESTAMP`, the same for all inputs): a repeated position is a stop.

    With drops "cells" (the default of every new setting) the material flows evenly over the stacking time, portion i from i / n to
    (i + 1) / n, while the stacker follows the piecewise linear path of the deposition variables, and every grid cell of the simulator that
    the path crosses gets the volume of the time spent over it (`continuous_drops`, as `MaterialDeposition.drops`). Drops "point" exist only
    to reproduce results from before simulator release 2026.3 (settings recorded without drops): every material portion at the stacker
    position of its timestamp, portion i at time i / (n - 1), the rows of `MaterialDeposition.point_drops`. It simulates all inputs in
    one call and evaluates them like `ReclaimedMaterialEvaluator`, with numpy for all inputs at once: at a coarse detail level pandas took
    several times as long as the simulation itself. `threads` simulations run at the same time in this process.
    """
    material = np.atleast_2d(np.asarray(material_variables, dtype=float))
    deposition = np.atleast_2d(np.asarray(deposition_variables, dtype=float))
    inputs, count = material.shape
    x_min, x_max = get_core_limits(bed_size_x, bed_size_z)

    settings = get_simulator_settings()
    deposition_timestamps = np.linspace(0, MAX_TIMESTAMP, deposition.shape[1]) if deposition_times is None else np.asarray(deposition_times, float)
    if len(deposition_timestamps) != deposition.shape[1]:
        raise ValueError("need one deposition time per deposition variable")
    if settings.drops == "point":
        material_timestamps = np.linspace(0, MAX_TIMESTAMP, count)
        data = np.empty((inputs, count, 4))
        data[:, :, 0] = [np.interp(material_timestamps, deposition_timestamps, row) for row in deposition]
        data[:, :, 1] = 0.5 * bed_size_z
        data[:, :, 2] = total_volume / count
        data[:, :, 3] = material
    else:
        cell_x, cell_z = drop_grid(settings.simulator, settings.ppm3, bed_size_x, bed_size_z)
        portion_times = np.linspace(0, MAX_TIMESTAMP, count + 1)
        volumes = np.full(count, total_volume / count)
        rows = [
            continuous_drops(volumes, material[i], portion_times, deposition_timestamps, deposition[i], None, cell_x, cell_z, z=0.5 * bed_size_z)
            for i in range(inputs)
        ]
        # inputs drop different numbers of times: rows without volume fill the batch up, the simulators skip them
        data = np.zeros((inputs, max(len(r) for r in rows), 4))
        data[:, :, :2] = [x_min, 0.5 * bed_size_z]
        for i, r in enumerate(rows):
            data[i, : len(r)] = r

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
        eight_height_scale=settings.eight_height_scale,
        lattice=settings.simulator == "lattice",
        cone=settings.simulator == "cone",
        formula=settings.simulator == "formula",
    )

    volume, quality = reclaimed["volume"], reclaimed["quality"]
    total = volume.sum(axis=1, keepdims=True)
    mean_quality = (quality * volume).sum(axis=1, keepdims=True) / total
    f1 = np.sqrt((volume * (quality - mean_quality) ** 2).sum(axis=1, keepdims=True) / total)[:, 0]
    deviation = get_ideal_stockpile_volumes(reclaimed["x"], total, x_min, x_max, rule=settings.ideal_stockpile) - volume
    f2 = np.sqrt(((deviation - deviation.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
    return np.stack([f1, f2], axis=1), np.stack([volume, quality], axis=1)
