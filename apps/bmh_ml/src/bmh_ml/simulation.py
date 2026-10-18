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
    deposition = np.atleast_2d(np.asarray(deposition_variables, dtype=float))
    deposition_timestamps = np.linspace(0, MAX_TIMESTAMP, deposition.shape[1]) if deposition_times is None else np.asarray(deposition_times, float)
    if len(deposition_timestamps) != deposition.shape[1]:
        raise ValueError("need one deposition time per deposition variable")
    reclaimed = reclaim_paths(material_variables, [(deposition_timestamps, row) for row in deposition], bed_size_x, bed_size_z, total_volume, threads)
    x_min, x_max = get_core_limits(bed_size_x, bed_size_z)
    objectives = get_objectives(reclaimed["x"], reclaimed["volume"], reclaimed["quality"], x_min, x_max)
    return objectives, np.stack([reclaimed["volume"], reclaimed["quality"]], axis=1)


def get_objectives(x: np.ndarray, volume: np.ndarray, quality: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """F1 (the volume-weighted standard deviation of the reclaimed quality) and F2 (the standard deviation of the slice volumes from those
    of the ideal stockpile of the reclaimed volume, by the rule of the active settings) of reclaimed profiles, shape (inputs, 2)."""
    total = volume.sum(axis=1, keepdims=True)
    mean_quality = (quality * volume).sum(axis=1, keepdims=True) / total
    f1 = np.sqrt((volume * (quality - mean_quality) ** 2).sum(axis=1, keepdims=True) / total)[:, 0]
    deviation = get_ideal_stockpile_volumes(x, total, x_min, x_max, rule=get_simulator_settings().ideal_stockpile) - volume
    f2 = np.sqrt(((deviation - deviation.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
    return np.stack([f1, f2], axis=1)


def reclaim_paths(
    material_variables,
    paths: list[tuple[np.ndarray, np.ndarray]],
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
    threads: int = 1,
    fraction: float = 1.0,
    mixing: bool = False,
) -> dict[str, np.ndarray]:
    """Stacks and reclaims many inputs with the active simulator settings, each with its own stacker path: `paths` holds one pair (times,
    positions) per input, the knots of a piecewise linear path with times from 0 to at most `MAX_TIMESTAMP` (a repeated position is a stop;
    paths may have different numbers of knots). One row of material variables per input, or one row for all.

    The material flows evenly over the stacking time as in `evaluate_sims_with_profiles`. With `fraction` below 1 stacking stops at that
    share of the stacking time, and the partial pile of the first `fraction` of the material is reclaimed: the stacker's path is cut there,
    a path that ends earlier ends the material flow with it. `mixing` also returns the share of each material portion in each reclaimed
    slice, the mixing weights W, from the same simulation (every portion carries an indicator parameter of its own): the reclaimed quality
    is W m for any material m, exactly where the simulation mixes linearly (the cone, the formula and the detailed simulation per seed).

    :return: reclaimer positions x, shape (slices,); the reclaimed volume and quality, shape (inputs, slices); with `mixing` the weights,
        shape (inputs, slices, portions) (0 for empty slices)
    """
    settings = get_simulator_settings()
    installed = get_installed_simulator_version()
    if settings.version != installed:
        raise RuntimeError(
            f"The active simulator settings are of simulator release {settings.version} (e.g. the level of a bundle built with it), but"
            f" {installed} is installed: results of different releases must not be mixed, simulate with settings of the installed release"
        )
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    material = np.atleast_2d(np.asarray(material_variables, dtype=float))
    if len(material) == 1 and len(paths) > 1:
        material = np.repeat(material, len(paths), axis=0)
    if len(material) != len(paths):
        raise ValueError("need one row of material variables per path, or one for all")
    parameter_columns = [f"portion_{j}" for j in range(material.shape[1])] if mixing else []
    columns = ["x", "z", "volume", "quality", *parameter_columns]
    data = _drop_data(material, paths, bed_size_x, bed_size_z, total_volume, fraction, mixing)
    reclaimed = simulate_batch(
        data,
        columns,
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
    result = {"x": reclaimed["x"], "volume": reclaimed["volume"], "quality": reclaimed["quality"]}
    if mixing:
        result["weights"] = np.stack([reclaimed[name] for name in parameter_columns], axis=-1)
    return result


def _drop_data(material: np.ndarray, paths, bed_size_x: float, bed_size_z: float, total_volume: float, fraction: float, mixing: bool) -> np.ndarray:
    """The simulator input of `reclaim_paths`: rows (x, z, volume, quality, with `mixing` one indicator per portion) per input."""
    settings = get_simulator_settings()
    inputs, count = material.shape
    width = 4 + (count if mixing else 0)
    if settings.drops == "point":
        if fraction < 1.0:
            raise ValueError("partial piles need continuous drops")
        material_timestamps = np.linspace(0, MAX_TIMESTAMP, count)
        data = np.empty((inputs, count, width))
        data[:, :, 0] = [np.interp(material_timestamps, times, x) for times, x in paths]
        data[:, :, 1] = 0.5 * bed_size_z
        data[:, :, 2] = total_volume / count
        data[:, :, 3] = material
        if mixing:
            data[:, :, 4:] = np.eye(count)
        return data
    x_min, _ = get_core_limits(bed_size_x, bed_size_z)
    end = fraction * MAX_TIMESTAMP
    cell_x, cell_z = drop_grid(settings.simulator, settings.ppm3, bed_size_x, bed_size_z)
    rows = []
    for i, (times, x) in enumerate(paths):
        times, x = np.asarray(times, dtype=float), np.asarray(x, dtype=float)
        # a path ending within rounding of the stop (e.g. a Chevron schedule scaled to the stacking time) ends with the flow
        stop = end if times[-1] >= end - 1e-9 * MAX_TIMESTAMP else times[-1]
        if times[-1] > stop:
            keep = times < stop
            times, x = np.append(times[keep], stop), np.append(x[keep], np.interp(stop, times, x))
        # portion j flows from j / n to (j + 1) / n of the stacking time; what would flow after the stop stays in the stream
        portion_times = np.linspace(0, MAX_TIMESTAMP, count + 1)
        clipped = np.minimum(portion_times, stop)
        volumes = total_volume / count * np.diff(clipped) / np.diff(portion_times)
        parameters = np.column_stack([material[i], np.eye(count)]) if mixing else material[i]
        rows.append(continuous_drops(volumes, parameters, clipped, times, x, None, cell_x, cell_z, z=0.5 * bed_size_z))
    # inputs drop different numbers of times: rows without volume fill the batch up, the simulators skip them
    data = np.zeros((inputs, max(len(r) for r in rows), width))
    data[:, :, :2] = [x_min, 0.5 * bed_size_z]
    for i, r in enumerate(rows):
        data[i, : len(r)] = r
    return data
