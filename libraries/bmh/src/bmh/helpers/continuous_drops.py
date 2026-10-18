"""Drops for a stacker that moves while the material flows.

A simulator stacks every row it gets at one point. Stacking each material portion at the one stacker position of its time (as
`MaterialDeposition.point_drops` does) turns a moving stacker into separate piles whenever the portions are large compared to the distance the
stacker travels meanwhile. `continuous_drops` instead cuts the stacker path where it crosses a cell boundary of the simulator's grid or a
portion starts or ends: every piece gets the volume that flows during it, and the pieces of one cell and portion become one drop. Every
cell along the path receives the volume of the time the stacker spends over it, the first and the last cell of a pass only the share of
their overlap, and a stacker standing still makes one drop. With the grid of the simulator (`drop_grid`) this is as fine as the simulator
resolves positions anyway, so finer drops change nothing.

The drops follow the path continuously except where a turnaround of the stacker crosses a cell boundary: while its tip lies inside a
cell, the pieces before and after it are one drop, once the tip enters the next cell they become two, and the results jump slightly
(0.1 % in F1 and 0.01 % in F2 for a 19-pass chevron in the cone simulation with 1 m cells). Finite differences and line searches should
know it. Merging only pieces of the same path segment removes the jumps but approximated a fine-grained reference no better
(measured 2026-10-17).
"""

import contextlib
import math

import blending_simulator_lib
import numpy as np

# Tests compare the C++ code of the bindings with the numpy code by setting this
_NUMPY_ONLY = False


def drop_grid(simulator: str, ppm3: float, bed_size_x: float, bed_size_z: float) -> tuple[float, float]:
    """Cell size in x and z at which a simulator distinguishes drop positions: the particle size of the fast simulator, the column
    spacing of the lattice, the grid cells of the cone simulator (about the particle size, adjusted to tile the bed exactly), the cells of
    the formula simulator along the bed (like the cone's) and one cell across it (it needs all drops on one line along the bed)."""
    size = ppm3 ** (-1.0 / 3.0)
    if simulator in ("fast", "detailed"):
        # the detailed simulation releases a stream of spheres of this size around each drop
        return size, size
    if simulator == "cone":
        return bed_size_x / max(1, round(bed_size_x / size)), bed_size_z / max(1, round(bed_size_z / size))
    if simulator == "formula":
        return bed_size_x / max(1, round(bed_size_x / size)), bed_size_z
    if simulator == "lattice":
        # BlendingSimulatorLattice: spheres of diameter d on rows sqrt(3) / 2 d apart, compressed vertically to a 45 degree pile
        vertical_scale = 1.0 / (4.0 / 3.0 * math.sqrt(math.pi / math.sqrt(3.0)))
        diameter = (math.sqrt(2.0) / ppm3 / vertical_scale) ** (1.0 / 3.0)
        return diameter, diameter * math.sqrt(3.0) / 2.0
    raise ValueError(f"unknown simulator {simulator}")


def continuous_drops(
    volumes: np.ndarray,
    parameters: np.ndarray,
    portion_times: np.ndarray,
    path_times: np.ndarray,
    path_x: np.ndarray,
    path_z: np.ndarray | None,
    cell_x: float,
    cell_z: float,
    z: float | None = None,
    with_times: bool = False,
) -> np.ndarray:
    """Rows (x, z, volume, parameters...) for a material flowing along a piecewise linear stacker path, with with_times=True rows (time, x,
    z, volume, parameters...) where time is the volume-weighted mean time of the drop's material.

    :param volumes: volume of each of the n portions
    :param parameters: parameters of each portion, shape (n,) or (n, k)
    :param portion_times: the n + 1 times at which the portions start and end; portion i flows evenly from portion_times[i] to
        portion_times[i + 1] (a portion of volume 0 is a gap without flow)
    :param path_times: times of the path's knots, increasing
    :param path_x: stacker position x at the knots
    :param path_z: stacker position z at the knots, or None for a constant z
    :param cell_x: cell size of the simulator's grid in x (`drop_grid`)
    :param cell_z: cell size in z
    :param z: the constant z if path_z is None
    :param with_times: also return the time of every drop (for callers that stack up to given times)
    """
    volumes = np.asarray(volumes, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim == 1:
        parameters = parameters[:, None]
    bounds_t = np.asarray(portion_times, dtype=float)
    tx = np.asarray(path_times, dtype=float)
    x = np.asarray(path_x, dtype=float)
    zs = np.full(len(tx), float(z)) if path_z is None else np.asarray(path_z, dtype=float)
    n = len(volumes)
    if len(bounds_t) != n + 1 or len(parameters) != n:
        raise ValueError("need n + 1 portion times and n parameter rows for n portions")
    native = getattr(blending_simulator_lib, "continuous_drops", None)
    if native is not None and not _NUMPY_ONLY:
        # the same algorithm in C++, about 50 times faster than the numpy code below, which stays for bindings built before it
        if not with_times:
            return np.asarray(native(volumes, parameters, bounds_t, tx, x, zs, cell_x, cell_z))
        with contextlib.suppress(TypeError):  # bindings built before the drop times have no with_times
            return np.asarray(native(volumes, parameters, bounds_t, tx, x, zs, cell_x, cell_z, with_times=True))

    start, end = bounds_t[0], bounds_t[-1]
    times = np.unique(np.concatenate([tx, bounds_t]))
    times = times[(times >= start) & (times <= end)]
    px, pz = np.interp(times, tx, x), np.interp(times, tx, zs)
    cuts = [times]
    for pos, c in ((px, cell_x), (pz, cell_z)):
        a, b, pa, pb = times[:-1], times[1:], pos[:-1], pos[1:]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        # segments with a cell boundary k c strictly between their ends
        for i in np.nonzero(np.ceil(hi / c) - np.floor(lo / c) >= 2)[0]:
            k = np.arange(np.floor(lo[i] / c) + 1, np.ceil(hi[i] / c))
            cuts.append(a[i] + (k * c - pa[i]) / (pb[i] - pa[i]) * (b[i] - a[i]))
    cut = np.unique(np.concatenate(cuts))
    mid = 0.5 * (cut[:-1] + cut[1:])
    portion = np.clip(np.searchsorted(bounds_t, mid, side="right") - 1, 0, n - 1)
    duration = np.diff(bounds_t)
    rate = np.divide(volumes, duration, out=np.zeros(n), where=duration > 0)
    piece = np.diff(cut) * rate[portion]
    keep = piece > 0
    mid, piece, portion = mid[keep], piece[keep], portion[keep]
    mx, mz = np.interp(mid, tx, x), np.interp(mid, tx, zs)

    # one drop per run of pieces in the same cell and portion, at the pieces' volume-weighted mean position, computed relative to the
    # run's first position so that pieces sharing one position (e.g. on a cell boundary) keep it exactly
    cell = np.floor(mx / cell_x).astype(np.int64) * 1_000_003 + np.floor(mz / cell_z).astype(np.int64)
    first = np.concatenate([[True], (cell[1:] != cell[:-1]) | (portion[1:] != portion[:-1])])
    group = np.cumsum(first) - 1
    head = np.nonzero(first)[0]
    volume = np.bincount(group, piece)
    gx = mx[head] + np.bincount(group, piece * (mx - mx[head][group])) / volume
    gz = mz[head] + np.bincount(group, piece * (mz - mz[head][group])) / volume
    rows = np.column_stack([gx, gz, volume, parameters[portion[first]]])
    if with_times:
        return np.column_stack([np.bincount(group, piece * mid) / volume, rows])
    return rows


def portion_bounds(timestamps: np.ndarray) -> np.ndarray:
    """The n + 1 times at which n material portions with the given timestamps start and end: the midpoints between neighbouring
    timestamps, the first and last portion extended by half their spacing, and the whole mapped linearly onto the span of the
    timestamps. Evenly spaced timestamps give n equal intervals, portion i from i / n to (i + 1) / n of the stacking time."""
    t = np.asarray(timestamps, dtype=float)
    if len(t) == 1:
        return np.array([t[0], t[0]])
    bounds = np.concatenate([[1.5 * t[0] - 0.5 * t[1]], 0.5 * (t[1:] + t[:-1]), [1.5 * t[-1] - 0.5 * t[-2]]])
    return t[0] + (bounds - bounds[0]) * (t[-1] - t[0]) / (bounds[-1] - bounds[0])


def material_deposition_drops(
    material_timestamps: np.ndarray,
    volumes: np.ndarray,
    parameters: np.ndarray,
    deposition_timestamps: np.ndarray,
    deposition_x: np.ndarray,
    deposition_z: np.ndarray,
    cell_x: float,
    cell_z: float,
    with_times: bool = False,
) -> np.ndarray:
    """Drops of a material (portions at timestamps, `portion_bounds`) flowing while the stacker follows a deposition (positions at
    timestamps, linear in between): `continuous_drops` on the simulator's grid."""
    bounds = portion_bounds(material_timestamps)
    if bounds[-1] <= bounds[0]:
        # a single portion (or portions without a time span) flows during the whole deposition
        bounds = np.linspace(deposition_timestamps[0], deposition_timestamps[-1], len(bounds))
    return continuous_drops(volumes, parameters, bounds, deposition_timestamps, deposition_x, deposition_z, cell_x, cell_z, with_times=with_times)
