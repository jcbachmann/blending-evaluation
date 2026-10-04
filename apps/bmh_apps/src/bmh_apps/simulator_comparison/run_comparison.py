#!/usr/bin/env python
"""
Run the simulator comparison: every simulation method on every deposition pattern and resolution, with data for the viewer.

    python -m bmh_apps.simulator_comparison.run_comparison all --out <dir> [--jobs N] [--timeout s]
    python -m bmh_apps.simulator_comparison.run_comparison one --out <dir> --pattern cone --resolution mid --method hcp

Each run writes <dir>/runs/<pattern>/<resolution>/<method>.json and, for the particle simulators, <method>.particles.txt with the
particles as base64 text (artifact hosts serve no binary files) of int16 x, y, z centers in cm (n x 3, row major) followed by uint8
quality from 0 to 255 (n), little endian. Methods with a stockpile geometry store snapshots for the time slider in the JSON: the cross
and long section at every snapshot time and, for the methods without particles, the height maps on a grid of SNAPSHOT_CELL as base64 int16
heights in cm (snapshots x z x x, row major). The command "all" runs every combination in separate processes, killing runs that exceed the
timeout, and writes <dir>/index.json and the input curves.
"""

import argparse
import base64
import itertools
import json
import logging
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Material
from bmh.helpers.reclaimed_material_evaluator import ReclaimedMaterialEvaluator
from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from bmh.simulation.mathematical_blending_simulator import MathematicalBlendingSimulator
from bmh.simulation.smooth_blending_simulator import SmoothBlendingSimulator

from .particle_simulators import GridSimulator, LatticeSimulator, ParticleSimulator
from .scenarios import (
    BASE_PATTERNS,
    BED_SIZE_X,
    BED_SIZE_Z,
    DURATION,
    END_VARIANTS,
    METHODS,
    PATTERNS,
    RADIUS,
    RESOLUTIONS,
    X_MAX,
    X_MIN,
    get_layer_end_times,
    make_material_deposition,
    parse_pattern,
)

RECLAIM_INCREMENT = 1.0
# The detailed simulator creates particles at this absolute height 5 m before the stacker position and throws them along z; 16 m lets
# them land at the stacker position on flat ground, while the default of half the bed depth equals the pile height and shifts the pile
DETAILED_DROP_HEIGHT = 16.0
HEIGHT_MAP_CELL = 0.25
SEED = 1
# Stockpile snapshots for the time slider: evenly spaced times plus the ends of up to SNAPSHOT_LAYERS layers (all layer ends are evaluated for
# the ridge variation)
SNAPSHOT_STEPS = 32
SNAPSHOT_LAYERS = 32
SNAPSHOT_CELL = 0.5
# Particle counts over time for the viewer to show the particles stacked up to any time
PARTICLE_COUNT_STEPS = 400

# Methods without a stockpile geometry do not depend on the particle resolution, they run once at their own resolution
RESOLUTION_FREE_METHODS = ("mathematical", "smooth")
# The detailed simulation takes minutes to hours per run, it only runs for these patterns and resolutions
DETAILED_RUNS = (("cone", "mid"), ("chevron-1", "mid"))


def create_simulator(method: str, ppm3: float):
    common = {"bed_size_x": BED_SIZE_X, "bed_size_z": BED_SIZE_Z}
    if method == "mathematical":
        return MathematicalBlendingSimulator(BED_SIZE_X, buffer_size=round(BED_SIZE_X / RECLAIM_INCREMENT))
    if method == "smooth":
        return SmoothBlendingSimulator(BED_SIZE_X, buffer_size=round(BED_SIZE_X / RECLAIM_INCREMENT), sigma_x=0.5 * RADIUS)
    if method in ("fast", "detailed"):
        detailed = method == "detailed"
        drop_height = DETAILED_DROP_HEIGHT if detailed else None
        return BslBlendingSimulator(**common, ppm3=ppm3, reclaimincrement=RECLAIM_INCREMENT, detailed=detailed, dropheight=drop_height, seed=SEED)
    if method == "grid":
        return GridSimulator(**common, ppm3=ppm3, reclaim_increment=RECLAIM_INCREMENT, seed=SEED)
    if method == "hcp":
        return LatticeSimulator(**common, ppm3=ppm3, reclaim_increment=RECLAIM_INCREMENT, angle_of_repose=45.0)
    raise ValueError(f"unknown method {method}")


def particle_height_map(sim: ParticleSimulator, cell: float = HEIGHT_MAP_CELL, count: int | None = None, first: int = 0) -> np.ndarray:
    """Top of the pile on a grid of the given cell size, rasterizing the footprint of particles first to count, indexed [z][x]."""
    nx = round(BED_SIZE_X / cell)
    nz = round(BED_SIZE_Z / cell)
    heights = np.zeros((nz, nx))
    p = np.asarray(sim.positions[first:count]).reshape(-1, 3)
    if not len(p):
        return heights
    size = sim.particle_size
    top = p[:, 1] + (sim.layer_distance if isinstance(sim, LatticeSimulator) else sim.particle_height)
    half = 0.5 * size
    steps = max(1, math.ceil(size / cell))
    for ox, oz in itertools.product(np.linspace(-half, half, steps + 1), repeat=2):
        xi = np.clip(((p[:, 0] + ox) / cell).astype(int), 0, nx - 1)
        zi = np.clip(((p[:, 2] + oz) / cell).astype(int), 0, nz - 1)
        np.maximum.at(heights, (zi, xi), top)
    return heights


def resample_height_map(heights: list[list[float]], cell: float = HEIGHT_MAP_CELL) -> np.ndarray:
    """Nearest neighbor resampling of a native height map onto a grid of the given cell size."""
    native = np.asarray(heights, dtype=float)
    nx = round(BED_SIZE_X / cell)
    nz = round(BED_SIZE_Z / cell)
    xi = np.minimum(((np.arange(nx) + 0.5) * native.shape[1] / nx).astype(int), native.shape[1] - 1)
    zi = np.minimum(((np.arange(nz) + 0.5) * native.shape[0] / nz).astype(int), native.shape[0] - 1)
    return native[np.ix_(zi, xi)]


def cone_angle(heights: np.ndarray) -> float | None:
    """Effective angle of repose in degrees: slope of the height over the distance from the bed center between 20 % and 80 % of the peak."""
    nz, nx = heights.shape
    x = (np.arange(nx) + 0.5) * HEIGHT_MAP_CELL - 0.5 * BED_SIZE_X
    z = (np.arange(nz) + 0.5) * HEIGHT_MAP_CELL - 0.5 * BED_SIZE_Z
    distance = np.hypot(*np.meshgrid(x, z))
    peak = heights.max()
    mask = (heights > 0.2 * peak) & (heights < 0.8 * peak)
    if peak <= 0 or mask.sum() < 3:
        return None
    slope = np.polyfit(distance[mask], heights[mask], 1)[0]
    return math.degrees(math.atan(-slope))


def snapshot_times(pattern: str) -> tuple[list[float], list[float], list[float]]:
    """Times to evaluate (stored ones and all layer ends), the stored snapshot times and the layer end times."""
    layer_ends = [round(t, 3) for t in get_layer_end_times(pattern)] if pattern != "cone" else []
    kept = layer_ends
    if len(layer_ends) > SNAPSHOT_LAYERS:
        kept = [layer_ends[round(i)] for i in np.linspace(0, len(layer_ends) - 1, SNAPSHOT_LAYERS)]
    stored = sorted({round(DURATION * (k + 1) / SNAPSHOT_STEPS, 3) for k in range(SNAPSHOT_STEPS)} | set(kept))
    return sorted(set(stored) | set(layer_ends)), stored, layer_ends


def ridge(height_map: np.ndarray) -> np.ndarray:
    """Pile height on the bed center line, the ridge of a chevron stockpile."""
    center = round(RADIUS / SNAPSHOT_CELL)
    return height_map[center - 1 : center + 1, :].max(axis=0)


def cross_section(height_map: np.ndarray) -> np.ndarray:
    return height_map[:, min(height_map.shape[1] - 1, int(0.5 * BED_SIZE_X / SNAPSHOT_CELL))]


def ridge_variation(maps: dict[float, np.ndarray], layer_ends: list[float]) -> float | None:
    """Mean over the layer ends of the standard deviation of the ridge height along the core."""
    if not layer_ends:
        return None
    x = (np.arange(next(iter(maps.values())).shape[1]) + 0.5) * SNAPSHOT_CELL
    core = (x > X_MIN) & (x < X_MAX)
    return float(np.mean([ridge(maps[t])[core].std() for t in layer_ends]))


def stack_rows(sim, rows: pd.DataFrame) -> None:
    for row in rows[["timestamp", "x", "z", "volume", "quality"]].itertuples(index=False):
        sim.stack(row.timestamp, row.x, row.z, row.volume, [row.quality])  # noqa: PD013


def reclaim_frame(sim) -> pd.DataFrame:
    rows = sim.reclaim()
    return pd.DataFrame({"x": [r[0] for r in rows], "volume": [r[1] for r in rows], "quality": [r[2][0] if r[2] else 0.0 for r in rows]})


def run_one(out: Path, pattern: str, resolution: str, method: str) -> dict:
    ppm3 = RESOLUTIONS[resolution]
    material_deposition = make_material_deposition(pattern)
    data = material_deposition.data
    evaluated, stored, layer_ends = snapshot_times(pattern)
    sim = create_simulator(method, ppm3)
    maps: dict[float, np.ndarray] | None = None
    slice_snapshots = None
    store_frames = False

    start = time.perf_counter()
    if isinstance(sim, BslBlendingSimulator):
        # Stack in chunks up to every snapshot time; the time includes taking the snapshots
        maps = {}
        stacked = 0
        for t in evaluated:
            chunk = data[(data["timestamp"] <= t + 1e-6)].iloc[stacked:]
            if len(chunk):
                sim.bsl.stack_list(chunk.to_numpy(), chunk.columns.to_list())
                stacked += len(chunk)
            maps[t] = resample_height_map(sim.get_heights(), SNAPSHOT_CELL)
        heights = resample_height_map(sim.get_heights())
        data_dict = sim.bsl.reclaim()
        reclaimed = pd.DataFrame({k: data_dict[k] for k in ("x", "volume", "quality")})
        runtime = time.perf_counter() - start
        store_frames = True
    else:
        stack_rows(sim, data)
        reclaimed = reclaim_frame(sim)
        runtime = time.perf_counter() - start
        if isinstance(sim, ParticleSimulator):
            heights = particle_height_map(sim)
            # Heights only grow, so the particles of each interval are added to the height map of the previous snapshot
            counts = np.searchsorted(np.asarray(sim.times), np.asarray(evaluated) + 1e-6, side="right")
            maps = {}
            current = np.zeros((round(BED_SIZE_Z / SNAPSHOT_CELL), round(BED_SIZE_X / SNAPSHOT_CELL)))
            previous = 0
            for t, count in zip(evaluated, counts, strict=True):
                current = np.maximum(current, particle_height_map(sim, SNAPSHOT_CELL, int(count), first=previous))
                maps[t] = current
                previous = int(count)
        else:
            heights = None
            # Models without geometry: reclaim the material stacked up to each snapshot time
            volumes, qualities = [], []
            for t in stored:
                partial = create_simulator(method, ppm3)
                stack_rows(partial, data[data["timestamp"] <= t + 1e-6])
                frame = reclaim_frame(partial)
                volumes.append(frame["volume"].round(4).tolist())
                qualities.append(frame["quality"].round(4).tolist())
            slice_snapshots = {"times": stored, "volume": volumes, "quality": qualities}

    reclaimed = reclaimed[["x", "volume", "quality"]]
    evaluator = ReclaimedMaterialEvaluator(Material.from_data(reclaimed.assign(timestamp=reclaimed["x"])), x_min=X_MIN, x_max=X_MAX)
    result = {
        "pattern": pattern,
        "resolution": resolution if method not in RESOLUTION_FREE_METHODS else None,
        "method": method,
        "ppm3": ppm3 if method not in RESOLUTION_FREE_METHODS else None,
        "runtime": runtime,
        "duration": DURATION,
        "metrics": {
            "F1": evaluator.get_single_parameter_stdev("quality"),
            "F2": evaluator.get_volume_stdev() if pattern != "cone" else None,
            "volume": float(reclaimed["volume"].sum()),
            "angle": cone_angle(heights) if heights is not None and pattern == "cone" else None,
            "ridge": ridge_variation(maps, layer_ends) if maps is not None else None,
        },
        "reclaim": {k: reclaimed[k].round(6).tolist() for k in ("x", "volume", "quality")},
        "ideal_volume": get_ideal_stockpile_volumes(reclaimed["x"].to_numpy(), reclaimed["volume"].sum(), X_MIN, X_MAX).round(6).tolist()
        if pattern != "cone"
        else None,
        "height_map": None if heights is None else {"cell": HEIGHT_MAP_CELL, "values": heights.round(3).tolist()},
        "particles": None,
        "snapshots": None,
        "slice_snapshots": slice_snapshots,
    }

    path = out / "runs" / pattern / (resolution if method not in RESOLUTION_FREE_METHODS else "any")
    path.mkdir(parents=True, exist_ok=True)
    if maps is not None:
        frames = np.stack([maps[t] for t in stored])
        result["snapshots"] = {
            "times": stored,
            "layer_end_times": layer_ends,
            "cell": SNAPSHOT_CELL,
            "nx": frames.shape[2],
            "nz": frames.shape[1],
            "cross": [cross_section(f).round(2).tolist() for f in frames],
            "long": [ridge(f).round(2).tolist() for f in frames],
            "frames": base64.b64encode(np.round(frames * 100.0).astype("<i2").tobytes()).decode("ascii") if store_frames else None,
        }
    if isinstance(sim, ParticleSimulator) and sim.positions:
        p = np.asarray(sim.positions)
        centers = p + np.array([0.0, 0.5 * sim.particle_height, 0.0])
        quality = np.clip(np.asarray(sim.parameters)[:, 0], 0.0, 1.0)
        data_bytes = np.round(centers * 100.0).astype("<i2").tobytes() + np.round(quality * 255.0).astype(np.uint8).tobytes()
        with open(path / f"{method}.particles.txt", "w") as file:
            file.write(base64.b64encode(data_bytes).decode("ascii"))
        count_times = np.linspace(0.0, DURATION, PARTICLE_COUNT_STEPS + 1)
        result["particles"] = {
            "file": f"{method}.particles.txt",
            "count": len(p),
            "size": sim.particle_size,
            "height": sim.particle_height,
            "shape": sim.shape,
            "lost": sim.lost_particles,
            # Particles are stored in stacking order: the first counts[i] were stacked up to count_times[i]
            "count_times": count_times.round(1).tolist(),
            "counts": np.searchsorted(np.asarray(sim.times), count_times + 1e-6, side="right").tolist(),
        }

    with open(path / f"{method}.json", "w") as file:
        json.dump(result, file, separators=(",", ":"))
    return result


def write_inputs(out: Path) -> None:
    for pattern in PATTERNS:
        data = make_material_deposition(pattern).data
        # Enough samples to show every pass of the stacker
        steps = min(len(data), max(1000, 8 * parse_pattern(pattern)[1]))
        sample = data.iloc[np.linspace(0, len(data) - 1, steps).astype(int)]
        path = out / "runs" / pattern
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "input.json", "w") as file:
            json.dump({k: sample[k].round(6).tolist() for k in ("timestamp", "x", "z", "volume", "quality")}, file, separators=(",", ":"))


def tasks(methods: list[str]) -> list[tuple[str, str, str]]:
    result = []
    for pattern, method in itertools.product(PATTERNS, methods):
        if method in RESOLUTION_FREE_METHODS:
            resolutions = ["mid"]
        elif method == "detailed":
            resolutions = [r for p, r in DETAILED_RUNS if p == pattern]
        else:
            resolutions = list(RESOLUTIONS)
        result += [(pattern, resolution, method) for resolution in resolutions]
    # Expensive runs first so that they do not delay the end
    cost = {"detailed": 64, "hcp": 3, "grid": 2}
    return sorted(result, key=lambda t: -cost.get(t[2], 1) * RESOLUTIONS[t[1]])


def run_all(out: Path, jobs: int, timeout: float, methods: list[str]) -> None:
    logger = logging.getLogger(__name__)
    write_inputs(out)
    status: dict[str, dict] = {}
    if (out / "index.json").exists():
        with open(out / "index.json") as file:
            status = json.load(file).get("status") or {}

    def execute(task: tuple[str, str, str]) -> None:
        pattern, resolution, method = task
        key = "/".join(task)
        command = [sys.executable, "-m", "bmh_apps.simulator_comparison.run_comparison", "one", "--out", str(out)]
        command += ["--pattern", pattern, "--resolution", resolution, "--method", method]
        start = time.perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603
            state = "done" if completed.returncode == 0 else "failed"
            if state == "failed":
                logger.error(f"{key} failed:\n{completed.stderr[-2000:]}")
        except subprocess.TimeoutExpired:
            state = "timeout"
        status[key] = {"state": state, "seconds": round(time.perf_counter() - start, 1)}
        finished.append(key)
        logger.info(f"{key}: {state} after {status[key]['seconds']} s ({len(finished)} of {len(all_tasks)})")

    all_tasks = tasks(methods)
    finished: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(execute, all_tasks))

    write_index(out, status)


def write_index(out: Path, status: dict[str, dict] | None = None) -> None:
    current = {"/".join(t) for t in tasks(list(METHODS))}
    runs = []
    for file in sorted((out / "runs").glob("*/*/*.json")):
        if file.name == "input.json":
            continue
        with open(file) as f:
            result = json.load(f)
        key = "/".join([result["pattern"], result["resolution"] or "mid", result["method"]])
        if key not in current:
            continue
        runs.append(
            {
                "pattern": result["pattern"],
                "resolution": result["resolution"],
                "method": result["method"],
                "path": str(file.relative_to(out)),
                "runtime": result["runtime"],
                "metrics": result["metrics"],
                "particles": result["particles"],
            }
        )
    index = {
        "bed": {"x": BED_SIZE_X, "z": BED_SIZE_Z, "x_min": X_MIN, "x_max": X_MAX},
        "reclaim_increment": RECLAIM_INCREMENT,
        "patterns": PATTERNS,
        "base_patterns": BASE_PATTERNS,
        "end_variants": END_VARIANTS,
        "resolutions": RESOLUTIONS,
        "methods": METHODS,
        "runs": runs,
        "status": {k: v for k, v in (status or {}).items() if k in current},
    }
    with open(out / "index.json", "w") as file:
        json.dump(index, file, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare stockpile simulation methods")
    parser.add_argument("command", choices=["all", "one", "index"])
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--pattern", choices=list(PATTERNS))
    parser.add_argument("--resolution", choices=list(RESOLUTIONS))
    parser.add_argument("--method", choices=list(METHODS))
    parser.add_argument("--methods", choices=list(METHODS), nargs="+", default=list(METHODS), help="Methods to run with the command all")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel runs")
    parser.add_argument("--timeout", type=float, default=3 * 3600.0, help="Seconds after which a run is killed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "one":
        result = run_one(args.out, args.pattern, args.resolution, args.method)
        logging.info(f"{args.pattern}/{args.resolution}/{args.method}: {result['runtime']:.1f} s, {result['metrics']}")
    elif args.command == "all":
        run_all(args.out, args.jobs, args.timeout, args.methods)
    else:
        write_index(args.out)


if __name__ == "__main__":
    main()
