"""Simulates every final solution of one T6 material, and the reference (Chevron at maximum speed, 19 layers, `chevron_deposition`),
REPEATS times at several particle resolutions (ppm3, particles per cubic meter; reclaim step fixed at 1 m) and keeps the single results.

Needs only verify.py (out/verify-material-<i>.npz: the material and the 83 final solutions of T6). The level L1 (ppm3 1) is the
resolution of the optimization and of the interpolation lines: its clouds are the noise clouds of the
interpolation chart, and their means represent the solutions there. All levels together are the resolution chart.

Writes out/clouds-material-<i>.npz: `ppm3` (levels), `samples` (levels x solutions x REPEATS x 2), `chevron` (levels x REPEATS x 2),
`seconds` (simulation time per run and level). Saved after every level; a rerun skips the levels already there."""

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bmh_ml.datasets.simulate import simulate_chunk
from bmh_ml.evaluation.chevron import chevron_deposition
from bmh_ml.front_study.common import get_out
from bmh_ml.settings import SimulatorSettings, simulator_settings

OUT = get_out()
N_JOBS = 14
REPEATS = 128
# particle edge 2 m, 1 m (the level of the optimization), 0.63 m, 0.4 m, 0.25 m, 0.16 m; names as in bmh_ml, L0 added here
LEVELS = {"L0": 0.125, "L1": 1.0, "L2": 4.0, "L3": 16.0, "L4": 64.0, "L5": 256.0}


def simulate_level(material: np.ndarray, deposition: np.ndarray, ppm3: float) -> tuple[np.ndarray, float]:
    """REPEATS single simulations of every row of `deposition` at the level, in parallel (one task per row), and seconds per run."""
    start = time.perf_counter()
    rows = len(deposition)
    # the worker processes inherit the setting when they are spawned
    with (
        simulator_settings(SimulatorSettings(ppm3=ppm3, reclaim_increment=1.0)),
        ProcessPoolExecutor(max_workers=N_JOBS, mp_context=multiprocessing.get_context("spawn")) as executor,
    ):
        parts = list(executor.map(simulate_chunk, [material[None, :]] * rows, [row[None, :] for row in deposition], [REPEATS] * rows))
    samples = np.concatenate([part[0] for part in parts])
    return samples, (time.perf_counter() - start) * N_JOBS / (len(deposition) * REPEATS)


def main(material_index: int):
    path = OUT / f"clouds-material-{material_index}.npz"
    d = np.load(OUT / f"verify-material-{material_index}.npz")  # the material and the 83 final solutions of T6, as stored
    material, deposition = d["material"], d["deposition"]
    done = dict(np.load(path)) if path.exists() else {}
    samples, chevron, seconds = [], [], []
    for name, ppm3 in LEVELS.items():
        if "ppm3" in done and ppm3 in done["ppm3"]:
            k = list(done["ppm3"]).index(ppm3)
            samples.append(done["samples"][k]), chevron.append(done["chevron"][k]), seconds.append(done["seconds"][k])
            continue
        rows = np.vstack([deposition, chevron_deposition()])
        result, per_run = simulate_level(material, rows, ppm3)
        samples.append(result[:-1]), chevron.append(result[-1]), seconds.append(per_run)
        mean, sd = result[:-1].mean(axis=1), result[:-1].std(axis=1, ddof=1)
        chevron_mean = result[-1].mean(axis=0)
        print(f"{name} (ppm3 {ppm3}): {per_run * 1000:.1f} ms per run; solutions F1 {mean[:, 0].mean():.4f} (sd of one run {sd[:, 0].mean():.4f}),"
              f" F2 {mean[:, 1].mean():.3f} ({sd[:, 1].mean():.3f}); Chevron F1 {chevron_mean[0]:.4f}, F2 {chevron_mean[1]:.3f}", flush=True)  # fmt: skip
        levels = list(LEVELS.values())[: len(samples)]
        np.savez_compressed(path, ppm3=np.array(levels), names=np.array(list(LEVELS)[: len(samples)]), samples=np.array(samples),
                            chevron=np.array(chevron), seconds=np.array(seconds))  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]))
