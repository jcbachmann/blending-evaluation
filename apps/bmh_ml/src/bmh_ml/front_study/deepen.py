"""Averages the interpolation lines over more simulations: every interior point of the refined lines (refine.py) is simulated again until
it has TARGET simulations, pooled with the ones it has (mean and standard deviation of all). The end points of every line become the
solutions as the chart shows them: the mean of their REPEATS single simulations at L1 (resolution.py).

The noise of a point shrinks with the square root of its simulations: from 8 to 32, by half. The steps of the simulator stay
(seeded_scan.py): averaging removes noise, not the deterministic structure.

Writes out/deep-material-<i>.npz with the arrays of refined-material-<i>.npz (`pair`, `position`, `f`, `sd`, `simulations`, `round`),
saved after every CHUNK points; a rerun continues where it stopped."""

import sys
import time

import numpy as np

from bmh_ml.datasets.simulate import simulate
from bmh_ml.front_study.common import get_out

OUT = get_out()
N_JOBS = 14
TARGET = 32
CHUNK = 40_000


def pool(n1, m1, s1, n2, m2, s2):
    """Mean and standard deviation of two groups of simulations from their sizes, means and standard deviations (per objective)."""
    n = (n1 + n2)[:, None]
    mean = (n1[:, None] * m1 + n2[:, None] * m2) / n
    squares = (n1[:, None] - 1) * s1**2 + n1[:, None] * m1**2 + (n2[:, None] - 1) * s2**2 + n2[:, None] * m2**2
    return mean, np.sqrt(np.maximum(squares - n * mean**2, 0) / (n - 1))


def main(material_index: int):
    path = OUT / f"deep-material-{material_index}.npz"
    base = np.load(OUT / f"lines-material-{material_index}.npz")
    clouds = np.load(OUT / f"clouds-material-{material_index}.npz")
    level = list(clouds["ppm3"]).index(1.0)
    labels = clouds["samples"][level].mean(axis=1)
    source = path if path.exists() else OUT / f"refined-material-{material_index}.npz"
    state = {k: v.copy() for k, v in np.load(source).items()}
    grid = int(state["position"].max())
    pairs, deposition, material = base["pairs"], base["deposition"], base["material"]

    end = (state["position"] == 0) | (state["position"] == grid)
    a_or_b = np.where(state["position"] == 0, pairs[state["pair"], 0], pairs[state["pair"], 1])
    state["f"][end] = labels[a_or_b[end]]
    state["simulations"][end] = clouds["samples"].shape[2]

    todo = np.flatnonzero(~end & (state["simulations"] < TARGET))
    print(f"{len(todo)} points to bring to {TARGET} simulations", flush=True)
    for start in range(0, len(todo), CHUNK):
        rows = todo[start : start + CHUNK]
        extra = TARGET - state["simulations"][rows]
        assert np.all(extra == extra[0]), "points of one chunk need the same number of simulations"
        t = state["position"][rows] / grid
        a, b = deposition[pairs[state["pair"][rows], 0]], deposition[pairs[state["pair"][rows], 1]]
        begin = time.perf_counter()
        y, y_sd = simulate(material[None, :], a + t[:, None] * (b - a), repeats=int(extra[0]), n_jobs=N_JOBS, chunk_size=2000)
        state["f"][rows], state["sd"][rows] = pool(state["simulations"][rows], state["f"][rows], state["sd"][rows], extra, y, y_sd)
        state["simulations"][rows] = TARGET
        np.savez_compressed(path, **state)
        print(f"{min(start + CHUNK, len(todo))} of {len(todo)} points done ({time.perf_counter() - begin:.0f} s for this chunk)", flush=True)
    interior = ~end
    standard_error = np.nanmean(state["sd"][interior], axis=0) / np.sqrt(TARGET)
    print(f"mean standard error of a point now: F1 {standard_error[0]:.4f}, F2 {standard_error[1]:.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
