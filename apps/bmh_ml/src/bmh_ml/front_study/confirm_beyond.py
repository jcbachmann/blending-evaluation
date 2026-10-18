"""Re-simulates the interpolated points that no final solution dominates (32 fresh repeats) to separate real improvements from noise.

Works on the deepened lines of deepen.py (out/deep-material-<i>.npz, 32 simulations per point): every interior point of every line, the
base points of interpolate.py and the ones refine.py added. The solutions are the means of their single simulations at L1 (resolution.py)."""

import sys
import time

import numpy as np
from bmh.helpers.fronts import dominated_by, get_nondominated

from bmh_ml.datasets.simulate import simulate
from bmh_ml.front_study.common import get_out
from bmh_ml.front_study.refine import GRID

OUT = get_out()

REPEATS = 32


def main(material_index: int):
    d = np.load(OUT / f"lines-material-{material_index}.npz")
    r = np.load(OUT / f"deep-material-{material_index}.npz")
    clouds = np.load(OUT / f"clouds-material-{material_index}.npz")
    labels = clouds["samples"][list(clouds["ppm3"]).index(1.0)].mean(axis=1)
    front = get_nondominated(labels)
    pairs, deposition = d["pairs"], d["deposition"]
    interior = np.flatnonzero((r["position"] > 0) & (r["position"] < GRID))
    beyond = interior[~dominated_by(r["f"][interior], labels)]
    pair, t = r["pair"][beyond], r["position"][beyond] / GRID
    a, b = deposition[pairs[pair, 0]], deposition[pairs[pair, 1]]
    start = time.perf_counter()
    y, _ = simulate(d["material"][None, :], a + t[:, None] * (b - a), repeats=REPEATS, n_jobs=14, chunk_size=2000)
    confirmed = ~dominated_by(y, labels)
    print(f"{len(beyond)} of {len(interior)} points re-simulated in {time.perf_counter() - start:.0f} s, still non-dominated: {confirmed.sum()}")
    print(f"front solutions dominated by a confirmed point: {dominated_by(labels[front], y[confirmed]).sum()} of {len(front)}")
    np.savez(OUT / f"beyond-material-{material_index}.npz", index=beyond, pair=pair, position=r["position"][beyond], y=y, confirmed=confirmed)


if __name__ == "__main__":
    main(int(sys.argv[1]))
