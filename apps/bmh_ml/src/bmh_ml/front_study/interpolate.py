"""Simulates straight lines in deposition space between every pair of the final-front solutions of one T6 material (verify.py first).

Point k of the line from a to b is a + k / SEGMENTS * (b - a); the end points carry the labels of the solutions (stored and fresh
16 repeats pooled), the interior points are simulated REPEATS times. The box of the deposition variables is convex, so every point is a
valid deposition."""

import sys
import time

import numpy as np

from bmh_ml.datasets.simulate import simulate
from bmh_ml.evaluation.transfer import get_nondominated
from bmh_ml.front_study.common import get_out

OUT = get_out()
N_JOBS = 14
SEGMENTS = 40
REPEATS = 8


def main(material_index: int):
    data = np.load(OUT / f"verify-material-{material_index}.npz")
    material, deposition = data["material"], data["deposition"]
    labels = (data["stored"] + data["fresh"]) / 2  # 32 simulations per solution
    front = get_nondominated(labels)
    pairs = np.array([(i, j) for i in range(len(deposition)) for j in range(i + 1, len(deposition))])
    t = np.arange(1, SEGMENTS) / SEGMENTS
    a, b = deposition[pairs[:, 0]], deposition[pairs[:, 1]]
    points = (a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]).reshape(-1, deposition.shape[1])
    print(f"{len(deposition)} solutions, front {len(front)}, {len(pairs)} pairs, {len(points)} points x {REPEATS} repeats")
    start = time.perf_counter()
    y, y_sd = simulate(material[None, :], points, repeats=REPEATS, n_jobs=N_JOBS, chunk_size=2000)
    print(f"simulated in {time.perf_counter() - start:.0f} s")
    lines = np.concatenate([labels[pairs[:, 0]][:, None], y.reshape(len(pairs), SEGMENTS - 1, 2), labels[pairs[:, 1]][:, None]], axis=1)
    np.savez(
        OUT / f"lines-material-{material_index}.npz",
        material=material, deposition=deposition, labels=labels, front=front, pairs=pairs, t=np.linspace(0, 1, SEGMENTS + 1),
        lines=lines, interior_sd=y_sd.reshape(len(pairs), SEGMENTS - 1, 2), chevron=data["chevron"],
    )  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]))
