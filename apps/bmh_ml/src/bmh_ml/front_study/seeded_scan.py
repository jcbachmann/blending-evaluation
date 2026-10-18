"""Scans short pieces of interpolation lines with a fixed simulator seed (deterministic simulations, v2026.2) to see the shape of the
response without noise: a staircase (piecewise constant with jumps, from the particle grid) or a continuous change."""

import sys

import numpy as np
from bmh.benchmark.material_deposition import MaterialDeposition
from bmh.helpers.bed_setup import X_MAX, X_MIN
from bmh.helpers.reclaimed_material_evaluator import ReclaimedMaterialEvaluator
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator

from bmh_ml.front_study.common import get_out
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME
from bmh_ml.simulation import generate_deposition, generate_material

OUT = get_out()


def seeded(material, deposition, seed: int) -> tuple[float, float]:
    md = MaterialDeposition(material=generate_material(list(material), TOTAL_VOLUME), deposition=generate_deposition(list(deposition), BED_SIZE_X, BED_SIZE_Z))
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, ppm3=1.0, reclaimincrement=1.0, seed=seed)
    e = ReclaimedMaterialEvaluator(reclaimed=sim.stack_reclaim(md), x_min=X_MIN, x_max=X_MAX)
    return e.get_single_parameter_stdev("quality"), e.get_volume_stdev()


def main(material_index: int):
    d = np.load(OUT / f"lines-material-{material_index}.npz")
    material, deposition, pairs, front = d["material"], d["deposition"], d["pairs"], d["front"]
    dist = np.linalg.norm(deposition[pairs[:, 0]] - deposition[pairs[:, 1]], axis=1)
    front_pair = np.flatnonzero(np.isin(pairs, front).all(axis=1))
    p = front_pair[np.argmax(dist[front_pair])]  # the longest line between two front solutions
    a, b = deposition[pairs[p, 0]], deposition[pairs[p, 1]]
    t = np.linspace(0.25, 0.275, 1001)  # one base segment (1/40 of the line) in 1,000 steps
    for seed in (1, 2):
        y = np.array([seeded(material, a + ti * (b - a), seed) for ti in t])
        change = np.any(np.abs(np.diff(y, axis=0)) > 1e-12, axis=1)
        steps = np.abs(np.diff(y, axis=0))[change]
        print(f"pair {p} ({dist[p]:.1f} m), seed {seed}: {change.sum()} of 1000 steps change F at all; F1 range {np.ptp(y[:, 0]):.4f},"
              f" F2 range {np.ptp(y[:, 1]):.3f}; largest single changes F1 {steps[:, 0].max():.4f}, F2 {steps[:, 1].max():.3f};"
              f" constant stretches: median {np.median(np.diff(np.flatnonzero(np.r_[True, change, True]))):.0f} steps")  # fmt: skip
        np.save(OUT / f"seeded-scan-material-{material_index}-seed{seed}.npy", np.column_stack([t, y]))


if __name__ == "__main__":
    main(int(sys.argv[1]))
