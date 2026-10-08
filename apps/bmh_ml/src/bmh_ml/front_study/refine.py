"""Adaptive refinement of the interpolation lines of interpolate.py: points are added where a line changes, not where it is flat.

Every line starts with the 41 points of interpolate.py (t = k/40). In each round, every segment between two neighbouring points whose
objectives differ by more than THRESHOLD noise units gets its midpoint, simulated REPEATS times like the other points. A noise unit is
the standard deviation of the difference of two such means (sqrt(2) times the standard error of one point, per objective), and the
two objectives are combined as a Euclidean length, so under pure noise the jump is chi-distributed with 2 degrees of freedom (99 % below
3.0). Segments are not split below 1/GRID of the line. When a round has more candidates than the remaining BUDGET, the largest jumps are
refined first. The state is saved after every round (out/refined-material-<i>.npz), and a rerun resumes from it.

Positions on a line are integers on a grid of GRID steps (t = position / GRID); the 41 base points are at multiples of GRID / 40."""

import sys
import time

import numpy as np

from bmh_ml.datasets.simulate import simulate
from bmh_ml.front_study.common import get_out

OUT = get_out()
N_JOBS = 14
REPEATS = 8
THRESHOLD = 4.0
GRID = 40 * 2**6  # down to 1/2560 of a line
BUDGET = 1_200_000  # new points, about 9 times the 132,717 interior points of interpolate.py
MAX_ROUNDS = 6


def load(material_index: int):
    path = OUT / f"refined-material-{material_index}.npz"
    base = np.load(OUT / f"lines-material-{material_index}.npz")
    if path.exists():
        r = np.load(path)
        return base, {k: r[k] for k in r.files}
    pairs, lines, sd = base["pairs"], base["lines"], base["interior_sd"]
    k = lines.shape[1]
    step = GRID // (k - 1)
    pair = np.repeat(np.arange(len(pairs)), k)
    position = np.tile(np.arange(k) * step, len(pairs))
    f = lines.reshape(-1, 2)
    point_sd = np.concatenate([np.full((len(pairs), 1, 2), np.nan), sd, np.full((len(pairs), 1, 2), np.nan)], axis=1).reshape(-1, 2)
    simulations = np.tile(np.r_[32, np.full(k - 2, REPEATS), 32], len(pairs))
    state = {"pair": pair, "position": position, "f": f, "sd": point_sd, "simulations": simulations, "round": np.zeros(len(pair), dtype=int)}
    return base, state


def noise_unit(state) -> np.ndarray:
    """Standard deviation of the difference of two points of REPEATS simulations, per objective, from the measured spread of all points."""
    interior = state["simulations"] == REPEATS
    return np.sqrt(2) * np.nanmean(state["sd"][interior], axis=0) / np.sqrt(REPEATS)


def candidates(state, unit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((state["position"], state["pair"]))
    pair, position, f = state["pair"][order], state["position"][order], state["f"][order]
    same = pair[1:] == pair[:-1]
    gap = position[1:] - position[:-1]
    jump = np.sqrt((((f[1:] - f[:-1]) / unit) ** 2).sum(axis=1))
    split = same & (gap > 1) & (jump > THRESHOLD)
    return pair[:-1][split], position[:-1][split] + gap[split] // 2, jump[split]


def main(material_index: int):
    base, state = load(material_index)
    material, deposition, pairs = base["material"], base["deposition"], base["pairs"]
    unit = noise_unit(state)
    print(f"noise unit of a jump: F1 {unit[0]:.4f}, F2 {unit[1]:.3f}; {len(state['pair'])} points")
    while state["round"].max() < MAX_ROUNDS:
        added = int((state["round"] > 0).sum())
        pair, position, jump = candidates(state, unit)
        if len(pair) == 0 or added >= BUDGET:
            break
        if len(pair) > BUDGET - added:
            keep = np.argsort(-jump)[: BUDGET - added]
            pair, position, jump = pair[keep], position[keep], jump[keep]
        round_ = state["round"].max() + 1
        t = position / GRID
        a, b = deposition[pairs[pair, 0]], deposition[pairs[pair, 1]]
        points = a + t[:, None] * (b - a)
        start = time.perf_counter()
        y, y_sd = simulate(material[None, :], points, repeats=REPEATS, n_jobs=N_JOBS, chunk_size=2000)
        state = {
            "pair": np.r_[state["pair"], pair],
            "position": np.r_[state["position"], position],
            "f": np.vstack([state["f"], y]),
            "sd": np.vstack([state["sd"], y_sd]),
            "simulations": np.r_[state["simulations"], np.full(len(pair), REPEATS)],
            "round": np.r_[state["round"], np.full(len(pair), round_)],
        }
        np.savez_compressed(OUT / f"refined-material-{material_index}.npz", **state)
        print(f"round {round_}: {len(pair)} points (median jump {np.median(jump):.1f} noise units) in {time.perf_counter() - start:.0f} s,"
              f" {len(state['pair'])} points in all", flush=True)  # fmt: skip
    order = np.lexsort((state["position"], state["pair"]))
    pair, position, f = state["pair"][order], state["position"][order], state["f"][order]
    finest = (pair[1:] == pair[:-1]) & (position[1:] - position[:-1] == 1)
    steps = finest & (np.sqrt((((f[1:] - f[:-1]) / unit) ** 2).sum(axis=1)) > THRESHOLD)
    remaining, _, _ = candidates(state, unit)
    print(f"done: {int((state['round'] > 0).sum())} points added; {steps.sum()} segments of the finest length 1/{GRID} still jump by more than"
          f" {THRESHOLD} noise units (steps located as finely as the grid allows); {len(remaining)} longer segments left above it")  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]))
