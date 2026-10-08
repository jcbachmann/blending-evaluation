"""Writes the resolution chart: templates/resolution.html with the clouds of resolution.py filled in, the final solutions of one T6 material
and Chevron simulated REPEATS times at every particle resolution (ppm3). Also writes resolution-material-<i>.csv (one row per solution
and level) and resolution-data-material-<i>.json (what the page embeds). Needs only numpy."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from bmh_ml.front_study.common import get_out, nondominated

TEMPLATE = Path(__file__).parent / "templates" / "resolution.html"
OUT = get_out()
# where the numbers come from; shown on the page and written to the JSON, so a figure made from them can cite its source
PROVENANCE = {
    "result": "bundle S2-v1 of the bmh_ml store, test set T6 (dataset S2-v1-T6-c6b60ab04f, built 2026-10-02 with code e9b1bc0)",
    "solutions": "the 83 final-front solutions of 3 NSGA-III runs (population 100, 25,000 simulations each, seeds 1 to 3) at L1, as stored in T6",
    "code": "computed with BlendingEvaluation commit dbe2be3 (bmh, bmh_ml and blending_simulator_lib unchanged up to b607412);"
    " scripts in apps/bmh_ml/src/bmh_ml/front_study",
    "simulator": "BlendingSimulator release v2026.2, fast simulation (BslBlendingSimulator), unseeded; the optimization of T6 ran on v2026.1",
    "computed": "2026-10-08, Intel i9-9900K, 14 worker processes",
}


def ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def quantize(samples: np.ndarray) -> list:
    """Single simulations as integers, F1 in 1e-4 and F2 in 1e-3, interleaved: one flat list per row of the leading dimensions."""
    q = np.stack([np.round(samples[..., 0] * 1e4), np.round(samples[..., 1] * 1e3)], axis=-1).astype(int)
    return q.reshape(*q.shape[:-2], -1).tolist()


def main(material_index: int):
    c = np.load(OUT / f"clouds-material-{material_index}.npz")
    d = np.load(OUT / f"verify-material-{material_index}.npz")  # the material and the 83 final solutions of T6, as stored
    ppm3, names, samples, chevron, seconds = c["ppm3"], [str(n) for n in c["names"]], c["samples"], c["chevron"], c["seconds"]
    timing_path = OUT / f"timing-material-{material_index}.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    if timing:  # one simulation in a single process (time_levels.py), else the throughput of the worker pool of resolution.py
        seconds = np.array([timing["levels"][n]["median_seconds"] for n in names])
    means, sds = samples.mean(axis=2), samples.std(axis=2, ddof=1)  # levels x solutions x 2
    chevron_means = chevron.mean(axis=1)
    l1 = list(ppm3).index(1.0)
    finest = int(np.argmax(ppm3))
    order = np.argsort(means[l1][:, 0])  # ids by F1 at L1, as in the interpolation chart
    ids = np.empty(len(order), dtype=int)
    ids[order] = np.arange(1, len(order) + 1)
    fronts = [nondominated(m) for m in means]
    l1_front = set(fronts[l1].tolist())

    levels = []
    for k in range(len(ppm3)):
        relative = means[k] / chevron_means[k]
        levels.append({
            "name": names[k],
            "ppm3": float(ppm3[k]),
            "edge": float(ppm3[k] ** (-1 / 3)),
            "ms_per_run": round(float(seconds[k]) * 1000, 2),
            "chevron": np.round(chevron_means[k], 4).tolist(),
            "noise_sd": np.round(sds[k].mean(axis=0), 5).tolist(),
            "front": fronts[k].tolist(),
            "l1_front_kept": len(l1_front & set(fronts[k].tolist())),
            "shift_vs_finest": np.round((means[k] - means[finest]).mean(axis=0), 4).tolist(),
            "abs_shift_vs_finest": np.round(np.abs(means[k] - means[finest]).mean(axis=0), 4).tolist(),
            "spearman_vs_finest": [round(spearman(means[k][:, j], means[finest][:, j]), 3) for j in range(2)],
            "relative_range": [np.round(relative.min(axis=0), 3).tolist(), np.round(relative.max(axis=0), 3).tolist()],
        })  # fmt: skip
        level = levels[-1]
        print(f"{names[k]} ppm3 {ppm3[k]:g}: noise sd F1 {sds[k][:, 0].mean():.4f} F2 {sds[k][:, 1].mean():.3f}; front {len(fronts[k])}"
              f" ({level['l1_front_kept']} of the L1 front); mean |shift| to {names[finest]} F1 {level['abs_shift_vs_finest'][0]:.4f}"
              f" F2 {level['abs_shift_vs_finest'][1]:.3f}; Spearman F1 {level['spearman_vs_finest'][0]:.3f} F2 {level['spearman_vs_finest'][1]:.3f};"
              f" Chevron F1 {chevron_means[k][0]:.4f} F2 {chevron_means[k][1]:.3f}")  # fmt: skip

    data = {
        "material_index": material_index,
        "material": np.round(d["material"], 3).tolist(),
        "deposition": np.round(d["deposition"], 2).tolist(),
        "id": ids.tolist(),
        "l1": l1,
        "finest": finest,
        "levels": levels,
        "means": np.round(means, 5).tolist(),
        "sds": np.round(sds, 5).tolist(),
        "samples": quantize(samples),  # levels x solutions x (repeats * 2)
        "chevron_samples": quantize(chevron),  # levels x (repeats * 2)
        "repeats": int(samples.shape[2]),
        "timing": "median of single simulations in one process (time_levels.py), " + timing["cpu"] if timing else "throughput of 14 worker processes",
        "provenance": PROVENANCE,
    }
    payload = json.dumps(data, separators=(",", ":"))
    (OUT / f"resolution-data-material-{material_index}.json").write_text(payload)
    with open(OUT / f"resolution-material-{material_index}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "id", "level", "ppm3", "f1_mean", "f2_mean", "f1_sd_single", "f2_sd_single", "simulations", "nondominated_at_level", "on_l1_front"])
        for k in range(len(ppm3)):
            front = set(fronts[k].tolist())
            for i in range(means.shape[1]):
                w.writerow([i, ids[i], names[k], f"{ppm3[k]:g}", f"{means[k, i, 0]:.5f}", f"{means[k, i, 1]:.4f}", f"{sds[k, i, 0]:.5f}",
                            f"{sds[k, i, 1]:.4f}", samples.shape[2], int(i in front), int(i in l1_front)])  # fmt: skip
    page = TEMPLATE.read_text().replace("/*DATA*/null", payload)
    target = OUT / f"resolution-material-{material_index}.html"
    target.write_text(page)
    print(f"{target} {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(int(sys.argv[1]))
