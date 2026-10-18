"""Writes the interpolation chart (templates/interpolation.html with the data filled in) and the same data in plain formats:
data-material-<i>.json (exactly what the page embeds), solutions-material-<i>.csv, pairs-material-<i>.csv and lines-material-<i>.csv.gz
(every point of the lines: refine.py placed them, deepen.py averaged them over 32 simulations).

The solutions are represented by the mean of their single simulations at L1 (resolution.py), the front is the non-dominated part of
these means."""

import base64
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from bmh.helpers.fronts import dominated_by, get_nondominated

from bmh_ml.front_study.common import get_out

TEMPLATE = Path(__file__).parent / "templates" / "interpolation.html"
OUT = get_out()


def values_at(r, position: int, n_pairs: int) -> np.ndarray:
    """F1 and F2 of every line at one grid position that every line has (a base point of interpolate.py), one row per pair."""
    rows = np.flatnonzero(r["position"] == position)
    values = np.full((n_pairs, 2), np.nan)
    values[r["pair"][rows]] = r["f"][rows]
    return values


def get_beyond(b) -> dict:
    """The confirmed points of confirm_beyond.py (still non-dominated after 32 fresh simulations): their pair and fresh objectives."""
    c = b["confirmed"]
    return {"pair": b["pair"][c].tolist(), "y": [[round(float(f1), 4), round(float(f2), 3)] for f1, f2 in b["y"][c]]}


def encode_lines(r, n_pairs: int, grid: int) -> dict:
    """All points of the refined lines in one gzipped, base64 encoded little-endian int16 array: the number of points of every pair, then
    per point (sorted by pair and position) the position on the grid, F1 in 1e-4 and F2 in 1e-3, each as differences to the previous
    point of the same pair (the first point of a pair absolute). The page sums the differences up again."""
    order = np.lexsort((r["position"], r["pair"]))
    pair, position, f = r["pair"][order], r["position"][order], r["f"][order]
    counts = np.bincount(pair, minlength=n_pairs)
    starts = np.r_[0, np.cumsum(counts)[:-1]]

    def delta(values: np.ndarray) -> np.ndarray:
        d = np.diff(values, prepend=0)
        d[starts] = values[starts]
        assert np.abs(d).max() < 2**15
        return d

    columns = [counts, delta(position), delta(np.round(f[:, 0] * 1e4).astype(np.int64)), delta(np.round(f[:, 1] * 1e3).astype(np.int64))]
    blob = np.concatenate(columns).astype("<i2").tobytes()
    return {"grid": grid, "points": len(pair), "data": base64.b64encode(gzip.compress(blob, 9, mtime=0)).decode()}


def write_csvs(material_index, labels, ids, front, deposition, clouds, pairs, r, grid):
    """One row per solution, one per pair (line), and one per point of every line (long format, gzipped)."""
    with open(OUT / f"solutions-material-{material_index}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "id", "front", "f1", "f2", "f1_sd_single", "f2_sd_single"] + [f"x{k + 1}" for k in range(deposition.shape[1])])
        sd = clouds.std(axis=1, ddof=1)
        for i in range(len(labels)):
            w.writerow([i, ids[i], int(i in set(front.tolist())), *np.round(labels[i], 6), *np.round(sd[i], 6), *np.round(deposition[i], 4)])
    front_set = set(front.tolist())
    with open(OUT / f"pairs-material-{material_index}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "row_a", "row_b", "id_a", "id_b", "both_front", "distance_m", "points"])
        counts = np.bincount(r["pair"], minlength=len(pairs))
        for p, (a, b) in enumerate(pairs):
            distance = round(float(np.linalg.norm(deposition[a] - deposition[b])), 4)
            w.writerow([p, a, b, ids[a], ids[b], int(a in front_set and b in front_set), distance, counts[p]])
    order = np.lexsort((r["position"], r["pair"]))
    with gzip.open(OUT / f"lines-material-{material_index}.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "position", "t", "f1", "f2", "f1_sd", "f2_sd", "simulations", "round"])
        for i in order:
            sd = ("", "") if np.isnan(r["sd"][i, 0]) else (f"{r['sd'][i, 0]:.5f}", f"{r['sd'][i, 1]:.4f}")
            f1, f2 = f"{r['f'][i, 0]:.5f}", f"{r['f'][i, 1]:.4f}"
            w.writerow([r["pair"][i], r["position"][i], f"{r['position'][i] / grid:.6f}", f1, f2, *sd, r["simulations"][i], r["round"][i]])


def main(material_index: int):
    d = np.load(OUT / f"lines-material-{material_index}.npz")
    check = json.loads((OUT / f"verify-material-{material_index}.json").read_text())
    pairs, deposition = d["pairs"], d["deposition"]
    c = np.load(OUT / f"clouds-material-{material_index}.npz")
    level = list(c["ppm3"]).index(1.0)
    clouds, chevron_cloud = c["samples"][level], c["chevron"][level]
    labels, chevron = clouds.mean(axis=1), chevron_cloud.mean(axis=0)
    front = get_nondominated(labels)
    r = np.load(OUT / f"deep-material-{material_index}.npz")
    r = {k: r[k] for k in r.files}
    grid = int(r["position"].max())
    b = np.load(OUT / f"beyond-material-{material_index}.npz")

    # number the solutions by F1, so the ids read along the front
    order = np.argsort(labels[:, 0])
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))

    interior = (r["position"] > 0) & (r["position"] < grid)
    confirmed = b["y"][b["confirmed"]]
    front_pair = np.isin(pairs, front).all(axis=1)
    close = np.linalg.norm(deposition[pairs[:, 0]] - deposition[pairs[:, 1]], axis=1)[b["pair"][b["confirmed"]]]
    mid = values_at(r, grid // 2, len(pairs))  # every line has its base points, the middle among them
    chord = (labels[pairs[:, 0]] + labels[pairs[:, 1]]) / 2
    stats = {
        "points": int(interior.sum()),
        "base_points": int(len(pairs) * (len(d["t"]) - 2)),
        "simulations_per_point": int(np.median(r["simulations"][interior])),
        "cloud": int(clouds.shape[1]),
        "front_size": len(front),
        "rounds": int(r["round"].max()),
        "beyond": len(b["confirmed"]),
        "confirmed": int(b["confirmed"].sum()),
        "front_beaten": int(dominated_by(labels[front], confirmed).sum()),
        "confirmed_distance_median": float(np.median(close)),
        "mid_f1": float(np.median(mid[front_pair, 0])),
        "mid_f2": float(np.median(mid[front_pair, 1])),
        "chord_f1": float(np.median(chord[front_pair, 0])),
        "chord_f2": float(np.median(chord[front_pair, 1])),
        "distance_median": float(np.median(np.linalg.norm(deposition[pairs[:, 0]] - deposition[pairs[:, 1]], axis=1))),
    }
    print(json.dumps(stats, indent=1))

    data = {
        "material_index": material_index,
        "material": np.round(d["material"], 3).tolist(),
        "chevron": np.round(chevron, 4).tolist(),
        "labels": [[round(float(a), 4), round(float(b), 3)] for a, b in labels],
        "id": (rank + 1).tolist(),
        "front": front.tolist(),
        "deposition": np.round(deposition, 2).tolist(),
        "pairs": pairs.tolist(),
        "lines": encode_lines(r, len(pairs), grid),
        "check": check,
        "beyond": get_beyond(b),
        # single simulations of every solution (resolution.py, level L1), quantized like the lines, interleaved per simulation
        "clouds": np.stack([np.round(clouds[..., 0] * 1e4), np.round(clouds[..., 1] * 1e3)], axis=-1).astype(int).reshape(len(clouds), -1).tolist(),
        "chevron_cloud": [round(v * q) for f1, f2 in chevron_cloud for v, q in ((f1, 1e4), (f2, 1e3))],
        "stats": stats,
    }
    payload = json.dumps(data, separators=(",", ":"))
    page = TEMPLATE.read_text().replace("/*DATA*/null", payload)
    (OUT / f"data-material-{material_index}.json").write_text(payload)
    write_csvs(material_index, labels, rank + 1, front, deposition, clouds, pairs, r, grid)
    target = OUT / f"front-interpolation-material-{material_index}.html"
    target.write_text(page)
    print(f"{target} {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(int(sys.argv[1]))
