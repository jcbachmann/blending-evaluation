"""The resolution ladder as a static figure for print (PDF, SVG, PNG at 300 dpi), from the clouds of resolution.py.

(a) every final solution at every particle resolution: its 128 single simulations (faint) and their mean, relative to Chevron at the same
    resolution, with the front (non-dominated means) of each resolution;
(b) the noise of one simulation (standard deviation, mean over the solutions) against the particle edge, relative to Chevron;
(c) the time of one simulation against the particle edge (timing-material-<i>.json of time_levels.py if present, else the throughput of
    resolution.py).

Usage: python -m bmh_ml.front_study.plot_resolution_figure <material_index>."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bmh.helpers.fronts import get_nondominated

from bmh_ml.front_study.common import get_out

OUT = get_out()
# the colors of the page (light theme): coarser than the optimizer orange, the optimizer's level grey, finer a blue ramp
COLORS = {"L0": "#eb6834", "L1": "#6f6d68", "L2": "#6da7ec", "L3": "#2a78d6", "L4": "#1c5cab", "L5": "#0d366b"}


def main(material_index: int):
    c = np.load(OUT / f"clouds-material-{material_index}.npz")
    names, ppm3, samples, chevron = [str(n) for n in c["names"]], c["ppm3"], c["samples"], c["chevron"]
    chevron_mean = chevron.mean(axis=1)  # levels x 2
    relative = samples / chevron_mean[:, None, None, :]
    means = relative.mean(axis=2)
    edge = ppm3 ** (-1 / 3)
    timing_path = OUT / f"timing-material-{material_index}.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        seconds, timing_label = np.array([timing["levels"][n]["median_seconds"] for n in names]), "one simulation, single process"
    else:
        seconds, timing_label = c["seconds"], "throughput of 14 processes, per run"

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "legend.frameon": False, "savefig.bbox": "tight"})  # fmt: skip
    fig = plt.figure(figsize=(7.2, 3.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.55, wspace=0.32)
    ax = fig.add_subplot(grid[:, 0])
    for k, name in enumerate(names):
        cloud = relative[k].reshape(-1, 2)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.6, color=COLORS[name], alpha=0.06, linewidths=0, rasterized=True)
    for k, name in enumerate(names):
        front = means[k][get_nondominated(means[k])]
        front = front[np.argsort(front[:, 0])]
        ax.step(front[:, 0], front[:, 1], where="post", color=COLORS[name], lw=0.9)
        ax.scatter(means[k][:, 0], means[k][:, 1], s=5, color=COLORS[name], edgecolors="white", linewidths=0.3, zorder=3,
                   label=f"{name}  ppm3 {ppm3[k]:g}, {edge[k]:.2g} m" + ("  optimized here" if ppm3[k] == 1 else ""))  # fmt: skip
    ax.set_xlabel("F1 relative to Chevron at the same resolution")
    ax.set_ylabel("F2 relative to Chevron at the same resolution")
    ax.set_title("(a) 83 final solutions, 128 simulations each", loc="left", fontsize=8)
    ax.legend(loc="upper right", fontsize=6.5, markerscale=2, handletextpad=0.3, title="level  ppm3, particle edge", title_fontsize=6.5)

    ax_noise = fig.add_subplot(grid[0, 1])
    noise = samples.std(axis=2, ddof=1).mean(axis=1) / chevron_mean  # levels x 2
    for j, (label, marker) in enumerate((("F1", "o"), ("F2", "s"))):
        ax_noise.plot(ppm3, noise[:, j], color="#4b5560", lw=0.8, marker=marker, ms=3, label=label)
    for k, name in enumerate(names):
        ax_noise.scatter(ppm3[k], noise[k, 0], s=12, color=COLORS[name], zorder=3, linewidths=0)
        ax_noise.scatter(ppm3[k], noise[k, 1], s=12, color=COLORS[name], zorder=3, linewidths=0, marker="s")
    ax_noise.set_xscale("log")
    ax_noise.set_yscale("log")
    ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1) if noise.min() / 1.5 <= t <= noise.max() * 1.5]
    ax_noise.set_yticks(ticks)
    ax_noise.set_yticklabels([f"{t:g}" for t in ticks])
    ax_noise.set_xlabel("ppm3 (particles per m³)")
    ax_noise.set_ylabel("sd of one simulation\n(relative to Chevron)")
    ax_noise.set_title("(b) noise", loc="left", fontsize=8)
    ax_noise.legend(fontsize=6.5)

    ax_time = fig.add_subplot(grid[1, 1])
    ax_time.plot(ppm3, seconds * 1000, color="#4b5560", lw=0.8)
    for k, name in enumerate(names):
        ax_time.scatter(ppm3[k], seconds[k] * 1000, s=12, color=COLORS[name], zorder=3, linewidths=0)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if seconds.min() * 1000 / 1.5 <= t <= seconds.max() * 1000 * 1.5]
    ax_time.set_yticks(ticks)
    ax_time.set_yticklabels([f"{t:g}" for t in ticks])
    ax_time.set_xlabel("ppm3 (particles per m³)")
    ax_time.set_ylabel("ms per simulation")
    ax_time.set_title(f"(c) cost ({timing_label})", loc="left", fontsize=8)
    for axis in (ax_noise, ax_time):
        axis.set_xticks(ppm3)
        axis.set_xticklabels([f"{p:g}" for p in ppm3])
        axis.minorticks_off()

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"resolution-figure-material-{material_index}.{suffix}", dpi=300)
    print(f"{OUT / f'resolution-figure-material-{material_index}'}.pdf/.svg/.png")


if __name__ == "__main__":
    main(int(sys.argv[1]))
