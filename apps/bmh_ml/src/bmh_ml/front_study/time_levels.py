"""Times one simulation at every particle resolution of resolution.py in a single process (no worker pool, no process start-up), on an
otherwise idle machine: the cost figures of the resolution chart. Simulates the first front solution of the material and Chevron
alternately, RUNS times per level (fewer at the slow levels), after one untimed warm-up run, and records median, quartiles and the CPU.

Writes out/timing-material-<i>.json."""

import json
import platform
import sys
import time

import numpy as np

from bmh_ml.evaluation.chevron import chevron_deposition
from bmh_ml.front_study.common import get_out
from bmh_ml.front_study.resolution import LEVELS
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, SimulatorSettings, simulator_settings
from bmh_ml.simulation import evaluate_sim

OUT = get_out()
RUNS = {"L0": 200, "L1": 200, "L2": 100, "L3": 50, "L4": 20, "L5": 8}


def cpu_name() -> str:
    try:
        return next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def main(material_index: int):
    d = np.load(OUT / f"verify-material-{material_index}.npz")
    material = list(d["material"])
    depositions = [list(d["deposition"][0]), list(chevron_deposition()[0])]
    result = {"cpu": cpu_name(), "python": platform.python_version(), "levels": {}}
    for name, ppm3 in LEVELS.items():
        with simulator_settings(SimulatorSettings(ppm3=ppm3, reclaim_increment=1.0)):
            evaluate_sim(material, depositions[0], BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME)  # warm-up
            seconds = []
            for r in range(RUNS[name]):
                start = time.perf_counter()
                evaluate_sim(material, depositions[r % 2], BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME)
                seconds.append(time.perf_counter() - start)
        q1, median, q3 = np.percentile(seconds, [25, 50, 75])
        result["levels"][name] = {"ppm3": ppm3, "runs": RUNS[name], "median_seconds": median, "q1_seconds": q1, "q3_seconds": q3}
        print(f"{name} (ppm3 {ppm3:g}): median {median * 1000:.1f} ms per simulation (quartiles {q1 * 1000:.1f} to {q3 * 1000:.1f}),"
              f" {RUNS[name]} runs", flush=True)  # fmt: skip
    (OUT / f"timing-material-{material_index}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
