"""Checks that the stored T6 front of one material of bundle S2-v1 can be recomputed from its stored parameters.

1. The materials of T6 are regenerated from the bundle seed (build_bundle: SeedSequence(seed).spawn(7)[6], random_materials(8)).
2. Every stored solution of the chosen material is simulated again with 16 repeats, as the stored labels were, and compared with them.
3. The three NSGA-III runs (population 100, 25,000 simulations, seeds 1 to 3) are run again and their final fronts labeled with 16 repeats.

The simulator of v2026.1, which produced T6, had no seed, so step 3 can only reproduce the front statistically, not bit for bit.
Run with BMH_ML_STORE pointing to the store (read only)."""

import json
import sys
import time

import numpy as np

from bmh_ml.datasets.generators import random_materials
from bmh_ml.datasets.simulate import simulate
from bmh_ml.datasets.store import load_bundle, load_dataset, use_bundle_simulator
from bmh_ml.evaluation.chevron import chevron_deposition
from bmh_ml.evaluation.transfer import TransferConfig, get_hypervolume, get_nondominated, run_simulator_fronts_many
from bmh_ml.front_study.common import get_out

OUT = get_out()
N_JOBS = 14
REPEATS = 16


def main(material_index: int):
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle("S2-v1")
    settings = use_bundle_simulator(bundle)
    t6 = load_dataset(bundle.tests["T6"])
    args = bundle.notes["arguments"]

    # 1. materials from the bundle seed, in generation order
    rng = np.random.default_rng(np.random.SeedSequence(bundle.notes["seed"]).spawn(7)[6])
    materials = random_materials(int(args["material_fronts"]), rng)
    order = [int(np.flatnonzero(np.all(materials == row, axis=1))[0]) for row in t6.material]
    print(f"T6 materials regenerated from seed {bundle.notes['seed']}: {all(np.array_equal(materials[i], m) for i, m in zip(order, t6.material, strict=True))}")
    rows = np.flatnonzero(np.array(order) == material_index)
    material = materials[material_index]
    deposition, stored, stored_sd = t6.deposition[rows], t6.y[rows], t6.y_noise_sd[rows]
    print(f"material {material_index}: {len(rows)} stored final-front solutions, simulator {settings}")

    # 2. the stored solutions simulated again
    start = time.perf_counter()
    fresh, fresh_sd = simulate(material[None, :], deposition, repeats=REPEATS, n_jobs=N_JOBS)
    print(f"re-simulated {len(rows)} x {REPEATS} in {time.perf_counter() - start:.1f} s")
    se = np.sqrt(stored_sd**2 / REPEATS + fresh_sd**2 / REPEATS)
    z = (fresh - stored) / se
    front_stored, front_fresh = get_nondominated(stored), get_nondominated(fresh)
    reference = stored[front_stored]
    check = {
        "material_index": material_index,
        "solutions": len(rows),
        "mean_difference": (fresh - stored).mean(axis=0).tolist(),
        "mean_abs_difference": np.abs(fresh - stored).mean(axis=0).tolist(),
        "z_mean": z.mean(axis=0).tolist(),
        "z_sd": z.std(axis=0, ddof=1).tolist(),
        "share_abs_z_below_2": (np.abs(z) < 2).mean(axis=0).tolist(),
        "correlation": [float(np.corrcoef(fresh[:, k], stored[:, k])[0, 1]) for k in range(2)],
        "front_stored": front_stored.tolist(),
        "front_fresh": front_fresh.tolist(),
        "front_overlap": len(set(front_stored) & set(front_fresh)),
        "hv_stored_front": get_hypervolume(stored[front_stored], reference),
        "hv_fresh_labels_of_stored_front": get_hypervolume(fresh[front_stored], reference),
        "hv_fresh_front": get_hypervolume(fresh[front_fresh], reference),
    }

    # 3. the optimization itself run again
    config = TransferConfig(seeds=(1, 2, 3), population_size=int(args["front_population"]), evaluations=int(args["front_evaluations"]), n_jobs=N_JOBS)
    start = time.perf_counter()
    ((rerun_deposition, _, rerun_seed),) = run_simulator_fronts_many([material[None, :]], config)
    rerun_deposition, unique = np.unique(rerun_deposition, axis=0, return_index=True)
    rerun_seed = rerun_seed[unique]
    rerun, _ = simulate(material[None, :], rerun_deposition, repeats=REPEATS, n_jobs=N_JOBS)
    rerun_front = get_nondominated(rerun)
    check.update(
        {
            "rerun_seconds": time.perf_counter() - start,
            "rerun_solutions": len(rerun),
            "rerun_front_size": len(rerun_front),
            "hv_rerun_front": get_hypervolume(rerun[rerun_front], reference),
            "hv_rerun_per_seed": {int(s): get_hypervolume(rerun[rerun_seed == s], reference) for s in config.seeds},
            "rerun_f1_range": [float(rerun[rerun_front, 0].min()), float(rerun[rerun_front, 0].max())],
            "rerun_f2_range": [float(rerun[rerun_front, 1].min()), float(rerun[rerun_front, 1].max())],
        }
    )

    chevron, _ = simulate(material[None, :], chevron_deposition(), repeats=64, n_jobs=N_JOBS)
    check["chevron"] = chevron[0].tolist()
    print(json.dumps(check, indent=1))
    (OUT / f"verify-material-{material_index}.json").write_text(json.dumps(check, indent=1))
    np.savez(
        OUT / f"verify-material-{material_index}.npz",
        material=material, deposition=deposition, stored=stored, stored_sd=stored_sd, fresh=fresh, fresh_sd=fresh_sd,
        rerun_deposition=rerun_deposition, rerun=rerun, rerun_seed=rerun_seed, chevron=chevron[0],
    )  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]))
