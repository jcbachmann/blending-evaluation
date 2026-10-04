"""How results at one detail level of the simulator carry over to another, without any model (phase 0 of the evaluation plan).

The baseline every model has to beat: if solutions optimized on the cheap simulator are already good on the detailed one, optimizing
on the cheap simulator is itself a fast stand-in for the detailed one. Two bundles labeled at two levels with the same seed have the same
random inputs, so their test sets compare directly. Their fronts optimized for the same materials (T6) are evaluated at both levels: at
each level, the front optimized at the other level is compared with the front optimized at that level, per material. The test sets `T6x`
of a bundle hold the T6 inputs of the other bundle, labeled at this bundle's level (`build_bundle --cross-from`).
"""

import argparse
import logging

import numpy as np

from bmh_ml.datasets.manifest import Dataset
from bmh_ml.datasets.store import Bundle, get_bundle_simulator, load_bundle, load_dataset, use_bundle_simulator
from bmh_ml.evaluation.chevron import get_chevron_hypervolume, get_chevron_objectives
from bmh_ml.evaluation.noise import OBJECTIVES
from bmh_ml.evaluation.transfer import get_hypervolume, get_nondominated, get_reference_front, split_by_material
from bmh_ml.tracking.runs import configure_mlflow, get_experiment_id, get_experiment_name, get_finite_metrics

RANDOM_SETS = ("T1", "T3", "T5")
TAIL = 0.1  # the best tenth of the solutions, where an optimizer works


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def get_paired_metrics(low: Dataset, high: Dataset) -> dict[str, float]:
    """Agreement of the same inputs labeled at two levels: rank correlation (all and in the best tenth at the low level), mean shift and
    rmse of the high level against the low one, the rmse relative to the spread of the objective."""
    if not (np.array_equal(low.deposition, high.deposition) and np.array_equal(low.full_material(), high.full_material())):
        raise ValueError(f"{low.name} and {high.name} do not have the same inputs; build both bundles with the same seed and sizes")
    metrics = {}
    for i, objective in enumerate(OBJECTIVES):
        a, b = low.y[:, i], high.y[:, i]
        tail = a <= np.quantile(a, TAIL)
        metrics |= {
            f"{objective}/spearman": spearman(a, b),
            f"{objective}/tail_spearman": spearman(a[tail], b[tail]),
            f"{objective}/shift": float(np.mean(b - a)),
            f"{objective}/rmse": float(np.sqrt(np.mean((b - a) ** 2))),
            f"{objective}/relative_rmse": float(np.sqrt(np.mean((b - a) ** 2)) / a.std()),
        }
    return metrics


def get_front_comparison(own: Dataset, other: Dataset, chevron: np.ndarray) -> dict[str, float]:
    """At one level, for one material: the front optimized at this level (`own`) against the solutions optimized at the other level
    (`other`), both labeled at this level. `hv_ratio`: hypervolume of the other level's solutions in objectives normalized to the own front,
    divided by the own front's; `chevron_hv`: hypervolume beyond Chevron at this level, for both; `nondominated_share`: share of the
    other level's solutions that no solution of the own front dominates."""
    reference = get_reference_front(own)
    union = np.vstack([own.y, other.y])
    nondominated = set(get_nondominated(union).tolist())
    return {
        "hv_ratio": get_hypervolume(other.y, reference) / get_hypervolume(reference, reference),
        "chevron_hv_own": get_chevron_hypervolume(own.y / chevron),
        "chevron_hv_other": get_chevron_hypervolume(other.y / chevron),
        "nondominated_share": float(np.mean([len(own.y) + i in nondominated for i in range(len(other.y))])),
    }


def compare_fronts(own_bundle: Bundle, other_set: str = "T6x") -> dict[str, float]:
    """`get_front_comparison` for every material of the own bundle's T6, at the own bundle's level; means and the worst material."""
    use_bundle_simulator(own_bundle)  # Chevron at this level
    own_parts = split_by_material(load_dataset(own_bundle.tests["T6"]))
    other_parts = {material.tobytes(): part for material, part in split_by_material(load_dataset(own_bundle.tests[other_set]))}
    rows = []
    for material, own in own_parts:
        other = other_parts[material.tobytes()]
        rows.append(get_front_comparison(own, other, get_chevron_objectives(material)))
    metrics = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
    metrics["hv_ratio_material_min"] = float(min(row["hv_ratio"] for row in rows))
    metrics["materials"] = float(len(rows))
    return metrics


def get_fidelity_metrics(low: Bundle, high: Bundle, low_cross: Bundle) -> dict[str, float]:
    """All comparisons between two levels: the random test sets, and the fronts at each level."""
    metrics = {}
    for name in RANDOM_SETS:
        if name in low.tests and name in high.tests:
            pair = get_paired_metrics(load_dataset(low.tests[name]), load_dataset(high.tests[name]))
            metrics |= {f"{name}/{key}": value for key, value in pair.items()}
    pair = get_paired_metrics(load_dataset(low.tests["T6"]), load_dataset(high.tests["T6x"]))  # the low level's optimized solutions
    metrics |= {f"T6low/{key}": value for key, value in pair.items()}
    pair = get_paired_metrics(load_dataset(low_cross.tests["T6x"]), load_dataset(high.tests["T6"]))  # the high level's optimized solutions
    metrics |= {f"T6high/{key}": value for key, value in pair.items()}
    metrics |= {f"at_high/{key}": value for key, value in compare_fronts(high).items()}  # low-optimized solutions at the high level
    metrics |= {f"at_low/{key}": value for key, value in compare_fronts(low_cross).items()}  # high-optimized solutions at the low level
    return metrics


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compares two detail levels of the simulator: random inputs and optimized fronts at both levels")
    parser.add_argument("--low", required=True, help="Bundle at the low level, with T6 (fronts optimized at the low level) and the random test sets")
    parser.add_argument("--high", required=True, help="Bundle at the high level, same seed, with T6 and T6x (the low bundle's T6 labeled at the high level)")
    parser.add_argument("--low-cross", required=True, help="Bundle at the low level whose T6x is the high bundle's T6 labeled at the low level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO)
    args = get_args(argv)
    low, high, low_cross = (load_bundle(name) for name in (args.low, args.high, args.low_cross))
    if get_bundle_simulator(low) != get_bundle_simulator(low_cross):
        raise ValueError(f"{low.name} and {low_cross.name} must be labeled at the same level")
    metrics = get_fidelity_metrics(low, high, low_cross)
    for name, value in sorted(metrics.items()):
        print(f"{name:<36}{value:>10.4f}")
    mlflow = configure_mlflow()
    levels = {"low_ppm3": get_bundle_simulator(low).ppm3, "high_ppm3": get_bundle_simulator(high).ppm3}
    with mlflow.start_run(experiment_id=get_experiment_id(get_experiment_name(low.scope)), run_name=f"fidelity-{low.name}-{high.name}") as run:
        mlflow.log_params({"low": low.name, "high": high.name, "low_cross": low_cross.name, **levels})
        mlflow.set_tags({"kind": "fidelity", "purpose": "Phase 0: how results of one detail level carry over to another, without a model"})
        mlflow.log_metrics(get_finite_metrics({f"fidelity/{name}": value for name, value in metrics.items()}))
    print(f"Run {run.info.run_id}")


if __name__ == "__main__":
    main()
