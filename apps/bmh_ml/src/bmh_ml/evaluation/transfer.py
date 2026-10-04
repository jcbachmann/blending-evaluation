"""The transfer test: is a model useful for optimization? Optimize the model, simulate the solutions it finds, and compare them with the best
front known from optimizing the simulator itself.

The reference front is the non-dominated part of the test set T2 (the solutions of the optimization runs on the simulator, labeled with the
mean of repeated simulations). The solutions found on the model are simulated with as many repeats, so both are judged without the noise of
the simulator, which would otherwise favor whichever solutions were lucky once.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bmh_ml.datasets.manifest import Dataset, make_features
from bmh_ml.datasets.simulate import simulate
from bmh_ml.evaluation.chevron import get_chevron_metrics, get_chevron_objectives
from bmh_ml.evaluation.noise import OBJECTIVES
from bmh_ml.models.base import Model
from bmh_ml.settings import DEPOSITION_LENGTH, X_MAX, X_MIN

LOADED_MODELS: dict[Path, Model] = {}
REFERENCE_POINT = 1.1  # in objectives normalized to the reference front (0 its best, 1 its worst value), so its extremes count as well


@dataclass
class TransferConfig:
    """A fixed budget, so the results of different models are comparable: one optimization per seed, the fronts simulated `repeats` times."""

    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    population_size: int = 100
    evaluations: int = 20_000
    repeats: int = 16
    reference_set: str = "T2"
    n_jobs: int | None = None  # processes of the simulation and of the optimizations, default all cores

    def params(self) -> dict[str, str]:
        return {f"transfer.{key}": str(value) for key, value in self.__dict__.items() if key != "n_jobs"}


@dataclass
class TransferResult:
    metrics: dict[str, float]
    deposition: np.ndarray  # the solutions found on the model, one row each
    seed: np.ndarray  # the seed of the optimization that found each solution
    predicted: np.ndarray  # objectives the model predicted for them
    simulated: np.ndarray  # objectives of the simulator, mean of the repeats
    simulated_sd: np.ndarray | None
    reference: np.ndarray  # the reference front
    chevron: np.ndarray | None = None  # F1 and F2 of the 19-pass Chevron for the material, the denominators of the relative objectives
    material_index: np.ndarray | None = None  # with several materials: the material of each solution (index into `materials`)
    reference_index: np.ndarray | None = None  # with several materials: the material of each row of `reference`


def get_problem_class():
    from pymoo.core.problem import Problem

    class ModelProblem(Problem):
        """Optimizes the deposition for a fixed material with the objectives predicted by a model of any scope."""

        def __init__(self, model: Model, scope: str, material: np.ndarray):
            self.model, self.scope, self.material = model, scope, material
            super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

        def _evaluate(self, x, out, *_args, **_kwargs):
            out["F"] = self.model.predict(make_features(self.scope, self.material, x))

    return ModelProblem


def optimize_once(model: Model | Path, scope: str, material: np.ndarray, seed: int, population_size: int, evaluations: int) -> tuple[np.ndarray, np.ndarray]:
    """The final front of one NSGA-III run on the model (or the model stored in a directory): depositions and predicted objectives."""
    from bmh_ml.experiment import optimize
    from bmh_ml.models.registry import load_model

    if isinstance(model, Path):
        if model not in LOADED_MODELS:  # a worker process runs many optimizations of the same model, it loads the model once
            LOADED_MODELS[model] = load_model(model)
        model = LOADED_MODELS[model]
    problem = get_problem_class()(model, scope, np.atleast_2d(material))
    objectives, variables = optimize(problem, population_size, evaluations, seed)
    return np.atleast_2d(variables), np.atleast_2d(objectives)


def optimize_model(
    model: Model, scope: str, material: np.ndarray, seeds: tuple[int, ...], population_size: int, evaluations: int, workers: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The final fronts of one NSGA-III run per seed on the model: depositions, predicted objectives and the seed of each solution."""
    return optimize_model_many(model, scope, [material], seeds, population_size, evaluations, workers)[0]


def optimize_model_many(
    model: Model, scope: str, materials: list[np.ndarray], seeds: tuple[int, ...], population_size: int, evaluations: int, workers: int | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`optimize_model` for several materials: the runs of all materials and seeds are independent single-threaded jobs, so they share
    one set of parallel processes (default: all cores) instead of one material after the other. One result per material."""
    import tempfile

    from bmh_ml.parallel import get_cpu_count, run_parallel

    tasks = [(material, seed) for material in materials for seed in seeds]
    workers = min(workers or get_cpu_count(), len(tasks))
    with tempfile.TemporaryDirectory() as directory:
        source: Model | Path = model
        if workers > 1:
            source = Path(directory) / "model"
            model.save(source)
        fronts = run_parallel(optimize_once, [(source, scope, material, seed, population_size, evaluations) for material, seed in tasks], workers, threads=1)
    results = []
    for i in range(len(materials)):
        own = fronts[i * len(seeds) : (i + 1) * len(seeds)]
        run_seeds = np.concatenate([np.full(len(front[0]), seed) for front, seed in zip(own, seeds, strict=True)])
        results.append((np.vstack([front[0] for front in own]), np.vstack([front[1] for front in own]), run_seeds))
    return results


def get_simulator_problem_class():
    from pymoo.core.problem import Problem

    class SimulatorProblem(Problem):
        """Optimizes the deposition for a fixed material with the objectives of single simulations, as an optimizer without a model would."""

        def __init__(self, material: np.ndarray):
            self.material = material
            super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

        def _evaluate(self, x, out, *_args, **_kwargs):
            out["F"] = simulate(self.material, x, 1, n_jobs=1)[0]

    return SimulatorProblem


def optimize_simulator_once(material: np.ndarray, seed: int, population_size: int, evaluations: int) -> tuple[np.ndarray, np.ndarray]:
    """The final front of one NSGA-III run on the simulator: depositions and the objectives the optimizer saw (one simulation each)."""
    from bmh_ml.experiment import optimize

    objectives, variables = optimize(get_simulator_problem_class()(np.atleast_2d(material)), population_size, evaluations, seed)
    return np.atleast_2d(variables), np.atleast_2d(objectives)


def run_simulator_baseline(reference_dataset: Dataset, config: TransferConfig) -> TransferResult:
    """The transfer test's yardstick: NSGA-III directly on the simulator with `config.evaluations` simulations per run, for each material of
    the reference dataset, the final fronts simulated `config.repeats` times and scored with the same metrics. `predicted` holds the single
    noisy simulations the optimizer saw."""
    parts = split_by_material(reference_dataset)
    start = time.perf_counter()
    fronts = run_simulator_fronts_many([material for material, _ in parts], config)
    optimize_seconds = (time.perf_counter() - start) / len(parts)
    results = [score_found_solutions(material, part, front, config, optimize_seconds) for (material, part), front in zip(parts, fronts, strict=True)]
    return combine_results(results)


def run_simulator_fronts(material: np.ndarray, config: TransferConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-III on the simulator, one run per seed in parallel processes: depositions, the objectives the optimizer saw, the seed of each."""
    return run_simulator_fronts_many([material], config)[0]


def run_simulator_fronts_many(materials: list[np.ndarray], config: TransferConfig) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`run_simulator_fronts` for several materials, all runs in one set of parallel processes. One result per material."""
    from bmh_ml.parallel import get_cpu_count, run_parallel

    tasks = [(material, seed, config.population_size, config.evaluations) for material in materials for seed in config.seeds]
    fronts = run_parallel(optimize_simulator_once, tasks, min(config.n_jobs or get_cpu_count(), len(tasks)), threads=1)
    results = []
    for i in range(len(materials)):
        own = fronts[i * len(config.seeds) : (i + 1) * len(config.seeds)]
        seeds = np.concatenate([np.full(len(front[0]), seed) for front, seed in zip(own, config.seeds, strict=True)])
        results.append((np.vstack([front[0] for front in own]), np.vstack([front[1] for front in own]), seeds))
    return results


def get_nondominated(objectives: np.ndarray) -> np.ndarray:
    """The rows of the non-dominated solutions (both objectives are minimized)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    return np.sort(NonDominatedSorting().do(objectives, only_non_dominated_front=True))


def get_reference_front(dataset: Dataset) -> np.ndarray:
    return dataset.y[get_nondominated(dataset.y)]


def normalize(objectives: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ideal, nadir = reference.min(axis=0), reference.max(axis=0)
    return (objectives - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)


def get_hypervolume(objectives: np.ndarray, reference: np.ndarray) -> float:
    """Hypervolume of the solutions in objectives normalized to the reference front, solutions beyond the reference point add nothing."""
    from pymoo.indicators.hv import HV

    points = normalize(objectives, reference)
    points = points[np.all(points < REFERENCE_POINT, axis=1)]
    if len(points) == 0:
        return 0.0
    return float(HV(ref_point=np.full(2, REFERENCE_POINT))(points))


def get_front_metrics(objectives: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """`hv_ratio`: hypervolume of the solutions divided by that of the reference front, 1 is as good, above 1 better.
    `igd_plus`: distance of the reference front to the solutions (normalized objectives, 0 means the reference front is reached)."""
    from pymoo.indicators.igd_plus import IGDPlus

    return {
        "hv_ratio": get_hypervolume(objectives, reference) / get_hypervolume(reference, reference),
        "igd_plus": float(IGDPlus(normalize(reference, reference))(normalize(objectives, reference))),
    }


def split_by_material(dataset: Dataset) -> list[tuple[np.ndarray, Dataset]]:
    """The rows of each material of a dataset, in the order the materials first appear."""
    materials, first, inverse = np.unique(dataset.full_material(), axis=0, return_index=True, return_inverse=True)
    parts = []
    for index in np.argsort(first):
        rows = np.flatnonzero(inverse.reshape(-1) == index)
        part = Dataset(dataset.name, dataset.generator, dataset.seed, dataset.repeats, materials[index][None, :], dataset.deposition[rows], dataset.y[rows])
        parts.append((materials[index][None, :], part))
    return parts


def run_transfer_test(model: Model, scope: str, reference_dataset: Dataset, config: TransferConfig) -> TransferResult:
    """Optimizes the model for each material of the reference dataset, simulates the solutions found and compares them with the reference
    front of that material (its non-dominated rows) and with Chevron for that material. With several materials the metrics are means over
    the materials, and `hv_ratio_material_min` is the worst material."""
    parts = split_by_material(reference_dataset)
    start = time.perf_counter()
    fronts = optimize_model_many(model, scope, [material for material, _ in parts], config.seeds, config.population_size, config.evaluations, config.n_jobs)
    optimize_seconds = (time.perf_counter() - start) / len(parts)
    results = [score_found_solutions(material, part, front, config, optimize_seconds) for (material, part), front in zip(parts, fronts, strict=True)]
    return combine_results(results)


def score_found_solutions(
    material: np.ndarray, reference_dataset: Dataset, front: tuple[np.ndarray, np.ndarray, np.ndarray], config: TransferConfig, optimize_seconds: float
) -> TransferResult:
    """Simulates the solutions an optimizer found for one material and scores them against its reference front and its Chevron."""
    deposition, predicted, seeds = front
    reference = get_reference_front(reference_dataset)
    start = time.perf_counter()
    simulated, simulated_sd = simulate(material, deposition, config.repeats, config.n_jobs)
    chevron = get_chevron_objectives(material, n_jobs=config.n_jobs)
    metrics = get_transfer_metrics(seeds, predicted, simulated, reference, chevron)
    metrics.update({"optimize_seconds": optimize_seconds, "simulate_seconds": time.perf_counter() - start})
    return TransferResult(metrics, deposition, seeds, predicted, simulated, simulated_sd, reference, chevron)


def combine_results(results: list[TransferResult]) -> TransferResult:
    """One result for several materials: the metrics averaged (times summed), the solutions stacked with the material of each."""
    if len(results) == 1:
        return results[0]
    metrics = {}
    for name in results[0].metrics:
        values = [result.metrics[name] for result in results]
        metrics[name] = float(np.sum(values)) if name.endswith("_seconds") or name == "solutions" else float(np.mean(values))
    metrics["materials"] = float(len(results))
    metrics["hv_ratio_material_min"] = float(min(result.metrics["hv_ratio"] for result in results))
    metrics["chevron_beaten_rate_material_min"] = float(min(result.metrics["chevron_beaten_rate"] for result in results))
    metrics["chevron_hv_ratio_material_min"] = float(min(result.metrics["chevron_hv_ratio"] for result in results))
    stack = lambda name: np.concatenate([getattr(result, name) for result in results])  # noqa: E731
    simulated_sd = stack("simulated_sd") if all(result.simulated_sd is not None for result in results) else None
    return TransferResult(
        metrics,
        stack("deposition"),
        stack("seed"),
        stack("predicted"),
        stack("simulated"),
        simulated_sd,
        stack("reference"),
        np.vstack([result.chevron for result in results]),
        material_index=np.concatenate([np.full(len(result.deposition), i) for i, result in enumerate(results)]),
        reference_index=np.concatenate([np.full(len(result.reference), i) for i, result in enumerate(results)]),
    )


def get_transfer_metrics(seeds: np.ndarray, predicted: np.ndarray, simulated: np.ndarray, reference: np.ndarray, chevron: np.ndarray) -> dict[str, float]:
    """The metrics of the solutions found on a model, from their predicted and simulated objectives and the seed of the run that found them.

    Against the reference front (normalized to it): `hv_ratio`, `igd_plus`, per run `hv_ratio_run_mean` and `hv_ratio_run_min`, and
    `predicted_hv_ratio`, what the model promised. Against Chevron (objectives divided by Chevron's, reference point (1, 1)): `chevron_hv`,
    `chevron_beaten_rate` (share of the solutions better than Chevron in both), `chevron_best_F1` and `chevron_best_F2`, per run
    `chevron_hv_run_mean`, the same for the reference front (`reference_chevron_hv`) and for the predictions (`predicted_chevron_hv`).
    `chevron_hv_ratio` (pooled, per run `chevron_hv_ratio_run_mean` and `_run_min`): the hypervolume beyond Chevron divided by the reference
    front's, the share of the known improvement over Chevron that the solutions reach. Unlike `hv_ratio` it does not depend on the span of
    the reference front: normalized to a front that spans only 0.02 in F1, a solution 0.003 worse than its worst F1 already counts nothing
    (measured 2026-10-04, PLAN.md section 5).
    """
    run_seeds = np.unique(seeds)
    metrics = get_front_metrics(simulated, reference)
    reference_chevron_hv = get_chevron_metrics(reference, chevron)["chevron_hv"]
    ratio = lambda objectives: get_chevron_metrics(objectives, chevron)["chevron_hv"] / reference_chevron_hv if reference_chevron_hv > 0 else np.nan  # noqa: E731
    per_run_ratio = [ratio(simulated[seeds == seed]) for seed in run_seeds]
    per_run = [get_front_metrics(simulated[seeds == seed], reference)["hv_ratio"] for seed in run_seeds]
    metrics.update(
        {
            "hv_ratio_run_mean": float(np.mean(per_run)),
            "hv_ratio_run_min": float(np.min(per_run)),
            "predicted_hv_ratio": get_front_metrics(predicted, reference)["hv_ratio"],
            "negative_rate": float(np.mean(np.any(predicted < 0, axis=1))),
            "solutions": float(len(simulated)),
            **get_chevron_metrics(simulated, chevron),
            "chevron_hv_run_mean": float(np.mean([get_chevron_metrics(simulated[seeds == seed], chevron)["chevron_hv"] for seed in run_seeds])),
            "reference_chevron_hv": reference_chevron_hv,
            "chevron_hv_ratio": ratio(simulated),
            "chevron_hv_ratio_run_mean": float(np.mean(per_run_ratio)),
            "chevron_hv_ratio_run_min": float(np.min(per_run_ratio)),
            "predicted_chevron_hv": get_chevron_metrics(predicted, chevron)["chevron_hv"],
            "chevron_F1": float(chevron[0]),
            "chevron_F2": float(chevron[1]),
        }
    )
    for i, objective in enumerate(OBJECTIVES):
        error = predicted[:, i] - simulated[:, i]
        metrics[f"{objective}/bias"] = float(np.mean(error))  # negative: the model promises better values than the simulator gives
        metrics[f"{objective}/rmse"] = float(np.sqrt(np.mean(error**2)))
    return metrics


def get_simulator_run_metrics(result_files: list[Path], reference_dataset: Dataset, chevron: np.ndarray) -> dict[str, np.ndarray]:
    """The metrics of each optimization run on the simulator, the yardstick for the per-run metrics of the transfer test: `hv_ratio`
    against the reference front, and against Chevron `chevron_hv`, `chevron_beaten_rate`, `chevron_best_F1`, `chevron_best_F2`.

    Its solutions are judged by their labels in the reference dataset (which contains the solutions of these runs), not by the single
    simulation the run saw, so the metrics are free of the luck of the noise.
    """
    labels = {deposition.tobytes(): y for deposition, y in zip(reference_dataset.deposition, reference_dataset.y, strict=True)}
    reference = get_reference_front(reference_dataset)
    rows = []
    for file in result_files:
        variables = np.array(json.loads(file.read_text())["variables"], dtype=float)[:, -DEPOSITION_LENGTH:]
        missing = [row for row in variables if row.tobytes() not in labels]
        if missing:
            raise ValueError(f"{len(missing)} solutions of {file.name} are not in the reference dataset {reference_dataset.name}")
        objectives = np.array([labels[row.tobytes()] for row in variables])
        rows.append({"hv_ratio": get_front_metrics(objectives, reference)["hv_ratio"], **get_chevron_metrics(objectives, chevron)})
    return {name: np.array([row[name] for row in rows]) for name in rows[0]}


def plot_transfer(result: TransferResult):
    """The solutions found on the model, as the model predicted them and as the simulator evaluates them, with the reference front."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.material_index is not None:  # several materials: the first one
        first = result.material_index == 0
        result = TransferResult(
            result.metrics,
            result.deposition[first],
            result.seed[first],
            result.predicted[first],
            result.simulated[first],
            None,
            result.reference[result.reference_index == 0],
            result.chevron[0],
        )
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    order = np.argsort(result.reference[:, 0])
    axis.plot(result.reference[order, 0], result.reference[order, 1], color="black", marker="o", markersize=3, label="reference front (simulator)")
    axis.scatter(result.predicted[:, 0], result.predicted[:, 1], s=6, alpha=0.4, label="found on the model, predicted")
    axis.scatter(result.simulated[:, 0], result.simulated[:, 1], s=6, alpha=0.6, label="found on the model, simulated")
    if result.chevron is not None:
        axis.scatter(*result.chevron, marker="*", s=200, color="red", zorder=5, label="Chevron, 19 passes")
        axis.axvline(result.chevron[0], color="red", linewidth=0.5, linestyle="--")
        axis.axhline(result.chevron[1], color="red", linewidth=0.5, linestyle="--")
    metrics = result.metrics
    axis.set(
        xlabel="F1",
        ylabel="F2",
        title=f"Transfer: hypervolume ratio {metrics['hv_ratio']:.3f} (per run {metrics['hv_ratio_run_mean']:.3f}), "
        f"{100 * metrics.get('chevron_beaten_rate', float('nan')):.0f} % better than Chevron in both",
    )
    axis.legend()
    return figure


def log_transfer(run_id: str, result: TransferResult, config: TransferConfig, with_plots: bool = True) -> None:
    """Logs the metrics (`transfer/...`), the settings and the solutions of a transfer test to a run, also to a finished one: logging by
    run id does not reopen the run, which would change its end time."""
    import tempfile

    from bmh_ml.evaluation.plots import close_figure
    from bmh_ml.tracking.runs import configure_mlflow, log_metrics_to_run

    client = configure_mlflow().MlflowClient()
    for key, value in config.params().items():
        client.log_param(run_id, key, value)
    log_metrics_to_run(run_id, {f"transfer/{name}": value for name, value in result.metrics.items()})
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "solutions.npz"
        arrays = {"deposition": result.deposition, "seed": result.seed, "predicted": result.predicted, "simulated": result.simulated}
        if result.simulated_sd is not None:
            arrays["simulated_sd"] = result.simulated_sd
        if result.chevron is not None:
            arrays["chevron"] = result.chevron
        if result.material_index is not None:
            arrays["material_index"], arrays["reference_index"] = result.material_index, result.reference_index
        np.savez_compressed(file, reference=result.reference, **arrays)
        client.log_artifact(run_id, str(file), "transfer")
    if with_plots:
        figure = plot_transfer(result)
        client.log_figure(run_id, figure, "transfer/front.png")
        close_figure(figure)
