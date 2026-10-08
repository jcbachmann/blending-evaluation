"""Pre-screening the offspring of NSGA-III at the expensive level with a model (phase 2, option A of the evaluation plan).

The refinement at the expensive level starts from the fronts of the cheap level and simulates every offspring. With a model, only the
offspring it rates best would be simulated. Whether that can pay is checked first (`python -m bmh_ml.prescreen`): the refinement runs as
it is, every generation's parents and offspring are recorded with the objectives the optimizer saw, and afterwards each model ranks the
offspring of every generation. An offspring "joins the front" if it is non-dominated among the parents and the offspring of its
generation (by the simulated objectives). If a model picks a share s of the offspring, the recall is the part of the joining offspring
among its picks; picking at random gives s. A recall well above s means that simulating only the picks keeps most of the progress per
generation at a fraction s of the simulations (gain recall / s).
"""

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymoo.core.callback import Callback

SHARES = (0.2, 0.3, 0.5)


@dataclass
class Generation:
    """One generation of NSGA-III: the population it started from and the offspring it made, with the objectives the optimizer saw."""

    parents_x: np.ndarray
    parents_f: np.ndarray
    offspring_x: np.ndarray
    offspring_f: np.ndarray


class GenerationRecorder(Callback):
    """Records every generation after the first: the population before it and its evaluated offspring."""

    def __init__(self):
        super().__init__()
        self.generations: list[Generation] = []
        self.previous: tuple[np.ndarray, np.ndarray] | None = None

    def notify(self, algorithm):
        if self.previous is not None:
            self.generations.append(Generation(*self.previous, algorithm.off.get("X").copy(), algorithm.off.get("F").copy()))
        self.previous = (algorithm.pop.get("X").copy(), algorithm.pop.get("F").copy())


def record_refinement(material: np.ndarray, initial: np.ndarray | None, population_size: int, evaluations: int, seed: int) -> list[Generation]:
    """NSGA-III on the simulator at the active level for one material, started from `initial`; its generations."""
    from bmh_ml.evaluation.transfer import get_simulator_problem_class
    from bmh_ml.experiment import optimize

    recorder = GenerationRecorder()
    optimize(get_simulator_problem_class()(np.atleast_2d(material)), population_size, evaluations, seed, initial, recorder)
    return recorder.generations


def joins_front(parents_f: np.ndarray, offspring_f: np.ndarray) -> np.ndarray:
    """For each offspring: is it non-dominated among the parents and all offspring?"""
    from bmh_ml.evaluation.transfer import get_nondominated

    nondominated = get_nondominated(np.vstack([parents_f, offspring_f]))
    mask = np.zeros(len(offspring_f), dtype=bool)
    mask[nondominated[nondominated >= len(parents_f)] - len(parents_f)] = True
    return mask


def preselect(parents_f: np.ndarray, predicted_f: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """The `count` offspring a model would have simulated: those with the best non-domination rank of their predictions among the parents
    (simulated) and all offspring (predicted), ties broken at random."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    _, rank = NonDominatedSorting().do(np.vstack([parents_f, predicted_f]), return_rank=True)
    offspring_rank = rank[len(parents_f) :]
    return np.lexsort((rng.random(len(predicted_f)), offspring_rank))[: max(1, count)]


def get_check_metrics(generations: list[Generation], predicted: list[np.ndarray], seed: int = 0) -> dict[str, float]:
    """Recall of the front-joining offspring for each share (pooled over the generations), its gain over random picks, the share of
    offspring that join the front, and the rank correlation of predicted and simulated objectives among the offspring of a generation."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    hits, joining, correlations = dict.fromkeys(SHARES, 0), 0, {0: [], 1: []}
    offspring = 0
    for generation, prediction in zip(generations, predicted, strict=True):
        truth = joins_front(generation.parents_f, generation.offspring_f)
        joining += int(truth.sum())
        offspring += len(truth)
        for share in SHARES:
            hits[share] += int(truth[preselect(generation.parents_f, prediction, round(share * len(prediction)), rng)].sum())
        for objective in (0, 1):
            correlations[objective].append(spearmanr(prediction[:, objective], generation.offspring_f[:, objective]).statistic)
    metrics = {"joining_share": joining / offspring, "generations": float(len(generations))}
    for share in SHARES:
        recall = hits[share] / joining if joining else float("nan")
        metrics[f"recall_at_{round(100 * share)}"] = recall
        metrics[f"gain_at_{round(100 * share)}"] = recall / share
    for objective, name in ((0, "F1"), (1, "F2")):
        metrics[f"{name}/offspring_spearman"] = float(np.nanmean(correlations[objective]))
    return metrics


_MODELS: dict[str, object] = {}  # models loaded by this process, by directory


@dataclass
class Prescreen:
    """Pre-screening for NSGA-III on the simulator: every generation makes `candidates` offspring, the models rate them and only the
    `simulated` best are simulated. F1 comes from the model in `f1_model`, F2 from the one in `f2_model` (directories of saved models,
    possibly the same)."""

    f1_model: str
    f2_model: str
    scope: str
    candidates: int
    simulated: int

    def predictor(self, material: np.ndarray):
        """A function from depositions to the predicted objectives for this material."""
        from bmh_ml.datasets.manifest import make_features
        from bmh_ml.models.registry import load_model

        for directory in {self.f1_model, self.f2_model}:
            if directory not in _MODELS:
                _MODELS[directory] = load_model(Path(directory))
        f1_model, f2_model = _MODELS[self.f1_model], _MODELS[self.f2_model]

        def predict(depositions: np.ndarray) -> np.ndarray:
            x = make_features(self.scope, np.repeat(np.atleast_2d(material), len(depositions), axis=0), depositions)
            if f1_model is f2_model:
                return f1_model.predict(x)
            return np.column_stack([f1_model.predict(x)[:, 0], f2_model.predict(x)[:, 1]])

        return predict


def optimize_prescreened(problem, prescreen: Prescreen, population_size: int, evaluations: int, seed: int, initial: np.ndarray | None = None):
    """NSGA-III like `experiment.optimize` with `evaluations` simulations in all: the first population is simulated completely, after that
    each generation simulates only the offspring the models rate best (`Prescreen`). Returns the objectives and depositions of the final
    non-dominated solutions."""
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.termination import NoTermination
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.util.reference_direction import UniformReferenceDirectionFactory

    from bmh_ml.experiment import get_initial_population

    if evaluations < population_size:
        raise ValueError(f"{evaluations} evaluations are not enough for the first population of {population_size}")
    sampling = FloatRandomSampling() if initial is None else get_initial_population(initial, population_size, problem, seed)
    algorithm = NSGA3(
        pop_size=population_size,
        n_offsprings=prescreen.candidates,
        ref_dirs=UniformReferenceDirectionFactory(2, n_points=population_size).do(),
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
    predict = prescreen.predictor(problem.material)
    rng = np.random.default_rng(seed)
    used = 0
    while used < evaluations:
        infills = algorithm.ask()
        if algorithm.is_initialized:
            count = min(prescreen.simulated, evaluations - used)
            infills = infills[preselect(algorithm.pop.get("F"), predict(infills.get("X")), count, rng)]
        algorithm.evaluator.eval(problem, infills)
        algorithm.tell(infills=infills)
        used += len(infills)
    result = algorithm.result()
    return np.atleast_2d(result.F), np.atleast_2d(result.X)


def predict_generations(model, scope: str, material: np.ndarray, generations: list[Generation]) -> list[np.ndarray]:
    """The model's objectives for the offspring of every generation of one material, in one batch."""
    from bmh_ml.datasets.manifest import make_features

    offspring = np.vstack([generation.offspring_x for generation in generations])
    predicted = model.predict(make_features(scope, np.repeat(np.atleast_2d(material), len(offspring), axis=0), offspring))
    return np.split(predicted, np.cumsum([len(generation.offspring_x) for generation in generations])[:-1])


def save_generations(path: Path, runs: list[list[Generation]]) -> None:
    """All recorded generations of all materials in one file (material, generation, parents and offspring with their objectives)."""
    arrays: dict[str, list] = {"material_index": [], "generation": [], "role": [], "x": [], "f": []}
    for material_index, generations in enumerate(runs):
        for number, generation in enumerate(generations):
            for role, x, f in ((0, generation.parents_x, generation.parents_f), (1, generation.offspring_x, generation.offspring_f)):
                arrays["material_index"].append(np.full(len(x), material_index))
                arrays["generation"].append(np.full(len(x), number))
                arrays["role"].append(np.full(len(x), role))  # 0 parents, 1 offspring
                arrays["x"].append(x)
                arrays["f"].append(f)
    np.savez_compressed(path, **{name: np.concatenate(value) for name, value in arrays.items()})


def run_check(run_ids: list[str], references: list[str], seed_population: list[str], evaluations: int, seed: int, population_size: int = 100) -> str:
    """The feasibility check on the materials of the references, at their level; logged as one run with the metrics of every model."""
    from bmh_ml.evaluate_run import load_run_model
    from bmh_ml.evaluation.transfer import split_by_material
    from bmh_ml.parallel import get_cpu_count, run_parallel
    from bmh_ml.tracking.environment import get_code_version
    from bmh_ml.tracking.runs import configure_mlflow, get_experiment_id, get_experiment_name, get_finite_metrics
    from bmh_ml.transfer import load_initial_populations, load_references

    reference, scope = load_references(references)
    initial = load_initial_populations(seed_population) if seed_population else {}
    materials = [material for material, _ in split_by_material(reference)]
    tasks = [(material, initial.get(material[0].tobytes()), population_size, evaluations, seed) for material in materials]
    logging.info(f"Recording the refinement of {len(materials)} materials, {evaluations} simulations each")
    runs = run_parallel(record_refinement, tasks, min(get_cpu_count(), len(tasks)), threads=1)

    mlflow = configure_mlflow()
    metrics, names = {}, []
    for run_id in run_ids:
        name = mlflow.get_run(run_id).info.run_name
        names.append(name)
        model = load_run_model(run_id)
        generations = [g for generations in runs for g in generations]
        predicted = [p for material, generations in zip(materials, runs, strict=True) for p in predict_generations(model, scope, material, generations)]
        model_metrics = get_check_metrics(generations, predicted, seed)
        logging.info(f"{name}: {model_metrics}")
        metrics |= {f"prescreen_{name}/{key}": value for key, value in model_metrics.items()}
    run_name = f"prescreen-check-{evaluations}" + ("-seeded" if seed_population else "")
    with mlflow.start_run(experiment_id=get_experiment_id(get_experiment_name(scope)), run_name=run_name) as run:
        mlflow.log_params(
            {
                "references": " ".join(references),
                "seed_population": " ".join(seed_population),
                "models": " ".join(names),
                "model_runs": " ".join(run_ids),
                "evaluations": evaluations,
                "seed": seed,
                "population_size": population_size,
            }
        )
        purpose = "Phase 2 option A: can a model pick the offspring that join the front?"
        mlflow.set_tags({"kind": "prescreen-check", "code_version": get_code_version(), "purpose": purpose})
        mlflow.log_metrics(get_finite_metrics(metrics))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generations.npz"
            save_generations(path, runs)
            mlflow.log_artifact(str(path))
    return run.info.run_id


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Can a model pick the offspring of NSGA-III at the expensive level that join the front?")
    parser.add_argument("--run-id", nargs="+", required=True, help="Training runs whose models rank the offspring")
    parser.add_argument("--references", nargs="+", required=True, metavar="BUNDLE:SET", help="The materials (and level) of the refinement")
    parser.add_argument("--seed-population", nargs="*", default=[], metavar="SPEC", help="Starting populations, as for transfer --seed-population")
    parser.add_argument("--evaluations", type=int, default=2000, help="Simulations per material")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO)
    args = get_args(argv)
    run_id = run_check(args.run_id, args.references, args.seed_population, args.evaluations, args.seed)
    print(f"Run {run_id}")


if __name__ == "__main__":
    main()
