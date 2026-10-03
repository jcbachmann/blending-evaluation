"""The refinement loop: training data in the region the optimizer works in (the operating region).

A model trained on random depositions is wrong exactly where the optimizer goes, because the good solutions are far from random ones. Each
round optimizes the current model, simulates the solutions it finds and perturbations around them, adds them to the training data and trains
again. The rounds are nested runs of one parent run, whose metrics show the effect of each round (step = round).

An interrupted loop continues with `--resume` (same arguments): the finished rounds are kept, the round that was interrupted starts again.
The rounds after the resume draw from new random streams, so the result is a valid loop but not identical to an uninterrupted one.
"""

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

from bmh_ml.datasets.generators import random_materials
from bmh_ml.datasets.manifest import SCOPE_FIXED_MATERIAL
from bmh_ml.datasets.simulate import build_dataset
from bmh_ml.datasets.store import Bundle, bundle_exists, load_bundle, load_dataset, load_manifest, save_bundle, save_dataset
from bmh_ml.evaluate_run import evaluate_run, load_run_model
from bmh_ml.evaluation.noise import OBJECTIVES
from bmh_ml.evaluation.transfer import TransferConfig, optimize_model_many
from bmh_ml.models.registry import MODELS
from bmh_ml.settings import DEPOSITION_LENGTH, X_MAX, X_MIN
from bmh_ml.tracking.annotate import annotate_run
from bmh_ml.tracking.runs import configure_mlflow, get_experiment_id, get_experiment_name, get_finite_metrics
from bmh_ml.train import parse_parameters, run_training

VALOP = "valop"
SEED_STRIDE = 1000  # optimization seeds of round k start at k * SEED_STRIDE, the transfer test uses small seeds
VALOP_SEED_OFFSET = 500
SUMMARY_METRICS = ("nrmse", "r2", "negative_rate")


@dataclass
class RefineConfig:
    base: str
    name: str
    model: str
    params: dict
    rounds: int = 4
    optimizations: int = 10
    population_size: int = 100
    evaluations: int = 20_000
    perturbations: int = 4
    perturbation_sd: float = 2.0
    valop_optimizations: int = 4
    valop_repeats: int = 8
    seed: int = 1
    start_run: str | None = None
    control: bool = False
    transfer: TransferConfig | None = None
    reference_set: str = "T2"
    n_jobs: int | None = None
    materials_per_round: int = 0  # 0: the fixed material of the reference set; else new random materials each round (scope S2)
    resume: bool = False  # continue an interrupted loop of the same name after its last finished round

    def params_to_log(self) -> dict:
        skip = {"params", "transfer", "n_jobs", "resume"}
        return {f"refine.{key}": value for key, value in self.__dict__.items() if key not in skip and value is not None} | self.params


def perturb(deposition: np.ndarray, count: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    """`count` copies of every deposition with normal noise on each position, kept within the bounds."""
    copies = np.repeat(deposition, count, axis=0)
    return np.clip(copies + rng.normal(0.0, sd, copies.shape), X_MIN, X_MAX)


def get_round_seeds(round_number: int, count: int, offset: int = 0) -> tuple[int, ...]:
    start = round_number * SEED_STRIDE + offset
    return tuple(range(start, start + count))


def get_fixed_material(bundle: Bundle, reference_set: str) -> np.ndarray:
    """The material the loop optimizes for: that of the reference set, where the operating region is measured."""
    if reference_set in bundle.tests:
        return load_dataset(bundle.tests[reference_set]).material[:1]
    if bundle.scope == SCOPE_FIXED_MATERIAL:
        return load_dataset(bundle.train).material[:1]
    raise ValueError(f"Bundle {bundle.name} has no set {reference_set} that defines the material")


def get_round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """The metrics of a round that the parent run shows as curves over the rounds."""
    return {name: value for name, value in metrics.items() if name.startswith("transfer/") or name.split("/")[-1] in SUMMARY_METRICS}


def get_front_errors(predicted: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    """How wrong the model was about the solutions it found, before they were added: the error the refinement corrects."""
    metrics = {"refine/front_negative_rate": float(np.mean(np.any(predicted < 0, axis=1)))}
    for i, objective in enumerate(OBJECTIVES):
        error = predicted[:, i] - simulated[:, i]
        metrics[f"refine/{objective}/front_bias"] = float(np.mean(error))
        metrics[f"refine/{objective}/front_rmse"] = float(np.sqrt(np.mean(error**2)))
    return metrics


def train_round(config: RefineConfig, bundle: str, run_name: str, round_number: int, source_run: str | None = None) -> tuple[str, dict[str, float]]:
    result = run_training(
        bundle,
        config.model,
        config.params,
        config.seed,
        run_name,
        with_plots=True,
        nested=True,
        tags={"refine": config.name, "round": str(round_number), **({"source_model_run": source_run} if source_run else {})},
        transfer=config.transfer,
    )
    return result.run_id, result.metrics


def get_round_bundle_name(config: RefineConfig, round_number: int) -> str:
    return config.name if round_number == config.rounds else f"{config.name}-r{round_number}"


@dataclass
class ResumeState:
    parent_run: str
    run_ids: list[str]  # the training runs of the finished rounds, round 0 first
    added: list[str]  # the datasets the finished rounds added
    pending_bundle: bool  # the interrupted round had saved its bundle already, only its training is missing


def get_resume_state(config: RefineConfig, base: Bundle) -> ResumeState:
    """What an interrupted loop of the same name left: its parent run, the finished rounds and their data. The runs it left open (the
    parent and the interrupted round) are closed as KILLED."""
    mlflow = configure_mlflow()
    client = mlflow.MlflowClient()
    experiment = get_experiment_id(get_experiment_name(base.scope))
    parents = client.search_runs(
        [experiment], f"attributes.run_name = 'refine-{config.name}' and tags.refine = '{config.name}'", order_by=["attributes.start_time DESC"]
    )
    if not parents:
        raise ValueError(f"No refinement loop {config.name} to resume")
    parent = parents[0].info.run_id
    rounds = {}
    for run in client.search_runs([experiment], f"tags.refine = '{config.name}' and tags.`mlflow.parentRunId` = '{parent}'"):
        if run.info.status == "FINISHED":
            rounds[int(run.data.tags["round"])] = run.info.run_id
        elif run.info.status == "RUNNING":
            client.set_terminated(run.info.run_id, "KILLED")
    if parents[0].info.status == "RUNNING":
        client.set_terminated(parent, "KILLED")
    if not (config.start_run or 0 in rounds):
        raise ValueError(f"Refinement loop {config.name} was interrupted in round 0, start it again under a new name")
    run_ids = [config.start_run or rounds[0]]
    while len(run_ids) in rounds:
        run_ids.append(rounds[len(run_ids)])
    finished = len(run_ids) - 1
    if finished == config.rounds:
        raise ValueError(f"Refinement loop {config.name} has finished all {config.rounds} rounds")
    pending_bundle = bundle_exists(get_round_bundle_name(config, finished + 1))
    last = get_round_bundle_name(config, finished + 1 if pending_bundle else finished) if finished or pending_bundle else None
    added = list(load_bundle(last).train_extra[len(base.train_extra) :]) if last else []
    return ResumeState(parent, run_ids, added, pending_bundle)


def get_valop_depositions(config: RefineConfig, base: Bundle, material: np.ndarray | None, run_ids: list[str], added: list[str]) -> list:
    """The validation optimizations of finished rounds again: the model of the round before, the materials the round added, the same
    seeds. They were only kept in memory by the interrupted loop."""
    depositions = []
    for k, dataset_id in enumerate(added, start=1):
        round_materials = [material] if material is not None else [m[None, :] for m in np.unique(load_dataset(dataset_id).full_material(), axis=0)]
        seeds = get_round_seeds(k, config.valop_optimizations, VALOP_SEED_OFFSET)
        fronts = optimize_model_many(
            load_run_model(run_ids[k - 1]), base.scope, round_materials, seeds, config.population_size, config.evaluations, config.n_jobs
        )
        depositions += [(round_material, found) for round_material, (found, _, _) in zip(round_materials, fronts, strict=True)]
    return depositions


def refine(config: RefineConfig) -> list[str]:
    """Runs the loop and returns the ids of the training runs, one per round (round 0 is the model on the base bundle)."""
    base = load_bundle(config.base)
    names = [f"{config.name}-r{k}" for k in range(1, config.rounds)] + [config.name] + ([f"{config.name}-control"] if config.control else [])
    existing = [name for name in names if bundle_exists(name)]
    if existing and not config.resume:
        raise FileExistsError(f"Bundles {existing} exist already, choose another name or continue the loop with --resume")
    material = get_fixed_material(base, config.reference_set) if not config.materials_per_round else None
    with_profiles = has_profiles(base)  # the added data keeps what the base data has, so a profile model can train on all of it
    mlflow = configure_mlflow()
    state = get_resume_state(config, base) if config.resume else None
    first = len(state.run_ids) if state else 1
    # Own streams, so the perturbations do not depend on the material count; a resumed loop continues with new streams
    rng = np.random.default_rng([config.seed, first] if state else config.seed)
    material_rng = np.random.default_rng([config.seed, 7, first] if state else [config.seed, 7])

    run_options = (
        {"run_id": state.parent_run} if state else {"experiment_id": get_experiment_id(get_experiment_name(base.scope)), "run_name": f"refine-{config.name}"}
    )
    with mlflow.start_run(**run_options) as parent:
        if state:
            run_ids, added = list(state.run_ids), list(state.added)
            run_id = run_ids[-1]
            valop_depositions = get_valop_depositions(config, base, material, run_ids, added)  # with the interrupted round if it saved its data
            logging.info(f"Resuming {config.name} after round {first - 1}")
        else:
            mlflow.log_params(config.params_to_log())
            mlflow.set_tags({"refine": config.name})
            if config.start_run:
                run_id, metrics = config.start_run, mlflow.get_run(config.start_run).data.metrics
            else:
                run_id, metrics = train_round(config, base.name, f"{config.name}-r0", 0)
            mlflow.log_metrics(get_finite_metrics(get_round_metrics(metrics)), step=0)
            run_ids, added, valop_depositions = [run_id], [], []

        for k in range(first, config.rounds + 1):
            start = time.perf_counter()
            name = get_round_bundle_name(config, k)
            if state and state.pending_bundle and k == first:  # interrupted after saving the round's data: only train again
                run_id, metrics = train_round(config, name, f"{config.name}-r{k}", k, source_run=run_id)
                run_ids.append(run_id)
                mlflow.log_metrics(get_finite_metrics({**get_round_metrics(metrics), "refine/round_seconds": time.perf_counter() - start}), step=k)
                continue
            model = load_run_model(run_id)
            seeds = get_round_seeds(k, config.optimizations)
            valop_seeds = get_round_seeds(k, config.valop_optimizations, VALOP_SEED_OFFSET)
            round_materials = [material] if material is not None else [m[None, :] for m in random_materials(config.materials_per_round, material_rng)]
            parts, predicted_parts, front_rows, offset = [], [], [], 0
            all_fronts = optimize_model_many(model, base.scope, round_materials, seeds + valop_seeds, config.population_size, config.evaluations, config.n_jobs)
            for round_material, (found, predicted_all, found_seeds) in zip(round_materials, all_fronts, strict=True):
                for_training = np.isin(found_seeds, seeds)
                deposition = found[for_training]
                valop_depositions.append((round_material, found[~for_training]))
                inputs = np.vstack([deposition, perturb(deposition, config.perturbations, config.perturbation_sd, rng)])
                parts.append((np.repeat(round_material, len(inputs), axis=0), inputs))
                predicted_parts.append(predicted_all[for_training])
                front_rows.append(np.arange(offset, offset + len(deposition)))  # the solutions come first, their perturbations after
                offset += len(inputs)
            fronts = np.concatenate(front_rows)
            solutions = len(fronts)
            materials = np.vstack([part[0] for part in parts])
            inputs = np.vstack([part[1] for part in parts])
            source = {"round": k, "model_run": run_id, "seeds": list(seeds), "front_solutions": solutions, "materials": len(round_materials)}
            source |= {"perturbations": config.perturbations, "perturbation_sd": config.perturbation_sd}
            settings = {"scope": base.scope}
            rows_material = materials if material is None else material
            dataset = build_dataset(
                f"{config.name}-add{k}", "refinement", config.seed, rows_material, inputs, 1, config.n_jobs, settings, source, with_profiles
            )
            added.append(save_dataset(dataset))
            front_errors = get_front_errors(np.vstack(predicted_parts), dataset.y[fronts])
            logging.info(f"Round {k}: {solutions} solutions found for {len(round_materials)} material(s), {len(inputs)} rows added, {front_errors}")

            val_extra = dict(base.val_extra)
            if k == config.rounds:
                val_extra[VALOP] = build_valop(config, base, material, valop_depositions, run_ids)
            notes = {"refine": config.name, "round": k, "base": base.name, "model": config.model, "params": config.params}
            save_bundle(Bundle(name, base.scope, base.train, base.val, base.tests, notes, [*base.train_extra, *added], val_extra))

            run_id, metrics = train_round(config, name, f"{config.name}-r{k}", k, source_run=run_id)
            run_ids.append(run_id)
            round_metrics = {**get_round_metrics(metrics), **front_errors, "refine/rows_added": float(len(inputs))}
            mlflow.log_metrics(get_finite_metrics({**round_metrics, "refine/round_seconds": time.perf_counter() - start}), step=k)

        if config.control:
            rows = sum(load_manifest(dataset_id)["size"] for dataset_id in added)
            run_ids.append(train_control(config, base, material, rows, rng, material_rng))
        mlflow.set_tags({"final_run": run_ids[config.rounds], "final_bundle": config.name})
        logging.info(f"Refinement {config.name} done in parent run {parent.info.run_id}")

    # The models of the earlier rounds on the sets of the final bundle (`<name>/valop/...`), for comparison. Needs the parent run closed.
    for earlier in run_ids[: config.rounds]:
        evaluate_run(earlier, config.name)
    annotate_run(parent.info.run_id)
    return run_ids


def has_profiles(bundle: Bundle) -> bool:
    return all(load_manifest(dataset_id).get("profiles", False) for dataset_id in [bundle.train, *bundle.train_extra])


def build_valop(config: RefineConfig, base: Bundle, material: np.ndarray | None, depositions: list[tuple[np.ndarray, np.ndarray]], run_ids: list[str]) -> str:
    """The operating-region validation set: solutions of separate optimizations of every round's model, never trained on, with repeats.
    `depositions` holds the material and the solutions of each optimization."""
    rows = np.unique(np.vstack([np.hstack([np.repeat(m, len(d), axis=0), d]) for m, d in depositions]), axis=0)
    materials, inputs = rows[:, :-DEPOSITION_LENGTH], rows[:, -DEPOSITION_LENGTH:]
    source = {"model_runs": run_ids, "optimizations_per_round": config.valop_optimizations, "refine": config.name}
    dataset = build_dataset(
        f"{config.name}-{VALOP}",
        "refinement-validation",
        config.seed,
        material if material is not None else materials,
        inputs,
        config.valop_repeats,
        config.n_jobs,
        {"scope": base.scope},
        source,
    )
    return save_dataset(dataset)


def train_control(
    config: RefineConfig, base: Bundle, material: np.ndarray | None, rows: int, rng: np.random.Generator, material_rng: np.random.Generator | None = None
) -> str:
    """The same number of rows added as random depositions (and, without a fixed material, random materials): the effect of more data
    without the operating region."""
    from bmh_ml.datasets.generators import random_depositions

    final = load_bundle(config.name)
    deposition = random_depositions(rows, rng)
    if material is None:
        material = random_materials(rows, material_rng)
    settings = {"scope": base.scope}
    dataset = build_dataset(f"{config.name}-control-add", "random", config.seed, material, deposition, 1, config.n_jobs, settings, None, has_profiles(base))
    name = f"{config.name}-control"
    notes = {"refine": config.name, "control_of": config.name, "base": base.name}
    save_bundle(Bundle(name, base.scope, base.train, base.val, base.tests, notes, [*base.train_extra, save_dataset(dataset)], final.val_extra))
    return train_round(config, name, name, -1)[0]


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = RefineConfig("", "", "", {})
    parser = argparse.ArgumentParser(description="Refinement loop: adds solutions found by optimizing the model to its training data, round by round")
    parser.add_argument("--base", required=True, help="Bundle to start from, its validation and test sets are kept")
    parser.add_argument("--name", required=True, help="Name of the final bundle, the rounds before are <name>-r<k>")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--param", nargs="*", default=[], metavar="KEY=VALUE", help="Parameters of the model")
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--optimizations", type=int, default=defaults.optimizations, help="NSGA-III runs on the model per round")
    parser.add_argument("--population-size", type=int, default=defaults.population_size)
    parser.add_argument("--evaluations", type=int, default=defaults.evaluations, help="Evaluations of the model per NSGA-III run")
    parser.add_argument("--perturbations", type=int, default=defaults.perturbations, help="Perturbed copies of every solution found")
    parser.add_argument("--perturbation-sd", type=float, default=defaults.perturbation_sd, help="Standard deviation of the perturbation of a position")
    parser.add_argument("--valop-optimizations", type=int, default=defaults.valop_optimizations, help="NSGA-III runs per round for the validation set")
    parser.add_argument("--valop-repeats", type=int, default=defaults.valop_repeats)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--start-run", help="Use the model of this run as round 0 instead of training one on the base bundle")
    parser.add_argument("--control", action="store_true", help="Also train on the base bundle plus as many random rows as were added")
    parser.add_argument("--transfer", action="store_true", help="Run the transfer test on the model of every round")
    parser.add_argument("--reference-set", default=defaults.reference_set, help="Test set that defines the material and the transfer reference")
    parser.add_argument("--n-jobs", type=int, help="Processes for the simulation (default: all cores)")
    parser.add_argument("--materials-per-round", type=int, default=0, help="Scope S2: optimize the model for this many new random materials each round")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted loop of this name after its last finished round")
    args = parser.parse_args(argv)
    args.params = parse_parameters(args.param)
    return args


def main(argv: list[str] | None = None):
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO)
    transfer = TransferConfig(reference_set=args.reference_set, n_jobs=args.n_jobs) if args.transfer else None
    options = {name: getattr(args, name) for name in RefineConfig.__dataclass_fields__ if name not in ("transfer",)}
    config = RefineConfig(**options, transfer=transfer)
    run_ids = refine(config)
    print(f"Runs of the rounds: {' '.join(run_ids)}")


if __name__ == "__main__":
    main()
