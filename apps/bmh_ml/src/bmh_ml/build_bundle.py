import argparse
import logging
from pathlib import Path

import numpy as np

from bmh_ml.datasets.generators import (
    find_result_files,
    load_fixed_material,
    load_front_depositions,
    random_depositions,
    random_materials,
    stress_depositions,
)
from bmh_ml.datasets.manifest import SCOPE_FIXED_MATERIAL, SCOPE_GENERAL_MATERIAL, SCOPES
from bmh_ml.datasets.simulate import build_dataset
from bmh_ml.datasets.store import Bundle, bundle_exists, get_bundle_simulator, get_evaluation_sets, load_bundle, load_dataset, save_bundle, save_dataset
from bmh_ml.evaluation.noise import format_noise_table, get_noise_summary
from bmh_ml.settings import DEPOSITION_LENGTH, TRAINING_DATA_FILE, SimulatorSettings, use_simulator_settings


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates the datasets of a bundle: training and validation data and the frozen test sets")
    parser.add_argument("--name", required=True, help="Name of the bundle, it is frozen afterwards")
    parser.add_argument(
        "--scope", choices=SCOPES, required=True, help="S1: one fixed material, models use the 20 deposition inputs. S2: any material, 70 inputs"
    )
    parser.add_argument("--train-size", type=int, default=250_000)
    parser.add_argument("--val-size", type=int, default=20_000)
    parser.add_argument("--test-size", type=int, default=5_000, help="Size of the random test set T1")
    parser.add_argument("--train-repeats", type=int, default=1, help="Simulations per training input, labels are their mean")
    parser.add_argument("--val-repeats", type=int, default=8)
    parser.add_argument("--test-repeats", type=int, default=16, help="Repeats of the test sets, they make the labels almost noise free")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=None, help="Processes for the simulation (default: all cores)")
    parser.add_argument("--material-from", default=f"csv:{TRAINING_DATA_FILE}", help="Fixed material: csv:<file>, results:<directory> or random")
    parser.add_argument("--fronts", type=Path, help="Directory with the results of the optimization on the simulation, for the test set T2")
    parser.add_argument("--surrogate-fronts", type=Path, help="Directory with the results of the optimization on the surrogate, for T2s")
    parser.add_argument("--t3-materials", type=int, default=20, help="S2: number of new materials of T3")
    parser.add_argument("--t3-depositions", type=int, default=50, help="S2: depositions per material of T3")
    parser.add_argument("--material-fronts", type=int, default=0, help="T6: new random materials with reference fronts found by NSGA-III on the simulator")
    parser.add_argument("--front-evaluations", type=int, default=25_000, help="T6: simulations per NSGA-III run")
    parser.add_argument("--front-seeds", type=int, default=3, help="T6: NSGA-III runs per material")
    parser.add_argument("--front-population", type=int, default=100, help="T6: population of the NSGA-III runs")
    parser.add_argument("--stress-random", type=int, default=200, help="Random inputs at the edges of the range in the stress test set T5")
    parser.add_argument("--tests-from", help="Use the validation and test sets of this bundle (same scope) and only generate new training data")
    parser.add_argument("--new-val", action="store_true", help="With --tests-from: generate a new validation set, keep only the test sets")
    parser.add_argument("--profiles", action="store_true", help="Store the reclaimed profiles of the training and validation data (for profile models)")
    parser.add_argument("--ppm3", type=float, default=1.0, help="Detail level of the simulator: particles per cubic meter (reclaim step fixed at 1 m)")
    parser.add_argument("--cross-from", help="Also label the inputs of test sets of this bundle at this bundle's level, as test sets <set>x")
    parser.add_argument("--cross-sets", nargs="+", default=["T6"], help="The test sets of --cross-from to label again (default: T6)")
    return parser.parse_args(argv)


def build_inputs(scope: str, n: int, rng: np.random.Generator, fixed_material: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if scope == SCOPE_FIXED_MATERIAL:
        return fixed_material, random_depositions(n, rng)
    return random_materials(n, rng), random_depositions(n, rng)


def build_bundle(args: argparse.Namespace) -> Bundle:
    if bundle_exists(args.name):
        raise FileExistsError(f"Bundle '{args.name}' already exists and is frozen, choose another name")
    seeds = np.random.SeedSequence(args.seed).spawn(7)
    names = ("material", "train", "val", "t1", "t3", "t5", "t6")
    rngs = {name: np.random.default_rng(seed) for name, seed in zip(names, seeds, strict=True)}
    scope = args.scope
    settings = {"scope": scope}
    simulator = SimulatorSettings(ppm3=args.ppm3)
    use_simulator_settings(simulator)  # every dataset of the bundle, the T6 fronts included, is simulated at this level

    fixed_material, material_source = None, {}
    needs_material = scope == SCOPE_FIXED_MATERIAL or args.fronts or args.surrogate_fronts
    if needs_material:
        fixed_material, material_source = load_fixed_material(args.material_from, rngs["material"])

    def build(name: str, generator: str, material, deposition, repeats: int, source: dict | None = None, with_profiles: bool = False) -> str:
        dataset = build_dataset(
            f"{args.name}-{name}",
            generator,
            args.seed,
            material,
            deposition,
            repeats,
            args.n_jobs,
            settings,
            {**material_source, **(source or {})},
            with_profiles=with_profiles,
        )
        dataset_id = save_dataset(dataset)
        logging.info(f"{name}: {len(dataset)} rows, repeats {repeats}, {dataset_id}")
        return dataset_id

    train_material, train_deposition = build_inputs(scope, args.train_size, rngs["train"], fixed_material)
    train = build("train", "random", train_material, train_deposition, args.train_repeats, with_profiles=args.profiles)

    def build_val() -> str:
        val_material, val_deposition = build_inputs(scope, args.val_size, rngs["val"], fixed_material)
        return build("val", "random", val_material, val_deposition, args.val_repeats, with_profiles=args.profiles)

    if args.tests_from:
        other = load_bundle(args.tests_from)
        if other.scope != scope:
            raise ValueError(f"Bundle {other.name} has scope {other.scope}, not {scope}")
        if get_bundle_simulator(other) != simulator:
            raise ValueError(f"The test sets of {other.name} were labeled with other simulator settings; use --cross-from to label them again")
        val = build_val() if args.new_val else other.val
        tests = {**other.tests, **build_cross_sets(args, build)}
        return finish(args, Bundle(args.name, scope, train, val, tests, {"tests_from": other.name}, val_extra=other.val_extra))

    val = build_val()

    tests = {**build_tests(args, scope, rngs, fixed_material, build), **build_cross_sets(args, build)}
    return finish(args, Bundle(args.name, scope, train, val, tests))


def build_cross_sets(args: argparse.Namespace, build) -> dict[str, str]:
    """The inputs of test sets of another bundle, labeled at this bundle's level (`<set>x`): the same solutions at two detail levels, e.g.
    the fronts optimized at one level evaluated at another."""
    if not args.cross_from:
        return {}
    other = load_bundle(args.cross_from)
    sets = {}
    for name in args.cross_sets:
        dataset = load_dataset(other.tests[name])
        source = {"cross_from": other.name, "set": name, "dataset": other.tests[name], "simulator": get_bundle_simulator(other).as_dict()}
        sets[f"{name}x"] = build(f"{name}x", f"{dataset.generator}-cross", dataset.material, dataset.deposition, args.test_repeats, source)
    return sets


def build_tests(args: argparse.Namespace, scope: str, rngs: dict, fixed_material: np.ndarray | None, build) -> dict[str, str]:
    """The frozen test sets of a new bundle, by name."""
    tests = {}
    material, deposition = build_inputs(scope, args.test_size, rngs["t1"], fixed_material)
    tests["T1"] = build("T1", "random", material, deposition, args.test_repeats)

    if args.fronts:
        deposition, source = load_front_depositions(find_result_files(args.fronts))
        tests["T2"] = build("T2", "simulator-fronts", fixed_material, deposition, args.test_repeats, source)
    if args.surrogate_fronts:
        deposition, source = load_front_depositions(find_result_files(args.surrogate_fronts))
        tests["T2s"] = build("T2s", "surrogate-fronts", fixed_material, deposition, args.test_repeats, source)

    if scope == SCOPE_GENERAL_MATERIAL:
        materials = random_materials(args.t3_materials, rngs["t3"])
        material = np.repeat(materials, args.t3_depositions, axis=0)
        deposition = random_depositions(len(material), rngs["t3"])
        tests["T3"] = build("T3", "unseen-materials", material, deposition, args.test_repeats, {"depositions_per_material": args.t3_depositions})

    if args.material_fronts:
        material, deposition, source = build_material_fronts(args, rngs["t6"])
        tests["T6"] = build("T6", "simulator-fronts-materials", material, deposition, args.test_repeats, source)

    deposition = stress_depositions(rngs["t5"], args.stress_random)
    material = fixed_material if scope == SCOPE_FIXED_MATERIAL else random_materials(len(deposition), rngs["t5"])
    tests["T5"] = build("T5", "stress", material, deposition, args.test_repeats)

    return tests


def build_material_fronts(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """New random materials and, for each, the final front solutions of NSGA-III runs on the simulator: one row per solution, the material
    repeated. The reference front of a material for the transfer test is the non-dominated part of its rows (after relabeling)."""
    from bmh_ml.evaluation.transfer import TransferConfig, run_simulator_fronts_many

    seeds = tuple(range(1, args.front_seeds + 1))
    config = TransferConfig(seeds=seeds, population_size=args.front_population, evaluations=args.front_evaluations, n_jobs=args.n_jobs)
    new_materials = random_materials(args.material_fronts, rng)
    fronts = run_simulator_fronts_many([material[None, :] for material in new_materials], config)  # all runs of all materials in parallel
    materials, depositions = [], []
    for material, front in zip(new_materials, fronts, strict=True):
        deposition = np.unique(front[0], axis=0)
        materials.append(np.repeat(material[None, :], len(deposition), axis=0))
        depositions.append(deposition)
        logging.info(f"T6: material {len(materials)} of {args.material_fronts}, {len(deposition)} front solutions")
    source = {"materials": args.material_fronts, "evaluations_per_run": args.front_evaluations, "runs_per_material": args.front_seeds}
    return np.vstack(materials), np.vstack(depositions), source


def finish(args: argparse.Namespace, bundle: Bundle) -> Bundle:
    bundle.notes.update({"seed": args.seed, "arguments": {k: str(v) for k, v in vars(args).items()}})
    save_bundle(bundle)
    return bundle


def show_noise(bundle: Bundle):
    summaries = {}
    for name, dataset_id in get_evaluation_sets(bundle).items():
        dataset = load_dataset(dataset_id)
        if dataset.y_noise_sd is not None:
            summaries[f"{bundle.name}/{name}"] = get_noise_summary(dataset)
    print(format_noise_table(summaries))


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO)
    args = get_args(argv)
    bundle = build_bundle(args)
    show_noise(bundle)
    print(f"Bundle '{bundle.name}' stored, train {bundle.train}, {DEPOSITION_LENGTH} deposition inputs")


if __name__ == "__main__":
    main()
