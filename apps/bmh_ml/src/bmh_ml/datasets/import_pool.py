"""Stores results computed outside the store as datasets: the output files of the compute pool's `simulate` and `optimize` jobs.

A `simulate` result (`results-<i>.npz`: y, y_sd, profiles, settings, versions) is paired with the input file it was computed from
(`inputs-<i>.npz`: material, deposition); an `optimize` result (deposition, objectives, material, settings, versions) holds its inputs
itself: a front becomes a dataset of its depositions, labeled with the objectives of the simulator it was optimized on. Every file of one
dataset must come from the same simulator settings and release; the job's seed and the pool image go into the dataset's source.
"""

import argparse
import dataclasses
import json
import logging
import re
from pathlib import Path

import numpy as np

from bmh_ml.datasets.manifest import Dataset, concatenate_datasets
from bmh_ml.datasets.store import save_dataset
from bmh_ml.settings import PROFILE_LENGTH, SimulatorSettings

SETTINGS_FIELDS = {field.name for field in dataclasses.fields(SimulatorSettings)}


def read_settings(stored: dict) -> tuple[SimulatorSettings, dict]:
    """The simulator settings a pool job recorded, and what it recorded besides them (seed, DEM parameters, versions)."""
    settings = json.loads(str(stored["settings"]))
    versions = json.loads(str(stored["versions"]))
    if settings.get("simulator") == "detailed":
        raise ValueError("results of the detailed simulation (DEM) cannot be stored yet: SimulatorSettings has no detailed simulator")
    simulator = SimulatorSettings.from_dict({key: value for key, value in settings.items() if key in SETTINGS_FIELDS})
    if versions.get("simulator") not in (None, "unknown", simulator.version):
        raise ValueError(f"the job recorded simulator {versions['simulator']} but settings of {simulator.version}")
    extra = {key: value for key, value in settings.items() if key not in SETTINGS_FIELDS}
    return simulator, {**extra, "image": versions.get("image"), "host": versions.get("host")}


def file_number(path: Path) -> int:
    match = re.search(r"(\d+)\.npz$", path.name)
    if match is None:
        raise ValueError(f"{path}: expected a file name ending in <number>.npz")
    return int(match.group(1))


def load_simulate_result(result_file: Path, input_file: Path, name: str, repeats: int) -> tuple[Dataset, SimulatorSettings]:
    with np.load(result_file) as stored:
        result = {key: stored[key] for key in stored.files}
    with np.load(input_file) as stored:
        inputs = {key: stored[key] for key in stored.files}
    simulator, extra = read_settings(result)
    y_sd = result["y_sd"]
    profiles = result.get("profiles")
    dataset = Dataset(
        name=name,
        generator="pool-simulate",
        seed=int(extra["seed"]) if extra.get("seed") is not None else 0,
        repeats=repeats,
        material=np.atleast_2d(inputs["material"]),
        deposition=inputs["deposition"],
        y=result["y"],
        y_noise_sd=None if np.isnan(y_sd).all() else y_sd,
        profiles=profiles if profiles is not None and profiles.shape[-1] == PROFILE_LENGTH else None,
        settings={"simulator": simulator.as_dict()},
        source={"files": [input_file.name, result_file.name], **extra},
    )
    return dataset, simulator


def load_optimize_result(result_file: Path, name: str) -> tuple[Dataset, SimulatorSettings]:
    with np.load(result_file) as stored:
        result = {key: stored[key] for key in stored.files}
    simulator, extra = read_settings(result)
    dataset = Dataset(
        name=name,
        generator="pool-optimize",
        seed=int(extra["seed"]) if extra.get("seed") is not None else 0,
        repeats=1,
        material=np.atleast_2d(result["material"])[:1],
        deposition=result["deposition"],
        y=result["objectives"],
        settings={"simulator": simulator.as_dict()},
        source={"files": [result_file.name], **extra},
    )
    return dataset, simulator


def import_results(name: str, results: list[Path], inputs: Path | None = None, repeats: int = 1) -> Dataset:
    """One dataset of the result files, in the order of their file numbers. `inputs`: the directory of the input files of `simulate`
    results (`inputs-<i>.npz` for `results-<i>.npz`); without it the files are `optimize` results."""
    results = sorted(results, key=file_number)
    if not results:
        raise ValueError("no result files")
    loaded = []
    for result_file in results:
        if inputs is None:
            loaded.append(load_optimize_result(result_file, name))
        else:
            loaded.append(load_simulate_result(result_file, inputs / f"inputs-{file_number(result_file)}.npz", name, repeats))
    settings = {simulator for _, simulator in loaded}
    if len(settings) > 1:
        raise ValueError(f"the files come from different simulator settings: {sorted(map(str, settings))}")
    datasets = [dataset for dataset, _ in loaded]
    dataset = concatenate_datasets(name, datasets)
    sources = [part.source for part in datasets]
    common = {key: value for key, value in sources[0].items() if key != "files" and all(source.get(key) == value for source in sources)}
    noise = [part.y_noise_sd for part in datasets]
    y_noise_sd = np.vstack(noise) if all(sd is not None for sd in noise) else None
    return dataclasses.replace(dataset, y_noise_sd=y_noise_sd, source={**common, "files": [file for source in sources for file in source["files"]]})


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stores the results of compute pool jobs (simulate, optimize) as one dataset")
    parser.add_argument("--name", required=True, help="Name of the dataset (its id adds a hash of the content)")
    parser.add_argument("results", nargs="+", type=Path, help="Result files (results-<i>.npz of simulate jobs, or the fronts of optimize jobs)")
    parser.add_argument("--inputs", type=Path, help="simulate results: the directory of their input files inputs-<i>.npz")
    parser.add_argument("--repeats", type=int, default=1, help="simulate results: the --repeats the jobs ran with")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.INFO)
    args = get_args(argv)
    dataset = import_results(args.name, args.results, args.inputs, args.repeats)
    dataset_id = save_dataset(dataset)
    print(f"Dataset '{dataset_id}' stored: {len(dataset)} rows, {dataset.settings['simulator']}")


if __name__ == "__main__":
    main()
