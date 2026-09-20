"""Inputs of the datasets: random, stress patterns, and the fronts found by the optimization experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np

from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH, MATERIAL_MAX, MATERIAL_MIN, X_MAX, X_MIN
from bmh_ml.training_data import load_fixed_material_variables
from bmh_ml.variables import generate_material_variables


def random_depositions(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(X_MIN, X_MAX, (n, DEPOSITION_LENGTH))


def random_materials(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.array([generate_material_variables(MATERIAL_LENGTH, MATERIAL_MIN, MATERIAL_MAX, rng=rng) for _ in range(n)])


def stress_depositions(rng: np.random.Generator, n_random_binary: int = 200) -> np.ndarray:
    """Depositions far from the random ones: at the edges, constant, alternating, ramps, sawtooth, steps and random extremes.

    Random depositions are smooth on average, but the optimization finds and exploits extremes, so a model has to be checked on them too.
    """
    n = DEPOSITION_LENGTH
    low, high = X_MIN, X_MAX
    position = np.arange(n)
    patterns = [np.full(n, low), np.full(n, high)]
    patterns += [np.full(n, level) for level in np.linspace(low, high, 9)[1:-1]]
    patterns += [np.where(position % 2 == 0, low, high), np.where(position % 2 == 0, high, low)]
    patterns += [np.linspace(low, high, n), np.linspace(high, low, n)]
    for period in (3, 4, 5, 8, 10):
        ramp = (position % period) / (period - 1)
        patterns += [low + (high - low) * ramp, high - (high - low) * ramp]
    for steps in (2, 3, 4, 5):
        stair = (position * steps // n) / (steps - 1)
        patterns += [low + (high - low) * stair, high - (high - low) * stair]
    patterns += [np.where(position < n // 2, low, high), np.where(position < n // 2, high, low)]
    patterns += [np.where(rng.random(n) < 0.5, low, high) for _ in range(n_random_binary)]
    patterns = np.array(patterns, dtype=float)
    # Some patterns coincide (e.g. two steps and the two halves), keep the first of each in the original order
    _, first = np.unique(patterns, axis=0, return_index=True)
    return patterns[np.sort(first)]


def get_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_front_depositions(files: list[Path]) -> tuple[np.ndarray, dict]:
    """The deposition variables of the solutions in result files of the optimization experiments, without duplicates.

    Results on the simulation contain the 20 deposition variables, results on the surrogate the 50 material and 20 deposition variables.
    """
    rows = []
    for file in files:
        variables = np.array(json.loads(file.read_text())["variables"], dtype=float)
        if variables.ndim != 2 or variables.shape[1] not in (DEPOSITION_LENGTH, MATERIAL_LENGTH + DEPOSITION_LENGTH):
            raise ValueError(f"{file} has variables of shape {variables.shape}, expected 20 or 70 columns")
        rows.append(variables[:, -DEPOSITION_LENGTH:])
    if not rows:
        raise ValueError("No result files given")
    depositions = np.unique(np.vstack(rows), axis=0)
    source = {"files": {file.name: get_file_hash(file) for file in files}}
    return depositions, source


def find_result_files(directory: Path) -> list[Path]:
    """The result files of a directory, without the files derived from them (recomputed results)."""
    files = sorted(path for path in directory.glob("*.json") if "_recomputed_" not in path.name)
    if not files:
        raise FileNotFoundError(f"No result files in {directory}")
    return files


def load_fixed_material(spec: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """The material of a dataset for the fixed material scope.

    `csv:<file>`: the first row of a training data file, `results:<directory>`: the material stored in a result file of the optimization
    (or the first 50 variables of a result on the surrogate), `random`: a new random material.
    """
    kind, _, argument = spec.partition(":")
    if kind == "random":
        return np.array(generate_material_variables(MATERIAL_LENGTH, MATERIAL_MIN, MATERIAL_MAX, rng=rng)), {"material": "random"}
    if kind == "csv":
        return load_fixed_material_variables(argument), {"material": spec}
    if kind == "results":
        for file in find_result_files(Path(argument)):
            result = json.loads(file.read_text())
            if "material_variables" in result:
                return np.array(result["material_variables"], dtype=float), {"material": spec, "file": file.name}
            variables = np.array(result["variables"], dtype=float)
            if variables.shape[1] == MATERIAL_LENGTH + DEPOSITION_LENGTH:
                return variables[0, :MATERIAL_LENGTH], {"material": spec, "file": file.name}
        raise ValueError(f"No result file in {argument} contains the material")
    raise ValueError(f"Unknown material '{spec}', use csv:<file>, results:<directory> or random")
