import argparse
import json

import numpy as np

from bmh_ml import optimize_simulation
from bmh_ml.optimize_simulation import HomogenizationProblemSimulation
from bmh_ml.settings import DEPOSITION_LENGTH, X_MAX, X_MIN

MATERIAL = np.linspace(5.0, 10.0, 50)


def test_problem_optimizes_the_deposition_within_the_stockpile_core():
    problem = HomogenizationProblemSimulation(material_variables=MATERIAL, threads=1)

    assert problem.n_var == DEPOSITION_LENGTH
    assert problem.n_obj == 2
    assert np.all(problem.xl == X_MIN)
    assert np.all(problem.xu == X_MAX)


def test_problem_simulates_a_generation_in_one_call(monkeypatch):
    calls = []
    original = optimize_simulation.evaluate_sims_with_profiles

    def counting(material, deposition, *args, **kwargs):
        calls.append((len(deposition), kwargs["threads"]))
        return original(material, deposition, *args, **kwargs)

    monkeypatch.setattr(optimize_simulation, "evaluate_sims_with_profiles", counting)
    problem = HomogenizationProblemSimulation(material_variables=MATERIAL, threads=2)
    rng = np.random.default_rng(0)

    first = problem.evaluate(rng.uniform(X_MIN, X_MAX, (4, DEPOSITION_LENGTH)))
    second = problem.evaluate(rng.uniform(X_MIN, X_MAX, (3, DEPOSITION_LENGTH)))

    assert first.shape == (4, 2)
    assert second.shape == (3, 2)
    assert np.isfinite(first).all()
    assert calls == [(4, 2), (3, 2)]  # one call per generation with all its solutions


def test_main_writes_one_result_per_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(optimize_simulation, "load_fixed_material_variables", lambda _training_data_file: MATERIAL)

    optimize_simulation.main(argparse.Namespace(verbose=False, runs=2, evaluations=[20], population_sizes=[10], training_data="data/training_data.csv"))

    results = sorted((tmp_path / "output" / "simulation").glob("*.json"))
    assert len(results) == 2
    result = json.loads(results[0].read_text())
    assert result["model"] == "simulation"
    assert result["material_variables"] == MATERIAL.tolist()  # the results say which material they are for
