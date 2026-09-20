import argparse
import json
import sys

import numpy as np
import pytest

from bmh_ml import optimize_simulation
from bmh_ml.optimize_simulation import HomogenizationProblemSimulation
from bmh_ml.settings import DEPOSITION_LENGTH, X_MAX, X_MIN

MATERIAL = np.linspace(5.0, 10.0, 50)


class SerialPool:
    """Stands in for a multiprocessing pool and counts how often it is used."""

    def __init__(self):
        self.starmap_calls = 0

    def starmap(self, function, arguments):
        self.starmap_calls += 1
        return [function(*args) for args in arguments]


def test_problem_optimizes_the_deposition_within_the_stockpile_core():
    problem = HomogenizationProblemSimulation(material_variables=MATERIAL, pool=SerialPool())

    assert problem.n_var == DEPOSITION_LENGTH
    assert problem.n_obj == 2
    assert np.all(problem.xl == X_MIN)
    assert np.all(problem.xu == X_MAX)


def test_problem_evaluates_all_solutions_of_a_generation_with_the_given_pool():
    pool = SerialPool()
    problem = HomogenizationProblemSimulation(material_variables=MATERIAL, pool=pool)
    rng = np.random.default_rng(0)

    first = problem.evaluate(rng.uniform(X_MIN, X_MAX, (4, DEPOSITION_LENGTH)))
    second = problem.evaluate(rng.uniform(X_MIN, X_MAX, (3, DEPOSITION_LENGTH)))

    assert first.shape == (4, 2)
    assert second.shape == (3, 2)
    assert np.isfinite(first).all()
    assert pool.starmap_calls == 2  # one call per generation on the same pool, no new pool


def test_main_uses_a_single_pool_for_all_runs(tmp_path, monkeypatch):
    pools = []

    class CountingPool(SerialPool):
        def __init__(self):
            super().__init__()
            pools.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(optimize_simulation, "Pool", CountingPool)
    monkeypatch.setattr(optimize_simulation, "load_fixed_material_variables", lambda: MATERIAL)

    optimize_simulation.main(argparse.Namespace(verbose=False, runs=2, evaluations=[20], population_sizes=[10]))

    assert len(pools) == 1
    assert pools[0].starmap_calls >= 4  # two runs with two generations each
    results = sorted((tmp_path / "output" / "simulation").glob("*.json"))
    assert len(results) == 2
    assert json.loads(results[0].read_text())["model"] == "simulation"


@pytest.mark.skipif(sys.platform != "linux", reason="worker processes are spawned on other platforms and are not needed to test the problem")
def test_real_process_pool_evaluates_solutions():
    from multiprocessing import Pool

    with Pool(2) as pool:
        problem = HomogenizationProblemSimulation(material_variables=MATERIAL, pool=pool)
        objectives = problem.evaluate(np.random.default_rng(1).uniform(X_MIN, X_MAX, (6, DEPOSITION_LENGTH)))

    assert objectives.shape == (6, 2)
    assert np.isfinite(objectives).all()
