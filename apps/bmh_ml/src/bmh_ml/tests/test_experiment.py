import argparse
import json
import os
from collections import Counter

import numpy as np
import pytest
from pymoo.problems import get_problem

from bmh_ml.experiment import add_experiment_arguments, get_design, run_experiments

MATERIAL = np.linspace(5.0, 10.0, 50)


def test_default_arguments_are_the_original_design():
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)

    args = parser.parse_args([])

    assert (args.runs, args.evaluations, args.population_sizes) == (30, [20000, 100000], [50, 100, 200])
    assert args.training_data == "data/training_data.csv"


def test_design_assigns_the_settings_of_the_original_latin_hypercube_design():
    design = get_design(30, [20000, 100000], [50, 100, 200])

    # (evaluations, population size, replicate), the settings of the first runs are those of the original implementation
    assert design[:3] == [(20000, 100, 0), (20000, 200, 0), (20000, 200, 1)]
    assert len(design) == 30
    assert Counter((evaluations, population) for evaluations, population, _ in design) == {
        (20000, 50): 3,
        (20000, 100): 7,
        (20000, 200): 5,
        (100000, 50): 7,
        (100000, 100): 3,
        (100000, 200): 5,
    }


def test_replicates_count_the_runs_with_the_same_settings():
    design = get_design(30, [20000, 100000], [50, 100, 200])

    replicates_by_setting: dict[tuple[int, int], list[int]] = {}
    for evaluations, population, replicate in design:
        replicates_by_setting.setdefault((evaluations, population), []).append(replicate)

    assert all(replicates == list(range(len(replicates))) for replicates in replicates_by_setting.values())


def test_design_only_uses_the_given_levels_and_is_deterministic():
    design = get_design(8, [100, 200, 300], [10, 20])

    assert design == get_design(8, [100, 200, 300], [10, 20])
    assert {evaluations for evaluations, _, _ in design} <= {100, 200, 300}
    assert {population for _, population, _ in design} <= {10, 20}
    assert all(isinstance(value, int) for run in design for value in run)


def make_args(runs: int = 3, evaluations: tuple[int, ...] = (40, 80), population_sizes: tuple[int, ...] = (10, 20)) -> argparse.Namespace:
    return argparse.Namespace(runs=runs, evaluations=list(evaluations), population_sizes=list(population_sizes))


def test_run_experiments_writes_one_file_per_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    problem = get_problem("zdt1", n_var=5)

    paths = run_experiments(problem, "test_model", make_args(), MATERIAL)

    design = get_design(3, [40, 80], [10, 20])
    assert paths == [
        os.path.join("output", "test_model", f"run_{run:02d}_evals{evaluations}_pop{population}_replicate{replicate:02d}.json")
        for run, (evaluations, population, replicate) in enumerate(design)
    ]
    for run, ((evaluations, population, replicate), path) in enumerate(zip(design, paths, strict=True)):
        result = json.loads((tmp_path / path).read_text())
        assert list(result) == ["model", "material_variables", "objectives", "variables", "parameters"]
        assert result["model"] == "test_model"
        assert result["parameters"] == {"run_id": run, "replicate": replicate, "population_size": population, "n_evaluations": evaluations}
        assert len(result["objectives"]) == len(result["variables"]) > 0
        assert len(result["objectives"][0]) == 2
        assert len(result["variables"][0]) == 5


def test_every_result_stores_the_fixed_material_and_extra_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    paths = run_experiments(get_problem("zdt1", n_var=5), "test_model", make_args(), MATERIAL, extra_parameters={"model_set": "some_set"})

    for path in paths:
        result = json.loads((tmp_path / path).read_text())
        assert result["material_variables"] == MATERIAL.tolist()
        assert result["parameters"]["model_set"] == "some_set"


def read_results(directory, paths) -> list[dict]:
    return [json.loads((directory / path).read_text()) for path in paths]


def test_replicates_repeat_the_same_problem_with_different_random_seeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(runs=3, evaluations=(60,), population_sizes=(10,))  # one setting, so all runs are replicates of each other

    results = read_results(tmp_path, run_experiments(get_problem("zdt1", n_var=5), "test_model", args, MATERIAL))

    assert [result["parameters"]["replicate"] for result in results] == [0, 1, 2]
    assert len({(result["parameters"]["population_size"], result["parameters"]["n_evaluations"]) for result in results}) == 1
    assert len({json.dumps(result["variables"]) for result in results}) == 3  # different random streams give different results
    assert all(result["material_variables"] == MATERIAL.tolist() for result in results)  # for the same material


def test_run_experiments_is_reproducible(tmp_path, monkeypatch):
    results = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.chdir(directory)
        paths = run_experiments(get_problem("zdt1", n_var=5), "test_model", make_args(runs=2), MATERIAL)
        results.append([(directory / path).read_bytes() for path in paths])

    assert results[0] == results[1]


def test_run_experiments_rejects_less_evaluations_than_one_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(runs=1, evaluations=(5,), population_sizes=(10,))

    with pytest.raises(ValueError, match="5 evaluations are not enough for one generation of 10 solutions"):
        run_experiments(get_problem("zdt1", n_var=5), "test_model", args, MATERIAL)
