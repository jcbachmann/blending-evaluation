import argparse
import json
from collections import Counter

import pytest
from pymoo.problems import get_problem

from bmh_ml.experiment import add_experiment_arguments, get_design, run_experiments


def test_default_arguments_are_the_original_design():
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)

    args = parser.parse_args([])

    assert (args.runs, args.evaluations, args.population_sizes) == (30, [20000, 100000], [50, 100, 200])


def test_design_is_the_original_latin_hypercube_design():
    design = get_design(30, [20000, 100000], [50, 100, 200])

    # (evaluations, population size, instance id), the first runs of the original implementation
    assert design[:3] == [(20000, 100, 28), (20000, 200, 3), (20000, 200, 9)]
    assert sorted(instance for _, _, instance in design) == list(range(30))  # every instance is used by exactly one run
    assert Counter((evaluations, population) for evaluations, population, _ in design) == {
        (20000, 50): 3,
        (20000, 100): 7,
        (20000, 200): 5,
        (100000, 50): 7,
        (100000, 100): 3,
        (100000, 200): 5,
    }


def test_design_only_uses_the_given_levels_and_is_deterministic():
    design = get_design(8, [100, 200, 300], [10, 20])

    assert design == get_design(8, [100, 200, 300], [10, 20])
    assert {evaluations for evaluations, _, _ in design} <= {100, 200, 300}
    assert {population for _, population, _ in design} <= {10, 20}
    assert all(isinstance(value, int) for run in design for value in run)


def make_args(runs: int = 3) -> argparse.Namespace:
    return argparse.Namespace(runs=runs, evaluations=[40, 80], population_sizes=[10, 20])


def test_run_experiments_writes_one_file_per_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    problem = get_problem("zdt1", n_var=5)

    paths = run_experiments(problem, "test_model", make_args())

    design = get_design(3, [40, 80], [10, 20])
    assert paths == [f"output/test_model/instance_{instance:02d}_run_{run:02d}_pop{population}.json" for run, (_, population, instance) in enumerate(design)]
    for run, ((evaluations, population, instance), path) in enumerate(zip(design, paths, strict=True)):
        result = json.loads((tmp_path / path).read_text())
        assert list(result) == ["model", "objectives", "variables", "parameters"]
        assert result["model"] == "test_model"
        assert result["parameters"] == {"instance_id": instance, "run_id": run, "population_size": population, "n_evaluations": evaluations}
        assert len(result["objectives"]) == len(result["variables"]) > 0
        assert len(result["objectives"][0]) == 2
        assert len(result["variables"][0]) == 5


def test_run_experiments_is_reproducible(tmp_path, monkeypatch):
    results = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.chdir(directory)
        paths = run_experiments(get_problem("zdt1", n_var=5), "test_model", make_args(runs=2))
        results.append([(directory / path).read_bytes() for path in paths])

    assert results[0] == results[1]


def test_run_experiments_rejects_less_evaluations_than_one_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(runs=1, evaluations=[5], population_sizes=[10])

    with pytest.raises(ValueError, match="5 evaluations are not enough for one generation of 10 solutions"):
        run_experiments(get_problem("zdt1", n_var=5), "test_model", args)
