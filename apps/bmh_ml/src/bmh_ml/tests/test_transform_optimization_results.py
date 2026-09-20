import json
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from bmh_ml import transform_optimization_results as transform
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH

MATERIAL = np.linspace(5.0, 10.0, MATERIAL_LENGTH)
OTHER_MATERIAL = np.linspace(6.0, 9.0, MATERIAL_LENGTH)
SOLUTIONS = 3


def surrogate_variables(material: np.ndarray) -> list[list[float]]:
    return [[*material.tolist(), *np.linspace(10 + solution, 40, DEPOSITION_LENGTH).tolist()] for solution in range(SOLUTIONS)]


def deposition_variables() -> list[list[float]]:
    return [np.linspace(10 + solution, 40, DEPOSITION_LENGTH).tolist() for solution in range(SOLUTIONS)]


def objectives() -> list[list[float]]:
    return [[0.1 * solution, 1.0 + solution] for solution in range(SOLUTIONS)]


def test_a_stored_material_is_used():
    data = {"material_variables": MATERIAL.tolist(), "variables": deposition_variables()}

    assert transform.get_result_material(data).tolist() == MATERIAL.tolist()


def test_results_on_the_surrogate_contain_their_material_in_the_variables():
    data = {"variables": surrogate_variables(MATERIAL)}

    assert transform.get_result_material(data).tolist() == MATERIAL.tolist()


def test_variables_with_more_than_one_material_are_an_error():
    data = {"variables": [*surrogate_variables(MATERIAL)[:1], *surrogate_variables(OTHER_MATERIAL)[:1]]}

    with pytest.raises(ValueError, match="more than one material"):
        transform.get_result_material(data)


def test_older_results_of_the_simulation_do_not_know_their_material():
    assert transform.get_result_material({"variables": deposition_variables()}) is None


def test_a_cached_recomputation_is_reused_for_the_same_material_and_model_set(tmp_path):
    path = str(tmp_path / "cache.json")
    context = transform.get_cache_context(MATERIAL, model_set="a")
    variables, recomputed = np.array([[1.0, 2.0]]), np.array([[0.5, 0.25]])
    transform.save_recomputed_results(path, variables, recomputed, context)

    cached = transform.load_recomputed_results(path, context)

    assert cached is not None
    assert cached[1].tolist() == recomputed.tolist()


@pytest.mark.parametrize(
    ("material", "model_set"),
    [(OTHER_MATERIAL, "a"), (MATERIAL, "b"), (MATERIAL, None)],
)
def test_a_cached_recomputation_of_another_material_or_model_set_is_ignored(tmp_path, material, model_set, caplog):
    path = str(tmp_path / "cache.json")
    transform.save_recomputed_results(path, np.array([[1.0]]), np.array([[2.0]]), transform.get_cache_context(MATERIAL, model_set="a"))

    with caplog.at_level(logging.INFO):
        assert transform.load_recomputed_results(path, transform.get_cache_context(material, model_set)) is None

    assert "computed for another material or model set" in caplog.text


def test_caches_of_older_versions_without_material_are_ignored(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"variables": [[1.0]], "objectives": [[2.0]]}))

    assert transform.load_recomputed_results(str(path), transform.get_cache_context(MATERIAL, model_set=None)) is None


def test_recompute_cached_computes_once_per_material(tmp_path):
    path = str(tmp_path / "result.json")
    calls = []

    def compute(variables):
        calls.append(len(calls))
        return np.full((len(variables), 2), float(len(calls)))

    variables = np.zeros((2, 3))
    for material in (MATERIAL, MATERIAL, OTHER_MATERIAL):
        transform.recompute_cached(path, "sim", variables, compute, transform.get_cache_context(material, model_set=None))

    assert len(calls) == 2  # the second call used the cache, the other material did not


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(transform, "plot_before_after", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transform, "plot_metric_scatter", lambda *_args, **_kwargs: None)


def write_result(directory, name: str, **data):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"objectives": objectives(), **data}))


@pytest.mark.usefixtures("no_plots")
def test_results_on_the_surrogate_are_recomputed_for_the_material_they_were_made_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    used = []
    monkeypatch.setattr(transform, "evaluate_sim", lambda material_variables, **_kwargs: (used.append(np.array(material_variables)), (0.5, 5.0))[1])
    write_result(tmp_path / "output" / "lstm_model", "old_format.json", variables=surrogate_variables(OTHER_MATERIAL))  # no stored material

    transform.recompute_lstm_results_with_simulation()

    assert len(used) == SOLUTIONS
    assert all(material.tolist() == OTHER_MATERIAL.tolist() for material in used)


class FakeSurrogate:
    model_set = "fake_set"

    @classmethod
    def load(cls, model_set: str):
        assert model_set == "fake_set"
        return cls()

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.last_material = x[0, :MATERIAL_LENGTH]
        return np.zeros((len(x), 2))


@pytest.mark.usefixtures("no_plots")
def test_results_on_the_simulation_use_their_stored_material_and_the_fallback_only_for_older_results(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "bmh_ml.surrogate", types.SimpleNamespace(Surrogate=FakeSurrogate))
    transform.get_fallback_material.cache_clear()
    columns = ["f1", "f2", *[f"m{i + 1}" for i in range(MATERIAL_LENGTH)], *[f"d{i + 1}" for i in range(DEPOSITION_LENGTH)]]
    training_data = pd.DataFrame([[0.0, 0.0, *OTHER_MATERIAL, *np.full(DEPOSITION_LENGTH, 20.0)]], columns=columns)
    training_data.to_csv(tmp_path / "training_data.csv", index=False)
    directory = tmp_path / "output" / "simulation"
    write_result(directory, "a_new_format.json", variables=deposition_variables(), material_variables=MATERIAL.tolist())
    write_result(directory, "b_old_format.json", variables=deposition_variables())

    with caplog.at_level(logging.WARNING):
        transform.recompute_simulation_results_with_lstm("fake_set", "training_data.csv")

    new_cache = json.loads((directory / "a_new_format_recomputed_lstm.json").read_text())
    old_cache = json.loads((directory / "b_old_format_recomputed_lstm.json").read_text())
    assert new_cache["material_variables"] == MATERIAL.tolist()
    assert old_cache["material_variables"] == OTHER_MATERIAL.tolist()  # the first row of the given training data
    assert new_cache["model_set"] == old_cache["model_set"] == "fake_set"
    assert caplog.text.count("without their material") == 1
