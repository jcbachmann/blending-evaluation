import json

import numpy as np
import pytest

from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.manifest import Dataset
from bmh_ml.evaluation.transfer import (
    TransferConfig,
    get_front_metrics,
    get_nondominated,
    get_reference_front,
    get_simulator_run_ratios,
    optimize_model,
    run_transfer_test,
)
from bmh_ml.models.registry import create_model

REFERENCE = np.array([[0.1, 10.0], [0.2, 5.0], [0.4, 3.0]])
SMALL_TRANSFER = TransferConfig(seeds=(1, 2), population_size=6, evaluations=12, repeats=2, reference_set="T1", n_jobs=1)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def make_reference_dataset(n: int = 6, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    deposition = random_depositions(n, rng)
    y = np.vstack([REFERENCE, REFERENCE[: n - len(REFERENCE)] + 1])  # the rows after the front are dominated
    return Dataset(name="ref", generator="test", seed=seed, repeats=2, material=random_materials(1, rng), deposition=deposition, y=y)


def test_the_non_dominated_solutions_are_found():
    objectives = np.array([[1.0, 1.0], [0.5, 2.0], [2.0, 2.0], [1.0, 1.0], [0.4, 3.0]])

    assert list(get_nondominated(objectives[[0, 1, 2, 4]])) == [0, 1, 3]


def test_the_reference_front_scores_one_and_worse_or_better_fronts_are_told_apart():
    assert get_front_metrics(REFERENCE, REFERENCE) == pytest.approx({"hv_ratio": 1.0, "igd_plus": 0.0})

    worse = get_front_metrics(REFERENCE * 1.05, REFERENCE)
    better = get_front_metrics(REFERENCE * 0.95, REFERENCE)
    assert worse["hv_ratio"] < 1 < better["hv_ratio"]
    assert worse["igd_plus"] > 0
    assert better["igd_plus"] == 0
    assert get_front_metrics(REFERENCE * 10, REFERENCE)["hv_ratio"] == 0  # beyond the reference point


def test_the_reference_front_is_the_non_dominated_part_of_the_reference_set():
    assert np.array_equal(get_reference_front(make_reference_dataset()), REFERENCE)


def test_simulator_runs_are_judged_by_the_labels_of_the_reference_set(tmp_path):
    reference = make_reference_dataset()
    best, dominated = tmp_path / "best.json", tmp_path / "dominated.json"
    best.write_text(json.dumps({"variables": reference.deposition[:3].tolist()}))
    dominated.write_text(json.dumps({"variables": reference.deposition[3:].tolist()}))

    ratios = get_simulator_run_ratios([best, dominated], reference)

    assert ratios[0] == pytest.approx(1)
    assert ratios[1] < 1
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"variables": random_depositions(1, np.random.default_rng(9)).tolist()}))
    with pytest.raises(ValueError, match="not in the reference"):
        get_simulator_run_ratios([unknown], reference)


def test_the_transfer_test_optimizes_the_model_and_simulates_what_it_finds():
    pytest.importorskip("sklearn")
    reference = make_reference_dataset()
    rng = np.random.default_rng(1)
    x = random_depositions(50, rng)
    model = create_model("ridge")
    model.fit(x, np.column_stack([x[:, 0] / 100, x[:, -1]]), x, np.column_stack([x[:, 0] / 100, x[:, -1]]), seed=0)

    result = run_transfer_test(model, "S1", reference, SMALL_TRANSFER)

    assert len(result.deposition) == len(result.simulated) == len(result.seed)
    assert set(result.seed) == {1, 2}
    assert result.simulated_sd is not None
    assert np.allclose(result.predicted, model.predict(result.deposition))
    for name in ("hv_ratio", "hv_ratio_run_mean", "igd_plus", "predicted_hv_ratio", "negative_rate", "F1/bias", "F2/rmse", "optimize_seconds"):
        assert np.isfinite(result.metrics[name]), name
    assert result.metrics["solutions"] == len(result.deposition)


def test_the_fronts_do_not_depend_on_the_number_of_workers():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    x = random_depositions(50, rng)
    y = np.column_stack([x[:, 0] / 100, x[:, -1]])
    model = create_model("ridge")
    model.fit(x, y, x, y, seed=0)
    material = random_materials(1, rng)

    one = optimize_model(model, "S1", material, (1, 2), population_size=6, evaluations=12, workers=1)
    two = optimize_model(model, "S1", material, (1, 2), population_size=6, evaluations=12, workers=2)

    for first, second in zip(one, two, strict=True):
        assert np.allclose(first, second)
