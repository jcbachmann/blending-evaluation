import numpy as np
import pytest

from bmh_ml.datasets.manifest import Dataset
from bmh_ml.evaluation.evaluate import evaluate_dataset, measure_throughput
from bmh_ml.evaluation.metrics import get_metrics, get_pairwise_accuracy
from bmh_ml.models.registry import create_model
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH


def make_truth(n: int = 2000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(1, 5, n)


def test_a_perfect_model_has_no_error():
    y = make_truth()

    metrics = get_metrics(y, y)

    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["bias"] == 0
    assert metrics["r2"] == 1
    assert metrics["spearman"] == pytest.approx(1)
    assert metrics["pairwise_accuracy"] == 1
    assert metrics["tail_rmse"] == 0


def test_the_error_of_a_model_with_noise_is_measured():
    y = make_truth(20_000)
    prediction = y + np.random.default_rng(1).normal(0, 0.1, len(y))

    metrics = get_metrics(y, prediction)

    assert metrics["rmse"] == pytest.approx(0.1, rel=0.05)
    assert metrics["mae"] == pytest.approx(0.1 * (2 / np.pi) ** 0.5, rel=0.05)
    assert abs(metrics["bias"]) < 0.01
    assert 0.99 < metrics["r2"] < 1
    assert "nrmse" not in metrics


def test_the_error_is_measured_against_the_noise_of_the_simulator():
    y = make_truth()

    metrics = get_metrics(y, y + 0.1, noise_sd=0.2)

    assert metrics["nrmse"] == pytest.approx(0.5)
    assert metrics["bias"] == pytest.approx(0.1)


def test_a_constant_model_explains_nothing():
    y = make_truth()

    metrics = get_metrics(y, np.full_like(y, y.mean()))

    assert metrics["r2"] == pytest.approx(0, abs=1e-9)
    assert np.isnan(metrics["spearman"])


def test_negative_predictions_are_counted():
    y = make_truth(100)
    prediction = y.copy()
    prediction[:25] = -1

    assert get_metrics(y, prediction)["negative_rate"] == 0.25


def test_the_tail_of_the_best_solutions_is_measured_separately():
    y = make_truth(1000)
    prediction = y.copy()
    best = y <= np.quantile(y, 0.1)
    prediction[best] += 0.5

    metrics = get_metrics(y, prediction)

    assert metrics["tail_rmse"] == pytest.approx(0.5)
    assert metrics["tail_bias"] == pytest.approx(0.5)
    assert metrics["rmse"] < 0.2


def test_a_model_that_reverses_the_order_ranks_every_pair_wrong():
    y = make_truth()

    assert get_pairwise_accuracy(y, -y) == 0
    assert get_metrics(y, -y)["spearman"] == pytest.approx(-1)


def test_pairs_that_the_noise_could_swap_are_left_out():
    y = np.array([0.0, 0.001] * 500)
    prediction = np.array([0.001, 0.0] * 500)  # right order except for tiny differences

    assert get_pairwise_accuracy(y, prediction, min_gap=0.0) < 1
    assert np.isnan(get_pairwise_accuracy(y, prediction, min_gap=0.01))


def make_dataset(n: int = 200, repeats: int = 1, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    deposition = rng.uniform(0, 1, (n, DEPOSITION_LENGTH))
    y = np.column_stack([deposition[:, 0] + 1, deposition[:, 1] + 2])
    return Dataset("test", "random", seed, repeats, rng.uniform(5, 10, (n, MATERIAL_LENGTH)), deposition, y, np.full((n, 2), 0.1) if repeats > 1 else None)


def test_a_model_is_evaluated_on_both_objectives():
    dataset = make_dataset()
    model = create_model("mean")
    model.fit(dataset.features("S2"), dataset.y, dataset.features("S2"), dataset.y, seed=0)

    result = evaluate_dataset(model, dataset, "S2")

    assert result.y_pred.shape == (200, 2)
    assert {"F1/rmse", "F2/rmse", "F1/r2", "F2/tail_rmse"} <= set(result.metrics)
    assert "F1/nrmse" not in result.metrics  # no repeated simulations, so no noise estimate


def test_the_noise_ceiling_is_reported_for_repeated_datasets():
    dataset = make_dataset(repeats=4)
    model = create_model("mean")
    model.fit(dataset.features("S1"), dataset.y, dataset.features("S1"), dataset.y, seed=0)

    metrics = evaluate_dataset(model, dataset, "S1").metrics

    assert metrics["F1/noise_sd"] == pytest.approx(0.1)
    assert 0 < metrics["F1/r2_ceiling"] <= 1
    assert metrics["F1/nrmse"] == pytest.approx(metrics["F1/rmse"] / 0.1)


def test_the_throughput_is_measured_for_every_batch_size():
    dataset = make_dataset()
    model = create_model("mean")
    model.fit(dataset.features("S2"), dataset.y, dataset.features("S2"), dataset.y, seed=0)

    throughput = measure_throughput(model, dataset.features("S2"), batch_sizes=(10, 1000), min_seconds=0.01)

    assert set(throughput) == {"throughput/batch_10", "throughput/batch_1000"}
    assert all(value > 0 for value in throughput.values())
