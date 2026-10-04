import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.manifest import Dataset
from bmh_ml.datasets.store import load_bundle
from bmh_ml.settings import SIMULATOR_ENVIRONMENT_VARIABLE

SMALL = ["--scope", "S2", "--seed", "3", "--val-size", "6", "--test-size", "8", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1"]
TESTS = ["--t3-materials", "1", "--t3-depositions", "4", "--stress-random", "1"]
FRONTS = ["--material-fronts", "2", "--front-evaluations", "40", "--front-seeds", "1", "--front-population", "8"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path / "store"


def test_two_levels_are_compared_on_random_inputs_and_on_the_fronts_optimized_at_each():
    pytest.importorskip("pymoo")
    from bmh_ml.fidelity import get_fidelity_metrics

    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", "--train-size", "10", *SMALL, *TESTS, *FRONTS]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", "--train-size", "10", *SMALL, *TESTS, *FRONTS, "--cross-from", "low"]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "low-x", "--train-size", "4", *SMALL, "--tests-from", "low", "--cross-from", "high"]))

    metrics = get_fidelity_metrics(load_bundle("low"), load_bundle("high"), load_bundle("low-x"))

    assert -1 <= metrics["T1/F1/spearman"] <= 1
    assert metrics["T1/F2/rmse"] >= 0
    assert "T6low/F1/spearman" in metrics  # the low level's optimized solutions at both levels
    assert "T6high/F2/shift" in metrics
    for level in ("at_high", "at_low"):
        assert metrics[f"{level}/materials"] == 2
        assert metrics[f"{level}/hv_ratio"] >= 0
        assert 0 <= metrics[f"{level}/nondominated_share"] <= 1
        assert metrics[f"{level}/hv_ratio_material_min"] <= metrics[f"{level}/hv_ratio"]


def test_paired_sets_must_have_the_same_inputs():
    from bmh_ml.fidelity import get_paired_metrics

    rng = np.random.default_rng(0)
    material = rng.uniform(5, 10, (1, 50))
    first = Dataset("a", "random", 0, 1, material, rng.uniform(10, 49, (5, 20)), rng.uniform(0, 1, (5, 2)))
    second = Dataset("b", "random", 0, 1, material, rng.uniform(10, 49, (5, 20)), rng.uniform(0, 1, (5, 2)))

    assert get_paired_metrics(first, first)["F1/spearman"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same inputs"):
        get_paired_metrics(first, second)
