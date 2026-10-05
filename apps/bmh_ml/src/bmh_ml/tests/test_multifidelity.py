import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.simulate import build_dataset
from bmh_ml.models.registry import create_model
from bmh_ml.settings import SIMULATOR_ENVIRONMENT_VARIABLE, SimulatorSettings, get_simulator_settings

SMALL = ["--scope", "S2", "--seed", "4", "--val-size", "6", "--test-size", "6", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1"]
TESTS = ["--t3-materials", "1", "--t3-depositions", "2", "--stress-random", "1"]
FRONTS = ["--material-fronts", "2", "--front-evaluations", "40", "--front-seeds", "1", "--front-population", "8"]
TRANSFER = {"seeds": (1,), "population_size": 6, "evaluations": 12, "repeats": 2, "n_jobs": 1}


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv("BMH_ML_TRACKING_URI", raising=False)
    return tmp_path / "store"


def profile_data(n: int, seed: int, repeats: int = 1):
    rng = np.random.default_rng(seed)
    dataset = build_dataset("d", "random", 0, random_materials(n, rng), random_depositions(n, rng), repeats=repeats, n_jobs=1, with_profiles=True)
    return dataset.features("S2"), dataset.y, dataset.profiles


def test_fine_tuning_continues_from_the_model_of_another_run(monkeypatch):
    pytest.importorskip("keras")
    from bmh_ml import evaluate_run

    (x, y, p), (xv, yv, pv) = profile_data(60, 1), profile_data(20, 2, repeats=2)
    start = create_model("profile_mlp", width=16, depth=1, epochs=3, material_scaling=True)
    start.fit(x, y, xv, yv, seed=0, profiles_train=p, profiles_val=pv, val_repeats=2)
    before = start.predict(xv).copy()
    monkeypatch.setattr(evaluate_run, "load_run_model", lambda _run_id: start)
    (x2, y2, p2) = profile_data(40, 3)

    tuned = create_model("profile_mlp", width=16, depth=1, epochs=3, material_scaling=True, init_run="cheap-run", learning_rate=3e-4)
    tuned.fit(x2, y2, xv, yv, seed=0, profiles_train=p2, profiles_val=pv, val_repeats=2)

    assert tuned.network is start.network  # the trained weights are the starting point
    assert np.array_equal(tuned.scaling["x_mean"], start.scaling["x_mean"])  # the scaling the network was trained with is kept
    assert not np.allclose(tuned.predict(xv), before)  # and the training went on
    other = create_model("profile_mlp", width=32, depth=1, epochs=1, material_scaling=True, init_run="cheap-run")
    with pytest.raises(ValueError, match="architecture"):
        other.fit(x2, y2, xv, yv, seed=0, profiles_train=p2, profiles_val=pv, val_repeats=2)


def test_a_model_is_tested_against_the_references_of_another_level():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    from bmh_ml.evaluation.transfer import TransferConfig
    from bmh_ml.train import run_training
    from bmh_ml.transfer import load_references, transfer_run_against

    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", "--train-size", "30", *SMALL, *TESTS, *FRONTS]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", "--train-size", "10", *SMALL, *TESTS, *FRONTS]))
    run = run_training("low", "ridge", {}, with_plots=False)

    result = transfer_run_against(run.run_id, ["high:T6"], "L4", TransferConfig(**TRANSFER), with_plots=False)

    assert get_simulator_settings() == SimulatorSettings(ppm3=4.0)  # the solutions were simulated at the level of the references
    assert result.metrics["materials"] == 2
    logged = mlflow.get_run(run.run_id).data
    assert "transfer_L4/chevron_beaten_rate" in logged.metrics
    assert logged.params["transfer_L4.references"] == "high:T6"
    assert "transfer/chevron_beaten_rate" not in logged.metrics  # the run's own test is a different one
    with pytest.raises(ValueError, match="mix simulator settings"):
        load_references(["high:T6", "low:T6"])
