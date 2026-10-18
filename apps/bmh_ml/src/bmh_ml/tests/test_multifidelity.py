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


def test_the_initial_population_takes_the_given_solutions_and_fills_up_at_random():
    pytest.importorskip("pymoo")
    from bmh_ml.experiment import get_initial_population
    from bmh_ml.problems import SimulatorProblem
    from bmh_ml.settings import X_MAX, X_MIN

    problem = SimulatorProblem(random_materials(1, np.random.default_rng(0)))
    given = random_depositions(3, np.random.default_rng(1))

    population = get_initial_population(np.vstack([given, given[:1]]), 8, problem, seed=0)

    assert population.shape == (8, 20)
    assert all(any(np.array_equal(row, solution) for row in population) for solution in given)  # duplicates count once
    assert population.min() >= X_MIN
    assert population.max() <= X_MAX
    assert len(get_initial_population(random_depositions(20, np.random.default_rng(2)), 8, problem, seed=0)) == 8


def test_the_simulator_baseline_can_start_from_the_fronts_of_another_level():
    mlflow = pytest.importorskip("mlflow")
    from bmh_ml.evaluation.transfer import TransferConfig
    from bmh_ml.transfer import simulator_baseline

    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", "--train-size", "4", *SMALL, *TESTS, *FRONTS]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", "--train-size", "4", *SMALL, *TESTS, *FRONTS]))
    config = TransferConfig(**{**TRANSFER, "evaluations": 16, "population_size": 8})

    run_id = simulator_baseline(None, config, False, ["high:T6"], "L4", ["low:T6"])

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "simulator-nsga3-16-L4-seeded"
    assert run.data.params["seed_population"] == "low:T6"
    assert "transfer_L4/chevron_beaten_rate" in run.data.metrics  # under the label of the models it is compared with
    with pytest.raises(ValueError, match="no starting population"):
        simulator_baseline(None, config, False, ["high:T6"], "L4", ["high:T1"])


def test_the_simulator_baseline_can_start_from_the_fronts_a_model_found():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    from bmh.helpers.fronts import get_nondominated

    from bmh_ml.evaluation.transfer import TransferConfig, split_by_material
    from bmh_ml.train import run_training
    from bmh_ml.transfer import load_initial_populations, simulator_baseline

    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", "--train-size", "30", *SMALL, *TESTS, *FRONTS]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", "--train-size", "4", *SMALL, *TESTS, *FRONTS]))
    model_run = run_training("low", "ridge", {}, with_plots=False)
    result = transfer_run_against_high(model_run.run_id)

    initial = load_initial_populations([f"run:{model_run.run_id}:L4"])

    assert get_simulator_settings() == SimulatorSettings(ppm3=4.0)  # loading the fronts keeps the active level
    for i, (material, _) in enumerate(split_by_material(load_dataset_of("high"))):
        rows = result.material_index == i
        expected = result.deposition[rows][get_nondominated(result.predicted[rows])]  # what the model itself calls its front
        assert np.array_equal(initial[material[0].tobytes()], expected)
    config = TransferConfig(**{**TRANSFER, "evaluations": 16, "population_size": 8})
    run_id = simulator_baseline(None, config, False, ["high:T6"], "L4", [f"run:{model_run.run_id}:L4"])
    assert mlflow.get_run(run_id).info.run_name == f"simulator-nsga3-16-L4-seeded-from-{mlflow.get_run(model_run.run_id).info.run_name}"


def test_the_simulator_baseline_can_let_a_model_choose_which_offspring_are_simulated():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    from bmh_ml.evaluation.transfer import TransferConfig
    from bmh_ml.train import run_training
    from bmh_ml.transfer import simulator_baseline

    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", "--train-size", "30", *SMALL, *TESTS, *FRONTS]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", "--train-size", "4", *SMALL, *TESTS, *FRONTS]))
    model_run = run_training("low", "ridge", {}, with_plots=False)
    config = TransferConfig(**{**TRANSFER, "evaluations": 16, "population_size": 8})

    run_id = simulator_baseline(None, config, False, ["high:T6"], "L4", None, [model_run.run_id], candidates=12, simulated=4)

    run = mlflow.get_run(run_id)
    assert run.info.run_name == "simulator-nsga3-16-L4-prescreened-12to4"
    assert run.data.params["prescreen"] == model_run.run_id
    assert run.data.params["prescreen_simulated"] == "4"
    assert "transfer_L4/chevron_beaten_rate" in run.data.metrics


def transfer_run_against_high(run_id: str):
    from bmh_ml.evaluation.transfer import TransferConfig
    from bmh_ml.transfer import transfer_run_against

    return transfer_run_against(run_id, ["high:T6"], "L4", TransferConfig(**TRANSFER), with_plots=False)


def load_dataset_of(bundle_name: str):
    from bmh_ml.datasets.store import load_bundle, load_dataset

    return load_dataset(load_bundle(bundle_name).tests["T6"])


def test_the_multifidelity_model_learns_from_the_cheap_result_of_each_input(tmp_path):
    pytest.importorskip("keras")
    from bmh_ml.datasets.store import load_bundle, load_dataset
    from bmh_ml.models.registry import load_model
    from bmh_ml.settings import use_simulator_settings

    args = ["--train-size", "40", "--profiles", *SMALL, *TESTS]
    build_bundle.build_bundle(build_bundle.get_args(["--name", "cheap", "--simulator", "cone", *args]))  # the default cheap level
    build_bundle.build_bundle(build_bundle.get_args(["--name", "detailed", "--ppm3", "4", *args]))
    detailed = load_bundle("detailed")
    train, val = load_dataset(detailed.train), load_dataset(detailed.val)
    use_simulator_settings(SimulatorSettings(ppm3=4.0))  # the level of the bundle, as training activates it
    model = create_model("multifidelity_mlp", width=16, depth=1, epochs=3, material_scaling=True, low_bundle="cheap", low_ppm3=1.0)

    model.fit(train.features("S2"), train.y, val.features("S2"), val.y, seed=0, profiles_train=train.profiles, profiles_val=val.profiles, val_repeats=2)

    assert len(model.low_results) >= len(train.y) + len(val.y)  # the cheap results of the same inputs, preloaded, not simulated
    rng = np.random.default_rng(9)
    new = np.hstack([random_materials(5, rng), random_depositions(5, rng)])
    known = len(model.low_results)
    prediction = model.predict(new)
    assert prediction.shape == (5, 2)
    assert np.all(np.isfinite(prediction))
    assert len(model.low_results) == known + 5  # new inputs are simulated at the cheap level once
    assert get_simulator_settings() == SimulatorSettings(ppm3=4.0)  # and the active level is restored
    model.save(tmp_path / "model")
    loaded = load_model(tmp_path / "model")
    loaded.low_results = dict(model.low_results)  # the same cheap results, so the prediction must be the same
    assert np.allclose(loaded.predict(new), prediction, atol=1e-4)


def test_the_models_of_an_ensemble_share_the_cheap_simulation_of_an_input():
    from bmh_ml.models.registry import create_model

    first = create_model("multifidelity_mlp", low_ppm3=2.0)
    second = create_model("multifidelity_mlp", low_ppm3=2.0)
    rng = np.random.default_rng(11)
    x = np.hstack([random_materials(3, rng), random_depositions(3, rng)])

    known = len(first.low_results)
    first.get_low_results(x)

    assert second.low_results is first.low_results
    assert len(second.low_results) == known + 3  # the second model finds them without simulating
    assert np.array_equal(second.get_low_results(x), first.get_low_results(x))
    assert create_model("multifidelity_mlp", low_ppm3=4.0).low_results is not first.low_results  # another level is another simulation


def test_a_model_stored_before_the_cheap_level_could_be_chosen_keeps_the_fast_simulator():
    from bmh_ml.models.registry import create_model, get_model_class

    assert create_model("multifidelity_mlp").low_settings_key() == (1.0, "cone", "cells", "integrated")  # new models
    stored = {
        k: v for k, v in create_model("multifidelity_mlp").params.items() if k not in ("low_simulator", "low_drops", "low_ideal_stockpile")
    }  # as saved before
    upgraded = create_model("multifidelity_mlp", **get_model_class("multifidelity_mlp").upgrade_params(stored))  # what load_model does
    assert upgraded.low_settings_key() == (1.0, "fast", "point", "point")


def test_the_cheap_level_must_match_the_low_bundle():
    pytest.importorskip("keras")
    from bmh_ml.models.registry import create_model

    build_bundle.build_bundle(build_bundle.get_args(["--name", "cheap-fast", "--train-size", "20", "--profiles", *SMALL, *TESTS]))
    model = create_model("multifidelity_mlp", low_bundle="cheap-fast")  # the bundle is the fast simulator, the model's level the cone
    with pytest.raises(ValueError, match="cheap level"):
        model.preload_low_results()
