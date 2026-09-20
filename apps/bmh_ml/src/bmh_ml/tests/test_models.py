import json

import numpy as np
import pytest

from bmh_ml.models.base import Model, get_objective_inputs
from bmh_ml.models.registry import MODELS, create_model, get_model_class, load_model
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH

N_FEATURES = MATERIAL_LENGTH + DEPOSITION_LENGTH


def make_data(n: int, seed: int, n_features: int = N_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """F1 depends on the material and the deposition, F2 on the deposition only, both with a little noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, n_features))
    deposition = x[:, -DEPOSITION_LENGTH:]
    material = x[:, : n_features - DEPOSITION_LENGTH]
    f1 = deposition[:, :5].sum(axis=1) * (1 + material.mean(axis=1) if material.size else 1) + 0.01 * rng.standard_normal(n)
    f2 = np.sin(3 * deposition[:, 0]) + deposition[:, 1] * deposition[:, 2] + 0.01 * rng.standard_normal(n)
    return x, np.column_stack([f1, f2])


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return 1 - float(np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


@pytest.fixture(scope="module")
def data():
    return make_data(1500, 0), make_data(300, 1), make_data(300, 2)


def test_every_model_name_resolves_to_a_model_class_without_its_heavy_package():
    for name in MODELS:
        model_class = get_model_class(name)
        assert model_class.name == name
        assert issubclass(model_class, Model)


def test_an_unknown_model_is_reported():
    with pytest.raises(ValueError, match="Unknown model 'forest'"):
        create_model("forest")


def test_unknown_parameters_are_rejected():
    with pytest.raises(ValueError, match="Unknown parameters"):
        create_model("mean", alpha=1)
    with pytest.raises(ValueError, match="alpha"):
        create_model("ridge", alhpa=1)


def test_the_parameters_are_the_defaults_and_the_given_values():
    model = create_model("ridge", alpha=5.0)

    assert model.params["alpha"] == 5.0
    assert model.params["deposition_only_f2"] is True
    assert model.describe()["model"] == "ridge"


def test_f2_uses_the_deposition_only_if_asked_to():
    x = np.arange(2 * N_FEATURES, dtype=float).reshape(2, N_FEATURES)

    assert get_objective_inputs(x, 0, True).shape == (2, N_FEATURES)
    assert np.array_equal(get_objective_inputs(x, 1, True), x[:, -DEPOSITION_LENGTH:])
    assert get_objective_inputs(x, 1, False).shape == (2, N_FEATURES)
    # the features of a fixed material are the deposition, there is nothing to cut
    assert get_objective_inputs(x[:, :DEPOSITION_LENGTH], 1, True).shape == (2, DEPOSITION_LENGTH)


def test_the_mean_model_predicts_the_training_mean(data, tmp_path):
    (x, y), (x_val, y_val), _ = data
    model = create_model("mean")

    model.fit(x, y, x_val, y_val, seed=0)
    prediction = model.predict(x_val)

    assert prediction.shape == (len(x_val), 2)
    assert np.allclose(prediction, y.mean(axis=0))
    model.save(tmp_path)
    assert np.array_equal(load_model(tmp_path).predict(x_val), prediction)


def test_the_stored_model_says_what_it_is(tmp_path):
    model = create_model("mean")
    model.save(tmp_path)

    assert json.loads((tmp_path / "model.json").read_text()) == {"name": "mean", "params": {}}


def test_ridge_learns_a_linear_relation_and_survives_saving(data, tmp_path):
    pytest.importorskip("sklearn")
    (x, y), (x_val, y_val), (x_test, y_test) = data
    model = create_model("ridge", alpha=0.1)

    model.fit(x, y, x_val, y_val, seed=0)
    prediction = model.predict(x_test)

    assert r2(y_test[:, 0], prediction[:, 0]) > 0.6  # F1 has an interaction that a linear model cannot capture
    model.save(tmp_path / "ridge")
    assert np.allclose(load_model(tmp_path / "ridge").predict(x_test), prediction)


def test_the_f2_prediction_does_not_depend_on_the_material(data):
    pytest.importorskip("sklearn")
    (x, y), (x_val, y_val), (x_test, _) = data
    model = create_model("ridge")
    model.fit(x, y, x_val, y_val, seed=0)
    changed = x_test.copy()
    changed[:, :MATERIAL_LENGTH] += 0.5

    assert np.array_equal(model.predict(x_test)[:, 1], model.predict(changed)[:, 1])
    assert not np.allclose(model.predict(x_test)[:, 0], model.predict(changed)[:, 0])


def test_all_inputs_are_used_for_f2_if_asked_to(data):
    pytest.importorskip("sklearn")
    (x, y), (x_val, y_val), (x_test, _) = data
    model = create_model("ridge", deposition_only_f2=False)
    model.fit(x, y, x_val, y_val, seed=0)
    changed = x_test.copy()
    changed[:, :MATERIAL_LENGTH] += 0.5

    assert not np.array_equal(model.predict(x_test)[:, 1], model.predict(changed)[:, 1])


def test_models_work_on_the_features_of_a_fixed_material():
    pytest.importorskip("sklearn")
    x, y = make_data(400, 0, DEPOSITION_LENGTH)
    x_val, y_val = make_data(100, 1, DEPOSITION_LENGTH)
    model = create_model("ridge")

    model.fit(x, y, x_val, y_val, seed=0)

    assert model.predict(x_val).shape == (100, 2)


def test_lightgbm_learns_the_nonlinear_relation_and_survives_saving(data, tmp_path):
    pytest.importorskip("lightgbm")
    (x, y), (x_val, y_val), (x_test, y_test) = data
    model = create_model("lightgbm", n_estimators=300, learning_rate=0.1, n_jobs=1)

    info = model.fit(x, y, x_val, y_val, seed=0)
    prediction = model.predict(x_test)

    assert r2(y_test[:, 0], prediction[:, 0]) > 0.9
    assert r2(y_test[:, 1], prediction[:, 1]) > 0.9
    assert info["F1/trees"] >= 1
    model.save(tmp_path / "lightgbm")
    assert np.allclose(load_model(tmp_path / "lightgbm").predict(x_test), prediction)


def test_lightgbm_with_the_same_seed_gives_the_same_model(data):
    pytest.importorskip("lightgbm")
    (x, y), (x_val, y_val), (x_test, _) = data
    predictions = []
    for _ in range(2):
        model = create_model("lightgbm", n_estimators=50, n_jobs=1)
        model.fit(x, y, x_val, y_val, seed=3)
        predictions.append(model.predict(x_test))

    assert np.array_equal(predictions[0], predictions[1])


@pytest.mark.parametrize(
    ("name", "params", "min_r2"),
    [
        ("mlp", {"width": 32, "depth": 2, "epochs": 40, "patience": 5}, 0.8),
        ("legacy_lstm", {"units": 16, "epochs": 30}, 0.2),  # unscaled labels and a fixed number of epochs make it a weak learner on purpose
    ],
)
def test_neural_networks_learn_and_survive_saving(name, params, min_r2, data, tmp_path):
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    (x, y), (x_val, y_val), (x_test, y_test) = data
    model = create_model(name, **params)

    info = model.fit(x, y, x_val, y_val, seed=0)
    prediction = model.predict(x_test)

    assert prediction.shape == (len(x_test), 2)
    assert np.all(np.isfinite(prediction))
    assert r2(y_test[:, 0], prediction[:, 0]) > min_r2
    assert info["F2/epochs"] >= 1
    model.save(tmp_path / name)
    assert np.allclose(load_model(tmp_path / name).predict(x_test), prediction, atol=1e-5)


def test_the_mlp_stops_early(data):
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    (x, y), (x_val, y_val), _ = data
    model = create_model("mlp", width=16, depth=1, epochs=300, patience=3, learning_rate=0.05)

    info = model.fit(x, y, x_val, y_val, seed=0)

    assert info["F1/epochs"] < 300
    assert info["F1/best_epoch"] <= info["F1/epochs"]


def test_the_legacy_model_trains_all_epochs_on_unscaled_labels():
    model_class = pytest.importorskip("bmh_ml.models.keras_models").LegacyLSTMModel

    assert model_class.defaults["epochs"] == 100
    assert model_class.defaults["patience"] == 0
    assert model_class.defaults["scale_targets"] is False
    assert model_class.defaults["batch_size"] == 32
