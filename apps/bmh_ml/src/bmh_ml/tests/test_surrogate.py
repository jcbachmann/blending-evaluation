import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

tf = pytest.importorskip("tensorflow")

from bmh_ml import train_lstm_model  # noqa: E402
from bmh_ml.optimize_model import HomogenizationProblemMl  # noqa: E402
from bmh_ml.settings import (  # noqa: E402
    DATA_DIR,
    DEPOSITION_LENGTH,
    MATERIAL_LENGTH,
    MODEL_F1_NAME,
    MODEL_F2_NAME,
    X_MAX,
    X_MIN,
    get_model_file,
    get_scaler_file,
)
from bmh_ml.surrogate import Surrogate  # noqa: E402

FEATURES = MATERIAL_LENGTH + DEPOSITION_LENGTH
MODEL_SET = "test_set"


def make_model(features: int):
    inputs = tf.keras.Input(shape=(1, features))
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(4)(inputs)))


def save_model_set(model_set: str):
    """Small untrained models in the files the scripts load: F1 takes all variables, F2 only the deposition."""
    (Path(DATA_DIR)).mkdir(exist_ok=True)
    make_model(FEATURES).save(get_model_file(MODEL_F1_NAME, model_set))
    make_model(DEPOSITION_LENGTH).save(get_model_file(MODEL_F2_NAME, model_set))
    scaler = StandardScaler().fit(np.random.default_rng(0).uniform(0, 50, (20, FEATURES)))
    with open(get_scaler_file(model_set), "wb") as f:
        pickle.dump(scaler, f)


@pytest.fixture
def surrogate_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_model_set(MODEL_SET)


@pytest.mark.usefixtures("surrogate_files")
def test_predict_returns_both_objectives_for_every_row():
    x = np.random.default_rng(1).uniform(5, 45, (7, FEATURES))

    predictions = Surrogate.load(MODEL_SET).predict(x)

    assert predictions.shape == (7, 2)
    assert np.isfinite(predictions).all()


@pytest.mark.usefixtures("surrogate_files")
def test_f2_only_depends_on_the_deposition():
    rng = np.random.default_rng(2)
    deposition = rng.uniform(X_MIN, X_MAX, DEPOSITION_LENGTH)
    x = np.array([np.concatenate([rng.uniform(5, 10, MATERIAL_LENGTH), deposition]) for _ in range(3)])

    predictions = Surrogate.load(MODEL_SET).predict(x)

    assert np.ptp(predictions[:, 1]) < 1e-6  # the same deposition with other materials, up to float32 rounding
    assert np.ptp(predictions[:, 0]) > 1e-4  # F1 depends on the material as well


@pytest.mark.usefixtures("surrogate_files")
def test_problem_fixes_the_material_and_bounds_the_deposition():
    material = np.linspace(5.0, 10.0, MATERIAL_LENGTH)

    problem = HomogenizationProblemMl(surrogate=Surrogate.load(MODEL_SET), material_variables=material)

    assert problem.n_var == FEATURES
    assert np.array_equal(problem.xl[:MATERIAL_LENGTH], material)
    assert np.array_equal(problem.xu[:MATERIAL_LENGTH], material)
    assert np.all(problem.xl[MATERIAL_LENGTH:] == X_MIN)
    assert np.all(problem.xu[MATERIAL_LENGTH:] == X_MAX)


@pytest.mark.usefixtures("surrogate_files")
def test_problem_evaluates_with_the_surrogate():
    surrogate = Surrogate.load(MODEL_SET)
    problem = HomogenizationProblemMl(surrogate=surrogate, material_variables=np.linspace(5.0, 10.0, MATERIAL_LENGTH))
    x = np.column_stack([np.tile(np.linspace(5.0, 10.0, MATERIAL_LENGTH), (4, 1)), np.random.default_rng(3).uniform(X_MIN, X_MAX, (4, DEPOSITION_LENGTH))])

    assert np.array_equal(problem.evaluate(x), surrogate.predict(x))


def test_models_written_by_the_trainer_can_be_used_by_the_surrogate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / DATA_DIR).mkdir()
    rng = np.random.default_rng(0)
    columns = ["f1", "f2", *[f"m{i + 1}" for i in range(MATERIAL_LENGTH)], *[f"d{i + 1}" for i in range(DEPOSITION_LENGTH)]]
    features = np.column_stack([rng.uniform(5, 10, (40, MATERIAL_LENGTH)), rng.uniform(X_MIN, X_MAX, (40, DEPOSITION_LENGTH))])
    training_data = pd.DataFrame(np.column_stack([rng.uniform(0, 1, 40), rng.uniform(5, 40, 40), features]), columns=columns)
    training_data.to_csv(tmp_path / DATA_DIR / "training_data.csv", index=False)

    train_lstm_model.main(argparse.Namespace(verbose=False, epochs=1, training_data="data/training_data.csv", model_set="trained"))

    predictions = Surrogate.load("trained").predict(np.full((2, FEATURES), 20.0))
    assert predictions.shape == (2, 2)
    assert np.isfinite(predictions).all()


@pytest.mark.usefixtures("surrogate_files")
def test_the_surrogate_knows_its_model_set():
    assert Surrogate.load(MODEL_SET).model_set == MODEL_SET


@pytest.mark.usefixtures("surrogate_files")
def test_an_unknown_model_set_is_reported_with_the_command_to_train_it():
    with pytest.raises(FileNotFoundError, match=r"Model set 'other' is incomplete.*--model-set other"):
        Surrogate.load("other")


@pytest.mark.usefixtures("surrogate_files")
def test_model_sets_are_kept_apart():
    save_model_set("second_set")  # other random weights, the files of the first set are not touched
    x = np.random.default_rng(4).uniform(5, 45, (3, FEATURES))

    first, second = Surrogate.load(MODEL_SET), Surrogate.load("second_set")

    assert first.model_set != second.model_set
    assert not np.array_equal(first.predict(x), second.predict(x))
