import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from bmh_ml import train_lstm_model  # noqa: E402
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH, MODEL_F1_NAME, MODEL_F2_NAME, SCALER_FILE, get_model_file  # noqa: E402
from bmh_ml.training_data import load_training_data  # noqa: E402


def write_training_data(directory, rows: int = 40):
    rng = np.random.default_rng(0)
    columns = ["f1", "f2", *[f"m{i + 1}" for i in range(MATERIAL_LENGTH)], *[f"d{i + 1}" for i in range(DEPOSITION_LENGTH)]]
    data = pd.DataFrame(
        np.column_stack(
            [rng.uniform(0, 1, rows), rng.uniform(5, 40, rows), rng.uniform(5, 10, (rows, MATERIAL_LENGTH)), rng.uniform(10, 49, (rows, DEPOSITION_LENGTH))]
        ),
        columns=columns,
    )
    (directory / "data").mkdir()
    data.to_csv(directory / "data" / "training_data.csv", index=False)


def train(epochs: int = 1):
    train_lstm_model.main(argparse.Namespace(verbose=False, epochs=epochs))


def load_models():
    return tf.keras.models.load_model(get_model_file(MODEL_F1_NAME)), tf.keras.models.load_model(get_model_file(MODEL_F2_NAME))


def test_training_writes_scaler_and_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_training_data(tmp_path)

    train()

    model_f1, model_f2 = load_models()
    assert model_f1.input_shape == (None, 1, MATERIAL_LENGTH + DEPOSITION_LENGTH)
    assert model_f2.input_shape == (None, 1, DEPOSITION_LENGTH)  # F2 only depends on the deposition
    with open(SCALER_FILE, "rb") as f:
        assert pickle.load(f).n_features_in_ == MATERIAL_LENGTH + DEPOSITION_LENGTH  # noqa: S301 - written by the training just above


def test_training_ignores_a_truncated_last_row_instead_of_training_on_nan(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    write_training_data(tmp_path)
    path = tmp_path / "data" / "training_data.csv"
    content = path.read_text()
    path.write_text(content[: content.rindex(",", 0, len(content) - 40)])

    with caplog.at_level(logging.WARNING):
        train()

    assert "Ignoring the incomplete last row" in caplog.text
    model_f1, _ = load_models()
    assert np.isfinite(model_f1.predict(np.full((1, 1, MATERIAL_LENGTH + DEPOSITION_LENGTH), 0.0), verbose=0)).all()
    assert len(load_training_data(str(path))) == 39


def test_training_refuses_corrupt_training_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_training_data(tmp_path)
    path = tmp_path / "data" / "training_data.csv"
    lines = path.read_text().splitlines()
    lines[5] = ",".join(lines[5].split(",")[:-3])
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="incomplete rows"):
        train()
