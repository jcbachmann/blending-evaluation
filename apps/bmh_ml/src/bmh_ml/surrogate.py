import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bmh_ml.settings import DEFAULT_MODEL_SET, MATERIAL_LENGTH, MODEL_F1_NAME, MODEL_F2_NAME, get_model_file, get_scaler_file


class Surrogate:
    """LSTM models predicting the objectives F1 and F2 for the variables of a material and a deposition."""

    def __init__(self, scaler: StandardScaler, model_f1: tf.keras.Model, model_f2: tf.keras.Model, model_set: str = DEFAULT_MODEL_SET):
        self.scaler = scaler
        self.model_f1 = model_f1
        self.model_f2 = model_f2
        self.model_set = model_set

    @classmethod
    def load(cls, model_set: str = DEFAULT_MODEL_SET) -> "Surrogate":
        scaler_file = get_scaler_file(model_set)
        model_files = [get_model_file(MODEL_F1_NAME, model_set), get_model_file(MODEL_F2_NAME, model_set)]
        missing = [file for file in (scaler_file, *model_files) if not os.path.exists(file)]
        if missing:
            raise FileNotFoundError(
                f"Model set '{model_set}' is incomplete, missing {missing}. Train it with: python -m bmh_ml.train_lstm_model --model-set {model_set}"
            )

        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)  # noqa: S301 - written by train_lstm_model.py
        return cls(scaler=scaler, model_f1=tf.keras.models.load_model(model_files[0]), model_f2=tf.keras.models.load_model(model_files[1]), model_set=model_set)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the objectives F1 and F2 for rows of material and deposition variables, shape (n, 70) -> (n, 2)."""
        x_scaled = self.scaler.transform(x)
        x_lstm = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        f1 = self.model_f1.predict(x_lstm, verbose=0).flatten()
        # F2 (reclaim volume deviation) only depends on the deposition, the material volume per time is constant
        f2 = self.model_f2.predict(x_lstm[:, :, MATERIAL_LENGTH:], verbose=0).flatten()
        return np.column_stack([f1, f2])
