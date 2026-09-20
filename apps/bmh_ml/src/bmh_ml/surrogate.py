import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bmh_ml.settings import MATERIAL_LENGTH, MODEL_F1_NAME, MODEL_F2_NAME, SCALER_FILE, get_model_file


class Surrogate:
    """LSTM models predicting the objectives F1 and F2 for the variables of a material and a deposition."""

    def __init__(self, scaler: StandardScaler, model_f1: tf.keras.Model, model_f2: tf.keras.Model):
        self.scaler = scaler
        self.model_f1 = model_f1
        self.model_f2 = model_f2

    @classmethod
    def load(cls) -> "Surrogate":
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)  # noqa: S301 - written by train_lstm_model.py
        return cls(
            scaler=scaler,
            model_f1=tf.keras.models.load_model(get_model_file(MODEL_F1_NAME)),
            model_f2=tf.keras.models.load_model(get_model_file(MODEL_F2_NAME)),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the objectives F1 and F2 for rows of material and deposition variables, shape (n, 70) -> (n, 2)."""
        x_scaled = self.scaler.transform(x)
        x_lstm = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        f1 = self.model_f1.predict(x_lstm, verbose=0).flatten()
        # F2 (reclaim volume deviation) only depends on the deposition, the material volume per time is constant
        f2 = self.model_f2.predict(x_lstm[:, :, MATERIAL_LENGTH:], verbose=0).flatten()
        return np.column_stack([f1, f2])
