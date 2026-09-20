"""Neural networks with Keras: a multilayer perceptron and the LSTM of the first experiments."""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bmh_ml.models.base import PerObjectiveModel

PREDICT_BATCH_SIZE = 8192


class KerasModel(PerObjectiveModel):
    """Shared training of the Keras models: standardized inputs, optional standardized labels, early stopping on the validation data."""

    defaults: ClassVar[dict[str, Any]] = {
        **PerObjectiveModel.defaults,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 200,
        "patience": 15,  # epochs without improvement of the validation loss, 0 trains all epochs
        "scale_targets": True,
    }

    def build(self, n_features: int):
        raise NotImplementedError

    def reshape(self, x: np.ndarray) -> np.ndarray:
        return x

    def fit_objective(self, x_train, y_train, x_val, y_val, seed):
        import keras
        from sklearn.preprocessing import StandardScaler

        params = self.params
        keras.utils.set_random_seed(seed)
        x_scaler = StandardScaler().fit(x_train)
        y_mean, y_std = (float(y_train.mean()), float(y_train.std())) if params["scale_targets"] else (0.0, 1.0)

        model = self.build(x_train.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="mean_squared_error")
        callbacks = []
        if params["patience"] > 0:
            callbacks.append(keras.callbacks.EarlyStopping(patience=params["patience"], restore_best_weights=True))
        history = model.fit(
            self.reshape(x_scaler.transform(x_train)),
            (y_train - y_mean) / y_std,
            validation_data=(self.reshape(x_scaler.transform(x_val)), (y_val - y_mean) / y_std),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        losses = history.history["val_loss"]
        info = {"epochs": float(len(losses)), "best_epoch": float(int(np.argmin(losses)) + 1), "val_loss": float(min(losses))}
        return (model, x_scaler, y_mean, y_std), info

    def predict_objective(self, estimator, x):
        model, x_scaler, y_mean, y_std = estimator
        prediction = model.predict(self.reshape(x_scaler.transform(x)), batch_size=PREDICT_BATCH_SIZE, verbose=0)
        return prediction.reshape(-1) * y_std + y_mean

    def save_objective(self, estimator, path: Path):
        import joblib

        model, x_scaler, y_mean, y_std = estimator
        path.mkdir(parents=True, exist_ok=True)
        model.save(path / "model.keras")
        joblib.dump({"x_scaler": x_scaler, "y_mean": y_mean, "y_std": y_std}, path / "scaling.joblib")

    def load_objective(self, path: Path):
        import joblib
        import keras

        scaling = joblib.load(path / "scaling.joblib")  # written by save_objective
        return keras.models.load_model(path / "model.keras"), scaling["x_scaler"], scaling["y_mean"], scaling["y_std"]


class MLPModel(KerasModel):
    """Fully connected network."""

    name = "mlp"
    defaults: ClassVar[dict[str, Any]] = {**KerasModel.defaults, "width": 256, "depth": 3, "activation": "relu", "dropout": 0.0}

    def build(self, n_features: int):
        import keras

        params = self.params
        layers: list[Any] = [keras.layers.Input(shape=(n_features,))]
        for _ in range(params["depth"]):
            layers.append(keras.layers.Dense(params["width"], activation=params["activation"]))
            if params["dropout"] > 0:
                layers.append(keras.layers.Dropout(params["dropout"]))
        layers.append(keras.layers.Dense(1))
        return keras.Sequential(layers)


class LegacyLSTMModel(KerasModel):
    """The model of the first experiments: an LSTM over a sequence of length one, which makes it a dense network with gates.

    It is kept as the reference the other models are compared with. It trains for a fixed number of epochs on the unscaled labels, like the
    original (`train_lstm_model`), except that it evaluates the validation data of the dataset.
    """

    name = "legacy_lstm"
    defaults: ClassVar[dict[str, Any]] = {
        **KerasModel.defaults,
        "units": 64,
        "dropout": 0.2,
        "batch_size": 32,
        "epochs": 100,
        "patience": 0,
        "scale_targets": False,
    }

    def reshape(self, x: np.ndarray) -> np.ndarray:
        return x.reshape((x.shape[0], 1, x.shape[1]))

    def build(self, n_features: int):
        import keras

        return keras.Sequential(
            [
                keras.layers.Input(shape=(1, n_features)),
                keras.layers.LSTM(self.params["units"], activation="relu"),
                keras.layers.Dropout(self.params["dropout"]),
                keras.layers.Dense(1),
            ]
        )
