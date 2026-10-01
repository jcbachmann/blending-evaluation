"""Neural networks with Keras: a multilayer perceptron and the LSTM of the first experiments."""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bmh_ml.models.base import Model, PerObjectiveModel
from bmh_ml.parallel import get_thread_limit

PREDICT_BATCH_SIZE = 8192


def limit_keras_threads() -> None:
    """Applies the thread limit of the process (see `parallel`) to TensorFlow, which only accepts it before its first operation."""
    limit = get_thread_limit()
    if limit:
        import tensorflow as tf

        try:
            tf.config.threading.set_intra_op_parallelism_threads(limit)
            tf.config.threading.set_inter_op_parallelism_threads(min(limit, 2))
        except RuntimeError:  # TensorFlow already runs, the limit of the environment variables applies
            pass


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
        limit_keras_threads()
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
        x = self.reshape(x_scaler.transform(x))
        if len(x) <= PREDICT_BATCH_SIZE:
            # `predict` sets up a data pipeline on every call, which made a batch of 100 (one generation of an optimizer) about 25 times slower
            prediction = np.asarray(model(x, training=False))
        else:
            prediction = model.predict(x, batch_size=PREDICT_BATCH_SIZE, verbose=0)
        return prediction.reshape(-1) * y_std + y_mean

    def save_objective(self, estimator, path: Path):
        import joblib

        model, x_scaler, y_mean, y_std = estimator
        path.mkdir(parents=True, exist_ok=True)
        model.save(path / "model.keras")
        joblib.dump({"x_scaler": x_scaler, "y_mean": y_mean, "y_std": y_std}, path / "scaling.joblib")

    def load_objective(self, path: Path):
        limit_keras_threads()
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


class ProfileMLPModel(Model):
    """Predicts the reclaimed profile (volume and quality of every slice) and computes F1 and F2 from it, as the simulator does.

    The profile carries far more information than the two objectives (120 numbers per simulation instead of 2), and objectives computed
    from a profile are consistent by construction: never negative, and F1 and F2 belong to one stockpile. The objectives of a mean
    profile are a little lower than the mean of the objectives of noisy simulations (the noise of the slices adds to their spread), so
    `noise_correction` adds a constant learned from the training labels: F = sqrt(F_profile^2 + c).
    """

    name = "profile_mlp"
    needs_profiles = True
    defaults: ClassVar[dict[str, Any]] = {
        "width": 512,
        "depth": 4,
        "activation": "relu",
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 512,
        "epochs": 200,
        "patience": 15,
        "noise_correction": True,
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.network = None
        self.scaling: dict[str, np.ndarray] = {}

    def build(self, n_features: int, n_outputs: int):
        import keras

        params = self.params
        layers: list[Any] = [keras.layers.Input(shape=(n_features,))]
        for _ in range(params["depth"]):
            layers.append(keras.layers.Dense(params["width"], activation=params["activation"]))
            if params["dropout"] > 0:
                layers.append(keras.layers.Dropout(params["dropout"]))
        layers.append(keras.layers.Dense(n_outputs))
        return keras.Sequential(layers)

    def fit(self, x_train, y_train, x_val, y_val, seed, profiles_train=None, profiles_val=None):  # noqa: ARG002 - y_val: the profiles stop the training
        limit_keras_threads()
        import keras

        from bmh_ml.evaluation.profile import get_profile_objectives

        if profiles_train is None or profiles_val is None:
            raise ValueError("profile_mlp needs the reclaimed profiles of the training and validation data, build the bundle with --profiles")
        params = self.params
        keras.utils.set_random_seed(seed)
        targets, targets_val = (np.asarray(p, dtype=np.float32).reshape(len(p), -1) for p in (profiles_train, profiles_val))
        target_std = targets.std(axis=0)
        self.scaling = {
            "x_mean": x_train.mean(axis=0),
            "x_std": np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0),
            "t_mean": targets.mean(axis=0),
            "t_std": np.where(target_std > 1e-6, target_std, 1.0),  # the slices at the ends are always empty
            "profile_shape": np.array(profiles_train.shape[1:]),
            "noise": np.zeros(2),
        }
        self.network = self.build(x_train.shape[1], targets.shape[1])
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="mean_squared_error")
        callbacks = [keras.callbacks.EarlyStopping(patience=params["patience"], restore_best_weights=True)] if params["patience"] > 0 else []
        history = self.network.fit(
            self.scale_inputs(x_train),
            (targets - self.scaling["t_mean"]) / self.scaling["t_std"],
            validation_data=(self.scale_inputs(x_val), (targets_val - self.scaling["t_mean"]) / self.scaling["t_std"]),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        if params["noise_correction"]:
            uncorrected = get_profile_objectives(self.predict_profiles(x_train))
            self.scaling["noise"] = np.maximum((np.asarray(y_train) ** 2 - uncorrected**2).mean(axis=0), 0.0)
        losses = history.history["val_loss"]
        return {
            "epochs": float(len(losses)),
            "best_epoch": float(int(np.argmin(losses)) + 1),
            "val_loss": float(min(losses)),
            "F1/noise_correction": float(self.scaling["noise"][0]),
            "F2/noise_correction": float(self.scaling["noise"][1]),
        }

    def scale_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.scaling["x_mean"]) / self.scaling["x_std"]

    def predict_profiles(self, x: np.ndarray) -> np.ndarray:
        """The predicted reclaimed profiles, shape (n, 2, slices)."""
        scaled = self.scale_inputs(x)
        if len(scaled) <= PREDICT_BATCH_SIZE:
            output = np.asarray(self.network(scaled, training=False))
        else:
            output = self.network.predict(scaled, batch_size=PREDICT_BATCH_SIZE, verbose=0)
        output = output * self.scaling["t_std"] + self.scaling["t_mean"]
        return output.reshape(len(x), *self.scaling["profile_shape"])

    def predict(self, x):
        from bmh_ml.evaluation.profile import get_profile_objectives

        objectives = get_profile_objectives(self.predict_profiles(x))
        return np.sqrt(objectives**2 + self.scaling["noise"])

    def save_files(self, directory: Path):
        self.network.save(directory / "model.keras")
        np.savez(directory / "scaling.npz", **self.scaling)

    def load_files(self, directory: Path):
        limit_keras_threads()
        import keras

        self.network = keras.models.load_model(directory / "model.keras")
        with np.load(directory / "scaling.npz") as stored:
            self.scaling = {name: stored[name] for name in stored.files}
