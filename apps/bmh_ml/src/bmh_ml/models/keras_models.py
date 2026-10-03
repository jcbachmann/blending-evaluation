"""Neural networks with Keras: the multilayer perceptron, the LSTM of the first experiments, and the models that predict the reclaimed profile."""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bmh_ml.models.base import Model, PerObjectiveModel
from bmh_ml.parallel import get_thread_limit
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH

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
    """Predicts the reclaimed profile (volume and quality of every slice) together with F1 and F2, and computes the objectives from it.

    The profile carries far more information than the two objectives (120 numbers per simulation instead of 2), and F2 computed from a
    predicted profile is consistent by construction (never negative, a real stockpile shape): measured on S1-v2, it generalizes to
    optimized depositions far better than a direct prediction. F1 is the spread of a quality profile that varies only slightly, so the
    small errors of every slice add up; it comes from the direct output by default (`f1_source`). Both outputs share one network and are
    trained together; `objective_weight` weights each direct objective output against each profile value in the loss.

    The objectives of a mean profile are a little lower than the mean objectives of noisy simulations (the noise of the slices adds to
    their spread). For an objective computed from the profile, `noise_correction` adds a constant measured on the validation data, whose
    labels and profiles are both means of R simulations: c = R/(R-1) * mean(y^2 - F(profile)^2), and F = sqrt(F_profile^2 + c).

    `material_scaling` (any material only) standardizes each material curve by its own mean and standard deviation and predicts F1 and the
    quality profile in these units. The reclaimed quality mixes the material linearly (measured: F1(a m + b) = |a| F1(m) within the
    noise), so F1 relative to the material's spread is the same quantity for materials that differ only in scale or offset.
    """

    name = "profile_mlp"
    needs_profiles = True
    SOURCES = ("profile", "head")
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
        "f1_source": "head",
        "f2_source": "profile",
        "objective_weight": 30.0,
        "material_scaling": False,
    }

    def __init__(self, **params):
        super().__init__(**params)
        for key in ("f1_source", "f2_source"):
            if self.params[key] not in self.SOURCES:
                raise ValueError(f"{key} must be one of {self.SOURCES}, got {self.params[key]!r}")
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

    def fit(self, x_train, y_train, x_val, y_val, seed, profiles_train=None, profiles_val=None, val_repeats: int = 1):
        limit_keras_threads()
        import keras

        if profiles_train is None or profiles_val is None:
            raise ValueError("profile_mlp needs the reclaimed profiles of the training and validation data, build the bundle with --profiles")
        params = self.params
        if params["material_scaling"]:
            x_train, y_train, profiles_train = to_relative_material(x_train, y_train, profiles_train)
            x_val, y_val, profiles_val = to_relative_material(x_val, y_val, profiles_val)
        keras.utils.set_random_seed(seed)
        flat = [np.asarray(p, dtype=np.float32).reshape(len(p), -1) for p in (profiles_train, profiles_val)]
        targets, targets_val = (np.hstack([profile, np.asarray(y, dtype=np.float32)]) for profile, y in zip(flat, (y_train, y_val), strict=True))
        target_std = targets.std(axis=0)
        n_profile = flat[0].shape[1]
        self.scaling = {
            "x_mean": x_train.mean(axis=0),
            "x_std": np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0),
            "t_mean": targets.mean(axis=0),
            "t_std": np.where(target_std > 1e-6, target_std, 1.0),  # the slices at the ends are always empty
            "profile_shape": np.array(profiles_train.shape[1:]),
            "noise": np.zeros(2),
        }
        weights = np.ones(targets.shape[1], dtype=np.float32)
        weights[n_profile:] = params["objective_weight"]
        weights = keras.ops.convert_to_tensor(weights / weights.mean())

        def weighted_mse(y_true, y_pred):
            return keras.ops.mean(weights * keras.ops.square(y_true - y_pred), axis=-1)

        self.network = self.build(x_train.shape[1], targets.shape[1])
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss=weighted_mse)
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
            self.scaling["noise"] = get_noise_correction(profiles_val, y_val, val_repeats)
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

    def predict_outputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The predicted profiles, shape (n, 2, slices), and the directly predicted objectives, shape (n, 2), relative to the material
        with `material_scaling`."""
        if self.params["material_scaling"]:
            x = to_relative_material(x)[0]
        scaled = self.scale_inputs(x)
        if len(scaled) <= PREDICT_BATCH_SIZE:
            output = np.asarray(self.network(scaled, training=False))
        else:
            output = self.network.predict(scaled, batch_size=PREDICT_BATCH_SIZE, verbose=0)
        output = output * self.scaling["t_std"] + self.scaling["t_mean"]
        n_profile = int(np.prod(self.scaling["profile_shape"]))
        return output[:, :n_profile].reshape(len(x), *self.scaling["profile_shape"]), output[:, n_profile:]

    def predict_profiles(self, x: np.ndarray) -> np.ndarray:
        profiles = self.predict_outputs(x)[0]
        if self.params["material_scaling"]:
            mean, spread = get_material_spread(x)
            profiles[:, 1] = profiles[:, 1] * spread[:, None] + mean[:, None]
        return profiles

    def predict(self, x):
        from bmh_ml.evaluation.profile import get_profile_objectives

        profiles, direct = self.predict_outputs(x)
        from_profile = np.sqrt(get_profile_objectives(profiles) ** 2 + self.scaling["noise"])
        sources = (self.params["f1_source"], self.params["f2_source"])
        y = np.column_stack([from_profile[:, i] if source == "profile" else np.maximum(direct[:, i], 0.0) for i, source in enumerate(sources)])
        if self.params["material_scaling"]:
            y[:, 0] *= get_material_spread(x)[1]
        return y

    def save_files(self, directory: Path):
        self.network.save(directory / "model.keras")
        np.savez(directory / "scaling.npz", **self.scaling)

    def load_files(self, directory: Path):
        limit_keras_threads()
        import keras

        self.network = keras.models.load_model(directory / "model.keras", compile=False)  # the weighted loss is only needed to train
        with np.load(directory / "scaling.npz") as stored:
            self.scaling = {name: stored[name] for name in stored.files}


class MixingMLPModel(ProfileMLPModel):
    """F1 as a quadratic form in the material whose matrix depends on the deposition alone, F2 from a predicted volume profile.

    The reclaimed quality of each slice is a volume-weighted mean of the material that went into it, so the quality profile is W m, with
    mixing weights W that depend on the deposition (and the noise) but not on the material m (measured: q(m1 + m2) = q(m1) + q(m2) and
    F1(a m + b) = |a| F1(m) within the noise). The expected square of F1 is therefore exactly m_c' B m_c, the centered material in a
    positive semi-definite matrix B(deposition). A network on the 20 deposition values predicts a factor L of B (50 x `rank`), and
    F1 = |L' m_c|. A new material then needs no new data: only how a deposition mixes has to be learned, from all materials at once.

    The same network predicts the reclaimed volume of every slice, F2 is computed from it as in `profile_mlp`. With `quality_weight` > 0
    it also predicts the mixing weights W (a softmax over the material for each slice) and learns the quality profile W m, weighted by the
    reclaimed volume of the slice as in F1; this only adds supervision, F1 always comes from the quadratic form. The material enters
    every output linearly, so the model is exact in scale and offset of the material by construction.
    """

    name = "mixing_mlp"
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
        "rank": 16,
        "quality_weight": 1.0,
        "objective_weight": 30.0,
    }

    def __init__(self, **params):
        Model.__init__(self, **params)
        self.network = None
        self.scaling = {}

    def build(self, n_slices: int):
        import keras
        from keras import ops

        params = self.params
        inputs = keras.layers.Input(shape=(MATERIAL_LENGTH + DEPOSITION_LENGTH,))
        material, hidden = inputs[:, :MATERIAL_LENGTH], inputs[:, MATERIAL_LENGTH:]
        for _ in range(params["depth"]):
            hidden = keras.layers.Dense(params["width"], activation=params["activation"])(hidden)
            if params["dropout"] > 0:
                hidden = keras.layers.Dropout(params["dropout"])(hidden)
        outputs = [keras.layers.Dense(n_slices, name="volume")(hidden)]
        if params["quality_weight"] > 0:
            logits = keras.layers.Reshape((n_slices, MATERIAL_LENGTH))(keras.layers.Dense(n_slices * MATERIAL_LENGTH, name="mixing")(hidden))
            outputs.append(ops.einsum("bsm,bm->bs", ops.softmax(logits, axis=-1), material))
        factor = keras.layers.Reshape((MATERIAL_LENGTH, params["rank"]))(keras.layers.Dense(MATERIAL_LENGTH * params["rank"], name="factor")(hidden))
        projected = ops.einsum("bm,bmk->bk", material, factor)
        outputs.append(ops.sqrt(ops.sum(ops.square(projected), axis=-1, keepdims=True) + 1e-8))
        return keras.Model(inputs, ops.concatenate(outputs, axis=-1))

    def scale_inputs(self, x: np.ndarray) -> np.ndarray:
        """The deposition standardized, the material centered per curve and divided by one constant, which keeps it linear."""
        mean, _ = get_material_spread(x)
        material = (x[:, :MATERIAL_LENGTH] - mean[:, None]) / self.scaling["material_scale"]
        deposition = (x[:, MATERIAL_LENGTH:] - self.scaling["x_mean"]) / self.scaling["x_std"]
        return np.hstack([material, deposition]).astype(np.float32)

    def get_targets(self, x: np.ndarray, y: np.ndarray, profiles: np.ndarray) -> np.ndarray:
        """Volume per slice (standardized), the quality per slice relative to the material mean, its loss weights (the volume of the
        slice relative to the mean slice volume, 0 for empty slices) and F1, in the units of the network's outputs."""
        volume = np.asarray(profiles[:, 0], dtype=np.float32)
        parts = [(volume - self.scaling["v_mean"]) / self.scaling["v_std"]]
        if self.params["quality_weight"] > 0:
            mean, _ = get_material_spread(x)
            quality = (np.asarray(profiles[:, 1], dtype=np.float32) - mean[:, None]) / self.scaling["material_scale"]
            parts += [np.where(volume > 0, quality, 0.0), np.maximum(volume, 0.0) / volume.mean()]
        parts.append(np.asarray(y[:, :1], dtype=np.float32) / self.scaling["f1_scale"])
        return np.hstack(parts).astype(np.float32)

    def fit(self, x_train, y_train, x_val, y_val, seed, profiles_train=None, profiles_val=None, val_repeats: int = 1):
        limit_keras_threads()
        import keras
        from keras import ops

        if profiles_train is None or profiles_val is None:
            raise ValueError("mixing_mlp needs the reclaimed profiles of the training and validation data, build the bundle with --profiles")
        params = self.params
        keras.utils.set_random_seed(seed)
        deposition, volume = x_train[:, MATERIAL_LENGTH:], np.asarray(profiles_train[:, 0], dtype=float)
        n_slices = volume.shape[1]
        self.scaling = {
            "x_mean": deposition.mean(axis=0),
            "x_std": np.where(deposition.std(axis=0) > 0, deposition.std(axis=0), 1.0),
            "material_scale": np.array(get_material_spread(x_train)[1].mean()),
            "v_mean": volume.mean(axis=0),
            "v_std": np.where(volume.std(axis=0) > 1e-6, volume.std(axis=0), 1.0),  # the slices at the ends are always empty
            "f1_scale": np.array(float(np.std(y_train[:, 0]))),
            "profile_shape": np.array(profiles_train.shape[1:]),
            "noise": np.zeros(2),
        }
        with_quality, quality_weight, objective_weight = params["quality_weight"] > 0, params["quality_weight"], params["objective_weight"]
        total_weight = n_slices + (quality_weight * n_slices if with_quality else 0.0) + objective_weight

        def loss(y_true, y_pred):
            error = ops.sum(ops.square(y_true[:, :n_slices] - y_pred[:, :n_slices]), axis=-1)
            if with_quality:
                quality_true, slice_weight = y_true[:, n_slices : 2 * n_slices], y_true[:, 2 * n_slices : 3 * n_slices]
                error += quality_weight * ops.sum(slice_weight * ops.square(quality_true - y_pred[:, n_slices : 2 * n_slices]), axis=-1)
            error += objective_weight * ops.square(y_true[:, -1] - y_pred[:, -1])
            return error / total_weight

        self.network = self.build(n_slices)
        self.network.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss=loss)
        callbacks = [keras.callbacks.EarlyStopping(patience=params["patience"], restore_best_weights=True)] if params["patience"] > 0 else []
        history = self.network.fit(
            self.scale_inputs(x_train),
            self.get_targets(x_train, y_train, profiles_train),
            validation_data=(self.scale_inputs(x_val), self.get_targets(x_val, y_val, profiles_val)),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        if params["noise_correction"]:
            self.scaling["noise"] = get_noise_correction(profiles_val, y_val, val_repeats)
        losses = history.history["val_loss"]
        return {
            "epochs": float(len(losses)),
            "best_epoch": float(int(np.argmin(losses)) + 1),
            "val_loss": float(min(losses)),
            "F2/noise_correction": float(self.scaling["noise"][1]),
        }

    def predict_outputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The predicted profiles, shape (n, 2, slices), the quality NaN without the quality output, and F1, shape (n,)."""
        scaled = self.scale_inputs(x)
        if len(scaled) <= PREDICT_BATCH_SIZE:
            output = np.asarray(self.network(scaled, training=False))
        else:
            output = self.network.predict(scaled, batch_size=PREDICT_BATCH_SIZE, verbose=0)
        n_slices = int(self.scaling["profile_shape"][-1])
        volume = output[:, :n_slices] * self.scaling["v_std"] + self.scaling["v_mean"]
        if self.params["quality_weight"] > 0:
            quality = output[:, n_slices : 2 * n_slices] * self.scaling["material_scale"] + get_material_spread(x)[0][:, None]
        else:
            quality = np.full_like(volume, np.nan)
        return np.stack([volume, quality], axis=1), output[:, -1] * self.scaling["f1_scale"]

    def predict_profiles(self, x: np.ndarray) -> np.ndarray:
        return self.predict_outputs(x)[0]

    def predict(self, x):
        from bmh_ml.evaluation.profile import get_profile_objectives

        profiles, f1 = self.predict_outputs(x)
        profiles[:, 1] = 0.0  # F2 depends on the volume only
        f2 = np.sqrt(get_profile_objectives(profiles)[:, 1] ** 2 + self.scaling["noise"][1])
        return np.column_stack([f1, f2])

    def save_files(self, directory: Path):
        self.network.save_weights(directory / "model.weights.h5")  # the architecture comes from the parameters
        np.savez(directory / "scaling.npz", **self.scaling)

    def load_files(self, directory: Path):
        limit_keras_threads()
        with np.load(directory / "scaling.npz") as stored:
            self.scaling = {name: stored[name] for name in stored.files}
        self.network = self.build(int(self.scaling["profile_shape"][-1]))
        self.network.load_weights(directory / "model.weights.h5")


def get_material_spread(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the standard deviation of each material curve in features of the any-material scope."""
    if x.shape[1] != MATERIAL_LENGTH + DEPOSITION_LENGTH:
        raise ValueError(f"Scaling by the material needs the material in the features (scope S2), got {x.shape[1]} columns")
    material = x[:, :MATERIAL_LENGTH]
    return material.mean(axis=1), np.maximum(material.std(axis=1), 1e-6)


def to_relative_material(x: np.ndarray, y: np.ndarray | None = None, profiles: np.ndarray | None = None):
    """Features with each material curve standardized by its own mean and standard deviation, F1 and the quality profile in the same
    units. Empty slices keep the quality 0 the simulator gives them."""
    mean, spread = get_material_spread(x)
    x = x.copy()
    x[:, :MATERIAL_LENGTH] = (x[:, :MATERIAL_LENGTH] - mean[:, None]) / spread[:, None]
    if y is not None:
        y = np.array(y, dtype=float)
        y[:, 0] /= spread
    if profiles is not None:
        profiles = np.array(profiles, dtype=np.float32)
        quality = (profiles[:, 1] - mean[:, None]) / spread[:, None]
        profiles[:, 1] = np.where(profiles[:, 0] > 0, quality, 0.0)
    return x, y, profiles


def get_noise_correction(profiles: np.ndarray, y: np.ndarray, repeats: int) -> np.ndarray:
    """What the noise of the slices adds to the square of an objective, from data alone: the labels and the profiles of a dataset are both
    means of `repeats` simulations, E[y^2] = F(true)^2 + c and E[F(mean profile)^2] = F(true)^2 + c / repeats. Zero without repeats."""
    from bmh_ml.evaluation.profile import get_profile_objectives

    if repeats < 2:
        return np.zeros(2)
    difference = np.asarray(y, dtype=float) ** 2 - get_profile_objectives(np.asarray(profiles, dtype=float)) ** 2
    return np.maximum(difference.mean(axis=0) * repeats / (repeats - 1), 0.0)
