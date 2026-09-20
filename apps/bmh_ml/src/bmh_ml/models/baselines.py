"""Models without a neural network: the mean, a linear model and gradient boosting."""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bmh_ml.models.base import Model, PerObjectiveModel


class MeanModel(Model):
    """Predicts the mean of the training labels. Every model has to be better than this."""

    name = "mean"

    def __init__(self, **params):
        super().__init__(**params)
        self.mean = np.zeros(2)

    def fit(self, x_train, y_train, x_val, y_val, seed):  # noqa: ARG002 - the interface of all models
        self.mean = y_train.mean(axis=0)
        return {}

    def predict(self, x):
        return np.tile(self.mean, (len(x), 1))

    def save_files(self, directory: Path):
        np.save(directory / "mean.npy", self.mean)

    def load_files(self, directory: Path):
        self.mean = np.load(directory / "mean.npy")


class RidgeModel(PerObjectiveModel):
    """Linear regression on standardized inputs with an L2 penalty."""

    name = "ridge"
    defaults: ClassVar[dict[str, Any]] = {**PerObjectiveModel.defaults, "alpha": 1.0}

    def fit_objective(self, x_train, y_train, x_val, y_val, seed):  # noqa: ARG002 - the interface of all models
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=self.params["alpha"])).fit(x_train, y_train), {}

    def predict_objective(self, estimator, x):
        return estimator.predict(x)

    def save_objective(self, estimator, path: Path):
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, path / "estimator.joblib")

    def load_objective(self, path: Path):
        import joblib

        return joblib.load(path / "estimator.joblib")  # written by save_objective


class LightGBMModel(PerObjectiveModel):
    """Gradient boosted trees, stopped early on the validation data."""

    name = "lightgbm"
    defaults: ClassVar[dict[str, Any]] = {
        **PerObjectiveModel.defaults,
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "n_jobs": 8,  # all 16 threads of the development machine were 10 to 40 times slower than 8, so the default is not "all"
    }

    def fit_objective(self, x_train, y_train, x_val, y_val, seed):
        import lightgbm

        params = self.params
        regressor = lightgbm.LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            n_jobs=min(params["n_jobs"], os.cpu_count() or 1),
            random_state=seed,
            verbose=-1,
        )
        regressor.fit(
            x_train,
            y_train,
            eval_X=x_val,
            eval_y=y_val,
            callbacks=[lightgbm.early_stopping(params["early_stopping_rounds"], verbose=False)],
        )
        return regressor.booster_, {"trees": float(regressor.booster_.num_trees())}

    def predict_objective(self, estimator, x):
        return estimator.predict(x)

    def save_objective(self, estimator, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        estimator.save_model(str(path / "booster.txt"))

    def load_objective(self, path: Path):
        import lightgbm

        return lightgbm.Booster(model_file=str(path / "booster.txt"))
