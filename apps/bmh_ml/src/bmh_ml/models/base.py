"""The interface between the pipeline and the models: a model predicts F1 and F2 for the inputs of a dataset scope.

The pipeline only uses `fit`, `predict`, `save` and `load`, so a framework or an algorithm is an implementation detail of a model class.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bmh_ml.settings import DEPOSITION_LENGTH

OBJECTIVES = ("F1", "F2")
MODEL_FILE = "model.json"


class Model(ABC):
    """Predicts both objectives, shape (n, 2), from features of shape (n, 20) for a fixed material or (n, 70) for any material."""

    name: ClassVar[str]
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params):
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise ValueError(f"Unknown parameters {unknown} for model '{self.name}', known: {sorted(self.defaults)}")
        self.params = {**self.defaults, **params}

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> dict[str, float]:
        """Trains the model. The validation data may be used to stop early, never to fit the weights. Returns numbers describing the training."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predictions of shape (n, 2)."""

    @abstractmethod
    def save_files(self, directory: Path) -> None:
        """Writes everything but the name and the parameters, which `save` stores."""

    @abstractmethod
    def load_files(self, directory: Path) -> None:
        """The counterpart of `save_files` on a model created with the stored parameters."""

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MODEL_FILE).write_text(json.dumps({"name": self.name, "params": self.params}, indent=2, sort_keys=True))
        self.save_files(directory)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, **self.params}


def get_objective_inputs(x: np.ndarray, objective: int, deposition_only_f2: bool) -> np.ndarray:
    """F2 (reclaim volume deviation) does not depend on the material, the volume per time is constant. Its models can use the deposition only."""
    if objective == 1 and deposition_only_f2:
        return x[:, -DEPOSITION_LENGTH:]
    return x


class PerObjectiveModel(Model):
    """A model made of one estimator for each objective, which is how F1 and F2 differ in their inputs and in their scale."""

    defaults: ClassVar[dict[str, Any]] = {"deposition_only_f2": True}

    def __init__(self, **params):
        super().__init__(**params)
        self.estimators: list[Any] = []

    @abstractmethod
    def fit_objective(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> tuple[Any, dict[str, float]]:
        """Trains the estimator of one objective from one column of the labels. Returns it and numbers describing the training."""

    @abstractmethod
    def predict_objective(self, estimator: Any, x: np.ndarray) -> np.ndarray:
        """Predictions of shape (n,)."""

    @abstractmethod
    def save_objective(self, estimator: Any, path: Path) -> None: ...

    @abstractmethod
    def load_objective(self, path: Path) -> Any: ...

    def fit(self, x_train, y_train, x_val, y_val, seed):
        self.estimators, info = [], {}
        for i, objective in enumerate(OBJECTIVES):
            deposition_only = self.params["deposition_only_f2"]
            estimator, objective_info = self.fit_objective(
                get_objective_inputs(x_train, i, deposition_only),
                y_train[:, i],
                get_objective_inputs(x_val, i, deposition_only),
                y_val[:, i],
                seed,
            )
            self.estimators.append(estimator)
            info.update({f"{objective}/{key}": value for key, value in objective_info.items()})
        return info

    def predict(self, x):
        columns = [
            self.predict_objective(estimator, get_objective_inputs(x, i, self.params["deposition_only_f2"])) for i, estimator in enumerate(self.estimators)
        ]
        return np.column_stack(columns)

    def save_files(self, directory):
        for objective, estimator in zip(OBJECTIVES, self.estimators, strict=True):
            self.save_objective(estimator, directory / objective)

    def load_files(self, directory):
        self.estimators = [self.load_objective(directory / objective) for objective in OBJECTIVES]
