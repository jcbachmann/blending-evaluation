"""Several trained models as one model for the optimizer (phase 2B of the evaluation plan: a model alone in the optimizer).

The optimizer sees the mean of the members' predictions. With `kappa` > 0 it sees the mean plus `kappa` times their spread: solutions on
which the members disagree look worse, so the optimizer is kept away from regions where a single model's errors can be exploited (an
uncertainty penalty; both objectives are minimized). `f1_source` sets where members that predict a profile take F1 from ("head", the
direct output, or "profile", computed from the predicted quality profile with the noise correction measured at training).
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from bmh_ml.models.base import Model


class EnsembleModel(Model):
    name = "ensemble"
    defaults: ClassVar[dict[str, Any]] = {"members": [], "kappa": 0.0, "f1_source": None}

    def __init__(self, **params):
        super().__init__(**params)
        self.models: list[Model] = []

    @classmethod
    def from_runs(cls, run_ids: list[str], kappa: float = 0.0, f1_source: str | None = None) -> "EnsembleModel":
        """The models of training runs as one ensemble."""
        from bmh_ml.evaluate_run import load_run_model

        ensemble = cls(members=list(run_ids), kappa=kappa, f1_source=f1_source)
        ensemble.use_members([load_run_model(run_id) for run_id in run_ids])
        return ensemble

    def use_members(self, models: list[Model]) -> None:
        if self.params["f1_source"] is not None:
            for model in models:
                if "f1_source" not in model.params:
                    raise ValueError(f"f1_source is set, but the member model '{model.name}' has no such option")
                model.params["f1_source"] = self.params["f1_source"]
        self.models = models

    def fit(self, x_train, y_train, x_val, y_val, seed):
        raise NotImplementedError("an ensemble is made of trained models; train its members")

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.stack([model.predict(x) for model in self.models])
        mean = predictions.mean(axis=0)
        if self.params["kappa"] and len(self.models) > 1:
            mean = mean + self.params["kappa"] * predictions.std(axis=0, ddof=1)
        return mean

    def save_files(self, directory: Path) -> None:
        for i, model in enumerate(self.models):
            model.save(directory / f"member-{i}")

    def load_files(self, directory: Path) -> None:
        from bmh_ml.models.registry import load_model

        self.use_members([load_model(directory / f"member-{i}") for i in range(len(self.params["members"]))])
