"""The optimization problems of bmh_ml: the deposition variables for a fixed material, within the core of the stockpile, with the objectives
F1 and F2 from the simulator (`SimulatorProblem`) or from a model (`ModelProblem`); or the variables of a deposition encoding that lets the
stacker stop (`StopSimulatorProblem`, `bmh_ml.encoding`). NSGA-III runs on them with `experiment.make_nsga3`."""

import numpy as np
from pymoo.core.problem import Problem

from bmh_ml.datasets.manifest import make_features
from bmh_ml.datasets.simulate import simulate
from bmh_ml.encoding import StopEncoding
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, DEPOSITION_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN
from bmh_ml.simulation import get_objectives, reclaim_paths


class SimulatorProblem(Problem):
    """The objectives of single simulations, as an optimizer without a model sees them. A generation goes to the simulator in one call,
    `threads` simulations at the same time (0 for one per core): no worker processes, nothing to pickle."""

    def __init__(self, material: np.ndarray, threads: int = 1):
        self.material = np.atleast_2d(np.asarray(material, dtype=float))
        self.threads = threads
        super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

    def _evaluate(self, x, out, *_args, **_kwargs):
        out["F"] = simulate(self.material, x, 1, n_jobs=self.threads)[0]


class StopSimulatorProblem(Problem):
    """F1 and F2 of single simulations of the paths of a `StopEncoding`; with a stop fraction, of the pile of the stacked part of the
    material, F2 against the ideal stockpile of its volume."""

    def __init__(self, material: np.ndarray, encoding: StopEncoding | None = None, threads: int = 1):
        self.material = np.atleast_2d(np.asarray(material, dtype=float))
        self.encoding = encoding or StopEncoding()
        self.threads = threads
        low, high = self.encoding.bounds
        super().__init__(n_var=self.encoding.n_var, n_obj=2, xl=low, xu=high)

    def _evaluate(self, x, out, *_args, **_kwargs):
        paths, _ = self.encoding.decode(x)
        reclaimed = reclaim_paths(self.material, paths, BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, threads=self.threads)
        out["F"] = get_objectives(reclaimed["x"], reclaimed["volume"], reclaimed["quality"], X_MIN, X_MAX)


class ModelProblem(Problem):
    """The objectives a model predicts: anything with `predict(features)` for the features of its scope (`make_features`), e.g. a model
    of the registry or the first LSTM surrogate (scope S2, material followed by deposition)."""

    def __init__(self, model, scope: str, material: np.ndarray):
        self.model, self.scope = model, scope
        self.material = np.atleast_2d(np.asarray(material, dtype=float))
        super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

    def _evaluate(self, x, out, *_args, **_kwargs):
        out["F"] = self.model.predict(make_features(self.scope, self.material, x))
