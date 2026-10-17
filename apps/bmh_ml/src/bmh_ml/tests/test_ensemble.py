import numpy as np
import pytest

from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.manifest import make_features
from bmh_ml.settings import SIMULATOR_ENVIRONMENT_VARIABLE


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv("BMH_ML_TRACKING_URI", raising=False)
    return tmp_path / "store"


def ridge_members(count: int, seed: int):
    pytest.importorskip("sklearn")
    from bmh_ml.models.registry import create_model

    rng = np.random.default_rng(seed)
    x = make_features("S2", random_materials(40, rng), random_depositions(40, rng))
    members = []
    for _ in range(count):
        model = create_model("ridge")
        y = rng.random((40, 2))  # different members
        model.fit(x, y, x, y, seed=0)
        members.append(model)
    return x, members


def test_the_ensemble_predicts_the_mean_and_with_kappa_adds_the_spread(tmp_path):
    from bmh_ml.models.ensemble import EnsembleModel
    from bmh_ml.models.registry import load_model

    x, members = ridge_members(3, 0)
    predictions = np.stack([member.predict(x) for member in members])
    plain = EnsembleModel(members=["a", "b", "c"])
    plain.use_members(members)
    cautious = EnsembleModel(members=["a", "b", "c"], kappa=1.5)
    cautious.use_members(members)

    assert np.allclose(plain.predict(x), predictions.mean(axis=0))
    assert np.allclose(cautious.predict(x), predictions.mean(axis=0) + 1.5 * predictions.std(axis=0, ddof=1))
    cautious.save(tmp_path / "ensemble")
    assert np.allclose(load_model(tmp_path / "ensemble").predict(x), cautious.predict(x))  # workers of the transfer test load it from disk


def test_f1_source_is_passed_to_the_members_and_rejected_for_models_without_it():
    from bmh_ml.models.ensemble import EnsembleModel
    from bmh_ml.models.registry import create_model

    profile_model = create_model("profile_mlp", material_scaling=True)
    ensemble = EnsembleModel(members=["a"], f1_source="profile")
    ensemble.use_members([profile_model])
    assert profile_model.params["f1_source"] == "profile"

    _, members = ridge_members(1, 1)
    with pytest.raises(ValueError, match="no such option"):
        EnsembleModel(members=["a"], f1_source="profile").use_members(members)
    with pytest.raises(NotImplementedError):
        ensemble.fit(None, None, None, None, 0)
