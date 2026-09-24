import numpy as np
import pytest

from bmh_ml.datasets.generators import random_materials
from bmh_ml.evaluation import chevron as chevron_module
from bmh_ml.evaluation.chevron import chevron_deposition, get_chevron_hypervolume, get_chevron_metrics, get_chevron_objectives
from bmh_ml.settings import DEPOSITION_LENGTH, X_MAX, X_MIN


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def test_the_chevron_alternates_between_the_ends_of_the_bed():
    deposition = chevron_deposition()

    assert deposition.shape == (1, DEPOSITION_LENGTH)
    assert list(deposition[0, :4]) == [X_MIN, X_MAX, X_MIN, X_MAX]


def test_only_what_beats_chevron_in_both_objectives_counts():
    assert get_chevron_hypervolume(np.array([[0.5, 0.5]])) == pytest.approx(0.25)
    assert get_chevron_hypervolume(np.array([[0.5, 1.2], [1.1, 0.1]])) == 0
    assert get_chevron_hypervolume(np.array([[0.5, 0.5], [0.8, 0.2]])) == pytest.approx(0.25 + 0.16 - 0.1)  # two boxes and their overlap

    metrics = get_chevron_metrics(np.array([[0.1, 20.0], [0.4, 5.0], [0.6, 4.0]]), chevron=np.array([0.5, 10.0]))

    assert metrics["chevron_beaten_rate"] == pytest.approx(1 / 3)
    assert metrics["chevron_best_F1"] == pytest.approx(0.2)
    assert metrics["chevron_best_F2"] == pytest.approx(0.4)
    assert metrics["chevron_hv"] == pytest.approx((1 - 0.8) * (1 - 0.5))


def test_the_reference_is_simulated_once_per_material_and_then_read(monkeypatch):
    material = random_materials(1, np.random.default_rng(0))
    calls = []
    original = chevron_module.simulate

    def counting_simulate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chevron_module, "simulate", counting_simulate)

    first = get_chevron_objectives(material, repeats=3, n_jobs=1)
    second = get_chevron_objectives(material, repeats=3, n_jobs=1)

    assert len(calls) == 1
    assert np.array_equal(first, second)
    assert np.all(first > 0)
    get_chevron_objectives(random_materials(1, np.random.default_rng(1)), repeats=3, n_jobs=1)
    assert len(calls) == 2
