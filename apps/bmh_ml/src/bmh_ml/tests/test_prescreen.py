import numpy as np
import pytest

from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.settings import SIMULATOR_ENVIRONMENT_VARIABLE


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv("BMH_ML_TRACKING_URI", raising=False)
    return tmp_path / "store"


PARENTS = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
OFFSPRING = np.array([[0.5, 5.0], [3.0, 3.0], [1.5, 1.5], [5.0, 5.0]])  # extends the front, dominated, dominates a parent, dominated


def test_an_offspring_joins_the_front_if_nothing_dominates_it():
    pytest.importorskip("pymoo")
    from bmh_ml.prescreen import joins_front

    assert joins_front(PARENTS, OFFSPRING).tolist() == [True, False, True, False]


def test_the_preselection_takes_the_offspring_with_the_best_predicted_rank():
    pytest.importorskip("pymoo")
    from bmh_ml.prescreen import preselect

    rng = np.random.default_rng(0)
    assert sorted(preselect(PARENTS, OFFSPRING, 0.5, rng).tolist()) == [0, 2]  # a perfect model picks exactly the joining ones
    wrong = OFFSPRING[::-1]  # predictions that swap good and bad
    assert sorted(preselect(PARENTS, wrong, 0.5, rng).tolist()) == [1, 3]
    assert len(preselect(PARENTS, OFFSPRING, 0.01, rng)) == 1  # at least one is always simulated


def test_a_perfect_model_recalls_every_joining_offspring_and_noise_does_not():
    pytest.importorskip("pymoo")
    pytest.importorskip("scipy")
    from bmh_ml.prescreen import Generation, get_check_metrics

    rng = np.random.default_rng(1)
    generations = [Generation(rng.random((20, 20)), rng.random((20, 2)), rng.random((40, 20)), rng.random((40, 2))) for _ in range(30)]

    perfect = get_check_metrics(generations, [g.offspring_f for g in generations])
    noise = get_check_metrics(generations, [rng.random((40, 2)) for _ in generations])

    assert perfect["recall_at_50"] == pytest.approx(1.0)
    assert perfect["F1/offspring_spearman"] == pytest.approx(1.0)
    assert 0.3 < noise["recall_at_50"] < 0.7  # random picks recall about the share
    assert abs(noise["F2/offspring_spearman"]) < 0.2
    assert 0 < perfect["joining_share"] < 1


def test_the_refinement_records_parents_and_offspring_of_every_generation(tmp_path):
    pytest.importorskip("pymoo")
    from bmh_ml.prescreen import record_refinement, save_generations

    rng = np.random.default_rng(2)
    material, start = random_materials(1, rng), random_depositions(3, rng)

    generations = record_refinement(material, start, 8, 32, seed=1)

    assert len(generations) == 3  # 4 generations of 8, the first is the start
    assert all(g.parents_x.shape == (8, 20) and g.offspring_f.shape[1] == 2 for g in generations)
    assert any(np.allclose(row, start[0]) for row in generations[0].parents_x)  # the run started from the given solutions
    save_generations(tmp_path / "g.npz", [generations])
    with np.load(tmp_path / "g.npz") as stored:
        assert set(stored["role"].tolist()) == {0, 1}
        assert stored["x"].shape[1] == 20
