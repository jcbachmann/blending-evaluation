import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.simulate import build_dataset
from bmh_ml.datasets.store import Bundle, get_bundle_simulator, load_bundle, load_dataset, save_dataset
from bmh_ml.settings import (
    SIMULATOR_ENVIRONMENT_VARIABLE,
    UNRECORDED_SIMULATOR_VERSION,
    SimulatorSettings,
    get_installed_simulator_version,
    get_simulator_settings,
    use_simulator_settings,
)

SMALL = ["--train-size", "20", "--val-size", "6", "--test-size", "6", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1", "--stress-random", "1"]
S2_SMALL = ["--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path / "store"


def test_the_default_level_is_the_simulator_of_all_earlier_data():
    assert get_simulator_settings() == SimulatorSettings(ppm3=1.0, reclaim_increment=1.0)  # the simulator's own step, one particle, is 1 m as well
    assert SimulatorSettings.from_dict(None) == SimulatorSettings(version=UNRECORDED_SIMULATOR_VERSION, drops="point", ideal_stockpile="point")


def test_the_active_level_reaches_the_simulator(monkeypatch):
    from bmh_ml import simulation

    created = []
    original = simulation.simulate_batch

    def recording(*args, **kwargs):
        created.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation, "simulate_batch", recording)
    rng = np.random.default_rng(0)
    material, deposition = random_materials(1, rng)[0], random_depositions(1, rng)[0]
    use_simulator_settings(SimulatorSettings(ppm3=4.0))

    simulation.evaluate_sim(material, deposition, 59, 20, 2500)

    assert created[-1]["ppm3"] == 4.0
    assert created[-1]["reclaimincrement"] == 1.0  # fixed, so every level reclaims the same 60 slices


def get_level_in_worker(_: int) -> float:
    return get_simulator_settings().ppm3


def test_worker_processes_simulate_at_the_level_of_the_process_that_starts_them():
    from bmh_ml.parallel import run_parallel

    use_simulator_settings(SimulatorSettings(ppm3=16.0))

    assert run_parallel(get_level_in_worker, [(1,), (2,)], workers=2) == [16.0, 16.0]


def test_a_dataset_records_the_level_of_its_labels():
    rng = np.random.default_rng(1)
    use_simulator_settings(SimulatorSettings(ppm3=4.0))

    dataset = build_dataset("d", "random", 0, random_materials(3, rng), random_depositions(3, rng), n_jobs=1, settings={"scope": "S2"})

    assert dataset.settings == {"scope": "S2", "simulator": SimulatorSettings(ppm3=4.0).as_dict()}
    assert load_dataset(save_dataset(dataset)).settings["simulator"]["ppm3"] == 4.0


def test_a_bundle_has_one_level():
    rng = np.random.default_rng(2)
    ids = []
    for ppm3 in (1.0, 4.0):
        use_simulator_settings(SimulatorSettings(ppm3=ppm3))
        ids.append(save_dataset(build_dataset(f"d{ppm3}", "random", 0, random_materials(2, rng), random_depositions(2, rng), n_jobs=1)))

    assert get_bundle_simulator(Bundle("same", "S2", ids[1], ids[1], {})) == SimulatorSettings(ppm3=4.0)
    with pytest.raises(ValueError, match="mixes simulator settings"):
        get_bundle_simulator(Bundle("mixed", "S2", ids[0], ids[1], {}))


def test_a_bundle_is_built_at_a_level_and_can_label_the_test_sets_of_another_level():
    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", *S2_SMALL]))
    high = build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", *S2_SMALL, "--cross-from", "low", "--cross-sets", "T1", "T3"]))

    low = load_bundle("low")
    assert get_bundle_simulator(low) == SimulatorSettings()
    assert get_bundle_simulator(high) == SimulatorSettings(ppm3=4.0)
    same_inputs = load_dataset(low.tests["T1"]), load_dataset(high.tests["T1x"])
    assert np.array_equal(same_inputs[0].deposition, same_inputs[1].deposition)  # the same solutions, labeled at the other level
    assert not np.allclose(same_inputs[0].y, same_inputs[1].y)
    assert same_inputs[1].source["cross_from"] == "low"
    assert np.array_equal(load_dataset(low.tests["T1"]).deposition, load_dataset(high.tests["T1"]).deposition)  # same seed, same inputs
    with pytest.raises(ValueError, match="other simulator settings"):
        build_bundle.build_bundle(build_bundle.get_args(["--name", "mixed", "--ppm3", "4", "--tests-from", "low", *S2_SMALL]))


def test_the_chevron_reference_is_cached_per_level():
    from bmh_ml.evaluation.chevron import get_reference_key

    material = random_materials(1, np.random.default_rng(3))
    default_key = get_reference_key(material, 64)
    use_simulator_settings(SimulatorSettings(ppm3=64.0))

    assert get_reference_key(material, 64) != default_key
    use_simulator_settings(SimulatorSettings())
    assert get_reference_key(material, 64) == default_key
    use_simulator_settings(SimulatorSettings(version="2099.1"))
    assert get_reference_key(material, 64) != default_key  # a new simulator release simulates its references anew


def test_chevron_references_cached_before_the_release_was_recorded_keep_their_keys():
    import hashlib
    import json

    from bmh_ml.evaluation.chevron import get_reference_key
    from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, DEPOSITION_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN

    material = random_materials(1, np.random.default_rng(4))
    use_simulator_settings(SimulatorSettings.from_dict({"ppm3": 64.0, "reclaim_increment": 1.0}))

    # The key as it was computed from 2026-10-04 to 2026-10-14
    digest = hashlib.sha256(np.ascontiguousarray(material, dtype=np.float64).tobytes())
    digest.update(json.dumps([BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, DEPOSITION_LENGTH, X_MIN, X_MAX, 64]).encode())
    digest.update(json.dumps({"ppm3": 64.0, "reclaim_increment": 1.0}, sort_keys=True).encode())
    assert get_reference_key(material, 64) == digest.hexdigest()[:16]


def test_records_without_release_are_of_the_simulator_before_releases_were_recorded():
    settings = SimulatorSettings.from_dict({"ppm3": 4.0, "reclaim_increment": 1.0})  # a manifest of 2026-10-04 to 2026-10-14

    expected = SimulatorSettings(
        ppm3=4.0, reclaim_angle=45.0, eight_likelihood=0.87, version=UNRECORDED_SIMULATOR_VERSION, simulator="fast", drops="point", ideal_stockpile="point"
    )
    assert settings == expected
    assert SimulatorSettings().version == get_installed_simulator_version()  # new results are of the installed release
    assert SimulatorSettings.from_dict(SimulatorSettings(ppm3=16.0).as_dict()) == SimulatorSettings(ppm3=16.0)


def test_results_of_another_release_are_not_mixed_with_the_installed_one():
    from bmh_ml import simulation

    rng = np.random.default_rng(0)
    material, deposition = random_materials(1, rng)[0], random_depositions(1, rng)[0]
    use_simulator_settings(SimulatorSettings(version="1999.1"))  # e.g. the level of a bundle built with an older simulator

    with pytest.raises(RuntimeError, match="must not be mixed"):
        simulation.evaluate_sim(material, deposition, 59, 20, 2500)


def test_new_settings_drop_continuously_and_old_records_read_as_point_drops():
    assert (SimulatorSettings().simulator, SimulatorSettings().drops) == ("fast", "cells")
    old = SimulatorSettings.from_dict({"ppm3": 4.0, "reclaim_increment": 1.0, "version": "2026.2"})
    assert (old.simulator, old.drops) == ("fast", "point")
    with pytest.raises(ValueError, match="unknown simulator"):
        SimulatorSettings(simulator="bullet")
    with pytest.raises(ValueError, match="unknown drops"):
        SimulatorSettings(drops="lines")
    assert SimulatorSettings().ideal_stockpile == "integrated"
    assert old.ideal_stockpile == "point"
    with pytest.raises(ValueError, match="unknown ideal stockpile rule"):
        SimulatorSettings(ideal_stockpile="mean")


def test_chevron_references_of_point_drops_keep_their_keys_and_continuous_drops_get_new_ones():
    from bmh_ml.evaluation.chevron import get_reference_key

    material = np.linspace(5.0, 10.0, 50)
    use_simulator_settings(SimulatorSettings.from_dict({"ppm3": 16.0, "reclaim_increment": 1.0, "version": get_installed_simulator_version()}))
    key = get_reference_key(material, 4)
    use_simulator_settings(SimulatorSettings(ppm3=16.0, simulator="fast", drops="point", ideal_stockpile="point"))
    assert get_reference_key(material, 4) == key
    use_simulator_settings(SimulatorSettings(ppm3=16.0))
    assert get_reference_key(material, 4) != key
    use_simulator_settings(SimulatorSettings(ppm3=16.0, simulator="cone"))
    assert len({key, get_reference_key(material, 4)}) == 2
