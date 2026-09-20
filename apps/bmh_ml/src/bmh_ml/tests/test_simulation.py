import math

import numpy as np
import pandas as pd
import pytest
from bmh.benchmark.material_deposition import Material
from bmh.helpers.math import stdev
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator

from bmh_ml.simulation import evaluate_sim, generate_deposition, generate_material

BED_SIZE_X = 59.0
BED_SIZE_Z = 20.0
X_MIN = 0.5 * BED_SIZE_Z
X_MAX = BED_SIZE_X - X_MIN


def make_reclaimed_material() -> Material:
    rng = np.random.default_rng(3)
    slices = 60
    return Material.from_data(
        pd.DataFrame(
            {
                "x": np.linspace(0.5, BED_SIZE_X, slices),
                "volume": rng.uniform(20.0, 50.0, slices),
                "quality": rng.uniform(5.0, 10.0, slices),
                "timestamp": np.linspace(0.0, 86400.0, slices),
            }
        )
    )


def reference_f1(reclaimed: Material) -> float:
    quality, volume = reclaimed.data["quality"], reclaimed.data["volume"]
    average = np.average(quality, weights=volume)
    return math.sqrt(np.average((quality - average) ** 2, weights=volume))


def reference_f2(reclaimed: Material) -> float:
    ideal_df = reclaimed.data.copy()
    height = get_stockpile_height(volume=ideal_df["volume"].sum(), core_length=X_MAX - X_MIN)
    ideal_df["x_diff"] = (ideal_df["x"] - ideal_df["x"].shift(1)).fillna(0.0)
    ideal_df["volume"] = ideal_df.apply(
        lambda row: get_stockpile_slice_volume(x=row["x"], core_length=X_MAX - X_MIN, height=height, x_min=X_MIN, x_diff=row["x_diff"]), axis=1
    )
    return stdev((ideal_df["volume"] - reclaimed.data["volume"]).to_numpy())


def test_evaluate_sim_returns_homogenization_and_volume_deviation(monkeypatch):
    # The simulation is stochastic, so it is replaced by a fixed reclaimed material to test the evaluation
    reclaimed = make_reclaimed_material()
    monkeypatch.setattr(BslBlendingSimulator, "stack_reclaim", lambda _self, _material_deposition: reclaimed)

    f1, f2 = evaluate_sim(
        material_variables=np.linspace(5, 10, 50),
        deposition_variables=np.linspace(X_MIN, X_MAX, 20),
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        total_volume=2500.0,
    )

    assert f1 == pytest.approx(reference_f1(reclaimed), rel=1e-12)
    assert f2 == pytest.approx(reference_f2(reclaimed), rel=1e-12)


def test_evaluate_sim_of_the_real_simulation_is_plausible():
    f1, f2 = evaluate_sim(
        material_variables=np.linspace(5, 10, 50),
        deposition_variables=np.linspace(X_MIN, X_MAX, 20),
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        total_volume=2500.0,
    )

    # The simulation is stochastic, only the order of magnitude is stable (about 1.4 and 16 in repeated runs)
    assert 0.5 < f1 < 3.0
    assert 5.0 < f2 < 40.0


def test_generated_material_and_deposition_have_the_requested_length():
    material = generate_material(list(np.linspace(5, 10, 50)), total_volume=2500.0)
    deposition = generate_deposition(list(np.linspace(X_MIN, X_MAX, 20)), bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z)

    assert len(material.data) == 50
    assert material.data["volume"].sum() == pytest.approx(2500.0)
    assert len(deposition.data) == 20
