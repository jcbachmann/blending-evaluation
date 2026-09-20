import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from ...benchmark.material_deposition import Material
from ..math import stdev
from ..reclaimed_material_evaluator import ReclaimedMaterialEvaluator, get_ideal_reclaimed_material
from ..stockpile_math import get_ideal_stockpile_volumes, get_stockpile_height, get_stockpile_slice_volume

X_MIN = 10.0
X_MAX = 49.0


def make_reclaimed_material(seed: int = 0, slices: int = 60) -> Material:
    rng = np.random.default_rng(seed)
    return Material.from_data(
        pd.DataFrame(
            {
                "x": np.linspace(0.5, X_MAX + X_MIN, slices),
                "volume": rng.uniform(20.0, 50.0, slices),
                "timestamp": np.linspace(0.0, 86400.0, slices),
                "quality": rng.uniform(5.0, 10.0, slices),
                "other": rng.uniform(0.0, 1.0, slices),
            }
        )
    )


def reference_ideal_volumes(reclaimed: Material) -> np.ndarray:
    """The row-wise computation the vectorized function replaced, kept as an independent reference."""
    ideal_df = reclaimed.data.copy()
    height = get_stockpile_height(volume=ideal_df["volume"].sum(), core_length=X_MAX - X_MIN)
    ideal_df["x_diff"] = (ideal_df["x"] - ideal_df["x"].shift(1)).fillna(0.0)
    return ideal_df.apply(
        lambda row: get_stockpile_slice_volume(x=row["x"], core_length=X_MAX - X_MIN, height=height, x_min=X_MIN, x_diff=row["x_diff"]),
        axis=1,
    ).to_numpy()


def test_ideal_volumes_equal_the_rowwise_reference():
    for seed in range(5):
        reclaimed = make_reclaimed_material(seed)

        ideal = get_ideal_stockpile_volumes(reclaimed.data["x"].to_numpy(), reclaimed.data["volume"].sum(), X_MIN, X_MAX)

        assert_array_equal(ideal, reference_ideal_volumes(reclaimed))


def test_volume_stdev_equals_the_rowwise_reference():
    for seed in range(5):
        reclaimed = make_reclaimed_material(seed)
        expected = stdev(reference_ideal_volumes(reclaimed) - reclaimed.data["volume"].to_numpy())

        assert ReclaimedMaterialEvaluator(reclaimed=reclaimed, x_min=X_MIN, x_max=X_MAX).get_volume_stdev() == expected


def test_ideal_reclaimed_material_is_homogeneous_with_the_ideal_volume_curve():
    reclaimed = make_reclaimed_material()
    original = reclaimed.data.copy()

    ideal = get_ideal_reclaimed_material(reclaimed, X_MIN, X_MAX)

    for parameter in ("quality", "other"):
        expected = np.average(original[parameter], weights=original["volume"])
        assert_array_equal(ideal.data[parameter].to_numpy(), np.full(len(original), expected))
    assert_array_equal(ideal.data["volume"].to_numpy(), reference_ideal_volumes(reclaimed))
    assert_array_equal(ideal.data["x"].to_numpy(), original["x"].to_numpy())
    assert_array_equal(ideal.data["timestamp"].to_numpy(), original["timestamp"].to_numpy())


def test_ideal_reclaimed_material_adds_no_columns_and_leaves_the_input_untouched():
    reclaimed = make_reclaimed_material()
    original = reclaimed.data.copy()

    ideal = get_ideal_reclaimed_material(reclaimed, X_MIN, X_MAX)

    # A stray column like x_diff would count as a parameter of the ideal material
    assert list(ideal.data.columns) == list(original.columns)
    assert sorted(ideal.get_parameter_columns()) == ["other", "quality"]
    pd.testing.assert_frame_equal(reclaimed.data, original)
