import pytest

from ...helpers.simple import generate_material_deposition
from ..mathematical_blending_simulator import MathematicalBlendingSimulator
from ..smooth_blending_simulator import SmoothBlendingSimulator

BED_SIZE_X = 60
BED_SIZE_Z = 10
TOTAL_VOLUME = 500.0


def make_material_deposition():
    return generate_material_deposition(
        quality=[1.0, 3.0, 2.0, 5.0, 4.0],
        total_volume=TOTAL_VOLUME,
        x_position=[5.0, 55.0, 5.0, 55.0],
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
    )


@pytest.mark.parametrize(
    "sim",
    [
        MathematicalBlendingSimulator(bed_size_x=BED_SIZE_X, buffer_size=30),
        SmoothBlendingSimulator(bed_size_x=BED_SIZE_X, buffer_size=30, sigma_x=5.0),
    ],
    ids=["mathematical", "smooth"],
)
def test_python_simulators_stack_reclaim(sim):
    reclaimed = sim.stack_reclaim(make_material_deposition())

    assert reclaimed.data["volume"].sum() == pytest.approx(TOTAL_VOLUME)
    assert list(reclaimed.data.columns) == ["timestamp", "volume", "quality"]
