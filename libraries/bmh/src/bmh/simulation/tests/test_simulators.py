import pytest
from pandas.testing import assert_frame_equal

from ...helpers.simple import generate_material_deposition
from ..bsl_blending_simulator import BslBlendingSimulator
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


def test_bsl_get_heights_position():
    sim = BslBlendingSimulator(bed_size_x=4, bed_size_z=3, reclaimangle=90.0, eight=0.0)
    # A single particle on the empty bed stays in the cell it is dropped on
    sim.stack(0.0, 2.0, 1.0, 1.0, [1.0])  # noqa: PD013

    heights = sim.get_heights()

    assert len(heights) == 3
    for z, row in enumerate(heights):
        assert len(row) == 4
        for x, height in enumerate(row):
            assert height == pytest.approx(1.0 if (x, z) == (2, 1) else 0.0)


def bsl_stack_reclaim_with_seed(seed: int):
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, eight=0.5, seed=seed)
    return sim.stack_reclaim(make_material_deposition()).data


def test_bsl_seed_repeatable():
    assert_frame_equal(bsl_stack_reclaim_with_seed(42), bsl_stack_reclaim_with_seed(42))


def test_bsl_seed_different():
    assert not bsl_stack_reclaim_with_seed(1).equals(bsl_stack_reclaim_with_seed(2))


def test_bsl_stack_reclaim():
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, seed=0)
    sim.stack(0.0, 20.0, 5.0, 100.0, [1.0, 10.0])  # noqa: PD013
    sim.stack(1.0, 40.0, 5.0, 100.0, [3.0, 30.0])  # noqa: PD013

    reclaimed = sim.reclaim()

    volume = sum(v for _, v, _ in reclaimed)
    assert volume == pytest.approx(200.0)
    for i, expected in enumerate([2.0, 20.0]):
        assert sum(v * q[i] for _, v, q in reclaimed) / volume == pytest.approx(expected)
    assert all(len(q) == 2 for _, _, q in reclaimed)


def test_bsl_reclaim_parameter_names():
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, seed=0)
    sim.stack(0.0, 20.0, 5.0, 10.0, [1.0, 2.0])  # noqa: PD013

    assert list(sim.bsl.reclaim().keys()) == ["x", "volume", "p_1", "p_2"]


def test_bsl_parameter_count_mismatch():
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z)
    sim.stack(0.0, 20.0, 5.0, 10.0, [1.0])  # noqa: PD013

    with pytest.raises(ValueError, match="expected 1 parameters"):
        sim.stack(1.0, 20.0, 5.0, 10.0, [1.0, 2.0])  # noqa: PD013


def test_bsl_stack_list_twice():
    material_deposition = make_material_deposition()
    data = material_deposition.data
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, seed=0)

    half = len(data) // 2
    sim.bsl.stack_list(data.iloc[:half].to_numpy(), data.columns.to_list())
    sim.bsl.stack_list(data.iloc[half:].to_numpy(), data.columns.to_list())
    reclaimed = sim.bsl.reclaim()

    parameter_columns = [c for c in data.columns if c not in ("timestamp", "x", "z", "volume")]
    assert list(reclaimed.keys()) == ["x", "volume", *parameter_columns]


def test_bsl_stack_list_different_columns():
    data = make_material_deposition().data
    sim = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z)
    sim.bsl.stack_list(data.to_numpy(), data.columns.to_list())

    renamed = data.rename(columns={"quality": "other"})
    with pytest.raises(ValueError, match="parameter columns differ"):
        sim.bsl.stack_list(renamed.to_numpy(), renamed.columns.to_list())


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
