import contextlib

import blending_simulator_lib
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from ...helpers.simple import generate_material_deposition
from ..bsl_blending_simulator import BslBlendingSimulator, simulate_batch
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


def test_bsl_stack_reclaim_array_matches_stack_reclaim():
    material_deposition = make_material_deposition()
    reclaimed = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, seed=7).stack_reclaim(material_deposition)
    data = material_deposition.data
    arrays = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, seed=7).stack_reclaim_array(data.to_numpy(), data.columns.to_list())

    assert set(arrays) == {"x", "volume", "quality"}
    for name, values in arrays.items():
        assert values.tolist() == reclaimed.data[name].tolist()


@pytest.mark.parametrize("batch_in_bindings", [True, False])
@pytest.mark.parametrize("lattice", [False, True])
def test_simulate_batch_matches_single_simulations(monkeypatch, batch_in_bindings, lattice):
    if not batch_in_bindings:
        monkeypatch.delattr(blending_simulator_lib, "simulate_batch", raising=False)
    data = make_material_deposition().data
    columns = data.columns.to_list()
    inputs = np.stack([data.to_numpy(), data.assign(x=60 - data["x"]).to_numpy()])

    with contextlib.nullcontext() if batch_in_bindings else pytest.warns(UserWarning, match="simulating serially"):
        batch = simulate_batch(inputs, columns, BED_SIZE_X, BED_SIZE_Z, ppm3=2.0, seed=11, lattice=lattice, threads=2)

    assert batch["volume"].shape == batch["quality"].shape == (2, len(batch["x"]))
    for i, input_data in enumerate(inputs):
        single = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, ppm3=2.0, seed=11 + i, lattice=lattice)
        expected = single.stack_reclaim_array(input_data, columns)
        np.testing.assert_array_equal(batch["x"], expected["x"])
        for name in ["volume", "quality"]:
            np.testing.assert_array_equal(batch[name][i], expected[name])


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


@pytest.mark.parametrize("ppm3", [1.0, 8.0, 64.0])
def test_bsl_default_reclaim_increment(ppm3):
    sim = BslBlendingSimulator(bed_size_x=10.0, bed_size_z=4.0, ppm3=ppm3, seed=0)
    sim.stack(0.0, 5.0, 2.0, 2.0, [1.0])  # noqa: PD013
    # Reclaimer positions from 0 m to the bed length in steps of one particle size
    size = ppm3 ** (-1.0 / 3.0)
    assert [row[0] for row in sim.reclaim()] == pytest.approx([i * size for i in range(round(10.0 / size) + 1)])


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


cone_available = pytest.mark.skipif(
    "cone" not in (blending_simulator_lib.simulate_batch.__doc__ or ""), reason="blending_simulator_lib built without the cone simulator"
)


@cone_available
def test_cone_conserves_the_volume_and_is_deterministic():
    data = make_material_deposition().data
    columns = data.columns.to_list()
    inputs = np.stack([data.to_numpy(), data.to_numpy()])
    batch = simulate_batch(inputs, columns, BED_SIZE_X, BED_SIZE_Z, ppm3=8.0, seed=1, cone=True, threads=2, reclaimincrement=1.0)
    np.testing.assert_allclose(batch["volume"].sum(axis=1), TOTAL_VOLUME)
    np.testing.assert_array_equal(batch["volume"][0], batch["volume"][1])  # the seed changes nothing
    single = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, ppm3=8.0, cone=True, reclaimincrement=1.0)
    expected = single.stack_reclaim_array(inputs[0], columns)
    np.testing.assert_array_equal(batch["volume"][0], expected["volume"])


@cone_available
def test_cone_needs_a_single_simulator():
    with pytest.raises(ValueError, match="one of the detailed, the lattice and the cone"):
        simulate_batch(np.zeros((1, 1, 4)), ["x", "z", "volume", "quality"], BED_SIZE_X, BED_SIZE_Z, lattice=True, cone=True)
