import base64
import json
import math

import numpy as np
import pytest

from ..particle_simulators import GridSimulator, LatticeSimulator, ParticleBuffer
from ..run_comparison import run_one
from ..scenarios import DURATION, X_MIN, get_start_cone_volume, make_material_deposition


def test_particle_buffer():
    buffer = ParticleBuffer(1.0)
    buffer.push(0.5, [2.0])
    assert list(buffer.pop_all()) == []
    buffer.push(2.0, [4.0])
    particles = list(buffer.pop_all())
    assert particles == [pytest.approx([3.6]), pytest.approx([3.6])]
    assert buffer.volume == pytest.approx(0.5)


def test_grid_single_particle():
    sim = GridSimulator(4.0, 3.0, ppm3=1.0, reclaim_angle=90.0)
    sim.stack(0.0, 2.0, 1.0, 1.0, [1.0])
    assert sim.positions == [pytest.approx((2.5, 0.0, 1.5))]
    reclaimed = sim.reclaim()
    assert [v for _, v, _ in reclaimed] == [0.0, 0.0, 1.0, 0.0]


def test_grid_particle_wider_than_slice():
    sim = GridSimulator(6.0, 4.0, ppm3=0.125, reclaim_angle=90.0)
    # Like the C++ code, x = 2 m rounds to cell 1, which spans x from 2 m to 4 m
    sim.stack(0.0, 2.0, 2.0, 8.0, [1.0])
    # The 2 m particle splits its 8 m³ over two 1 m slices
    assert [v for _, v, _ in sim.reclaim()] == pytest.approx([0.0, 0.0, 4.0, 4.0, 0.0, 0.0])


def test_grid_seed_repeatable():
    def stack(seed):
        sim = GridSimulator(20.0, 20.0, ppm3=1.0, eight_likelihood=0.5, seed=seed)
        sim.stack(0.0, 10.0, 10.0, 300.0, [1.0])
        return sim.positions

    assert stack(3) == stack(3)
    assert stack(3) != stack(4)


@pytest.mark.parametrize("angle", [None, 45.0, 30.0])
def test_lattice_particle_volume(angle):
    sim = LatticeSimulator(20.0, 20.0, ppm3=8.0, angle_of_repose=angle)
    # Close-packed spheres fill a / sqrt(2) cubed each, compressed by the vertical scale
    assert sim.a**3 * sim.vertical_scale / np.sqrt(2.0) == pytest.approx(1.0 / 8.0)


def test_lattice_angle_of_repose():
    scale = math.tan(math.radians(45.0)) / math.tan(math.radians(LatticeSimulator.NATIVE_ANGLE_OF_REPOSE))
    compressed = LatticeSimulator(20.0, 20.0, ppm3=8.0, angle_of_repose=45.0)
    # Same horizontal spacing: the uncompressed particles have the volume of the compressed ones divided by the scale
    native = LatticeSimulator(20.0, 20.0, ppm3=8.0 * scale, angle_of_repose=None)
    assert native.a == pytest.approx(compressed.a)
    compressed.stack(0.0, 10.0, 10.0, 1600 / 8.0, [1.0])
    native.stack(0.0, 10.0, 10.0, 1600 / (8.0 * scale), [1.0])
    # The same sites fill, only the layers are lower, so that slopes of the native angle become 45°
    assert len(native.positions) == len(compressed.positions) == 1600
    assert native.heights == compressed.heights
    assert compressed.layer_distance == pytest.approx(scale * native.layer_distance)


def test_lattice_particles_supported():
    sim = LatticeSimulator(10.0, 10.0, ppm3=8.0, angle_of_repose=None)
    sim.stack(0.0, 5.0, 5.0, 40.0, [1.0])

    assert len(sim.positions) == 320
    assert sim.lost_particles == 0
    # Every particle above the ground rests on three filled sites
    for (xi, zi), height in sim.heights.items():
        for yi in range(1, height):
            assert all(sim.is_filled(*s) for s in sim.get_spaces_below(xi, yi, zi))
    # Spheres do not overlap
    p = np.asarray(sim.positions)
    distances = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1) + np.eye(len(p)) * 1e9
    assert distances.min() == pytest.approx(sim.a, rel=1e-6)


@pytest.mark.parametrize("method", ["grid", "hcp", "mathematical", "fast"])
def test_run_one(tmp_path, method):
    result = run_one(tmp_path, "chevron-1", "low", method)

    assert result["metrics"]["volume"] == pytest.approx(5046.8, rel=0.01)
    assert result["metrics"]["F1"] > 0.0
    files = list(tmp_path.glob("runs/chevron-1/*/*.json"))
    assert len(files) == 1
    with open(files[0]) as file:
        assert json.load(file)["method"] == method
    if result["particles"] is not None:
        count = result["particles"]["count"]
        with open(files[0].parent / f"{method}.particles.txt") as file:
            assert len(base64.b64decode(file.read())) == count * 7


@pytest.mark.parametrize("layers", [1, 5, 25])
def test_chevron_start_cone(layers):
    data = make_material_deposition(f"chevron-{layers}").data
    cone_volume = get_start_cone_volume(5046.8, layers)
    at_start = data[data["timestamp"] <= DURATION * cone_volume / data["volume"].sum()]
    # The stacker stays at the start until the cone of the first layer height is stacked, then travels
    assert (at_start["x"] == X_MIN).all()
    # Up to one material step of 5046.8 m³ / 4000
    assert at_start["volume"].sum() == pytest.approx(cone_volume, abs=1.3)
    assert data["x"].iloc[len(at_start) + 10] > X_MIN


@pytest.mark.parametrize(("ppm3", "angle"), [(1.0, 45.0), (8.0, 45.0), (8.0, 30.0)])
def test_cpp_lattice_matches_python(ppm3, angle):
    from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator

    python = LatticeSimulator(20.0, 10.0, ppm3=ppm3, angle_of_repose=angle)
    cpp = BslBlendingSimulator(bed_size_x=20.0, bed_size_z=10.0, ppm3=ppm3, lattice=True, latticeangle=angle, record_particles=True)
    rng = np.random.default_rng(0)
    # One particle per call, along a chevron path and at random spots including the bed edges
    xs = np.concatenate([np.linspace(2.0, 18.0, 300), np.linspace(18.0, 2.0, 300), rng.uniform(0.0, 20.0, 400)])
    zs = np.concatenate([np.full(600, 5.0), rng.uniform(0.0, 10.0, 400)])
    for i, (x, z) in enumerate(zip(xs, zs, strict=True)):
        python.stack(float(i), float(x), float(z), 1.0 / ppm3, [float(i)])
        cpp.stack(float(i), float(x), float(z), 1.0 / ppm3, [float(i)])

    particles = cpp.get_particles()
    p = np.asarray(python.positions)
    centers = p + np.array([0.0, 0.5 * python.particle_height, 0.0])
    assert len(particles["position"]) == len(p) > 900
    np.testing.assert_allclose(particles["position"], centers, atol=1e-6)
    np.testing.assert_allclose(particles["parameters"][:, 0], np.asarray(python.parameters)[:, 0])
