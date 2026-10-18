import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from ..bed_setup import STACKING_TIME
from ..simple import generate_deposition, generate_material, generate_material_deposition


def test_a_material_has_equal_portions_evenly_spread_over_the_stacking_time():
    material = generate_material([5.0, 7.0, 6.0, 9.0], total_volume=2500)

    assert material.data["timestamp"].tolist() == pytest.approx([0.0, STACKING_TIME / 3, 2 * STACKING_TIME / 3, STACKING_TIME])
    assert material.data["volume"].tolist() == [625.0] * 4
    assert material.data["quality"].tolist() == [5.0, 7.0, 6.0, 9.0]
    assert material.get_volume() == 2500


def test_a_deposition_runs_along_the_middle_of_the_bed_while_the_reclaimer_covers_the_bed():
    deposition = generate_deposition([10.0, 49.0, 10.0], bed_size_x=59, bed_size_z=20, duration=100.0)

    assert deposition.data["timestamp"].tolist() == [0.0, 50.0, 100.0]
    assert deposition.data["x"].tolist() == [10.0, 49.0, 10.0]
    assert deposition.data["z"].tolist() == [10.0] * 3
    assert (deposition.meta.bed_size_x, deposition.meta.bed_size_z) == (59, 20)
    assert deposition.meta.reclaim_x_per_s == pytest.approx(0.59)


def test_the_material_deposition_interpolates_the_material_to_portions_of_100_m3_and_builds_the_rest_alike():
    quality, x = [1.0, 3.0, 2.0], [5.0, 55.0, 5.0, 55.0]

    md = generate_material_deposition(quality, total_volume=500.0, x_position=x, bed_size_x=60, bed_size_z=10)

    assert md.material.data["quality"].tolist() == pytest.approx(np.interp(np.linspace(0, 2, 5), [0, 1, 2], quality))
    assert_frame_equal(md.material.data, generate_material(np.interp(np.linspace(0, 2, 5), [0, 1, 2], quality), 500.0).data)
    assert_frame_equal(md.deposition.data, generate_deposition(x, 60, 10).data)
    assert md.deposition.meta.reclaim_x_per_s == generate_deposition(x, 60, 10).meta.reclaim_x_per_s
