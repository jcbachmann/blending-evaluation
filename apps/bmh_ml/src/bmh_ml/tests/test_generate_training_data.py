import argparse
import json
import sys

import numpy as np
import pytest

from bmh_ml import generate_training_data
from bmh_ml.generate_training_data import generate, get_columns, get_task_sizes
from bmh_ml.settings import (
    BED_SIZE_X,
    BED_SIZE_Z,
    DEPOSITION_LENGTH,
    MATERIAL_LENGTH,
    MATERIAL_MAX,
    MATERIAL_MIN,
    TOTAL_VOLUME,
    X_MAX,
    X_MIN,
)
from bmh_ml.training_data import load_training_data

COLUMN_COUNT = 2 + MATERIAL_LENGTH + DEPOSITION_LENGTH


def make_args(materials: int = 4, depositions_per_material: int = 3, n_jobs: int = 2, output_file: str = "data/training_data.csv") -> argparse.Namespace:
    return argparse.Namespace(
        output_file=output_file,
        n_jobs=n_jobs,
        materials=materials,
        depositions_per_material=depositions_per_material,
        deposition_length=DEPOSITION_LENGTH,
        material_length=MATERIAL_LENGTH,
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        total_volume=TOTAL_VOLUME,
        material_min=MATERIAL_MIN,
        material_max=MATERIAL_MAX,
    )


def test_columns_are_the_objectives_followed_by_material_and_deposition():
    columns = get_columns(MATERIAL_LENGTH, DEPOSITION_LENGTH)

    assert len(columns) == COLUMN_COUNT
    assert columns[:3] == ["f1", "f2", "m1"]
    assert columns[2 + MATERIAL_LENGTH - 1 : 2 + MATERIAL_LENGTH + 1] == ["m50", "d1"]
    assert columns[-1] == "d20"


def test_generate_returns_the_rows_of_all_materials():
    rows = generate(make_args(depositions_per_material=3), materials=2)

    assert rows.shape == (6, COLUMN_COUNT)
    assert np.isfinite(rows).all()
    material = rows[:, 2 : 2 + MATERIAL_LENGTH]
    assert (material[:3] == material[0]).all()  # the depositions of a material share it
    assert (material[3:] == material[3]).all()
    assert not np.array_equal(material[0], material[3])  # the next material is another one
    assert (material >= MATERIAL_MIN - 1).all()
    assert (material <= MATERIAL_MAX + 1).all()
    deposition = rows[:, 2 + MATERIAL_LENGTH :]
    assert (deposition >= X_MIN).all()
    assert (deposition <= X_MAX).all()
    assert len({tuple(row) for row in deposition}) == 6  # every deposition is drawn on its own


@pytest.mark.parametrize(
    ("materials", "depositions_per_material", "n_jobs"),
    [(5000, 50, 16), (100000, 1, 16), (7, 3, 16), (3, 50, 2), (1, 1, 1), (17, 1, 4)],
)
def test_task_sizes_cover_all_materials_with_reasonable_tasks(materials, depositions_per_material, n_jobs):
    sizes = get_task_sizes(materials, depositions_per_material, n_jobs)

    assert sum(sizes) == materials
    assert all(size >= 1 for size in sizes)
    assert all(size * depositions_per_material <= max(1000, depositions_per_material) for size in sizes)
    assert len(sizes) >= min(n_jobs, materials)  # every job gets a task if there is enough to do


def test_default_tasks_have_about_a_thousand_rows():
    sizes = get_task_sizes(5000, 50, 16)

    assert set(sizes) == {20}
    assert len(sizes) == 250


@pytest.mark.skipif(sys.platform != "linux", reason="worker processes are spawned on other platforms and are not needed to test the file")
def test_main_writes_the_training_data_and_its_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    generate_training_data.main(make_args(materials=6, depositions_per_material=3))

    data = load_training_data("data/training_data.csv")
    assert data.shape == (18, COLUMN_COUNT)
    assert list(data.columns) == get_columns(MATERIAL_LENGTH, DEPOSITION_LENGTH)
    params = json.loads((tmp_path / "data" / "training_data_params.json").read_text())
    assert params["materials"] * params["depositions_per_material"] == 18
