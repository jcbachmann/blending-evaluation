import json
import logging

import numpy as np
import pandas as pd
import pytest

from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH
from bmh_ml.training_data import load_fixed_material_variables, load_training_data

COLUMNS = ["f1", "f2", *[f"m{i + 1}" for i in range(MATERIAL_LENGTH)], *[f"d{i + 1}" for i in range(DEPOSITION_LENGTH)]]


def write_training_data(directory, rows: int = 6) -> pd.DataFrame:
    data = pd.DataFrame(np.arange(rows * len(COLUMNS), dtype=float).reshape(rows, len(COLUMNS)) + 0.5, columns=COLUMNS)
    data.to_csv(directory / "training_data.csv", index=False)
    return data


def test_complete_training_data_is_loaded_unchanged(tmp_path):
    expected = write_training_data(tmp_path)

    pd.testing.assert_frame_equal(load_training_data(str(tmp_path / "training_data.csv")), expected)


def test_a_truncated_last_row_is_ignored_with_a_warning(tmp_path, caplog):
    expected = write_training_data(tmp_path)
    path = tmp_path / "training_data.csv"
    content = path.read_text()
    path.write_text(content[: content.rindex(",", 0, len(content) - 40)])  # cut the last row off in the middle

    with caplog.at_level(logging.WARNING):
        data = load_training_data(str(path))

    pd.testing.assert_frame_equal(data, expected.iloc[:-1])
    assert "Ignoring the incomplete last row" in caplog.text


def test_an_incomplete_row_in_the_middle_is_an_error(tmp_path):
    write_training_data(tmp_path)
    path = tmp_path / "training_data.csv"
    lines = path.read_text().splitlines()
    lines[3] = ",".join(lines[3].split(",")[:-5])  # the fourth data row (index 2) misses values
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=r"1 incomplete rows \(first: row 2\)"):
        load_training_data(str(path))


def test_fewer_rows_than_generated_is_a_warning(tmp_path, caplog):
    write_training_data(tmp_path, rows=6)
    (tmp_path / "training_data_params.json").write_text(json.dumps({"materials": 10, "depositions_per_material": 5}))

    with caplog.at_level(logging.WARNING):
        data = load_training_data(str(tmp_path / "training_data.csv"))

    assert len(data) == 6
    assert "has 6 rows, but 50 rows were generated" in caplog.text


def test_the_expected_number_of_rows_is_not_a_warning(tmp_path, caplog):
    write_training_data(tmp_path, rows=6)
    (tmp_path / "training_data_params.json").write_text(json.dumps({"materials": 2, "depositions_per_material": 3}))

    with caplog.at_level(logging.WARNING):
        load_training_data(str(tmp_path / "training_data.csv"))

    assert caplog.text == ""


def test_fixed_material_variables_are_the_material_of_the_first_row(tmp_path):
    data = write_training_data(tmp_path)

    material = load_fixed_material_variables(str(tmp_path / "training_data.csv"))

    assert material.dtype == np.float64
    assert material.tolist() == data.iloc[0][[f"m{i + 1}" for i in range(MATERIAL_LENGTH)]].tolist()


def test_fixed_material_variables_only_need_the_first_row(tmp_path):
    data = write_training_data(tmp_path, rows=2)
    path = tmp_path / "training_data.csv"
    path.write_text(path.read_text() + "this,is,not,a,valid,row\n")

    assert load_fixed_material_variables(str(path)).tolist() == data.iloc[0][[f"m{i + 1}" for i in range(MATERIAL_LENGTH)]].tolist()


def test_fixed_material_variables_are_exactly_the_values_of_the_file(tmp_path):
    # The default float parser of pandas is off by one ULP for some digits
    path = tmp_path / "training_data.csv"
    header = ",".join(COLUMNS)
    values = ["0.12345678901234567", "0.5", *["6.0612244897959187"] * MATERIAL_LENGTH, *["20.0"] * DEPOSITION_LENGTH]
    path.write_text(header + "\n" + ",".join(values) + "\n")

    assert load_fixed_material_variables(str(path)).tolist() == [float("6.0612244897959187")] * MATERIAL_LENGTH
