import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from jmetal.util.solution import read_solutions

from bmh_apps.funvar import export_fun
from bmh_apps.funvar.fun_var_results import FunVarResults


def make_run(root: Path, run_dir: str, fun_rows: str):
    path = root / run_dir
    path.mkdir(parents=True)
    (path / "OBJ").write_text("['F1', 'F2']\n")
    (path / "FUN").write_text(fun_rows)


@pytest.fixture
def runs(tmp_path):
    # Two runs sharing the parameter a=1, so the derived label is "a=1"
    make_run(tmp_path, "a=1,+run=0", "0.1 0.4\n0.3 0.2\n")
    make_run(tmp_path, "a=1,+run=1", "0.5 0.5\n0.2 0.3\n")
    return tmp_path


def run_export(monkeypatch, runs: Path, *options: str):
    monkeypatch.setattr(sys, "argv", ["export_fun", f"{runs}/*/", *options])
    export_fun.main()


def exported(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.FUN"))


@pytest.mark.parametrize(
    ("label", "suffix", "expected"),
    [
        ("a=1", "ab12", "a=1 ab12.FUN"),
        ("a=1", "", "a=1.FUN"),
        ("", "ab12", "ab12.FUN"),
        ("a/b\\c", "ab12", "a_b_c ab12.FUN"),  # a label must not move the file into another directory
    ],
)
def test_get_file_name(label, suffix, expected):
    assert export_fun.get_file_name(label, suffix) == expected


def test_random_suffix_is_separated_by_a_space(monkeypatch, runs):
    run_export(monkeypatch, runs)

    (file,) = exported(runs)
    assert re.fullmatch(r"a=1 [0-9a-f]{4}\.FUN", file.name)
    assert len(pd.read_csv(file, sep=" ", header=None)) == 4


def test_skip_random_suffix_gives_the_label_full_control(monkeypatch, runs):
    run_export(monkeypatch, runs, "--skip-random-suffix", "--label", "my label")

    assert [file.name for file in exported(runs)] == ["my label.FUN"]


def test_non_dominated_export_only_contains_efficient_front(monkeypatch, runs):
    run_export(monkeypatch, runs, "--non-dominated", "--skip-random-suffix")

    (file,) = exported(runs)
    assert file.name == "a=1 (non-dominated).FUN"
    exported_rows = pd.read_csv(file, sep=" ", header=None)
    assert sorted(zip(exported_rows[0], exported_rows[1], strict=True)) == [(0.1, 0.4), (0.2, 0.3), (0.3, 0.2)]


ALL_ROWS = [(0.1, 0.4), (0.2, 0.3), (0.3, 0.2), (0.5, 0.5)]


def test_export_has_no_header_and_the_objectives_are_in_an_obj_file(monkeypatch, runs):
    run_export(monkeypatch, runs, "--skip-random-suffix", "--label", "front")

    lines = (runs / "front.FUN").read_text().splitlines()
    assert len(lines) == 4
    assert all(re.fullmatch(r"[0-9.]+ [0-9.]+", line) for line in lines)
    assert (runs / "front.OBJ").read_text() == "['F1', 'F2']\n"


def test_export_can_be_read_back_by_funvarresults(monkeypatch, runs):
    run_export(monkeypatch, runs, "--skip-random-suffix", "--label", "front")

    results = FunVarResults.from_file(str(runs / "front.FUN"), fun_only=True)

    assert results.fun_columns == ["F1", "F2"]
    assert sorted(zip(results.df["F1"], results.df["F2"], strict=True)) == ALL_ROWS


def test_export_can_be_read_by_jmetal_as_reference_front(monkeypatch, runs):
    run_export(monkeypatch, runs, "--skip-random-suffix", "--label", "front")

    solutions = read_solutions(str(runs / "front.FUN"))

    assert sorted(tuple(solution.objectives) for solution in solutions) == ALL_ROWS


def test_export_keeps_the_full_float_precision(monkeypatch, tmp_path):
    make_run(tmp_path, "a=1,+run=0", "0.12345678901234567 0.98765432109876543\n")
    make_run(tmp_path, "a=1,+run=1", "0.5 0.5\n")

    monkeypatch.setattr(sys, "argv", ["export_fun", f"{tmp_path}/*/", "--skip-random-suffix", "--label", "front"])
    export_fun.main()

    results = FunVarResults.from_file(str(tmp_path / "front.FUN"), fun_only=True)
    assert (0.12345678901234567, 0.98765432109876543) in list(zip(results.df["F1"], results.df["F2"], strict=True))


def test_export_with_a_path_separator_in_the_label_stays_in_the_directory(monkeypatch, runs):
    run_export(monkeypatch, runs, "--skip-random-suffix", "--label", "E2 F1/Ash")

    assert [file.name for file in exported(runs)] == ["E2 F1_Ash.FUN"]
