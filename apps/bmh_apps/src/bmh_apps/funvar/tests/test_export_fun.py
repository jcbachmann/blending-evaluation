import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from bmh_apps.funvar import export_fun


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
    ],
)
def test_get_file_name(label, suffix, expected):
    assert export_fun.get_file_name(label, suffix) == expected


def test_random_suffix_is_separated_by_a_space(monkeypatch, runs):
    run_export(monkeypatch, runs)

    (file,) = exported(runs)
    assert re.fullmatch(r"a=1 [0-9a-f]{4}\.FUN", file.name)
    assert len(pd.read_csv(file, sep=" ")) == 4


def test_skip_random_suffix_gives_the_label_full_control(monkeypatch, runs):
    run_export(monkeypatch, runs, "--skip-random-suffix", "--label", "my label")

    assert [file.name for file in exported(runs)] == ["my label.FUN"]


def test_non_dominated_export_only_contains_efficient_front(monkeypatch, runs):
    run_export(monkeypatch, runs, "--non-dominated", "--skip-random-suffix")

    (file,) = exported(runs)
    assert file.name == "a=1 (non-dominated).FUN"
    exported_rows = pd.read_csv(file, sep=" ")
    assert sorted(zip(exported_rows["F1"], exported_rows["F2"], strict=True)) == [(0.1, 0.4), (0.2, 0.3), (0.3, 0.2)]
