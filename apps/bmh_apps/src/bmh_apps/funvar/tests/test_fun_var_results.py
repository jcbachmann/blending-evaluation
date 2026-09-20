from pathlib import Path

import pytest

from bmh_apps.funvar.fun_var_results import FunVarResults, cleanup_part, get_filename_without_extension, is_parameter

UUID_TAIL = "-0000-0000-0000-000000000000"


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        ("+run=1", "run=1"),
        ("optimization.precondition_population=True", "precondition=True"),
        ("optimization.max_evaluations=1000", "evaluations=1000"),
        ("experiment=foo", "foo"),
        (f"EXPERIMENT-1234abcd{UUID_TAIL}", "E1234abcd"),
        ("something", "something"),
    ],
)
def test_cleanup_part(part, expected):
    assert cleanup_part(part) == expected


@pytest.mark.parametrize(
    ("entry", "expected"),
    [("precondition=True", True), ("evaluations=1000", True), ("run=3", False), ("E1234abcd", False), ("Eabcd", False)],
)
def test_is_parameter(entry, expected):
    assert is_parameter(entry) == expected


def test_get_filename_without_extension(tmp_path):
    assert get_filename_without_extension("some/path/FUN") == "some/path/"
    assert get_filename_without_extension(str(tmp_path)) == f"{tmp_path}/"
    assert get_filename_without_extension(f"{tmp_path}/") == f"{tmp_path}/"
    with pytest.raises(Exception, match="Invalid file extension"):
        get_filename_without_extension("some/path/results.txt")


def make_run(root: Path, experiment: str, run_dir: str):
    path = root / experiment / run_dir
    path.mkdir(parents=True)
    (path / "OBJ").write_text("['F1', 'F2']\n")
    (path / "FUN").write_text("0.1 0.2\n0.3 0.4\n")


def test_from_files_splits_runs_and_parameters_of_one_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_EXPERIMENTS_PATH", str(tmp_path))
    experiment = f"EXPERIMENT-1234abcd{UUID_TAIL}"
    make_run(tmp_path, experiment, "optimization.precondition_population=True,+run=0")
    make_run(tmp_path, experiment, "optimization.precondition_population=False,+run=1")

    results = FunVarResults.from_files(["E1234abcd"], fun_only=True)

    assert results.len() == 4
    assert set(zip(results.df["run"], results.df["parameters"], strict=True)) == {("run=0", "precondition=True"), ("run=1", "precondition=False")}
    assert results.label == ""
    assert results.fun_columns == ["F1", "F2"]


def test_from_files_treats_experiments_as_runs_and_shared_parts_as_label(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_EXPERIMENTS_PATH", str(tmp_path))
    # The second experiment id is short, its shortened label must still not be treated as a parameter
    make_run(tmp_path, f"EXPERIMENT-1234abcd{UUID_TAIL}", "optimization.precondition_population=True,+run=0")
    make_run(tmp_path, f"EXPERIMENT-abcd{UUID_TAIL}", "optimization.precondition_population=True,+run=0")

    results = FunVarResults.from_files(["E1234abcd", "Eabcd"], fun_only=True)

    assert set(results.df["run"]) == {"E1234abcd", "Eabcd"}
    assert set(results.df["parameters"]) == {""}
    assert results.label == "precondition=True run=0"


def test_from_files_without_matching_results_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_EXPERIMENTS_PATH", str(tmp_path))

    with pytest.raises(ValueError, match=r"No results \(OBJ and FUN files\) found for: .*missing"):
        FunVarResults.from_files([f"{tmp_path}/missing/*/"], fun_only=True)


def test_from_files_names_the_resolved_path_of_an_experiment_without_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_EXPERIMENTS_PATH", str(tmp_path))
    (tmp_path / f"EXPERIMENT-1234abcd{UUID_TAIL}").mkdir()

    with pytest.raises(ValueError, match=r"EXPERIMENT-1234abcd.*/\*/"):
        FunVarResults.from_files(["E1234abcd"], fun_only=True)
