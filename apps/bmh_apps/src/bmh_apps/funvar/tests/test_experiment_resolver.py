import os
from pathlib import Path

import pytest

from bmh_apps.funvar.experiment_resolver import ExperimentResolver, shorten_experiment_label

UUID_TAIL = "-0000-0000-0000-000000000000"


def make_experiment(root: Path, name: str, parent: str = "") -> str:
    path = root / parent / name
    path.mkdir(parents=True)
    return os.path.abspath(path)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (f"EXPERIMENT-1234abcd{UUID_TAIL}", "E1234abcd"),
        ("E1234abcd", "E1234abcd"),
        ("run=1", "run=1"),
    ],
)
def test_shorten_experiment_label(label, expected):
    assert shorten_experiment_label(label) == expected


def test_resolves_experiment_id_to_run_directories(tmp_path):
    path = make_experiment(tmp_path, f"EXPERIMENT-1234abcd{UUID_TAIL}", parent="nested")

    resolved = ExperimentResolver(str(tmp_path)).resolve(["E1234abcd", "other/FUN"])

    assert resolved == [path + "/*/", "other/FUN"]


def test_resolves_experiment_directory_with_short_name(tmp_path):
    path = make_experiment(tmp_path, "Eabcd1234")

    assert ExperimentResolver(str(tmp_path)).resolve(["Eabcd1234"]) == [path + "/*/"]


def test_plain_paths_do_not_index_experiments(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("must not index")

    monkeypatch.setattr(ExperimentResolver, "index_experiments", fail)

    assert ExperimentResolver(str(tmp_path)).resolve(["a/FUN", "b/"]) == ["a/FUN", "b/"]


def test_unknown_experiment_raises(tmp_path):
    with pytest.raises(ValueError, match="not indexed"):
        ExperimentResolver(str(tmp_path)).resolve(["E1234abcd"])


def test_ambiguous_experiment_raises(tmp_path):
    make_experiment(tmp_path, f"EXPERIMENT-1234abcd{UUID_TAIL}")
    make_experiment(tmp_path, "EXPERIMENT-1234abce-1111-1111-1111-111111111111")

    with pytest.raises(ValueError, match="not unique"):
        ExperimentResolver(str(tmp_path)).resolve(["E1234abc"])


def test_experiment_without_root_directory_raises():
    with pytest.raises(ValueError, match="BMH_EXPERIMENTS_PATH"):
        ExperimentResolver().resolve(["E1234abcd"])
