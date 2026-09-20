import pandas as pd
import plotly.graph_objects as go
import pyperclip
import pytest

from bmh_apps.funvar import plot_fun
from bmh_apps.funvar.fun_var_results import FunVarResults


@pytest.fixture(autouse=True)
def clipboard(monkeypatch):
    # No browser and no real clipboard in tests
    copied = []
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pyperclip, "copy", copied.append)
    return copied


def make_results(directory, dimensions: int) -> FunVarResults:
    fun_columns = [f"F{i + 1}" for i in range(dimensions)]
    results_df = pd.DataFrame({column: [0.5, 0.7, 0.9] for column in fun_columns})
    results_df["file_path"] = str(directory)
    results_df["run"] = ["run=0", "run=0", "run=1"]
    results_df["parameters"] = ["", "", ""]
    return FunVarResults(df=results_df, fun_columns=fun_columns, misc_columns=["file_path", "run", "parameters"])


def exported_files(directory, suffix: str) -> list[str]:
    return sorted(path.name for path in directory.glob(f"plot fun *{suffix}"))


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4, 5])
def test_plots_are_exported_and_copied(tmp_path, clipboard, dimensions):
    results = make_results(tmp_path, dimensions)
    plots = {
        1: lambda: plot_fun.plot_fun_1d(results, "label"),
        2: lambda: plot_fun.plot_fun_2d(results, results.df, "label", auto_scale=False),
        3: lambda: plot_fun.plot_fun_3d(results, results.df, "label", auto_scale=False),
        4: lambda: plot_fun.plot_fun_4d(results, "label"),
        5: lambda: plot_fun.plot_fun_5d(results, "label"),
    }

    plots[dimensions]()

    assert len(exported_files(tmp_path, ".html")) == (2 if dimensions == 5 else 1)
    assert len(exported_files(tmp_path, ".txt")) == (2 if dimensions == 5 else 1)
    assert clipboard
    assert clipboard[0].startswith("##### Graph")


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("auto_scale", [False, True])
def test_plot_without_non_dominated_overlay_is_exported(tmp_path, dimensions, auto_scale):
    # plot_fun --non-dominated filters the results itself and passes no overlay
    results = make_results(tmp_path, dimensions)
    plot = plot_fun.plot_fun_2d if dimensions == 2 else plot_fun.plot_fun_3d

    plot(results, None, "label", auto_scale=auto_scale)

    assert len(exported_files(tmp_path, ".html")) == 1


def test_export_survives_missing_clipboard(tmp_path, monkeypatch):
    def fail(_content):
        raise pyperclip.PyperclipException("no clipboard mechanism")

    monkeypatch.setattr(pyperclip, "copy", fail)

    plot_fun.plot_fun_1d(make_results(tmp_path, 1), "label")

    assert len(exported_files(tmp_path, ".html")) == 1
    assert len(exported_files(tmp_path, ".txt")) == 1
