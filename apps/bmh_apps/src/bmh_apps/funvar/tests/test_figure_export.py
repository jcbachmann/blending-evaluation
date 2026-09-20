import logging
import re
import sys

import plotly.graph_objects as go
import pyperclip
import pytest
from pandas import DataFrame

from bmh_apps.funvar.figure_export import export_fig
from bmh_apps.funvar.fun_var_results import FunVarResults


@pytest.fixture(autouse=True)
def clipboard(monkeypatch):
    # No browser and no real clipboard in tests
    copied = []
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pyperclip, "copy", copied.append)
    return copied


def make_results(*directories) -> FunVarResults:
    return FunVarResults(df=DataFrame({"file_path": [str(directory) for directory in directories]}))


def test_export_writes_html_and_text_to_the_common_directory(tmp_path, monkeypatch, clipboard):
    monkeypatch.setattr(sys, "argv", ["plot_fun", "some/path", "--non-dominated"])
    results = make_results(tmp_path / "run0", tmp_path / "run1")

    export_fig("2d", go.Figure(), results, "my label")

    (html,) = tmp_path.glob("*.html")
    (text,) = tmp_path.glob("*.txt")
    assert re.fullmatch(r"plot fun 2d my label [0-9a-f]{4}\.html", html.name)
    assert text.name == html.name.replace(".html", ".txt")
    content = text.read_text()
    assert content.startswith("##### Graph\n")
    assert "Command: `plot_fun some/path --non-dominated`" in content
    assert f"[{html.name}](file://" in content
    assert clipboard == [content]


def test_export_survives_missing_clipboard(tmp_path, monkeypatch, caplog):
    def fail(_content):
        raise pyperclip.PyperclipException("no clipboard mechanism")

    monkeypatch.setattr(pyperclip, "copy", fail)

    with caplog.at_level(logging.WARNING):
        export_fig("1d", go.Figure(), make_results(tmp_path), "label")

    assert len(list(tmp_path.glob("*.html"))) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 1
    assert "Could not copy graph to clipboard" in caplog.text
