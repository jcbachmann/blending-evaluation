import argparse
import logging
from pathlib import Path

import plotly.graph_objects as go
import pytest

from bmh_apps.benchmark.evaluate_runtime import evaluate_log_file_runtime, main, parse_run_directory

START = "[2026-01-01 10:00:00,000][__main__][INFO] - start\n"


def log_line(time: str) -> str:
    return f"[2026-01-01 {time}][__main__][INFO] - message\n"


def make_log(directory: Path, name: str, content: str, run_dir: str = "") -> str:
    path = directory / run_dir
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(content)
    return str(path / name)


def test_runtime_is_hours_between_first_and_last_log_line(tmp_path):
    log = make_log(tmp_path, "run.log", START + log_line("11:00:00,500") + log_line("12:30:00,000"))

    assert evaluate_log_file_runtime(log) == pytest.approx(2.5)


def test_runtime_ignores_trailing_lines_without_timestamp(tmp_path):
    content = START + log_line("11:00:00,000") + "Traceback (most recent call last):\n  File ...\n[1, 2, 3]\n"

    assert evaluate_log_file_runtime(make_log(tmp_path, "run.log", content)) == pytest.approx(1.0)


def test_runtime_of_single_line_log_is_zero(tmp_path):
    assert evaluate_log_file_runtime(make_log(tmp_path, "run.log", START)) == 0.0


@pytest.mark.parametrize("content", ["", "no timestamp here\n"])
def test_log_without_timestamp_raises(tmp_path, content):
    with pytest.raises(ValueError, match="No log line with a timestamp"):
        evaluate_log_file_runtime(make_log(tmp_path, "run.log", content))


def test_parse_run_directory():
    assert parse_run_directory("+experiment=mining-f2,+run=3,optimization.max_evaluations=1000") == {
        "+experiment": "mining-f2",
        "+run": "3",
        "optimization.max_evaluations": "1000",
    }
    assert parse_run_directory("multirun") == {}


def run_main(logs: Path, parameter: str = "optimization.max_evaluations"):
    main(argparse.Namespace(logfile=[str(logs / "*" / "run.log")], parameter=parameter, verbose=False))


def test_main_plots_runtime_per_parameter_value(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig, *_args, **_kwargs: shown.append(fig))
    make_log(tmp_path, "run.log", START + log_line("12:30:00,000"), "optimization.max_evaluations=100,+run=0")
    make_log(tmp_path, "run.log", START + log_line("13:30:00,000"), "optimization.max_evaluations=100,+run=1")
    make_log(tmp_path, "run.log", START + log_line("11:00:00,000"), "optimization.max_evaluations=200,+run=0")

    run_main(tmp_path)

    (fig,) = shown
    (trace,) = fig.data
    assert sorted(zip(trace.x, trace.y, strict=True)) == [(100.0, pytest.approx(2.5)), (100.0, pytest.approx(3.5)), (200.0, pytest.approx(1.0))]


def test_main_skips_unusable_logs(tmp_path, monkeypatch, caplog):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig, *_args, **_kwargs: shown.append(fig))
    make_log(tmp_path, "run.log", START + log_line("11:00:00,000"), "optimization.max_evaluations=100,+run=0")
    make_log(tmp_path, "run.log", START, "other=1,+run=0")
    make_log(tmp_path, "run.log", "", "optimization.max_evaluations=100,+run=1")

    with caplog.at_level(logging.WARNING):
        run_main(tmp_path)

    assert len(shown) == 1
    assert len(shown[0].data[0].y) == 1
    assert sum("Skipping" in message for message in caplog.messages) == 2


def test_main_without_usable_logs_does_not_plot(tmp_path, monkeypatch, caplog):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig, *_args, **_kwargs: shown.append(fig))

    with caplog.at_level(logging.ERROR):
        run_main(tmp_path)

    assert not shown
    assert "No usable log files found" in caplog.messages
