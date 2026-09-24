import pandas as pd
import pytest

from bmh_ml import report, ui


def make_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["aaaaaaaa1111", "bbbbbbbb2222", "cccccccc3333"],
            "tags.mlflow.runName": ["good", None, "bad"],
            "params.model": ["mlp", "ridge", "mean"],
            "params.bundle": ["B", "B", "B"],
            "params.seed": ["1", "1", "1"],
            "metrics.T1/F1/nrmse": [1.0, 5.0, 20.0],
            "metrics.T2/F2/nrmse": [3.0, 2.0, 30.0],
            "metrics.T2/F2/r2": [0.5, 0.9, -1.0],
            "metrics.T1/F1/r2_ceiling": [0.99, 0.99, 0.99],
            "metrics.T2/F2/r2_ceiling": [0.98, 0.98, 0.98],
            "metrics.train/seconds": [10.0, 1.0, 0.0],
            "metrics.throughput/batch_10000": [1e5, 1e7, 1e9],
        }
    )


def test_the_leaderboard_is_sorted_by_the_operating_region_by_default():
    table = report.build_leaderboard(make_runs())

    assert list(table["model"]) == ["ridge", "mlp", "mean"]
    assert list(table.columns[:4]) == ["run", "model", "bundle", "seed"]
    assert "T1/F1/nrmse" in table.columns
    assert "T2/F2/nrmse" in table.columns


def test_runs_without_a_name_are_shown_by_their_id():
    table = report.build_leaderboard(make_runs())

    assert set(table["run"]) == {"good", "bbbbbbbb", "bad"}


def test_the_leaderboard_can_be_sorted_by_another_column():
    table = report.build_leaderboard(make_runs(), sort_by="T1/F1/nrmse")

    assert list(table["model"]) == ["mlp", "ridge", "mean"]


def test_for_r2_the_highest_value_comes_first():
    table = report.build_leaderboard(make_runs(), metric="r2")

    assert list(table["model"]) == ["ridge", "mlp", "mean"]


def test_only_the_top_runs_are_shown():
    assert len(report.build_leaderboard(make_runs(), top=2)) == 2


def test_the_cost_of_the_models_is_part_of_the_leaderboard():
    table = report.build_leaderboard(make_runs())

    assert {"train s", "pred/s (10k)"} <= set(table.columns)


def test_a_missing_metric_or_sort_column_is_reported():
    with pytest.raises(ValueError, match="No run has the metric"):
        report.build_leaderboard(make_runs(), metric="pairwise_accuracy")
    with pytest.raises(ValueError, match="Cannot sort by"):
        report.build_leaderboard(make_runs(), sort_by="T9/F1/nrmse")


def test_the_noise_ceilings_of_the_sets_are_listed():
    ceilings = report.get_noise_ceilings(make_runs())

    assert ceilings.loc["T1", "F1"] == 0.99
    assert ceilings.loc["T2", "F2"] == 0.98
    assert "F2" not in ceilings.loc["T1"].dropna()


def test_the_table_is_written_as_markdown():
    markdown = report.to_markdown(report.build_leaderboard(make_runs(), top=1))

    lines = markdown.splitlines()
    assert lines[0].startswith("| run | model |")
    assert lines[1].startswith("|---|")
    assert "ridge" in lines[2]


def test_the_values_are_formatted_compactly():
    assert report.format_value(0.123456) == "0.1235"
    assert report.format_value(float("nan")) == ""
    assert report.format_value(None) == ""
    assert report.format_value(1.4e9) == "1.4e+09"
    assert report.format_value("mlp") == "mlp"


def test_the_html_escapes_its_content():
    table = pd.DataFrame({"run": ["<script>"], "value": [1.0]})

    page = report.to_html(table, "Title <b>", "note")

    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "Title &lt;b&gt;" in page


def test_the_ui_uses_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path))

    command = ui.get_ui_command("127.0.0.1", 5001)

    assert command[1:4] == ["-m", "mlflow", "ui"]
    assert f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}" in command
    assert (tmp_path / "artifacts").as_uri() in command
    assert command[-2:] == ["--port", "5001"]


def test_the_transfer_test_is_shown_and_can_be_sorted_by():
    runs = make_runs()
    runs["metrics.transfer/hv_ratio"] = [0.5, None, 0.9]

    table = report.build_leaderboard(runs, sort_by="transfer hv")

    assert list(table["model"]) == ["mean", "mlp", "ridge"]


def test_runs_without_metrics_like_the_parents_of_sweeps_are_left_out():
    runs = pd.concat([make_runs(), pd.DataFrame({"run_id": ["dddddddd4444"], "tags.mlflow.runName": ["sweep-x"]})], ignore_index=True)

    assert "sweep-x" not in set(report.build_leaderboard(runs)["run"])
