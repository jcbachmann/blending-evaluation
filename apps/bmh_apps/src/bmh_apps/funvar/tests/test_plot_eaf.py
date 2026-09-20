import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyperclip
import pytest

from bmh_apps.funvar import plot_eaf


@pytest.fixture(autouse=True)
def shown_figures(monkeypatch):
    # No browser and no real clipboard in tests
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig, *_args, **_kwargs: shown.append(fig))
    monkeypatch.setattr(pyperclip, "copy", lambda _content: None)
    return shown


def make_front_runs(offset: float, runs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Points of several runs approximating the front f2 = 1.1 - f1 + offset, and the set of each point."""
    rng = np.random.default_rng(seed)
    points, sets = [], []
    for run in range(1, runs + 1):
        f1 = np.sort(rng.uniform(0.1, 1.0, 15))
        f2 = 1.1 - f1 + offset + rng.normal(0, 0.02, 15)
        points.extend(zip(f1, f2, strict=True))
        sets.extend([run] * 15)
    return np.array(points), np.array(sets)


def area_by_sign(rectangles: np.ndarray) -> tuple[float, float]:
    area = (rectangles[:, 2] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 1])
    return float(area[rectangles[:, 4] > 0].sum()), float(area[rectangles[:, 4] < 0].sum())


def test_best_and_worst_surface_of_two_runs():
    points, sets = np.array([[1.0, 3.0], [3.0, 1.0]]), np.array([1, 2])

    surfaces = plot_eaf.compute_attainment_surfaces(points, sets)

    assert surfaces[0].tolist() == [[1.0, 3.0], [3.0, 1.0]]  # attained by at least one run
    assert surfaces[100].tolist() == [[3.0, 3.0]]  # attained by all runs


def test_surfaces_of_identical_runs_equal_the_front():
    front = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
    points, sets = np.vstack([front] * 3), np.repeat([1, 2, 3], 3)

    surfaces = plot_eaf.compute_attainment_surfaces(points, sets)

    for surface in surfaces.values():
        assert surface.tolist() == [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]


def test_difference_is_positive_where_the_first_group_attains_more():
    points_a, sets_a = make_front_runs(0.0, 5, seed=1)
    points_b, sets_b = make_front_runs(0.2, 5, seed=2)
    ranges = plot_eaf.get_axis_ranges(np.vstack([points_a, points_b]), auto_scale=False)

    a_better = plot_eaf.clip_rectangles(plot_eaf.compute_eaf_difference(points_a, sets_a, points_b, sets_b, 5), ranges)
    b_better = plot_eaf.clip_rectangles(plot_eaf.compute_eaf_difference(points_b, sets_b, points_a, sets_a, 5), ranges)

    positive, negative = area_by_sign(a_better)
    assert positive > 5 * negative
    assert area_by_sign(b_better)[1] > 5 * area_by_sign(b_better)[0]


def test_difference_levels_are_integer_intervals_of_the_run_share():
    points_a, sets_a = make_front_runs(0.0, 11, seed=1)
    points_b, sets_b = make_front_runs(0.2, 11, seed=2)

    rectangles = plot_eaf.compute_eaf_difference(points_a, sets_a, points_b, sets_b, 5)

    assert set(np.unique(rectangles[:, 4])) <= {-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert len(np.unique(rectangles[:, 4])) > 5  # a clear difference uses many of the intervals


def test_to_levels():
    counts = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 8.0, 11.0, -1.0, -11.0])

    # 11 runs in 5 intervals: 1-2 runs -> 1, 3-4 -> 2, 5-6 -> 3, 7-8 -> 4, 9-11 -> 5
    assert plot_eaf.to_levels(counts, runs=11, intervals=5).tolist() == [0, 1, 1, 2, 3, 4, 5, -1, -5]
    assert plot_eaf.to_levels(np.array([1.0, -3.0, 5.0]), runs=5, intervals=5).tolist() == [1, -3, 5]


def test_difference_requires_the_same_number_of_runs():
    points_a, sets_a = make_front_runs(0.0, 6, seed=1)
    points_b, sets_b = make_front_runs(0.2, 4, seed=2)

    with pytest.raises(ValueError, match="same number of runs in both groups, got 6 and 4"):
        plot_eaf.compute_eaf_difference(points_a, sets_a, points_b, sets_b, 5)


def test_equalize_run_counts_keeps_the_first_runs_of_the_larger_group():
    points_a, sets_a = make_front_runs(0.0, 6, seed=1)
    points_b, sets_b = make_front_runs(0.2, 4, seed=2)

    trimmed_a, trimmed_sets_a, trimmed_b, trimmed_sets_b = plot_eaf.equalize_run_counts(points_a, sets_a, points_b, sets_b)

    assert set(trimmed_sets_a) == set(trimmed_sets_b) == {1, 2, 3, 4}
    assert np.array_equal(trimmed_a, points_a[sets_a <= 4])
    assert np.array_equal(trimmed_b, points_b)
    assert plot_eaf.compute_eaf_difference(trimmed_a, trimmed_sets_a, trimmed_b, trimmed_sets_b, 5).size > 0


def test_sets_are_numbered_by_sorted_path():
    group = pd.DataFrame({"F1": [1.0, 2.0, 3.0], "F2": [3.0, 2.0, 1.0], "file_path": ["b/", "a/", "b/"]})

    _, sets = plot_eaf.get_points_and_sets(group, ["F1", "F2"])

    assert sets.tolist() == [2, 1, 2]


def test_clip_rectangles_bounds_infinite_rectangles_and_drops_invisible_ones():
    rectangles = np.array(
        [
            [0.2, 0.5, np.inf, np.inf, 1.0],  # unbounded
            [2.0, 0.5, 3.0, 0.7, 1.0],  # right of the visible range
            [0.3, 0.6, 0.3, 0.9, 1.0],  # no width
        ]
    )

    clipped = plot_eaf.clip_rectangles(rectangles, ((0.0, 1.5), (0.0, 1.5)))

    assert clipped.tolist() == [[0.2, 0.5, 1.5, 1.5, 1.0]]


def test_axis_ranges():
    points = np.array([[0.2, 0.4], [0.6, 0.4]])

    assert plot_eaf.get_axis_ranges(points, auto_scale=False) == ((0.0, 1.5), (0.0, 1.5))
    x_range, y_range = plot_eaf.get_axis_ranges(points, auto_scale=True)
    assert x_range == pytest.approx((0.18, 0.62))
    assert y_range == pytest.approx((0.35, 0.45))  # a constant objective still gets a visible range


def test_difference_colors_are_symmetric_diverging_steps():
    colors = plot_eaf.get_difference_colors(5, "#2a78d6", "#eb6834")

    assert colors[0] == plot_eaf.NEUTRAL_COLOR
    assert colors[5] == "#2a78d6"
    assert colors[-5] == "#eb6834"

    def brightness(color: str) -> int:
        return sum(int(color[i : i + 2], 16) for i in (1, 3, 5))

    for sign in (1, -1):
        brightnesses = [brightness(colors[sign * level]) for level in range(1, 6)]
        assert brightnesses == sorted(brightnesses, reverse=True)  # the stronger the difference, the darker
        assert brightnesses[0] < brightness(plot_eaf.NEUTRAL_COLOR)


def group_df(parameters: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"parameters": parameters, "file_path": [f"run{i}" for i in range(len(parameters))]})


def test_get_groups():
    groups = plot_eaf.get_groups(group_df(["b=2", "a=1", "b=2"]))

    assert list(groups) == ["a=1", "b=2"]
    assert [len(group) for group in groups.values()] == [1, 2]
    assert list(plot_eaf.get_groups(group_df(["", ""]))) == [plot_eaf.ALL_RUNS]


def test_runs_without_parameters_next_to_runs_with_parameters_are_the_defaults():
    # E.g. random start populations are the config default and therefore not part of the run directory names
    groups = plot_eaf.get_groups(group_df(["", "precondition=True", ""]))

    assert list(groups) == [plot_eaf.DEFAULT_PARAMETERS, "precondition=True"]
    assert [len(group) for group in groups.values()] == [2, 1]


def test_select_groups():
    groups = plot_eaf.get_groups(group_df(["a=1", "b=2", "c=3"]))

    assert list(plot_eaf.select_groups(groups, None)) == ["a=1", "b=2", "c=3"]
    assert list(plot_eaf.select_groups(groups, ["c=3", "a=1"])) == ["c=3", "a=1"]
    with pytest.raises(ValueError, match=r"unknown: \['x=9'\]"):
        plot_eaf.select_groups(groups, ["a=1", "x=9"])
    with pytest.raises(ValueError, match="two different groups"):
        plot_eaf.select_groups(groups, ["a=1", "a=1"])


def test_more_groups_than_colors_require_a_selection():
    groups = plot_eaf.get_groups(group_df([f"p={i}" for i in range(9)]))

    with pytest.raises(ValueError, match="cannot be distinguished by color"):
        plot_eaf.select_groups(groups, None)


def test_surface_plot_has_one_legend_entry_and_color_per_group():
    points, sets = make_front_runs(0.0, 3, seed=1)
    surfaces = {name: plot_eaf.compute_attainment_surfaces(points, sets) for name in ("a=1", "b=2")}

    fig = plot_eaf.plot_attainment_surfaces(surfaces, ranges=((0.0, 1.5), (0.0, 1.5)), x_title="F1", y_title="F2", label="label", auto_scale=False)

    lines = [trace for trace in fig.data if trace.mode == "lines"]
    assert len(lines) == 6
    assert [trace.name for trace in lines if trace.showlegend] == ["a=1", "b=2"]
    assert {trace.line.color for trace in lines[:3]} == {plot_eaf.CATEGORICAL_COLORS[0]}
    assert {trace.line.color for trace in lines[3:]} == {plot_eaf.CATEGORICAL_COLORS[1]}
    assert [trace.line.dash for trace in lines[:3]] == ["dot", "solid", "dash"]


def test_difference_plot_uses_the_group_colors_for_both_sides():
    points_a, sets_a = make_front_runs(0.0, 5, seed=1)
    points_b, sets_b = make_front_runs(0.2, 5, seed=2)
    ranges = ((0.0, 1.5), (0.0, 1.5))
    rectangles = plot_eaf.clip_rectangles(plot_eaf.compute_eaf_difference(points_a, sets_a, points_b, sets_b, 5), ranges)

    fig = plot_eaf.plot_eaf_difference(
        rectangles, group_a="a=1", group_b="b=2", intervals=5, ranges=ranges, x_title="F1", y_title="F2", label="", auto_scale=False
    )

    legend_entries = {trace.name: trace.marker.color for trace in fig.data if trace.name and "attains more" in trace.name}
    assert legend_entries == {"a=1 attains more": plot_eaf.CATEGORICAL_COLORS[0], "b=2 attains more": plot_eaf.CATEGORICAL_COLORS[1]}
    filled = [trace for trace in fig.data if trace.fill == "toself"]
    assert filled
    assert all(not np.isinf(np.array(trace.x, dtype=float)).any() for trace in filled)


def make_run(root: Path, precondition: str, run: int, offset: float, objectives: int = 2):
    path = root / f"+experiment=mining-f1-f2,+run={run},optimization.precondition_population={precondition}"
    path.mkdir(parents=True)
    columns = ["F1/Ash (%)", "F2", "F3"][:objectives]
    (path / "OBJ").write_text(f"{columns}\n")
    rng = np.random.default_rng(run)
    f1 = np.sort(rng.uniform(0.1, 1.0, 10))
    rows = [[f, 1.1 - f + offset + rng.normal(0, 0.02), 0.5][:objectives] for f in f1]
    (path / "FUN").write_text("".join(" ".join(str(value) for value in row) + "\n" for row in rows))


def exported(directory: Path, suffix: str) -> list[str]:
    return sorted(path.name for path in directory.glob(f"plot fun eaf*{suffix}"))


def run_main(monkeypatch, root: Path, *options: str):
    monkeypatch.setattr(sys, "argv", ["plot_eaf", f"{root}/*/", *options])
    plot_eaf.main()


@pytest.fixture
def two_groups(tmp_path):
    for run in range(4):
        make_run(tmp_path, "false", run, offset=0.2)
        make_run(tmp_path, "true", run, offset=0.0)
    return tmp_path


def test_main_exports_surfaces_and_difference_for_two_groups(monkeypatch, two_groups, shown_figures):
    run_main(monkeypatch, two_groups)

    assert len(exported(two_groups, ".html")) == 2
    assert len(exported(two_groups, ".txt")) == 2
    surfaces_fig, difference_fig = shown_figures
    assert [trace.name for trace in surfaces_fig.data if trace.showlegend is not False] == ["precondition=false", "precondition=true", "Reference Point"]
    assert "precondition=false vs. precondition=true" in difference_fig.layout.title.text
    assert surfaces_fig.layout.xaxis.title.text == "F1 Homogenization Efficiency Ratio for Ash (%)"


def test_main_compare_restricts_to_the_selected_groups(monkeypatch, two_groups, shown_figures):
    for run in range(4):
        make_run(two_groups, "maybe", run, offset=0.1)

    run_main(monkeypatch, two_groups, "--compare", "precondition=true", "precondition=maybe")

    surfaces_fig, difference_fig = shown_figures
    assert [trace.name for trace in surfaces_fig.data if trace.showlegend and trace.name != "Reference Point"] == ["precondition=true", "precondition=maybe"]
    assert "precondition=true vs. precondition=maybe" in difference_fig.layout.title.text


def test_main_with_more_than_two_groups_only_exports_surfaces(monkeypatch, two_groups, caplog):
    for run in range(4):
        make_run(two_groups, "maybe", run, offset=0.1)

    run_main(monkeypatch, two_groups)

    assert len(exported(two_groups, ".html")) == 1
    assert "needs exactly two groups" in caplog.text


def test_main_with_a_single_group_only_exports_surfaces(monkeypatch, tmp_path):
    for run in range(3):
        make_run(tmp_path, "true", run, offset=0.0)

    run_main(monkeypatch, tmp_path)

    assert len(exported(tmp_path, ".html")) == 1


def test_main_requires_two_objectives(monkeypatch, tmp_path):
    for run in range(2):
        make_run(tmp_path, "true", run, offset=0.0, objectives=3)

    with pytest.raises(ValueError, match="exactly two objectives"):
        run_main(monkeypatch, tmp_path)


def test_main_drop_columns_selects_two_of_three_objectives(monkeypatch, tmp_path):
    for run in range(3):
        make_run(tmp_path, "true", run, offset=0.0, objectives=3)
        make_run(tmp_path, "false", run, offset=0.2, objectives=3)

    run_main(monkeypatch, tmp_path, "--drop-columns", "F3")

    assert len(exported(tmp_path, ".html")) == 2
    assert "drop [F3]" in exported(tmp_path, ".html")[0]


def test_main_auto_scale_uses_the_data_range(monkeypatch, two_groups, shown_figures):
    run_main(monkeypatch, two_groups, "--auto-scale")

    surfaces_fig, _ = shown_figures
    assert tuple(surfaces_fig.layout.xaxis.range) != (0.0, 1.5)


def test_main_difference_plot_contains_regions_favouring_the_better_group(monkeypatch, two_groups, shown_figures):
    run_main(monkeypatch, two_groups)

    _, difference_fig = shown_figures
    filled = [trace for trace in difference_fig.data if trace.fill == "toself"]
    assert filled
    assert {trace.fillcolor for trace in filled} <= set(plot_eaf.get_difference_colors(5, *plot_eaf.CATEGORICAL_COLORS[:2]).values())
    # "true" has the better fronts, so the regions between the fronts are attained more often by it
    assert any("precondition=true attains" in trace.text for trace in filled)
    assert not any("precondition=false attains" in trace.text for trace in filled)


def test_main_trims_groups_with_different_run_counts(monkeypatch, tmp_path, caplog, shown_figures):
    for run in range(4):
        make_run(tmp_path, "false", run, offset=0.2)
    for run in range(3):
        make_run(tmp_path, "true", run, offset=0.0)

    run_main(monkeypatch, tmp_path)

    assert "using the first 3 runs" in caplog.text
    assert len(shown_figures) == 2


def test_main_compares_two_experiments_given_by_id(monkeypatch, tmp_path, shown_figures):
    # The E2 workflow: one experiment with random and one with preconditioned start populations
    monkeypatch.setenv("BMH_EXPERIMENTS_PATH", str(tmp_path))
    random_experiment = tmp_path / "EXPERIMENT-11111111-0000-0000-0000-000000000000"
    preconditioned_experiment = tmp_path / "EXPERIMENT-22222222-0000-0000-0000-000000000000"
    for run in range(4):
        make_run(random_experiment, "false", run, offset=0.2)
        make_run(preconditioned_experiment, "true", run, offset=0.0)
    monkeypatch.setattr(sys, "argv", ["plot_eaf", "E11111111", "E22222222"])

    plot_eaf.main()

    surfaces_fig, difference_fig = shown_figures
    assert [trace.name for trace in surfaces_fig.data if trace.showlegend][:2] == ["precondition=false", "precondition=true"]
    assert "precondition=false vs. precondition=true" in difference_fig.layout.title.text


def test_rename_groups():
    groups = plot_eaf.get_groups(group_df(["a=1", "b=2"]))

    renamed = plot_eaf.rename_groups(groups, ["first", "second"])

    assert list(renamed) == ["first", "second"]
    assert renamed["first"] is groups["a=1"]
    with pytest.raises(ValueError, match="one label per group, got 1 labels for 2 groups"):
        plot_eaf.rename_groups(groups, ["only"])
    with pytest.raises(ValueError, match="must be unique"):
        plot_eaf.rename_groups(groups, ["same", "same"])


def test_main_labels_are_used_for_legend_title_and_hover(monkeypatch, two_groups, shown_figures):
    run_main(monkeypatch, two_groups, "--labels", "random", "preconditioned")

    surfaces_fig, difference_fig = shown_figures
    # Groups are sorted by their parameters, "precondition=false" is the random one
    assert [trace.name for trace in surfaces_fig.data if trace.showlegend] == ["random", "preconditioned"]
    assert "random vs. preconditioned" in difference_fig.layout.title.text
    assert any("preconditioned attains" in trace.text for trace in difference_fig.data if trace.fill == "toself")


def test_main_labels_follow_the_order_of_compare(monkeypatch, two_groups, shown_figures):
    run_main(monkeypatch, two_groups, "--compare", "precondition=true", "precondition=false", "--labels", "preconditioned", "random")

    surfaces_fig, _ = shown_figures
    assert [trace.name for trace in surfaces_fig.data if trace.showlegend] == ["preconditioned", "random"]
    assert surfaces_fig.data[0].line.color == plot_eaf.CATEGORICAL_COLORS[0]  # the first group in the order of --compare keeps the first color


def test_main_labels_need_one_label_per_group(monkeypatch, two_groups):
    with pytest.raises(ValueError, match="one label per group"):
        run_main(monkeypatch, two_groups, "--labels", "random")
