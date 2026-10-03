import json

import pytest

from bmh_ml.tracking.views import TAG_PREFIX, decode_state, encode_state, get_views, install_views


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def test_a_view_state_survives_encoding():
    for _, state in get_views().values():
        encoded = encode_state(state)
        assert encoded.startswith("deflate;")
        assert decode_state(encoded) == state


def test_every_chart_belongs_to_a_section_of_its_view():
    for name, state in get_views().values():
        sections = {section["uuid"] for section in state.get("compareRunSections", [])}
        assert all(chart["metricSectionId"] in sections for chart in state.get("compareRunCharts", [])), name


def test_each_scope_shows_the_test_sets_that_matter_for_it():
    s1, s2 = (json.dumps(get_views(scope)) for scope in ("S1", "S2"))

    assert "T2/F2/r2" in s1  # the operating region of the fixed material
    assert "T6" not in s1
    assert "T6/F1/r2" in s2  # the new materials of the transfer test
    assert "transfer/hv_ratio_material_min" in s2
    assert "T2/" not in s2


def test_installing_again_updates_the_views_instead_of_adding_copies():
    mlflow = pytest.importorskip("mlflow")
    from bmh_ml.tracking.runs import get_experiment_id

    experiment_id = get_experiment_id("S1-fixed-material")

    first = install_views("http://ui")
    second = install_views("http://ui")

    tags = {key: json.loads(value) for key, value in mlflow.get_experiment(experiment_id).tags.items() if key.startswith(TAG_PREFIX)}
    assert len(tags) == len(get_views())
    assert first == second
    assert any("compareRunsMode=CHART" in link for link in first)
