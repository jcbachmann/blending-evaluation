import pytest

from ..scenarios import PATTERNS, get_layer_end_times, get_pattern_volume, make_material_deposition


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_patterns(pattern):
    data = make_material_deposition(pattern).drops(1.0, 1.0)
    assert data["volume"].sum() == pytest.approx(get_pattern_volume(pattern))
    if pattern != "cone":
        assert len(get_layer_end_times(pattern)) == int(pattern.split("-")[1])
