import pytest

from bmh_apps.funvar.objective_resolver import prettify_objective


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("F1", "F1 Homogenization Efficiency Ratio"),
        ("F1/Ash (%)", "F1 Homogenization Efficiency Ratio for Ash (%)"),
        ("F2", "F2 Standard Deviation from Ideal Stockpile Reclaim Volume"),
        ("F3", "F3 Total Travel Distance"),
        ("F4", "F4 Maximum Travel Speed"),
        ("Custom", "Custom"),
    ],
)
def test_prettify_objective(objective, expected):
    assert prettify_objective(objective) == expected
