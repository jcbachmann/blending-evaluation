import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bmh_apps.funvar.fun_var_math import filter_efficient_front


def filter_efficient_front_loop(df: pd.DataFrame, fun_columns: list[str]) -> pd.DataFrame:
    # the filter as it was first written
    df = df.sort_values(by=fun_columns)
    i = 0
    while i < df.shape[0]:
        df = df[np.any(df[fun_columns] <= df[fun_columns].iloc[i], axis=1)]
        i += 1
    return df


def test_rows_beaten_in_every_objective_are_removed_but_weakly_dominated_rows_stay():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 2.0], "f2": [4.0, 4.0, 3.0, 2.0], "label": list("abcd")})

    assert filter_efficient_front(df, ["f1", "f2"])["label"].tolist() == ["a", "d", "b"]


@pytest.mark.parametrize("objectives", [2, 3])
def test_the_filter_keeps_the_rows_and_order_of_the_first_implementation(objectives):
    rng = np.random.default_rng(objectives)
    columns = [f"f{k}" for k in range(objectives)]
    for _ in range(30):
        df = pd.DataFrame(rng.integers(0, 6, (int(rng.integers(1, 40)), objectives)).astype(float), columns=columns)
        df["label"] = np.arange(len(df))
        assert_frame_equal(filter_efficient_front(df, columns), filter_efficient_front_loop(df, columns))
