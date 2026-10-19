import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bmh_apps.funvar.fun_var_math import filter_efficient_front


def pareto_front_loop(df: pd.DataFrame, fun_columns: list[str]) -> pd.DataFrame:
    # every row against every other: keep the rows no other row dominates
    df = df.sort_values(by=fun_columns)
    values = df[fun_columns].to_numpy()
    dominated = [any(np.all(other <= row) and np.any(other < row) for other in values) for row in values]
    return df[~np.array(dominated)]


def test_rows_dominated_by_another_row_are_removed_also_when_equal_in_some_objectives():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 2.0, 2.0], "f2": [4.0, 4.0, 3.0, 2.0, 2.0], "label": list("abcde")})

    # b is as good as a in f2 and worse in f1, c is beaten by d in both; the equal rows d and e both stay
    assert filter_efficient_front(df, ["f1", "f2"])["label"].tolist() == ["a", "d", "e"]


@pytest.mark.parametrize("objectives", [2, 3])
def test_the_filter_keeps_the_rows_and_order_of_a_pairwise_pareto_check(objectives):
    rng = np.random.default_rng(objectives)
    columns = [f"f{k}" for k in range(objectives)]
    for _ in range(30):
        df = pd.DataFrame(rng.integers(0, 6, (int(rng.integers(1, 40)), objectives)).astype(float), columns=columns)
        df["label"] = np.arange(len(df))
        assert_frame_equal(filter_efficient_front(df, columns), pareto_front_loop(df, columns))
