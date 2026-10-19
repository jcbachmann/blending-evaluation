import pandas as pd
from bmh.helpers.fronts import get_nondominated


def filter_relevant_region(df: pd.DataFrame, fun_columns: list[str], threshold: float = 1):
    for fun_column in fun_columns:
        df = df[df[fun_column] < threshold]
    return df


def filter_efficient_front(df: pd.DataFrame, fun_columns: list[str]):
    """The Pareto front: the rows no other row dominates (`bmh.helpers.fronts.get_nondominated`; equal rows are kept), sorted by the
    objectives. Rows as good as another in some objectives and worse in the rest (weakly dominated) are removed."""
    df = df.sort_values(by=fun_columns)
    return df.iloc[get_nondominated(df[fun_columns].to_numpy(dtype=float))]
