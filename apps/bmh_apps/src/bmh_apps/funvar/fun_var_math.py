import pandas as pd
from bmh.helpers.fronts import get_weakly_nondominated


def filter_relevant_region(df: pd.DataFrame, fun_columns: list[str], threshold: float = 1):
    for fun_column in fun_columns:
        df = df[df[fun_column] < threshold]
    return df


def filter_efficient_front(df: pd.DataFrame, fun_columns: list[str]):
    """The rows no other row beats in every objective, sorted by the objectives. Unlike a Pareto filter (`bmh.helpers.fronts.get_nondominated`)
    this keeps weakly dominated rows: as good as another row in some objectives and worse in the rest."""
    df = df.sort_values(by=fun_columns)
    return df.iloc[get_weakly_nondominated(df[fun_columns].to_numpy(dtype=float))]
