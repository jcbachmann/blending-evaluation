import numpy as np
import pandas as pd


def filter_relevant_region(df: pd.DataFrame, fun_columns: list[str], threshold: float = 1):
    for fun_column in fun_columns:
        df = df[df[fun_column] < threshold]
    return df


def filter_efficient_front(df: pd.DataFrame, fun_columns: list[str]):
    df = df.sort_values(by=fun_columns)
    i = 0
    while i < df.shape[0]:
        df = df[np.any(df[fun_columns] <= df[fun_columns].iloc[i], axis=1)]
        i += 1
    return df
