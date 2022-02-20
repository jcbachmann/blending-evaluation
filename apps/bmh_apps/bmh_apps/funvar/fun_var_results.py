import logging
import os
from dataclasses import dataclass

import pandas as pd

VALID_EXTENSIONS = ['FUN', 'VAR', 'OBJ']


def get_filename_without_extension(file_path: str) -> str:
    if file_path[-3:] not in VALID_EXTENSIONS:
        raise Exception("Invalid file extension")
    return file_path[:-3]


def read_fun_columns_file(file_path: str) -> list[str]:
    if not file_path.endswith('OBJ'):
        raise Exception("Invalid file extension")
    with open(file_path, 'r') as f:
        return eval(f.readline())


def read_fun_file(file_path: str, columns: list[str]) -> pd.DataFrame:
    if not file_path.endswith('FUN'):
        raise Exception("Invalid file extension")
    return pd.read_csv(file_path, sep=' ', header=None, index_col=False, names=columns)


def read_var_file(file_path: str) -> pd.DataFrame:
    if not file_path.endswith('VAR'):
        raise Exception("Invalid file extension")
    df = pd.read_csv(file_path, sep=' ', header=None, index_col=False)
    # As var files contain a trailing delimiter, we need to remove the empty column
    df = df.drop(df.columns[-1], axis=1)
    return df


@dataclass
class FunVarResults:
    df: pd.DataFrame = None
    misc_columns: list[str] = None
    fun_columns: list[str] = None
    var_columns: list[str] = None

    def merge(self, fun_var_results: 'FunVarResults'):
        if self.fun_columns is None:
            self.fun_columns = fun_var_results.fun_columns
        elif self.fun_columns != fun_var_results.fun_columns:
            raise Exception("Cannot merge results with different fun columns")

        if self.var_columns is None:
            self.var_columns = fun_var_results.var_columns
        elif self.var_columns != fun_var_results.var_columns:
            raise Exception("Cannot merge results with different var columns")

        if self.misc_columns is None:
            self.misc_columns = fun_var_results.misc_columns
        elif self.misc_columns != fun_var_results.misc_columns:
            raise Exception("Cannot merge results with different misc columns")

        self.df = fun_var_results.df if self.df is None else self.df.append(fun_var_results.df, ignore_index=True)

    @staticmethod
    def from_file(file_path: str) -> 'FunVarResults':
        file_path_without_extension = get_filename_without_extension(file_path)

        fun_columns = read_fun_columns_file(file_path_without_extension + 'OBJ')
        fun_df = read_fun_file(file_path_without_extension + 'FUN', fun_columns)
        var_df = read_var_file(file_path_without_extension + 'VAR')
        df = fun_df.join(var_df)

        df['run'] = os.path.basename(file_path_without_extension)
        df['individual'] = df.index
        df['run_individual'] = df['run'].astype(str) + ' - ' + df['individual'].astype(str)
        misc_columns = ['run', 'individual', 'run_individual']

        return FunVarResults(
            df=df,
            fun_columns=fun_columns,
            var_columns=var_df.columns.tolist(),
            misc_columns=misc_columns
        )

    @staticmethod
    def from_files(file_paths: list[str]) -> 'FunVarResults':
        logging.debug(f"Reading {len(file_paths)} files from")

        all_results = FunVarResults()
        for file_path in file_paths:
            fun_var_results = FunVarResults.from_file(file_path)
            all_results.merge(fun_var_results)
        logging.info(f"Read {len(all_results.df)} rows from {len(file_paths)} files")
        return all_results
