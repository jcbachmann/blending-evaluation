import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import List

import pandas as pd

VALID_EXTENSIONS = ['FUN', 'VAR', 'OBJ']


def get_filename_without_extension(file_path: str) -> str:
    if file_path[-3:] not in VALID_EXTENSIONS:
        raise Exception(f"Invalid file extension '{file_path[-3:]}'")
    return file_path[:-3]


def read_fun_columns_file(file_path: str) -> list[str]:
    if not file_path.endswith('OBJ'):
        raise Exception("Invalid file extension")
    with open(file_path, 'r') as f:
        return eval(f.readline())


def read_fun_file(file_path: str, columns: List[str]) -> pd.DataFrame:
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
    misc_columns: List[str] = None
    fun_columns: List[str] = None
    var_columns: List[str] = None
    runs: List[str] = None
    label: str = ''

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

        self.df = fun_var_results.df if self.df is None else pd.concat(
            [self.df, fun_var_results.df], ignore_index=True
        )

    @staticmethod
    def from_file(file_path: str, fun_only: bool = False) -> 'FunVarResults':
        file_path_without_extension = get_filename_without_extension(file_path)

        fun_columns = read_fun_columns_file(file_path_without_extension + 'OBJ')
        fun_df = read_fun_file(file_path_without_extension + 'FUN', fun_columns)
        if fun_only:
            df = fun_df
            var_columns = None
        else:
            var_df = read_var_file(file_path_without_extension + 'VAR')
            df = fun_df.join(var_df)
            var_columns = var_df.columns.tolist()

        df['file_path'] = file_path_without_extension
        misc_columns = ['file_path']

        return FunVarResults(
            df=df,
            fun_columns=fun_columns,
            var_columns=var_columns,
            misc_columns=misc_columns,
        )

    @staticmethod
    def from_files(file_paths: List[str], fun_only: bool = False) -> 'FunVarResults':
        logging.debug(f"Reading {len(file_paths)} files from")

        all_results = FunVarResults()
        file_count = 0
        for file_path in file_paths:
            for file in glob.glob(file_path):
                file_count += 1
                fun_var_results = FunVarResults.from_file(file, fun_only)
                all_results.merge(fun_var_results)
        logging.info(f"Read {len(all_results.df)} rows from {file_count} files")

        if len(all_results.df['file_path']) > 0:
            file_paths = all_results.df['file_path'].to_list()
            common_path = os.path.commonpath(file_paths)
            file_paths = [file_path.replace(common_path, '') for file_path in file_paths]
            run_parts = [re.split(r'[/,]', file_path) for file_path in file_paths]
            run_parts = [[part for part in run if part != ''] for run in run_parts]
            run_parts = [[part[1:] if part.startswith('+') else part for part in run] for run in run_parts]
            sample, remaining = run_parts[0], run_parts[1:]
            redundant = []
            for part in sample:
                if all(part in run for run in remaining):
                    redundant.append(part)
            run_parts = [[part for part in run if part not in redundant] for run in run_parts]
            all_results.df['run'] = [' '.join(run) for run in run_parts]
            all_results.label = ' '.join(redundant)
        else:
            all_results.df['run'] = ''

        all_results.df['individual'] = all_results.df.index
        all_results.df['run_individual'] = all_results.df['run'] + ' - ' + all_results.df['individual'].astype(str)
        all_results.misc_columns.extend(['run', 'individual', 'run_individual'])

        return all_results

    def len(self):
        return 0 if self.df is None else self.df.shape[0]

    def drop_columns(self, drop_columns: List[str]):
        self.df = self.df.drop(drop_columns, axis=1)
        self.fun_columns = [column for column in self.fun_columns if column not in drop_columns]
