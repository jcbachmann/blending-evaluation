import sys
from typing import Union, Optional

import pandas as pd
from pandas import DataFrame

from bmh_apps.helpers.stacker import Stacker
from bmh_apps.helpers.stacker_printer import StackerPrinter


def read_material(filepath: str, col_timestamp: str = 'timestamp', col_volume: str = 'volume',
                  cols_p: [str] = None) -> DataFrame:
    df = pd.read_csv(filepath, delimiter='\t', index_col=None)
    if cols_p is None:
        # Use all columns except for timestamp and volume
        cols_p = list(df.columns.drop(col_timestamp, col_volume))
    required_cols = [col_timestamp, col_volume] + cols_p
    if not set(required_cols).issubset(df.columns):
        raise Exception(f'required columns ({required_cols}) not found in material file')
    material = DataFrame()
    material['timestamp'] = df[col_timestamp]
    material['volume'] = df[col_volume]
    for i, col_p in enumerate(cols_p):
        material[col_p] = df[col_p]
    return material


def read_path(filepath: str, col_path: str = 'path', col_part: Optional[str] = None,
              col_timestamp: Optional[str] = None) -> DataFrame:
    required_cols = [col_path]
    if col_part is not None:
        required_cols += [col_part]
    if col_timestamp is not None:
        required_cols += [col_timestamp]
    path = pd.read_csv(filepath, delimiter='\t', index_col=None)
    if not set(required_cols).issubset(path.columns):
        raise Exception('required columns (%s) not found in path file', required_cols)
    path['path'] = path[col_path]
    if col_part is not None:
        path['part'] = path[col_part]
    if col_timestamp is not None:
        path['timestamp'] = path[col_timestamp]
    return path


def stack_with_printer(
        length: float,
        depth: float,
        material: Union[str, DataFrame],
        stacker_path: Union[str, DataFrame],
        header: bool = True,
        out_buffer=sys.stdout.buffer
):
    printer = StackerPrinter(header=header, out_buffer=out_buffer)

    if isinstance(material, str):
        material = read_material(material)

    if isinstance(stacker_path, str):
        stacker_path = read_path(stacker_path)

    Stacker(
        length,
        depth,
        status=printer.status
    ).run(
        material,
        stacker_path,
        callback=printer.out
    )
    printer.flush()
