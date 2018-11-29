import sys
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame


def read_material(filepath: str, col_timestamp: str = 'timestamp', col_volume: str = 'volume',
                  cols_p: List[str] = None) -> DataFrame:
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


def merge_material_path(length: float, depth: float, material: DataFrame, path: DataFrame) -> DataFrame:
    # Stacker path parameters
    min_pos = depth / 2
    max_pos = length - depth / 2

    # Total volume in cubic meters
    t_total = material['timestamp'].max()
    if 'timestamp' not in path.columns:
        # No timestamps provided - generate time stamps
        if 'part' in path.columns:
            # Position relative to time is known
            path['timestamp'] = path['part'] / path['part'].max() * t_total
        else:
            n = len(path.index)
            if n > 1:
                path['timestamp'] = [t_total * i / (n - 1) for i in range(n)]
            else:
                path['timestamp'] = [0]

    material_with_path = material.copy()
    material_with_path['z'] = depth / 2
    material_with_path['x'] = np.interp(material_with_path['timestamp'], path['timestamp'], path['path']) * (
            max_pos - min_pos) + min_pos

    return material_with_path


class StrToBytesWrapper:
    def __init__(self, bytes_buffer):
        self.bytes_buffer = bytes_buffer

    def write(self, s):
        self.bytes_buffer.write(s.encode('utf-8'))


def print_merged_material_path(material_path: DataFrame):
    first_cols = ['timestamp', 'x', 'z', 'volume']
    col_order = first_cols + list(set(material_path.columns) - set(first_cols))
    material_path.to_csv(StrToBytesWrapper(sys.stdout.buffer), index=False, columns=col_order, sep=' ')
    sys.stdout.buffer.flush()
