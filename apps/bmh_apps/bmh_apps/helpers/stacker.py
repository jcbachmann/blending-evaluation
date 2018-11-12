import numpy as np
from pandas import DataFrame


class Stacker:
    def __init__(self, length: float, depth: float, status=None):
        # Blending bed parameters
        self.length = length
        self.depth = depth

        # Status message callback
        self.status = lambda msg: None if status is None else status

    def run(self, material: DataFrame, path: DataFrame, callback) -> None:
        self.status('Starting stacker')

        # Stacker path parameters
        min_pos = self.depth / 2
        max_pos = self.length - self.depth / 2

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

        out_material = material.copy()
        out_material['z'] = self.depth / 2
        out_material['x'] = np.interp(out_material['timestamp'], path['timestamp'], path['path']) * (
                max_pos - min_pos) + min_pos

        try:
            callback(out_material)
        except IOError:
            self.status('Stopping stacker due to IOError')

        self.status('Stacker stopped')
