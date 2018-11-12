#!/usr/bin/env python
import csv
import math
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator
from pandas import DataFrame


class RoundnessEvaluator:
    def __init__(self, dist_seg_size, angle_seg_count):
        self.dist_seg_size = dist_seg_size
        self.angle_seg_count = angle_seg_count
        self.cx = None
        self.cz = None
        self.dist_seg_count = None
        self.shapes_main = None
        self.shapes_range = None

    def simulate(self, likelihood, pos, volume, run):
        print(f'processing volume {volume} with likelihood {likelihood} (run {run})')
        simulator = BslBlendingSimulator(
            bed_size_x=pos * 2.0,
            bed_size_z=pos * 2.0,
            eight=likelihood,
        )
        simulator.stack(0, pos, pos, volume, [])
        heights = simulator.get_heights()

        self.add_from_heights(heights)

    def add(self, x_abs, z_abs, height):
        x = x_abs - self.cx
        z = z_abs - self.cz

        if x == 0 and z == 0:  # center
            for shape in self.shapes_main:
                shape[0][0] += 1
                shape[0][1] += height

            for shape in self.shapes_range:
                shape[0][0] += 1
                shape[0][1] += height
        else:
            distance = math.hypot(x, z)
            distance_seg = min(int(distance / self.dist_seg_size), self.dist_seg_count - 1)
            angle = math.degrees(math.atan2(z, x) + math.pi)

            angle_seg_main = min(int(abs((angle % 90) - 45) * 3 / 45), 2)
            self.shapes_main[angle_seg_main][distance_seg][0] += 1
            self.shapes_main[angle_seg_main][distance_seg][1] += height

            angle_seg_range = min(int(abs((angle % 90) - 45) * self.angle_seg_count / 45), self.angle_seg_count - 1)
            self.shapes_range[angle_seg_range][distance_seg][0] += 1
            self.shapes_range[angle_seg_range][distance_seg][1] += height

    def add_from_file(self, input_file):
        df = pd.read_csv(input_file, header=None, delimiter='\t', index_col=None)

        # Calculate weighted center
        df_cols = DataFrame()
        cols = len(df.columns)
        for c in range(cols):
            df_cols[c] = df[c] * c

        df_rows = df.transpose()
        rows = len(df_rows.columns)
        for c in range(rows):
            df_rows[c] = df_rows[c] * c

        self.cx = df_cols.sum().sum() / df.sum().sum()
        self.cz = df_rows.sum().sum() / df.sum().sum()
        self.dist_seg_count = int(math.hypot(self.cx, self.cz) / self.dist_seg_size + 0.5)

        self.shapes_main = [[[0, 0] for _ in range(self.dist_seg_count)] for _ in range(3)]
        self.shapes_range = [[[0, 0] for _ in range(self.dist_seg_count)] for _ in range(self.angle_seg_count)]

        with open(input_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            z_abs = 0
            for row in reader:
                x_abs = 0
                for value in row:
                    if value != '':
                        v = float(value)
                        if v > 0:
                            self.add(x_abs, z_abs, v)
                    x_abs += 1
                z_abs += 1

    def add_from_heights(self, heights: List[List[float]]):
        # Calculate weighted center
        self.cx = np.average(range(len(heights)), weights=[sum(col) for col in heights])
        self.cz = np.average(range(len(heights[0])), weights=[sum(row) for row in list(map(list, zip(*heights)))])

        self.dist_seg_count = int(math.hypot(self.cx, self.cz) / self.dist_seg_size + 0.5)

        self.shapes_main = [[[0, 0] for _ in range(self.dist_seg_count)] for _ in range(3)]
        self.shapes_range = [[[0, 0] for _ in range(self.dist_seg_count)] for _ in range(self.angle_seg_count)]

        z_abs = 0
        for row in heights:
            x_abs = 0
            for v in row:
                if v > 0:
                    self.add(x_abs, z_abs, v)
                x_abs += 1
            z_abs += 1

    def evaluate(self):
        min_shape = None
        max_shape = None

        for shape in self.shapes_range:
            if min_shape is not None:
                min_shape = [
                    min(entry[1] / entry[0], min_shape_entry) if entry[0] > 0 else min_shape_entry for
                    entry, min_shape_entry in zip(shape, min_shape)
                ]
                max_shape = [
                    max(entry[1] / entry[0], max_shape_entry) if entry[0] > 0 else max_shape_entry for
                    entry, max_shape_entry in zip(shape, max_shape)
                ]
            else:
                min_shape = [entry[1] / entry[0] if entry[0] > 0 else 0 for entry in shape]
                max_shape = list(min_shape)

        diff = 0
        max_total = 0
        for i in range(len(min_shape)):
            diff += max_shape[i] - min_shape[i]
            max_total += max_shape[i]

        return diff / max_total

    def plot(self, label, c):
        color_map = cm.get_cmap('brg')
        min_shape = None
        max_shape = None

        for shape in self.shapes_range:
            if min_shape is not None:
                min_shape = [
                    min(entry[1] / entry[0], min_shape_entry) if entry[0] > 0 else min_shape_entry for
                    entry, min_shape_entry in zip(shape, min_shape)
                ]
                max_shape = [
                    max(entry[1] / entry[0], max_shape_entry) if entry[0] > 0 else max_shape_entry for
                    entry, max_shape_entry in zip(shape, max_shape)
                ]
            else:
                min_shape = [entry[1] / entry[0] if entry[0] > 0 else 0 for entry in shape]
                max_shape = list(min_shape)

        i = 0
        for shape in self.shapes_main:
            plt.plot(
                [x * self.dist_seg_size for x in range(len(shape) + 1)],
                [entry[1] / entry[0] if entry[0] > 0 else None for entry in shape] + [0],
                label=f'Shape {i * 45 / 2:.1f}°' if c == 0 else None,
                linestyle='-' if i == 0 else '--' if i == 1 else ':',
                color=color_map(c),
                alpha=0.5
            )
            i += 1

        plt.fill_between(
            [x * self.dist_seg_size for x in range(len(min_shape))],
            min_shape,
            max_shape,
            alpha=0.2,
            facecolor=color_map(c),
            label=label
        )
        plt.xlabel('distance from center')
        plt.ylabel('height')
