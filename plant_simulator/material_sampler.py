import logging

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from ciglobal.cimath import weighted_avg_and_std
from .material_handler import MaterialHandler


def average_sample_group(sample_group):
    regroup = list(map(list, zip(*sample_group)))

    average = [np.average(regroup[0])]
    for i in range(1, len(regroup), 2):
        average += [np.average(regroup[i]),
                    np.average(regroup[i + 1], weights=regroup[i]) if sum(regroup[i]) > 0 else 0]

    return average


def calculate_stats(start, samples, columns):
    logging.debug('Calculating stats')
    regroup = list(map(list, zip(*samples)))

    data = [
        ('start', start),
        ('end', samples[-1][0])
    ]
    for i in range(1, len(regroup), 2):
        tph_average = np.average(regroup[i])
        tph_std = np.std(regroup[i])
        data.extend(
            [
                (f'{columns[i]} min', np.min(regroup[i])),
                (f'{columns[i]} max', np.max(regroup[i])),
                (f'{columns[i]} average', tph_average),
                (f'{columns[i]} std_low', np.subtract(tph_average, tph_std)),
                (f'{columns[i]} std_high', np.add(tph_average, tph_std))
            ]
        )

        non_zero = np.array(regroup[i + 1])[np.nonzero(regroup[i])]
        is_zero = sum(regroup[i]) > 0
        q_average, q_std = weighted_avg_and_std(regroup[i + 1], weights=regroup[i]) if is_zero else (0, 0)
        data.extend(
            [
                (f'{columns[i + 1]} min', np.min(non_zero)),
                (f'{columns[i + 1]} max', np.max(non_zero)),
                (f'{columns[i + 1]} average', q_average),
                (f'{columns[i + 1]} std_low', np.subtract(q_average, q_std)),
                (f'{columns[i + 1]} std_high', np.add(q_average, q_std))
            ] if is_zero else [
                (f'{columns[i + 1]} min', 0),
                (f'{columns[i + 1]} max', 0),
                (f'{columns[i + 1]} average', 0),
                (f'{columns[i + 1]} std_low', 0),
                (f'{columns[i + 1]} std_high', 0)
            ]
        )

    return dict(data)


class MaterialSampler:
    def __init__(self, buffer_size: int, group_size: int, stats_size: int, stats_period: float):
        """
        Class sampling material handlers each simulation step, grouping samples and calculating statistics for
        evaluation.
        :param buffer_size: maximum amount of samples to keep in buffer
        :param group_size: amount of samples to group for evaluation
        :param stats_size: amount of samples to consider for statistics calculation
        :param stats_period: period in seconds between two statistics evaluations
        """
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.stats_size = stats_size
        self.stats_period = stats_period
        self.material_handlers = []
        self.samples = []
        self.sample_group = []
        self.stats = []
        self.last_stats = 0

    def put(self, label: str, material_handler: MaterialHandler):
        """
        Register sampling point for which sample() should acquire data
        :param label: label of the sampling point
        :param material_handler: sampling point
        """
        self.material_handlers.append((label, material_handler))

    def sample(self, time):
        """
        Acquire and group samples from all registered sampling points
        :param time: current time stamp
        """
        # Acquire data
        sample_row = [time]

        for label, material_handler in self.material_handlers:
            material_handler_samples = material_handler.sample()
            for tph, quality in material_handler_samples:
                sample_row.extend([tph, quality])
        # Add data to sample group
        self.sample_group.append(sample_row)

        if len(self.sample_group) > self.group_size:
            # Average sample group
            sample_group_row = average_sample_group(self.sample_group)
            self.sample_group = []
            self.samples.append(sample_group_row)

            # Truncate buffer to maximum size
            if len(self.samples) > self.buffer_size:
                self.samples.pop(0)

        if time - self.last_stats > self.stats_period:
            self.stats.append(calculate_stats(self.last_stats, self.samples[-self.stats_size:], self.get_columns()))
            self.last_stats = time

    def evaluate(self):
        df = DataFrame(data=self.samples, columns=self.get_columns())
        df = df.set_index('time')

        axs = df.filter(regex='.* quality').replace([0], value=[None]).plot.hist(
            subplots=True,
            grid=True,
            title='Material Quality Histogram',
            bins=15
        )
        for ax in axs:
            ax.set_xlabel('Quality')
            ax.set_ylabel('Amount')

        df_quality = df.filter(regex='.* quality').replace([0], value=[None])
        ma = df_quality.rolling(20).mean()
        mstd = df_quality.rolling(20).std()
        ax = ma.plot.line(
            layout=(len(self.material_handlers), 1),
            grid=True,
            title='Material Quality Graph 5 Minute Average'
        )
        for col in df_quality.columns:
            ax.fill_between(df_quality.index, ma[col] - 2 * mstd[col], ma[col] + 2 * mstd[col], alpha=0.3, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Quality')
        ax.set_xlim(0, None)

        df_tph = df.filter(regex='.* tph')
        ma = df_tph.rolling(20).mean()
        mstd = df_tph.rolling(20).std()
        ax = ma.plot.line(
            layout=(len(self.material_handlers), 1),
            grid=True,
            title='Material Volume Graph 5 Minute Average'
        )
        for col in df_tph.columns:
            ax.fill_between(df_tph.index, ma[col] - 2 * mstd[col], ma[col] + 2 * mstd[col], alpha=0.3, zorder=-1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Tons per Hour')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        plt.show()

    def get_diff_live(self, start):
        df = DataFrame(data=self.samples[start:], columns=self.get_columns())
        return df.to_dict(orient='list')

    def get_diff_stats(self, start):
        df = DataFrame(data=self.stats[start:])
        return df.to_dict(orient='list')

    def get_columns(self):
        columns = ['time']
        for label, material_handler in self.material_handlers:
            n = len(material_handler.sample())
            if n > 1:
                for i in range(n):
                    columns.extend([f'{label} tph {i}', f'{label} quality {i}'])
            else:
                columns.extend([f'{label} tph', f'{label} quality'])

        return columns
