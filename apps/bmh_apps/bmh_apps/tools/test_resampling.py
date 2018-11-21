#!/usr/bin/env python
import unittest

import numpy as np
import pandas as pd


# Resampling experiment
def resample(df, rule: str):
    df = df.append(pd.DataFrame([[0, 0, 0]], columns=df.columns))
    # df['t'] = pd.to_timedelta(df['t'], unit='s')
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df.set_index('t', inplace=True)
    resampler = df.resample(rule, closed='right', label='right')
    df = resampler.backfill()
    df = resampler.interpolate()
    df = resampler.agg({'q': np.mean, 'v': np.sum})
    df.reset_index(inplace=True)
    # df['t'] = (df['t'] / np.timedelta64(1, 's')).astype(int)
    df['t'] = (df['t'].astype(int) // 10 ** 9)
    df.drop(0, axis=0, inplace=True)
    return df


@unittest.skip
class TestResampling(unittest.TestCase):
    def test_downsampling_t(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'v': [1, 1, 1, 1, 1, 1]
        })

        r = resample(df, '3s').to_dict(orient='list')
        self.assertEqual([3, 6], r['t'])

    def test_downsampling_q(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'v': [1, 1, 1, 1, 1, 1]
        })

        r = resample(df, '3s').to_dict(orient='list')
        self.assertEqual([4 / 3, 5 / 3], r['q'])

    def test_downsampling_v(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'v': [1, 1, 1, 1, 1, 1]
        })

        r = resample(df, '3s').to_dict(orient='list')
        self.assertEqual([3, 3], r['v'])

    def test_upsampling_t(self):
        df = pd.DataFrame({
            't': [10, 20],
            'q': [1, 2],
            'v': [2, 4]
        })

        r = resample(df, '5s').to_dict(orient='list')
        self.assertEqual([5, 10, 15, 20], r['t'])

    def test_upsampling_q(self):
        df = pd.DataFrame({
            't': [10, 20],
            'q': [1, 2],
            'v': [2, 4]
        })

        r = resample(df, '5s').to_dict(orient='list')
        self.assertEqual([1, 1, 2, 2], r['q'])

    def test_upsampling_v(self):
        df = pd.DataFrame({
            't': [10, 20],
            'q': [1, 2],
            'v': [2, 4]
        })

        r = resample(df, '5s').to_dict(orient='list')
        self.assertEqual([1, 1, 2, 2], r['v'])

    def test_combined_t(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
            'q': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            'v': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        })

        r = resample(df, '5s').to_dict(orient='list')
        self.assertEqual([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], r['t'])

    def test_combined_q(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
            'q': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            'v': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        })

        r = resample(df, '5s').to_dict(orient='list')
        self.assertEqual([], r['q'])

    def test_combined_v(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6, 7, 8, 10, 30, 100],
            'q': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            'v': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        })

        r = resample(df, '5s').to_dict(orient='list')
        self.assertEqual([], r['v'])

    def test_volume_weighting_t(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'v': [1, 2, 1, 2, 1, 2]
        })

        r = resample(df, '3s').to_dict(orient='list')
        self.assertEqual([3, 6], r['t'])

    def test_volume_weighting_q(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'v': [1, 2, 1, 2, 1, 2]
        })

        r = resample(df, '3s').to_dict(orient='list')
        self.assertEqual([1.5, 1.8], r['q'])

    def test_volume_weighting_v(self):
        df = pd.DataFrame({
            't': [1, 2, 3, 4, 5, 6],
            'q': [1, 2, 1, 2, 1, 2],
            'v': [1, 2, 1, 2, 1, 2]
        })

        r = resample(df, '3s').to_dict(orient='list')
        self.assertEqual([4, 5], r['v'])
