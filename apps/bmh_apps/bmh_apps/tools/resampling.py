import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Material


def resample(material: Material, rule: str):
    df = material.data.copy()
    df = df.append(pd.DataFrame([[0 for _ in df.columns]], columns=df.columns))
    df = df.sort_values(by=['timestamp'])
    df['t_diff'] = df['timestamp'] - df['timestamp'].shift(+1)
    df.fillna(method='backfill', inplace=True)
    df['tph'] = df['volume'] / df['t_diff']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    resampler = df.resample(rule, closed='right', label='right')

    def parameter_average(s: pd.Series):
        weights = df['volume'][s.index]
        if weights.sum() > 0.0:
            return np.average(s, weights=weights)
        else:
            return float('nan')

    def tph_average(s: pd.Series):
        weights = df['t_diff'][s.index]
        if weights.sum() > 0.0:
            return np.average(s, weights=weights)
        else:
            return float('nan')

    func_dict = {
        'volume': np.sum,
        'tph': tph_average
    }
    func_dict.update({
        p: parameter_average for p in material.get_parameter_columns()
    })
    df = resampler.apply(func_dict).fillna(method='backfill')
    df.reset_index(inplace=True)
    df['timestamp'] = (df['timestamp'].astype(int) // 10 ** 9)
    df['volume'] = df['tph'] * (df['timestamp'] - df['timestamp'].shift(+1))
    df.drop(0, axis=0, inplace=True)
    df.drop(['tph'], axis=1, inplace=True)
    return Material.from_data(df, identifier=f'{material.meta.identifier} resampled {rule}')
