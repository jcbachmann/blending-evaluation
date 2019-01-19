import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Material


def get_resampled_max_timestamp(df: pd.DataFrame, rule: str) -> float:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    df = df.resample(rule, closed='right', label='right').sum()
    df.reset_index(inplace=True)
    return df['timestamp'].values[-1].astype(int) / 10 ** 9


def resample(material: Material, rule: str):
    # Create new material
    df = material.data.copy()

    # Preparation
    pad = {col: [0.0] for col in df.columns}
    df = df.append(pd.DataFrame(pad)).sort_values(by=['timestamp'])

    max_timestamp = df['timestamp'].values[-1]
    resampled_max_timestamp = get_resampled_max_timestamp(df.copy(), rule)

    if max_timestamp < resampled_max_timestamp:
        pad['timestamp'][0] = resampled_max_timestamp
        df = df.append(pd.DataFrame(pad))

    df['vpt'] = df['volume'] / (df['timestamp'] - df['timestamp'].shift(+1))
    df.drop(['volume'], axis=1, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    # Actual resampling

    # Upsampling to 1s
    # - slow but does the job for the moment
    # - does not support data with higher resolution than 1s
    df = df.resample('1s', closed='right', label='right').apply(np.mean).fillna(method='backfill')

    if rule != '1s':
        # Downsampling to final resolution
        def parameter_average(s: pd.Series):
            weights = df['vpt'][s.index]
            return np.average(s, weights=weights) if weights.sum() > 0.0 else float('nan')

        funcs = {p: parameter_average for p in material.get_parameter_columns()}
        funcs.update({'vpt': np.average})
        df = df.resample(rule, closed='right', label='right').apply(funcs)

    # Cleanup
    df.reset_index(inplace=True)
    df['timestamp'] = (df['timestamp'].astype(int) // 10 ** 9)
    df['volume'] = df['vpt'] * (df['timestamp'] - df['timestamp'].shift(+1))
    df.drop(0, axis=0, inplace=True)
    df.drop(['vpt'], axis=1, inplace=True)

    return Material.from_data(
        df,
        identifier=f'{material.meta.identifier} resampled {rule}',
        category=material.meta.category
    )
