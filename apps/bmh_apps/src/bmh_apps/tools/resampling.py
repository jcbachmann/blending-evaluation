import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Material


def get_resampled_max_timestamp(df: pd.DataFrame, rule: str) -> float:
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")

    df = df.resample(rule, closed="right", label="right").sum()
    df = df.reset_index()
    return df["timestamp"].iloc[-1].timestamp()


def resample(material: Material, rule: str):
    # Create new material
    df = material.data.copy()

    # Preparation
    pad = {col: [0.0] for col in df.columns}
    df = pd.concat([df, pd.DataFrame(pad)]).sort_values(by=["timestamp"])

    max_timestamp = df["timestamp"].iloc[-1]
    resampled_max_timestamp = get_resampled_max_timestamp(df.copy(), rule)

    if max_timestamp < resampled_max_timestamp:
        pad["timestamp"][0] = resampled_max_timestamp
        df = pd.concat([df, pd.DataFrame(pad)])

    df["vpt"] = df["volume"] / (df["timestamp"] - df["timestamp"].shift(+1))
    df = df.drop(["volume"], axis=1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")

    # Actual resampling

    # Upsampling to 1s
    # - slow but does the job for the moment
    # - does not support data with higher resolution than 1s
    df = df.resample("1s", closed="right", label="right").apply(np.mean).bfill()

    if rule != "1s":
        # Downsampling to final resolution
        def parameter_average(s: pd.Series):
            weights = df["vpt"][s.index]
            return np.average(s, weights=weights) if weights.sum() > 0.0 else float("nan")

        funcs = dict.fromkeys(material.get_parameter_columns(), parameter_average)
        funcs.update({"vpt": np.average})
        df = df.resample(rule, closed="right", label="right").apply(funcs)

    # Cleanup
    df = df.reset_index()
    df["timestamp"] = df["timestamp"].astype(int)
    df["volume"] = df["vpt"] * (df["timestamp"] - df["timestamp"].shift(+1))
    df = df.drop(0, axis=0)
    df = df.drop(["vpt"], axis=1)

    return Material.from_data(
        df,
        identifier=f"{material.meta.identifier} resampled {rule}",
        category=material.meta.category,
    )
