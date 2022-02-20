import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from seaborn import utils
from seaborn.palettes import color_palette
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def pretty_line_plot(data, x_col=None, unique_col=None, split_col=None, y_col=None, estimator=np.mean):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if split_col is None:
        split_col = pd.Series(1, index=data.index)
        n_cond = 1
        legend_name = None
    else:
        n_cond = len(data[split_col].unique())
        legend_name = split_col

    if unique_col is None:
        unique_col = x_col

    current_palette = utils.get_color_cycle()
    if len(current_palette) < n_cond:
        colors = color_palette('husl', n_cond)
    else:
        colors = color_palette(n_colors=n_cond)

    data = data.sort_values(x_col)

    for c, (cond, df_c) in enumerate(data.groupby(split_col, sort=False)):
        x = np.array([
            df_r[x_col].values.astype(np.float)[0]
            for i, (g, df_r) in enumerate(df_c.groupby(unique_col, sort=False))
        ])

        color = colors[c]

        ax.fill_between(
            x,
            [min(df_r[y_col].values) for r, (group, df_r) in enumerate(df_c.groupby(unique_col, sort=False))],
            [max(df_r[y_col].values) for r, (group, df_r) in enumerate(df_c.groupby(unique_col, sort=False))],
            facecolor=color,
            alpha=0.2
        )

        low = []
        high = []
        for r, (group, df_r) in enumerate(df_c.groupby(unique_col, sort=False)):
            a = df_r[y_col].values
            v_low, v_high = st.t.interval(0.9, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
            low.append(v_low)
            high.append(v_high)

        ax.fill_between(x, low, high, facecolor=color, alpha=0.2)

        central_data = [
            estimator(df_r[y_col].values)
            for r, (group, df_r) in enumerate(df_c.groupby(unique_col, sort=False))
        ]
        ax.plot(x, central_data, color=color, label=cond, marker='', linestyle='-')

    # ax.set_xlim(x.min(), x.max())

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if legend_name is not None:
        ax.legend(loc=0, title=legend_name)

    return ax


def pretty_scatter_plot(data, x_col=None, split_col=None, y_col=None, log_scale=False, equal=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if split_col is None:
        split_col = pd.Series(1, index=data.index)
        n_cond = 1
        legend_name = None
    else:
        n_cond = len(data[split_col].unique())
        legend_name = split_col

    current_palette = utils.get_color_cycle()
    if len(current_palette) < n_cond:
        colors = color_palette('husl', n_cond)
    else:
        colors = color_palette(n_colors=n_cond)

    data = data.sort_values(x_col)

    if equal:
        v_min = min(data[x_col].min(), data[y_col].min())
        v_max = max(data[x_col].max(), data[y_col].max())
        ax.plot([v_min, v_max], [v_min, v_max], color='black', label='1:1', marker='', linestyle='-')

    for c, (cond, df_c) in enumerate(data.groupby(split_col, sort=False)):
        x = df_c[x_col].values
        y = df_c[y_col].values

        regression = LinearRegression()
        regression.fit(x.reshape(-1, 1), y)
        y_prediction = regression.predict(x.reshape(-1, 1))

        label = str(cond) if legend_name is not None else ''
        label += f' (coeff={regression.coef_[0]:.3f}, R²={r2_score(y, y_prediction):.1f})'

        ax.plot(x, y_prediction, color=colors[c], marker='', linestyle='-')
        ax.plot(x, y, color=colors[c], label=label, marker='x', linestyle='')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if log_scale:
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')

    if equal:
        ax.set_aspect('equal')

    if legend_name is not None:
        ax.legend(loc=0, title=legend_name)
    else:
        ax.legend(loc=0)

    return ax
