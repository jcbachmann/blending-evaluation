"""Plots of the predictions of a model, one figure for each test set."""

import numpy as np

from bmh_ml.evaluation.noise import OBJECTIVES


def plot_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict[str, float]):
    """Predicted against true values and the error against the true values, for F1 and F2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for i, objective in enumerate(OBJECTIVES):
        true, pred = y_true[:, i], y_pred[:, i]
        low, high = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        parity, residual = axes[i]
        parity.scatter(true, pred, s=4, alpha=0.4)
        parity.plot([low, high], [low, high], color="black", linewidth=1)
        parity.set(xlabel=f"{objective} simulator", ylabel=f"{objective} model", title=f"{name} {objective}: R2 {metrics[f'{objective}/r2']:.4f}")
        residual.scatter(true, pred - true, s=4, alpha=0.4)
        residual.axhline(0, color="black", linewidth=1)
        residual.set(xlabel=f"{objective} simulator", ylabel="model - simulator", title=f"rmse {metrics[f'{objective}/rmse']:.4f}")
    return figure


def close_figure(figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)
