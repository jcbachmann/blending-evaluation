"""How much of the variation of the labels is noise of the simulator, and so the best a model can reach."""

import numpy as np

from bmh_ml.datasets.manifest import Dataset

OBJECTIVES = ("F1", "F2")


def get_noise_summary(dataset: Dataset) -> dict[str, dict[str, float]]:
    """Per objective: the noise of a single simulation, the variance of the labels and the highest R2 a model can reach.

    The noise is the pooled standard deviation of the repeated simulations of the same input. `r2_ceiling_labels` is the R2 of a model that
    predicts the true mean exactly, measured against these labels (the mean of `repeats` simulations), `r2_ceiling_single` measured against
    a single simulation, as a model would be judged in the optimization. Needs a dataset with repeats.
    """
    if dataset.y_noise_sd is None:
        raise ValueError(f"Dataset {dataset.name} has no repeated simulations, so its noise is unknown")
    summary = {}
    for i, objective in enumerate(OBJECTIVES):
        noise_variance = float(np.mean(dataset.y_noise_sd[:, i] ** 2))
        label_variance = float(np.var(dataset.y[:, i]))
        true_variance = max(label_variance - noise_variance / dataset.repeats, 0.0)
        summary[objective] = {
            "noise_sd": noise_variance**0.5,
            "label_sd": label_variance**0.5,
            "r2_ceiling_labels": true_variance / (true_variance + noise_variance / dataset.repeats),
            "r2_ceiling_single": true_variance / (true_variance + noise_variance),
        }
    return summary


def format_noise_table(summaries: dict[str, dict[str, dict[str, float]]]) -> str:
    """A table with a row per dataset and objective."""
    lines = [f"{'dataset':<28}{'obj':<5}{'noise sd':>10}{'label sd':>10}{'R2 ceiling (labels)':>21}{'R2 ceiling (single run)':>25}"]
    for name, summary in summaries.items():
        for objective, values in summary.items():
            lines.append(
                f"{name:<28}{objective:<5}{values['noise_sd']:>10.4f}{values['label_sd']:>10.4f}"
                f"{values['r2_ceiling_labels']:>21.4f}{values['r2_ceiling_single']:>25.4f}"
            )
    return "\n".join(lines)
