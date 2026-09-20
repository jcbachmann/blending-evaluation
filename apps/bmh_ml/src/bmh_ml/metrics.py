import numpy as np


def get_accuracy(expected: list[float], predicted: list[float]) -> dict[str, float]:
    """Coefficient of determination (1 is a perfect prediction, 0 is as good as predicting the mean) and the mean absolute error."""
    expected_values = np.asarray(expected, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    residual_sum = float(np.sum((expected_values - predicted_values) ** 2))
    total_sum = float(np.sum((expected_values - expected_values.mean()) ** 2))
    return {
        "r2": 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan"),
        "mean_absolute_error": float(np.mean(np.abs(expected_values - predicted_values))),
    }
