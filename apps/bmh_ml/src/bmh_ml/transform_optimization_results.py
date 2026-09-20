import argparse
import json
import logging
import os
import pickle
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf

from bmh_ml.simulation import evaluate_sim

# Constants aligned with optimize_model.py and optimize_simulation.py
MATERIAL_LENGTH: int = 50
DEPOSITION_LENGTH: int = 20
BED_SIZE_X: int = 59
BED_SIZE_Z: int = 20
TOTAL_VOLUME: int = 2500


def get_recomputed_cache_path(original_path: str, suffix: str) -> str:
    return os.path.splitext(original_path)[0] + f"_recomputed_{suffix}.json"


def save_recomputed_results(path: str, variables: np.ndarray, objectives: np.ndarray) -> None:
    with open(path, "w") as f:
        json.dump({"variables": variables.tolist(), "objectives": objectives.tolist()}, f)


def load_recomputed_results(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            return (np.array(data["variables"]), np.array(data["objectives"]))
    return None


def compute_with_simulation(dep_batch: np.ndarray, material_variables) -> np.ndarray:
    # dep_batch shape: (n, 20). Deposition variables are absolute positions (no scaling).
    results: list[tuple[float, float]] = []
    for dep in dep_batch:
        f1, f2 = evaluate_sim(
            material_variables=material_variables,
            deposition_variables=dep,
            bed_size_x=BED_SIZE_X,
            bed_size_z=BED_SIZE_Z,
            total_volume=TOTAL_VOLUME,
        )
        results.append((float(f1), float(f2)))
    return np.array(results, dtype=float)


def compute_with_lstm(dep_batch: np.ndarray, material_variables, scaler, lstm_model_f1, lstm_model_f2) -> np.ndarray:
    # Build full input vectors: material (fixed) + deposition
    n = dep_batch.shape[0]
    full_x = np.hstack([np.tile(material_variables, (n, 1)), dep_batch.astype(float)])  # shape (n, 70)
    x_scaled = scaler.transform(full_x)
    x_lstm = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
    f1 = lstm_model_f1.predict(x_lstm, verbose=0).flatten()
    f2 = lstm_model_f2.predict(x_lstm[:, :, MATERIAL_LENGTH:], verbose=0).flatten()  # F2 only depends on the deposition
    return np.column_stack([f1, f2]).astype(float)


def plot_before_after(
    original: np.ndarray,
    recomputed: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    fig = go.Figure()

    # Original points in red
    fig.add_trace(
        go.Scatter(
            x=original[:, 0],
            y=original[:, 1],
            mode="markers",
            name="original",
            marker={"color": "red", "size": 8},
        )
    )

    # Recomputed points in blue
    fig.add_trace(
        go.Scatter(
            x=recomputed[:, 0],
            y=recomputed[:, 1],
            mode="markers",
            name="recomputed",
            marker={"color": "blue", "size": 8},
        )
    )

    # Connect corresponding points
    for (o1, o2), (r1, r2) in zip(original, recomputed, strict=False):
        fig.add_trace(
            go.Scatter(
                x=[o1, r1],
                y=[o2, r2],
                mode="lines",
                line={"color": "gray", "width": 0.5},
                showlegend=False,
                opacity=0.7,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="F1",
        yaxis_title="F2",
        width=700,
        height=600,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_image(out_path)


def plot_metric_scatter(
    original: np.ndarray,
    recomputed: np.ndarray,
    metric_index: int,
    title: str,
    out_path: str,
) -> None:
    """Scatter plot comparing a single metric (F1 or F2) original vs recomputed.

    x: original[:, metric_index]
    y: recomputed[:, metric_index]
    """
    fig = go.Figure()
    x = original[:, metric_index]
    y = recomputed[:, metric_index]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="points",
            marker={"color": "blue", "size": 8},
        )
    )

    # Add y=x reference line for guidance
    min_val = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    max_val = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="y=x",
            line={"color": "gray", "dash": "dash"},
        )
    )

    axis_label = f"F{metric_index + 1}"
    fig.update_layout(
        title=title,
        xaxis_title=f"original {axis_label}",
        yaxis_title=f"recomputed {axis_label}",
        width=700,
        height=600,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_image(out_path)


def load_result_file(path: str, expected_variable_length: int) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    variables = np.asarray(data["variables"], dtype=float)
    objectives = np.asarray(data["objectives"], dtype=float)
    if variables.shape[1] != expected_variable_length:
        logging.warning(f"Unexpected variable length in {path}: {variables.shape[1]}")
    return variables, objectives


def recompute_cached(path: str, cache_suffix: str, variables: np.ndarray, compute: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    cache_path = get_recomputed_cache_path(path, cache_suffix)
    cached_results = load_recomputed_results(cache_path)
    if cached_results is not None:
        return cached_results[1]
    recomputed = compute(variables)
    save_recomputed_results(cache_path, variables, recomputed)
    return recomputed


def save_comparison_plots(path: str, objectives: np.ndarray, recomputed: np.ndarray, cache_suffix: str, description: str) -> None:
    base_no_ext = os.path.splitext(path)[0]
    file_name = os.path.basename(path)

    img_out = f"{base_no_ext}_recomputed_with_{cache_suffix}.png"
    plot_before_after(objectives, recomputed, f"{file_name} — {description}", img_out)
    logging.info(f"Saved plot: {img_out}")

    # Additional 1D scatter plots: F1 and F2
    img_f1 = f"{base_no_ext}_recomputed_with_{cache_suffix}_F1.png"
    img_f2 = f"{base_no_ext}_recomputed_with_{cache_suffix}_F2.png"
    plot_metric_scatter(objectives, recomputed, 0, f"{file_name} — F1: {description}", img_f1)
    plot_metric_scatter(objectives, recomputed, 1, f"{file_name} — F2: {description}", img_f2)
    logging.info(f"Saved plots: {img_f1}, {img_f2}")


def iter_json_files(root: str) -> Iterable[str]:
    for name in sorted(os.listdir(root)):
        # Skip the cache files written by get_recomputed_cache_path, they are not optimization results
        if name.endswith(".json") and "_recomputed_" not in name:
            yield os.path.join(root, name)


def recompute_lstm_results_with_simulation(material_variables: np.ndarray) -> None:
    lstm_dir = os.path.join("output", "lstm_model")
    if not os.path.isdir(lstm_dir):
        logging.info("No directory output/lstm_model found — skipping")
        return

    for path in iter_json_files(lstm_dir):
        try:
            # variables here are full vectors (70): first 50 materials (fixed), last 20 depositions
            variables, objectives = load_result_file(path, MATERIAL_LENGTH + DEPOSITION_LENGTH)
            dep_vars = variables[:, -DEPOSITION_LENGTH:]
            recomputed = recompute_cached(path, "sim", dep_vars, lambda dep: compute_with_simulation(dep, material_variables))
            save_comparison_plots(path, objectives, recomputed, "sim", "LSTM original vs Simulation recomputed")
        except Exception as e:
            logging.exception(f"Failed to process {path}: {e}")


def recompute_simulation_results_with_lstm(material_variables: np.ndarray) -> None:
    sim_dir = os.path.join("output", "simulation")
    if not os.path.isdir(sim_dir):
        logging.info("No directory output/simulation found — skipping")
        return

    logging.info("Loading scaler and LSTM models")
    with open("data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # noqa: S301 - written by train_lstm_model.py
    lstm_model_f1 = tf.keras.models.load_model("data/lstm_model_f1_random_training_data.keras")
    lstm_model_f2 = tf.keras.models.load_model("data/lstm_model_f2_random_training_data.keras")

    for path in iter_json_files(sim_dir):
        try:
            # variables here are deposition-only (20)
            variables, objectives = load_result_file(path, DEPOSITION_LENGTH)
            recomputed = recompute_cached(path, "lstm", variables, lambda dep: compute_with_lstm(dep, material_variables, scaler, lstm_model_f1, lstm_model_f2))
            save_comparison_plots(path, objectives, recomputed, "lstm", "SIM original vs LSTM recomputed")
        except Exception as e:
            logging.exception(f"Failed to process {path}: {e}")


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Load fixed material variables (same as used during optimization)
    logging.info("Loading training data to obtain fixed material variables")
    training_data = pd.read_csv("data/training_data.csv")
    example_row = training_data.iloc[0]
    material_variables = example_row.iloc[2 : 2 + MATERIAL_LENGTH].to_numpy().astype(float)

    recompute_lstm_results_with_simulation(material_variables)
    recompute_simulation_results_with_lstm(material_variables)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
