import argparse
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bmh_ml.metrics import get_accuracy
from bmh_ml.settings import (
    BED_SIZE_X,
    BED_SIZE_Z,
    DEPOSITION_LENGTH,
    MATERIAL_LENGTH,
    MATERIAL_MAX,
    MATERIAL_MIN,
    TOTAL_VOLUME,
    X_MAX,
    X_MIN,
    add_model_set_argument,
)
from bmh_ml.simulation import evaluate_sim
from bmh_ml.surrogate import Surrogate
from bmh_ml.variables import generate_deposition_variables, generate_material_variables


def plot_linked_f1_f2(
    predicted: np.ndarray,
    expected: np.ndarray,
    title: str = "Predicted vs Expected in F1-F2 space",
) -> None:
    fig = go.Figure()

    # Predicted points in blue
    fig.add_trace(
        go.Scatter(
            x=predicted[:, 0],
            y=predicted[:, 1],
            mode="markers",
            name="Predicted",
            marker={"color": "blue", "size": 8},
        )
    )

    # Expected points in red
    fig.add_trace(
        go.Scatter(
            x=expected[:, 0],
            y=expected[:, 1],
            mode="markers",
            name="Expected",
            marker={"color": "red", "size": 8},
        )
    )

    # Connect corresponding points
    for (p1, p2), (e1, e2) in zip(predicted, expected, strict=False):
        fig.add_trace(
            go.Scatter(
                x=[p1, e1],
                y=[p2, e2],
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
    )

    # Show interactive figure
    fig.show()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info(f"Evaluating model set '{args.model_set}'")
    surrogate = Surrogate.load(args.model_set)

    f1_predicted = []
    f2_predicted = []
    f1_expected_values = []
    f2_expected_values = []

    for _ in range(100):
        material_variables = generate_material_variables(
            material_length=MATERIAL_LENGTH,
            material_min=MATERIAL_MIN,
            material_max=MATERIAL_MAX,
        )
        deposition_variables = generate_deposition_variables(deposition_length=DEPOSITION_LENGTH, x_min=X_MIN, x_max=X_MAX)
        f1_expected, f2_expected = evaluate_sim(
            material_variables=material_variables,
            deposition_variables=deposition_variables,
            bed_size_x=BED_SIZE_X,
            bed_size_z=BED_SIZE_Z,
            total_volume=TOTAL_VOLUME,
        )
        f1, f2 = surrogate.predict(np.concatenate([material_variables, deposition_variables]).reshape(1, -1))[0]
        f1_predicted.append(f1)
        f2_predicted.append(f2)
        f1_expected_values.append(f1_expected)
        f2_expected_values.append(f2_expected)

    for objective, expected, predicted in (("F1", f1_expected_values, f1_predicted), ("F2", f2_expected_values, f2_predicted)):
        accuracy = get_accuracy(expected, predicted)
        logging.info(f"{objective}: R2 {accuracy['r2']:.3f}, mean absolute error {accuracy['mean_absolute_error']:.4f}")

    df_f1 = pd.DataFrame({"Expected": f1_expected_values, "Predicted": f1_predicted})
    df_f2 = pd.DataFrame({"Expected": f2_expected_values, "Predicted": f2_predicted})

    fig1 = px.scatter(df_f1, x="Expected", y="Predicted", title="F1 Predictions vs Expected")
    fig1.add_scatter(
        x=[min(f1_expected_values), max(f1_expected_values)],
        y=[min(f1_expected_values), max(f1_expected_values)],
        line={"color": "red", "dash": "dash"},
        name="Perfect Prediction",
    )
    fig1.show()

    fig2 = px.scatter(df_f2, x="Expected", y="Predicted", title="F2 Predictions vs Expected")
    fig2.add_scatter(
        x=[min(f2_expected_values), max(f2_expected_values)],
        y=[min(f2_expected_values), max(f2_expected_values)],
        line={"color": "red", "dash": "dash"},
        name="Perfect Prediction",
    )
    fig2.show()

    # Linked plot: (f1,f2) predicted vs (f1_expected,f2_expected)
    predicted_arr = np.column_stack([f1_predicted, f2_predicted]).astype(float)
    expected_arr = np.column_stack([f1_expected_values, f2_expected_values]).astype(float)
    plot_linked_f1_f2(
        predicted_arr,
        expected_arr,
        title="Linked Predicted vs Expected in F1-F2 space",
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    add_model_set_argument(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
