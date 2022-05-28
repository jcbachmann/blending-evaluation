import argparse
import os

import dash.html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, Input, Output, dash_table


def main(args: argparse.Namespace):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.config['suppress_callback_exceptions'] = True

    app.layout = dash.html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dash_table.DataTable(
                            [{"Run": f"Run {i}"} for i, _ in enumerate(args.experiment)],
                            [{"name": "Run", "id": "Run"}],
                            id='runs',
                            style_table={'max-height': '500px', 'overflowY': 'auto'}
                        ),
                    ),
                    style={'height': '100%'}
                ),
                xl=1, lg=2, sm=2
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(id='front-evolution', style={'height': '100%'})
                    ),
                    style={'height': '100%'}
                ),
                xl=11, lg=10, sm=10,
            ),
        ], className='g-0', style={'flex': '1'}),
        dbc.Row(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(
                        [
                            dcc.Graph(id='quality-indicators', style={'flex': '1'}),
                            dcc.Slider(
                                min=0, max=0, step=100, value=0,
                                id='generation-slider', updatemode='drag', marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={'display': 'flex', 'flexDirection': 'column'}
                    )
                ], style={'height': '100%'})
            ),
            className='g-0',
            style={'height': '300px'}
        )
    ], style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

    def get_row(active_cell):
        return active_cell['row'] if active_cell else 0

    @app.callback(
        Output('generation-slider', 'max'),
        Input('runs', 'active_cell')
    )
    def render_content(active_cell):
        quality_indicators_df = pd.read_csv(
            os.path.join(args.experiment[get_row(active_cell)], 'quality_indicators.csv'))
        return len(quality_indicators_df) - 1

    @app.callback(
        Output('front-evolution', 'figure'),
        Input('runs', 'active_cell'),
        Input('generation-slider', 'value')
    )
    def update_figure(active_cell, generation: int):
        objective_labels = ['F1/Ash (%)', 'F1/Sulphur (%)', 'F2']
        fun_df = pd.read_csv(
            os.path.join(args.experiment[get_row(active_cell)], 'fronts', f'FUN.{generation}'),
            sep=' ',
            header=None,
            index_col=False,
            names=objective_labels
        )

        reference_df = pd.read_csv(
            os.path.join(args.reference),
            sep=' ',
            header=None,
            index_col=False,
            names=objective_labels
        )

        layout = go.Layout(
            legend=dict(
                orientation="h",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(range=[0, 2], title=objective_labels[0]),
                yaxis=dict(range=[0, 2], title=objective_labels[1]),
                zaxis=dict(range=[0, 2], title=objective_labels[2]),
                aspectmode='cube',
            ),
            hovermode='closest',
            uirevision=True
        )

        data = []

        trace = go.Scatter3d(
            x=reference_df[objective_labels[0]],
            y=reference_df[objective_labels[1]],
            z=reference_df[objective_labels[2]],
            mode='markers',
            marker=dict(
                color='black',
                size=2,
                symbol='circle',
                line=dict(
                    color='#236FA4',
                    width=1
                ),
                opacity=0.8
            ),
            name='Reference front'
        )
        data.append(trace)

        trace = go.Scatter3d(
            x=[1],
            y=[1],
            z=[1],
            mode='markers',
            marker=dict(
                color='red',
                size=8,
                symbol='circle',
                line=dict(
                    color='#236FA4',
                    width=1
                ),
                opacity=0.8
            ),
            name='Reference point'
        )
        data.append(trace)

        trace = go.Scatter3d(
            x=fun_df[objective_labels[0]],
            y=fun_df[objective_labels[1]],
            z=fun_df[objective_labels[2]],
            mode='markers',
            marker=dict(
                color='#236FA4',
                size=4,
                symbol='circle',
                line=dict(
                    color='#236FA4',
                    width=1
                ),
                opacity=0.8
            ),
            name='Front approximation'
        )
        data.append(trace)

        fig = go.Figure(data=data, layout=layout)

        return fig

    @app.callback(
        Output('quality-indicators', 'figure'),
        Input('runs', 'active_cell')
    )
    def update_figure(active_cell):
        quality_indicators_df = pd.read_csv(
            os.path.join(args.experiment[get_row(active_cell)], 'quality_indicators.csv'))
        fig = px.line(quality_indicators_df)
        fig.update_layout(
            legend=dict(
                orientation="h",
                x=0,
                y=1,
                xanchor="left",
                yanchor="top",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            uirevision=True
        )
        fig.update_yaxes(range=[0, 0.5])

        return fig

    app.run_server(debug=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, nargs='+', help='Path to directory containing experiment results')
    parser.add_argument("--reference", type=str, required=True, help='Path to reference front')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
