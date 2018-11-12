from typing import List, Callable, Optional

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from .plot_server import PlotServer

app = dash.Dash('dash-plot-server')
server = app.server
global_all_callback: Optional[Callable] = None
global_pop_callback: Optional[Callable] = None
global_path_callback: Optional[Callable] = None

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='scatter',
            hoverData={'points': [{'customdata': 0}]}
        )
    ]),
    html.Div([
        dcc.Graph(id='path'),
    ]),
    dcc.Interval(id='scatter-update', interval=500)
])


def get_margin_range(data: List[float], relative_margin: float = 0.1):
    if len(data) == 0:
        return None

    min_val = min(data)
    max_val = max(data)
    diff = max_val - min_val
    offset = diff * relative_margin
    return [min_val - offset, max_val + offset]


@app.callback(Output('scatter', 'figure'), [Input('scatter-update', 'n_intervals')])
def update_scatter(_interval):
    all_data = global_all_callback(0)
    pop_data = global_pop_callback()

    all_data_scatter = go.Scattergl(
        name='All Evaluations',
        x=all_data['f1'],
        y=all_data['f2'],
        customdata=list(range(len(all_data['f1']))),
        marker=go.scattergl.Marker(
            symbol='x',
            color='#0000FF',
            opacity=0.5,
            colorscale='Viridis',
        ),
        mode='markers'
    )

    pop_data_scatter = go.Scattergl(
        name='Population',
        x=pop_data['f1'],
        y=pop_data['f2'],
        marker=go.scattergl.Marker(
            symbol='circle-open',
            color='#FF0000',
            size=10,
        ),
        mode='markers'
    )

    layout = go.Layout(
        height=600,
        xaxis=dict(
            range=get_margin_range(pop_data['f1']),
            title='f1 Homogenization Effect'
        ),
        yaxis=dict(
            range=get_margin_range(pop_data['f2']),
            title='f2 Volume StDev'
        ),
        margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
        hovermode='closest',
    )

    return go.Figure(data=[all_data_scatter, pop_data_scatter], layout=layout)


@app.callback(Output('path', 'figure'), [Input('scatter', 'hoverData')])
def update_path(hover_data):
    path_id = hover_data['points'][0]['customdata']
    path = global_path_callback(path_id)

    layout = go.Layout(
        height=700,
    )

    path = go.Scattergl(
        name='Population',
        x=path,
        y=list(range(len(path))),
        line=go.scattergl.Line(
            color='#FF0000',
        ),
        mode='lines'
    )

    return go.Figure(data=[path], layout=layout)


class DashPlotServer(PlotServer):
    def __init__(self, all_callback, pop_callback, path_callback):
        super().__init__(all_callback, pop_callback, path_callback)

        global global_all_callback
        global global_pop_callback
        global global_path_callback
        global_all_callback = all_callback
        global_pop_callback = pop_callback
        global_path_callback = path_callback

    def serve(self) -> None:
        app.run_server(port=self.port)