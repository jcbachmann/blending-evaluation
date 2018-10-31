import threading

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

app = dash.Dash('dash-plot-server')
server = app.server
global_all_callback = None
global_pop_callback = None
global_pop_size = 0

app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='scatter',
            hoverData={'points': [{'customdata': 'path1'}]}
        )
    ]),
    dcc.Interval(id='scatter-update', interval=500)
])


@app.callback(Output('scatter', 'figure'), [Input('scatter-update', 'n_intervals')])
def update_scatter(_interval):
    all_data = global_all_callback(0)
    pop_data = global_pop_callback()
    all_data_df = pd.DataFrame(data=all_data)
    pop_data_df = pd.DataFrame(data=pop_data)

    all_data_scatter = go.Scattergl(
        name='All Evaluations',
        x=all_data_df['f1'],
        y=all_data_df['f2'],
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
        x=pop_data_df['f1'],
        y=pop_data_df['f2'],
        marker=go.scattergl.Marker(
            symbol='circle-open',
            color='#FF0000',
            size=10,
        ),
        mode='markers'
    )

    x_min = pop_data_df['f1'].min()
    x_max = pop_data_df['f1'].max()
    x_range = x_max - x_min
    x_offset = x_range * 0.1
    y_min = pop_data_df['f2'].min()
    y_max = pop_data_df['f2'].max()
    y_range = y_max - y_min
    y_offset = y_range * 0.1

    layout = go.Layout(
        height=800,
        width=800,
        xaxis=dict(
            range=[x_min - x_offset, x_max + x_offset],
            title='f1 Homogenization Effect'
        ),
        yaxis=dict(
            range=[y_min - y_offset, y_max + y_offset],
            title='f2 Volume StDev'
        ),
        margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
        hovermode='closest',
    )

    return go.Figure(data=[all_data_scatter, pop_data_scatter], layout=layout)


class PlotServer:
    PORT = 5001

    def __init__(self, all_callback, pop_callback, pop_size):
        global global_all_callback
        global global_pop_callback
        global global_pop_size
        global_all_callback = all_callback
        global_pop_callback = pop_callback
        global_pop_size = pop_size

    def serve(self):
        app.run_server(port=PlotServer.PORT)

    def serve_background(self):
        t = threading.Thread(target=self.serve)
        t.start()
