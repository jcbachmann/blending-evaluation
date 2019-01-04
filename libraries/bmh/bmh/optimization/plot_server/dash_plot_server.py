import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from .plot_server import PlotServer, PlotServerInterface

app = dash.Dash('dash-plot-server')
global_plot_server_interface: PlotServerInterface = None

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


@app.callback(Output('scatter', 'figure'), [Input('scatter-update', 'n_intervals')])
def update_scatter(_interval):
    all_data = global_plot_server_interface.get_new_solutions(0)
    pop_data = global_plot_server_interface.get_population()

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
            range=[0, 2],
            title='f1 Homogenization Effect'
        ),
        yaxis=dict(
            range=[0, 2],
            title='f2 Volume StDev'
        ),
        margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
        hovermode='closest',
    )

    return go.Figure(data=[all_data_scatter, pop_data_scatter], layout=layout)


@app.callback(Output('path', 'figure'), [Input('scatter', 'hoverData')])
def update_path(hover_data):
    deposition = global_plot_server_interface.get_deposition(hover_data['points'][0]['customdata'])

    path = go.Scattergl(
        name='Population',
        x=deposition.data['timestamp'],
        y=deposition.data['x'],
        line=go.scattergl.Line(
            color='#FF0000',
        ),
        mode='lines'
    )

    layout = go.Layout(
        height=700,
        yaxis=dict(range=[0, deposition.meta.bed_size_x])
    )

    return go.Figure(data=[path], layout=layout)


class DashPlotServer(PlotServer):
    def __init__(self, plot_server_interface: PlotServerInterface):
        super().__init__(plot_server_interface)

        global global_plot_server_interface
        global_plot_server_interface = plot_server_interface

    def serve(self) -> None:
        app.server.env = 'development'
        app.run_server(port=self.port)

    def stop(self) -> None:
        raise NotImplementedError()
        # This does not work :(
        # with app.server.test_request_context():
        #     from flask import request
        #     if not 'werkzeug.server.shutdown' in request.environ:
        #         raise RuntimeError('Not running the development server')
        #     request.environ['werkzeug.server.shutdown']()
