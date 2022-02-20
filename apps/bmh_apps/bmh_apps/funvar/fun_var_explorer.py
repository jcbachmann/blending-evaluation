import argparse
import logging

import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from .fun_var_math import filter_efficient_front
from .fun_var_results import FunVarResults


def get_fun_figure(results: FunVarResults, selected_points, selected_range, previous_visible):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=results.df[results.fun_columns[0]],
        y=results.df[results.fun_columns[1]],
        mode='markers',
        marker=dict(size=5),
        selectedpoints=selected_points,
        customdata=results.df.index,
        visible=previous_visible[0] if len(previous_visible) > 0 else True,
        name='All fronts',
    ))

    efficient_df = filter_efficient_front(results.df, results.fun_columns)
    fig.add_trace(go.Scattergl(
        x=efficient_df[results.fun_columns[0]],
        y=efficient_df[results.fun_columns[1]],
        mode='markers',
        marker=dict(
            size=8,
            color='rgba(0, 0, 0, 0)',
            line_width=1,
            line_color='red'
        ),
        customdata=efficient_df.index,
        visible=previous_visible[1] if len(previous_visible) > 1 else True,
        name='Efficient Front',
    ))
    fig.update_layout(
        title='Objective Space',
        xaxis_title=results.fun_columns[0],
        yaxis_title=results.fun_columns[1],
        dragmode='select',
        xaxis_range=[0, 2],
        yaxis_range=[0, 2],
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    if selected_range:
        fig.add_shape(dict(
            type='rect',
            line=dict(width=1, dash='dot', color='darkgrey'),
            x0=selected_range['x'][0],
            x1=selected_range['x'][1],
            y0=selected_range['y'][0],
            y1=selected_range['y'][1]
        ))

    return fig


def get_var_figure(results: FunVarResults, selected_points):
    df = results.df.loc[selected_points].melt(
        id_vars=results.misc_columns,
        value_vars=results.var_columns,
        var_name='var',
        value_name='value'
    )
    fig = px.line(df, x='var', y='value', line_group='run_individual', color='run')
    fig.update_layout(
        title='Individual Solutions',
        xaxis_title='Variable',
        yaxis_title='Position',
    )
    fig.update_traces(opacity=0.5)

    return fig


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)

    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
    app.layout = html.Div([
        html.Div(
            dcc.Graph(id='g1', config={'displayModeBar': True}),
            #            dcc.Graph(id='g1', config={'displayModeBar': False}),
            className='five columns'
        ),
        html.Div(
            dcc.Graph(id='g2', config={'displayModeBar': False}),
            className='seven columns'
        ),
    ], className='row')

    @app.callback(
        output=dict(g1=Output('g1', 'figure'), g2=Output('g2', 'figure')),
        inputs=dict(selection=Input('g1', 'selectedData')),
        state=dict(previous_figure=State('g1', 'figure'))
    )
    def callback(selection, previous_figure):
        previous_visible = [
            trace['visible'] if 'visible' in trace.keys() else True
            for trace in previous_figure['data']
        ] if previous_figure else []
        selected_points = [p['customdata'] for p in selection['points'] if 'customdata' in p] if selection else []
        selected_range = selection['range'] if selection and 'range' in selection else None

        return dict(
            g1=get_fun_figure(results, selected_points, selected_range, previous_visible),
            g2=get_var_figure(results, selected_points)
        )

    app.run_server(debug=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
