import logging
import threading

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FuncTickFormatter, Range1d
# noinspection PyUnresolvedReferences
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.server.server import Server
from pandas import DataFrame
from tornado.ioloop import IOLoop


def create_live_graphs(source, palette, columns):
    p_live_q = figure(
        plot_height=300,
        tools='xpan,xwheel_zoom,reset',
        x_axis_type=None,
        y_axis_label='Quality'
    )
    p_live_q.x_range.follow = 'end'
    p_live_q.x_range.follow_interval = 86400000
    p_live_q.x_range.range_padding = 0

    i = 0
    for col in columns:
        if ' quality' in col:
            p_live_q.line(x='time', y=col, source=source, color=palette[i],
                          legend=col.replace(' quality', ''))
            i += 1
    p_live_q.legend.location = 'bottom_left'
    p_live_q.legend.orientation = 'horizontal'
    p_live_q.legend.click_policy = 'hide'
    p_live_q.yaxis.formatter = FuncTickFormatter(code="""
        return (100 * tick).toFixed(1) + ' %'
    """)
    p_live_q.y_range = Range1d(0.25, 0.37)

    # Live tph graph
    p_live_tph = figure(
        plot_height=250,
        x_range=p_live_q.x_range,
        tools='xpan,xwheel_zoom,reset',
        x_axis_type='datetime',
        x_axis_label='Time',
        y_axis_label='Tons per Hour'
    )
    i = 0
    for col in columns:
        if ' tph' in col:
            p_live_tph.line(x='time', y=col, source=source, color=palette[i])
            i += 1

    return p_live_q, p_live_tph


def create_stats_graphs(source, palette, p_reference, columns):
    p_stats_q = figure(
        plot_height=300,
        x_range=p_reference.x_range,
        tools='xpan,xwheel_zoom,reset',
        x_axis_type=None,
        y_axis_label='Quality'
    )

    i = 0
    for col in columns:
        if ' quality' in col:
            p_stats_q.line(x='time', y=f'{col} min', source=source, color=palette[i])
            p_stats_q.line(x='time', y=f'{col} max', source=source, color=palette[i])
            p_stats_q.line(x='time', y=f'{col} std_low', source=source, color=palette[i], line_width=2)
            p_stats_q.line(x='time', y=f'{col} std_high', source=source, color=palette[i], line_width=2)
            p_stats_q.line(x='time', y=f'{col} average', source=source, color=palette[i],
                           legend=col.replace(' quality', ''), line_width=3)
            i += 1
    p_stats_q.legend.location = 'bottom_left'
    p_stats_q.legend.orientation = 'horizontal'
    p_stats_q.legend.click_policy = 'hide'
    p_stats_q.yaxis.formatter = FuncTickFormatter(code="""
            return (100 * tick).toFixed(1) + ' %'
        """)
    p_stats_q.y_range = Range1d(0.25, 0.37)

    # Stats tph graph
    p_stats_tph = figure(
        plot_height=250,
        x_range=p_reference.x_range,
        tools='xpan,xwheel_zoom,reset',
        x_axis_type='datetime',
        x_axis_label='Time',
        y_axis_label='Tons per Hour'
    )
    i = 0
    for col in columns:
        if ' tph' in col:
            p_stats_tph.line(x='time', y=f'{col} min', source=source, color=palette[i])
            p_stats_tph.line(x='time', y=f'{col} max', source=source, color=palette[i])
            p_stats_tph.line(x='time', y=f'{col} std_low', source=source, color=palette[i], line_width=2)
            p_stats_tph.line(x='time', y=f'{col} std_high', source=source, color=palette[i], line_width=2)
            p_stats_tph.line(x='time', y=f'{col} average', source=source, color=palette[i], line_width=3)
            i += 1

    return p_stats_q, p_stats_tph


def get_stats_columns(columns):
    stats_columns = ['start', 'end']

    for col in columns:
        stats_columns.extend(
            [
                f'{col} min',
                f'{col} max',
                f'{col} average',
                f'{col} std_low',
                f'{col} std_high'
            ] if ' tph' in col or ' quality' in col else [col]
        )

    return stats_columns


class PlotServer:
    PORT = 5001

    def __init__(self, columns):
        self.live_data_callback = None
        self.stats_data_callback = None
        self.columns = columns
        self.logger = logging.getLogger(__name__)

    def set_data_callback(self, live_data_callback, stats_data_callback):
        self.live_data_callback = live_data_callback
        self.stats_data_callback = stats_data_callback

    def make_document(self, doc: Document):
        doc.title = 'Plant Simulation'

        # Create data sources
        live_source = ColumnDataSource(DataFrame(columns=self.columns).to_dict(orient='list'))
        stats_source = ColumnDataSource(DataFrame(columns=get_stats_columns(self.columns)).to_dict(orient='list'))

        # Add and arrange graphs
        palette = Category10[10]
        p_live_q, p_live_tph = create_live_graphs(live_source, palette, self.columns)
        # p_stats_q, p_stats_tph = create_stats_graphs(stats_source, palette, p_live_q, self.columns)
        doc.add_root(gridplot([
            [p_live_q],
            [p_live_tph],
            # [p_stats_q],
            # [p_stats_tph]
        ], toolbar_location='left', plot_width=1000))

        # Periodically update graphs
        def update():
            live_start = len(live_source.data['time'])
            live_data = self.live_data_callback(live_start)
            if 'time' in live_data:
                live_data['time'] = [t * 1000 for t in live_data['time']]
                live_source.stream(live_data, 57600)

            # stats_start = len(stats_source.data['time'])
            # stats_data = self.stats_data_callback(stats_start)
            # if 'start' in stats_data:
            #     stats_data['time'] = [t * 1000 for t in stats_data['start']]
            #     stats_source.stream(stats_data, 1000)

        doc.add_periodic_callback(update, 500)

    def serve(self):
        self.logger.info(f'Opening Bokeh application on http://localhost:{PlotServer.PORT}/')
        apps = {'/': Application(FunctionHandler(self.make_document))}

        server = Server(apps, port=PlotServer.PORT, io_loop=IOLoop())
        server.start()

        server.io_loop.add_callback(server.show, '/')
        server.io_loop.start()

    def serve_background(self):
        t = threading.Thread(target=self.serve)
        t.start()
