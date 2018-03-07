import threading

import pandas as pd
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
# noinspection PyUnresolvedReferences
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop


class PlotServer:
    PORT = 5001

    def __init__(self, columns):
        self.data_callback = None
        self.columns = columns

    def set_data_callback(self, data_callback):
        self.data_callback = data_callback

    def make_document(self, doc: Document):
        doc.title = 'Plant Simulation'

        source = ColumnDataSource(pd.DataFrame(columns=self.columns).to_dict(orient='list'))

        def update():
            start = len(source.data['time'])
            data = self.data_callback(start)
            data['time'] = [t * 1000 for t in data['time']]
            source.stream(data, 5760)

        p = figure(
            plot_height=500,
            tools='xpan,xwheel_zoom,xbox_zoom,reset',
            x_axis_type=None,
            y_axis_label='Quality'
        )
        p.x_range.follow = 'end'
        p.x_range.follow_interval = 7200000
        p.x_range.range_padding = 0

        palette = Category10[10]

        i = 0
        for col in self.columns:
            if ' quality ' in col:
                p.line(x='time', y=col, source=source, color=palette[i],
                       legend=col.replace('quality ', ''))
                i += 1
        p.legend.location = 'bottom_left'
        p.legend.orientation = 'horizontal'
        p.legend.click_policy = 'mute'

        p2 = figure(
            plot_height=250,
            x_range=p.x_range,
            tools='xpan,xwheel_zoom,xbox_zoom,reset',
            x_axis_type='datetime',
            x_axis_label='Time',
            y_axis_label='Tons per Hour'
        )
        i = 0
        for col in self.columns:
            if ' tph ' in col:
                p2.line(x='time', y=col, source=source, color=palette[i])
                i += 1

        doc.add_root(gridplot([[p], [p2]], toolbar_location='left', plot_width=1000))
        doc.add_periodic_callback(update, 500)

    def serve(self):
        print(f'Opening Bokeh application on http://localhost:{PlotServer.PORT}/')
        apps = {'/': Application(FunctionHandler(self.make_document))}

        server = Server(apps, port=PlotServer.PORT, io_loop=IOLoop())
        server.start()

        server.io_loop.add_callback(server.show, '/')
        server.io_loop.start()

    def serve_background(self):
        t = threading.Thread(target=self.serve)
        t.start()
