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


class PlotServer:
    PORT = 5001

    def __init__(self, columns):
        self.doc = None
        self.data_callback = None
        self.source = ColumnDataSource(pd.DataFrame(columns=columns).to_dict(orient='list'))

        self.p = figure(
            plot_height=500,
            tools='xpan,xwheel_zoom,xbox_zoom,reset',
            x_axis_type=None,
            y_axis_label='Quality'
        )
        self.p.x_range.follow = 'end'
        self.p.x_range.follow_interval = 7200000
        self.p.x_range.range_padding = 0

        palette = Category10[10]

        c = 0
        for column in columns:
            if ' quality ' in column:
                self.p.line(x='time', y=column, source=self.source, color=palette[c],
                            legend=column.replace('quality ', ''))
                c += 1
        self.p.legend.location = 'bottom_left'
        self.p.legend.orientation = 'horizontal'
        self.p.legend.click_policy = 'mute'

        self.p2 = figure(
            plot_height=250,
            x_range=self.p.x_range,
            tools='xpan,xwheel_zoom,xbox_zoom,reset',
            x_axis_type='datetime',
            x_axis_label='Time',
            y_axis_label='Tons per Hour'
        )
        c = 0
        for column in columns:
            if ' tph ' in column:
                self.p2.line(x='time', y=column, source=self.source, color=palette[c])
                c += 1

    def update(self):
        data = self.data_callback()
        data['time'] = [t * 1000 for t in data['time']]
        self.source.stream(data, 5760)

    def set_data_callback(self, data_callback):
        self.data_callback = data_callback

    def make_document(self, doc: Document):
        doc.title = 'Plant Simulation'
        doc.add_root(gridplot([[self.p], [self.p2]], toolbar_location='left', plot_width=1000))
        doc.add_periodic_callback(self.update, 500)
        self.doc = doc

    def serve(self):
        print(f'Opening Bokeh application on http://localhost:{PlotServer.PORT}/')
        apps = {'/': Application(FunctionHandler(self.make_document))}

        server = Server(apps, port=PlotServer.PORT)
        server.start()

        server.io_loop.add_callback(server.show, '/')
        server.io_loop.start()

    def serve_background(self):
        t = threading.Thread(target=self.serve)
        t.start()
