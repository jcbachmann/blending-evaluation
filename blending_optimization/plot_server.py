import threading

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Range1d
# noinspection PyUnresolvedReferences
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop


class PlotServer:
    PORT = 5001

    def __init__(self, data_callback):
        self.data_callback = data_callback

    def make_document(self, doc: Document):
        doc.title = 'Optimization'

        source = ColumnDataSource({'f1': [], 'f2': []})
        palette = Category10[10]

        p = figure(
            plot_height=600,
            plot_width=600,
            tools='pan,wheel_zoom,reset',
            x_axis_label='f1 Homogenization Effect',
            y_axis_label='f2 Volume StDev'
        )

        p.scatter(x='f1', y='f2', source=source, color=palette[0], legend='Solutions', marker='x')
        p.legend.location = 'top_right'
        p.x_range = Range1d(0, 0.3)
        p.y_range = Range1d(0, 2)

        doc.add_root(gridplot([[p]], toolbar_location='left'))

        def update():
            start = len(source.data['f1'])
            new_data = self.data_callback(start)
            source.stream(new_data, 50000)

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
