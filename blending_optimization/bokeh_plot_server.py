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

    def __init__(self, all_callback, pop_callback, path_callback):
        self.all_callback = all_callback
        self.pop_callback = pop_callback
        self.path_callback = path_callback

    def make_document(self, doc: Document):
        doc.title = 'Optimization'

        all_source = ColumnDataSource({'f1': [], 'f2': []})
        pop_source = ColumnDataSource({'f1': [], 'f2': []})
        path_source = ColumnDataSource({'x': [], 'i': []})
        palette = Category10[10]

        scatter_fig = figure(
            plot_height=600,
            plot_width=600,
            tools='pan,wheel_zoom,reset,hover,tap',
            x_axis_label='f1 Homogenization Effect',
            y_axis_label='f2 Volume StDev'
        )

        scatter_fig.scatter(
            x='f1', y='f2', source=all_source, legend='All Evaluations',
            marker='x', size=5, line_color=palette[0], alpha=0.7
        )
        scatter_fig.scatter(
            x='f1', y='f2', source=pop_source, legend='Population',
            marker='o', size=8, line_color=palette[1], fill_alpha=0
        )
        scatter_fig.legend.location = 'top_right'
        scatter_fig.x_range = Range1d(0, 0.3)
        scatter_fig.y_range = Range1d(0, 2)

        def path_callback(_attr, _old, new):
            path = self.path_callback(new[0])
            path_source.data['i'] = list(range(len(path)))
            path_source.data['x'] = path

        all_source.selected.on_change('indices', path_callback)

        path_fig = figure(
            plot_height=400,
            plot_width=900,
            tools='pan,wheel_zoom,reset,hover',
            x_axis_label='Position',
            y_axis_label='Layer'
        )
        path_fig.line(x='x', y='i', source=path_source)

        doc.add_root(gridplot([[scatter_fig], [path_fig]], toolbar_location='left'))

        def update():
            start = len(all_source.data['f1'])
            all_data = self.all_callback(start)
            all_source.stream(all_data)

            pop_data = self.pop_callback()
            pop_source.stream(pop_data, len(pop_data['f1']))

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
