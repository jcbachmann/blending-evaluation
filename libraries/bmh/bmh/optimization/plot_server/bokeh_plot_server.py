from typing import Optional

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Range1d
# noinspection PyUnresolvedReferences
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure
from bokeh.server.server import Server

from .plot_server import PlotServer, PlotServerInterface


class BokehPlotServer(PlotServer):
    def __init__(self, plot_server_interface: PlotServerInterface):
        super().__init__(plot_server_interface)

        self.server: Optional[Server] = None
        self.do_reset = False

    def make_document(self, doc: Document) -> None:
        doc.title = 'Optimization'

        all_source = ColumnDataSource({'f1': [], 'f2': [], 'color': []})
        pop_source = ColumnDataSource({'f1': [], 'f2': []})
        best_source = ColumnDataSource({'f1': [], 'f2': []})
        selected_source = ColumnDataSource({'f1': [], 'f2': []})

        best_path_source = ColumnDataSource({'timestamp': [], 'x': []})
        selected_path_source = ColumnDataSource({'timestamp': [], 'x': []})

        palette = Category10[10]

        scatter_fig = figure(
            title='Solutions',
            plot_width=750,
            plot_height=750,
            tools='pan,wheel_zoom,reset,tap,zoom_in,zoom_out,box_zoom,undo,redo,save',
            x_axis_label='f1 Homogenization Effect',
            y_axis_label='f2 Volume StDev',
            x_range=Range1d(0, 2),
            y_range=Range1d(0, 2),
        )
        scatter_fig.scatter(
            x='f1', y='f2', source=all_source, legend='All Evaluations',
            marker='x', size=5, line_color='color', alpha=0.7,
            nonselection_line_alpha=1.0
        )
        scatter_fig.scatter(
            x='f1', y='f2', source=pop_source, legend='Population',
            marker='o', size=8, line_color=palette[1], fill_alpha=0,
        )
        scatter_fig.scatter(
            x='f1', y='f2', source=best_source, legend='Best',
            marker='o', size=11, line_color=palette[2], fill_alpha=0.5,
        )
        scatter_fig.scatter(
            x='f1', y='f2', source=selected_source, legend='Selected',
            marker='o', size=11, line_color=palette[3], fill_alpha=0.5,
        )
        scatter_fig.legend.location = 'top_right'

        path_fig = figure(
            title='Deposition',
            plot_width=750,
            plot_height=400,
            tools='pan,wheel_zoom,reset,hover',
            x_axis_label='Timestamp',
            y_axis_label='Position',
            x_range=Range1d(0, None),
            y_range=Range1d(0, None),
        )
        path_fig.line(x='timestamp', y='x', legend='Best Deposition', source=best_path_source, color=palette[2])
        path_fig.line(x='timestamp', y='x', legend='Selected Deposition', source=selected_path_source, color=palette[3])
        path_fig.legend.location = 'top_right'

        doc.add_root(gridplot([[scatter_fig], [path_fig]], toolbar_location='left'))

        def path_selected_callback(_attr, _old, new):
            if len(new) > 0:
                solution = self.plot_server_interface.get_solution(new[0])
                selected_path_source.data = {
                    'timestamp': solution.deposition.data['timestamp'],
                    'x': solution.deposition.data['x']
                }
                selected_source.data = {
                    'f1': [solution.objectives[0]],
                    'f2': [solution.objectives[1]]
                }

        all_source.selected.on_change('indices', path_selected_callback)

        def update() -> None:
            if self.do_reset:
                all_source.data = {'f1': [], 'f2': [], 'color': []}
                pop_source.data = {'f1': [], 'f2': []}
                best_source.data = {'f1': [], 'f2': []}
                selected_source.data = {'f1': [], 'f2': []}
                best_path_source.data = {'timestamp': [], 'x': []}
                selected_path_source.data = {'timestamp': [], 'x': []}

                self.do_reset = False

            start = len(all_source.data['f1'])
            all_data = self.plot_server_interface.get_new_solutions(start)
            all_data['color'] = [Viridis256[min(int((i + start) / 100), 255)] for i in range(len(all_data['f1']))]
            all_source.stream(all_data)

            pop_source.data = self.plot_server_interface.get_population()

            best_solution = self.plot_server_interface.get_best_solution()
            if best_solution:
                best_path_source.data = {
                    'timestamp': best_solution.deposition.data['timestamp'],
                    'x': best_solution.deposition.data['x']
                }
                path_fig.x_range.end = best_solution.deposition.meta.time
                path_fig.y_range.end = best_solution.deposition.meta.bed_size_x
                best_source.data = {'f1': [best_solution.objectives[0]], 'f2': [best_solution.objectives[1]]}

        doc.add_periodic_callback(update, 500)

    def serve(self) -> None:
        self.logger.info(f'Opening Bokeh application on http://localhost:{self.port}/')
        apps = {'/': Application(FunctionHandler(self.make_document))}

        self.server = Server(apps, port=self.port)
        self.server.start()

        self.server.io_loop.add_callback(self.server.show, '/')
        self.server.io_loop.start()

    def stop(self) -> None:
        if self.server:
            self.server.stop()
            self.server.io_loop.stop()

    def reset(self) -> None:
        self.do_reset = True
