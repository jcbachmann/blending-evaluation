import logging
from typing import Optional

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Range1d, DataRange1d
# noinspection PyUnresolvedReferences
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure
from bokeh.server.server import Server

from .plot_server import PlotServer, PlotServerInterface


class BokehPlotServer(PlotServer):
    def __init__(self, plot_server_interface: PlotServerInterface, port: int = PlotServer.DEFAULT_PORT):
        super().__init__(plot_server_interface, port)

        self.server: Optional[Server] = None
        self.do_reset = False
        self.logger = logging.getLogger(__name__)

    def make_document(self, doc: Document) -> None:
        doc.title = 'Optimization'

        all_source = ColumnDataSource({'f1': [], 'f2': [], 'color': []})
        pop_source = ColumnDataSource({'f1': [], 'f2': []})
        best_source = ColumnDataSource({'f1': [], 'f2': []})
        selected_source = ColumnDataSource({'f1': [-1.0], 'f2': [-1.0]})
        reference_source = ColumnDataSource({'f1': [], 'f2': []})

        best_path_source = ColumnDataSource({'timestamp': [], 'x': []})
        selected_path_source = ColumnDataSource({'timestamp': [], 'x': []})
        reference_path_source = ColumnDataSource({'timestamp': [], 'x': []})

        parameter_labels = self.plot_server_interface.get_material().get_parameter_columns()
        material_source_data = {p: [] for p in parameter_labels}
        material_source_data.update({'timestamp': [], 'tonnage': []})
        material_input_source = ColumnDataSource(material_source_data)
        best_path_material_output_source = ColumnDataSource(material_source_data)
        selected_path_material_output_source = ColumnDataSource(material_source_data)
        reference_path_material_output_source = ColumnDataSource(material_source_data)

        progress_source = ColumnDataSource({'t_start': [0.0], 't_end': [0.0]})

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
        scatter_fig.scatter(
            x='f1', y='f2', source=reference_source, legend='Reference',
            marker='*', size=11, line_color=palette[4], fill_alpha=0.5,
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
            x_axis_type='datetime',
        )
        path_fig.line(x='timestamp', y='x', legend='Best', source=best_path_source, color=palette[2])
        path_fig.line(x='timestamp', y='x', legend='Selected', source=selected_path_source, color=palette[3])
        path_fig.line(x='timestamp', y='x', legend='Reference', source=reference_path_source, color=palette[4],
                      line_dash='4 4', alpha=0.5)
        path_fig.ray(x='t_start', y=0, color='black', length=0, angle=90, angle_units='deg', alpha=0.5,
                     source=progress_source)
        path_fig.legend.location = 'top_right'

        material_input_fig = figure(
            title='Material Input',
            plot_width=750,
            plot_height=400,
            tools='pan,wheel_zoom,reset,hover',
            x_axis_label='Timestamp',
            y_axis_label='Parameter',
            x_range=Range1d(0, None),
            x_axis_type='datetime',
        )
        for i, p in enumerate(parameter_labels):
            material_input_fig.line(
                x='timestamp', y=p, legend=f'Parameter {p}', source=material_input_source, color=palette[i]
            )
        material_input_fig.legend.location = 'top_right'

        material_input_fig_volume = figure(
            title='Material Input Tonnage',
            plot_width=750,
            plot_height=400,
            tools='pan,wheel_zoom,reset,hover',
            x_axis_label='Timestamp',
            y_axis_label='tph',
            x_range=Range1d(0, None),
            y_range=DataRange1d(start=0),
            x_axis_type='datetime',
        )
        material_input_fig_volume.line(x='timestamp', y='tonnage', source=material_input_source, color='black')

        material_output_fig = figure(
            title='Material Output',
            plot_width=750,
            plot_height=400,
            tools='pan,wheel_zoom,reset,hover',
            x_axis_label='Timestamp',
            y_axis_label='Parameter',
            x_axis_type='datetime',
            x_range=Range1d(0, None),
            y_range=material_input_fig.y_range,
        )
        for i, p in enumerate(parameter_labels):
            material_output_fig.line(
                x='timestamp', y=p, legend=f'Best Parameter {p}', source=best_path_material_output_source,
                color=palette[2 + i * 3]
            )
            material_output_fig.line(
                x='timestamp', y=p, legend=f'Selected Parameter {p}', source=selected_path_material_output_source,
                color=palette[3 + i * 3]
            )
            material_output_fig.line(
                x='timestamp', y=p, legend=f'Reference Parameter {p}', source=reference_path_material_output_source,
                color=palette[4 + i * 3], alpha=0.5
            )
        material_output_fig.legend.location = 'top_right'

        material_output_fig_volume = figure(
            title='Material Output Tonnage',
            plot_width=750,
            plot_height=400,
            tools='pan,wheel_zoom,reset,hover',
            x_axis_label='Timestamp',
            y_axis_label='tph',
            x_axis_type='datetime',
            x_range=material_output_fig.x_range,
            y_range=material_input_fig_volume.y_range,
        )
        material_output_fig_volume.line(
            x='timestamp', y='tonnage', legend=f'Best', source=best_path_material_output_source,
            color=palette[2]
        )
        material_output_fig_volume.line(
            x='timestamp', y='tonnage', legend=f'Selected', source=selected_path_material_output_source,
            color=palette[3]
        )
        material_output_fig_volume.line(
            x='timestamp', y='tonnage', legend=f'Reference', source=reference_path_material_output_source,
            color=palette[4], alpha=0.5
        )
        material_output_fig_volume.legend.location = 'top_right'

        doc.add_root(gridplot([
            [scatter_fig], [path_fig], [material_input_fig, material_output_fig],
            [material_input_fig_volume, material_output_fig_volume]
        ], toolbar_location='left'))

        def path_selected_callback(_attr, _old, new):
            if len(new) > 0:
                solution = self.plot_server_interface.get_solution(new[0])
                selected_path_source.data = {
                    'timestamp': solution.deposition.data['timestamp'] * 1000,
                    'x': solution.deposition.data['x']
                }
                selected_source.data = {
                    'f1': [solution.objectives[0]],
                    'f2': [solution.objectives[1]]
                }

                df = solution.reclaimed_material.data.copy()
                df['tonnage'] = 3600 * df['volume'] / (df['timestamp'] - df['timestamp'].shift(1).fillna(0))
                df = df[df['tonnage'] > 0.0]
                data = {p: df[p] for p in parameter_labels}
                data.update({'timestamp': df['timestamp'] * 1000, 'tonnage': df['tonnage']})
                selected_path_material_output_source.data = data

        all_source.selected.on_change('indices', path_selected_callback)

        # noinspection PyBroadException
        def update() -> None:
            try:
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
                        'timestamp': best_solution.deposition.data['timestamp'] * 1000,
                        'x': best_solution.deposition.data['x']
                    }
                    path_fig.x_range.end = best_solution.deposition.meta.time * 1000
                    path_fig.y_range.end = best_solution.deposition.meta.bed_size_x
                    best_source.data = {'f1': [best_solution.objectives[0]], 'f2': [best_solution.objectives[1]]}
                    df = best_solution.reclaimed_material.data.copy()
                    df['tonnage'] = 3600 * df['volume'] / (df['timestamp'] - df['timestamp'].shift(1).fillna(0))
                    df = df[df['tonnage'] > 0.0]
                    data = {p: df[p] for p in parameter_labels}
                    data.update({'timestamp': df['timestamp'] * 1000, 'tonnage': df['tonnage']})
                    best_path_material_output_source.data = data

                reference = self.plot_server_interface.get_reference()
                if reference:
                    reference_path_source.data = {
                        'timestamp': reference.deposition.data['timestamp'] * 1000,
                        'x': reference.deposition.data['x']
                    }
                    reference_source.data = {'f1': [reference.objectives[0]], 'f2': [reference.objectives[1]]}

                    df = reference.reclaimed_material.data.copy()
                    df['tonnage'] = 3600 * df['volume'] / (df['timestamp'] - df['timestamp'].shift(1).fillna(0))
                    df = df[df['tonnage'] > 0.0]
                    data = {p: df[p] for p in parameter_labels}
                    data.update({'timestamp': df['timestamp'] * 1000, 'tonnage': df['tonnage']})
                    reference_path_material_output_source.data = data
                    material_output_fig.x_range.end = reference.reclaimed_material.meta.time * 1000

                material = self.plot_server_interface.get_material()
                if material:
                    data = {p: material.data[p] for p in parameter_labels}
                    df = material.data.copy()
                    df['tonnage'] = 3600 * df['volume'] / (df['timestamp'] - df['timestamp'].shift(1).fillna(0))
                    data.update({'timestamp': df['timestamp'] * 1000, 'tonnage': df['tonnage']})
                    material_input_fig.x_range.end = material.meta.time * 1000
                    material_input_fig_volume.x_range.end = material.meta.time * 1000
                    material_input_source.data = data

                progress = self.plot_server_interface.get_progress()
                if progress:
                    progress_source.data = {
                        't_start': [progress['t_start'] * 1000],
                    }
            except Exception as e:
                self.logger.error(f'Periodic callback: {e}')

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
