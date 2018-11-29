import io
import json
from typing import Optional

import tornado
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
from matplotlib.backends.backend_webagg_core import (FigureManagerWebAgg, new_figure_manager_given_figure)
from matplotlib.figure import Figure
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, PeriodicCallback

from .plot_server import PlotServer

html_content = '''
<html>
    <head>
        <link rel="stylesheet" href="_static/css/page.css" type="text/css">
        <link rel="stylesheet" href="_static/css/boilerplate.css" type="text/css" />
        <link rel="stylesheet" href="_static/css/fbm.css" type="text/css" />
        <link rel="stylesheet" href="_static/jquery/css/themes/base/jquery-ui.min.css" >
        <style>
            body {
                margin: 30px;
            }
            
            .ui-dialog-titlebar {
                display: none;
            }
        </style>
        <script src="_static/jquery/js/jquery-1.11.3.min.js"></script>
        <script src="_static/jquery/js/jquery-ui.min.js"></script>
        <script src="mpl.js"></script>
        <script>
            function ondownload(figure, format) {
                window.open('download.' + format, '_blank');
            }
            
            function init() {
                var websocket_type = mpl.get_websocket_type();
                var websocket = new websocket_type("%(ws_uri)sws");
                var fig = new mpl.figure(%(fig_id)s, websocket, ondownload, $('div#figure'));
            }
            
            $(document).ready(init);
        </script>
        <title>Blending Evaluation</title>
    </head>
    <body>
        <div id="figure"></div>
    </body>
</html>
'''


class MyApplication(tornado.web.Application):
    class MainPage(tornado.web.RequestHandler):
        def get(self):
            manager = self.application.manager
            ws_uri = 'ws://{req.host}/'.format(req=self.request)
            content = html_content % {'ws_uri': ws_uri, 'fig_id': manager.num}
            self.write(content)

    class MplJs(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Content-Type', 'application/javascript')
            js_content = FigureManagerWebAgg.get_javascript()

            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        def get(self, fmt):
            manager = self.application.manager

            mimetypes = {
                'ps': 'application/postscript',
                'eps': 'application/postscript',
                'pdf': 'application/pdf',
                'svg': 'image/svg+xml',
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'tif': 'image/tiff',
                'emf': 'application/emf'
            }

            self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))

            buff = io.BytesIO()
            manager.canvas.figure.savefig(buff, format=fmt)
            self.write(buff.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        supports_binary = True

        def open(self):
            manager = self.application.manager
            manager.add_web_socket(self)
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            manager = self.application.manager
            manager.remove_web_socket(self)

        def on_message(self, message):
            message = json.loads(message)
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = self.application.manager
                manager.handle_json(message)

        def send_json(self, content):
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            if self.supports_binary:
                self.write_message(blob, binary=True)
            else:
                data_uri = 'data:image/png;base64,{0}'.format(
                    blob.encode('base64').replace('\n', ''))
                self.write_message(data_uri)

    def __init__(self, figure):
        self.figure = figure
        self.manager = new_figure_manager_given_figure(id(figure), figure)

        super().__init__([
            (r'/_static/(.*)', tornado.web.StaticFileHandler, {'path': FigureManagerWebAgg.get_static_file_path()}),
            ('/', self.MainPage),
            ('/mpl.js', self.MplJs),
            ('/ws', self.WebSocket),
            (r'/download.([a-z0-9.]+)', self.Download),
        ])


class MplPlotServer(PlotServer):
    def __init__(self, all_callback, pop_callback, path_callback):
        super().__init__(all_callback, pop_callback, path_callback)

        self.figure, self.all_plot, self.pop_plot = MplPlotServer.create_figure()
        self.application = MyApplication(self.figure)
        self.http_server: Optional[HTTPServer] = None
        self.io_loop: Optional[IOLoop] = None

    @staticmethod
    def create_figure():
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Blending Evaluation')

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)

        all_plot, = ax.plot(
            [], [], label='All Solutions',
            marker='x', markeredgecolor='blue', markersize=4, markeredgewidth=0.5,
            linestyle='None'
        )
        pop_plot, = ax.plot(
            [], [], label='Population',
            marker='o', markeredgecolor='red', markerfacecolor='None', markersize=7, markeredgewidth=0.5,
            linestyle='None'
        )

        ax.legend()

        return fig, all_plot, pop_plot

    def update_figure(self) -> None:
        all_data = self.all_callback(0)
        self.all_plot.set_data(all_data['f1'], all_data['f2'])

        pop_data = self.pop_callback()
        self.pop_plot.set_data(pop_data['f1'], pop_data['f2'])

        self.figure.canvas.draw_idle()

    def serve(self) -> None:
        self.http_server = HTTPServer(self.application)
        self.http_server.listen(self.port)
        self.logger.info(f'http://127.0.0.1:{self.port}/')

        self.io_loop = IOLoop.current()
        PeriodicCallback(self.update_figure, 100).start()
        self.io_loop.start()

    def stop(self) -> None:
        if self.http_server:
            self.http_server.stop()

        if self.io_loop:
            self.io_loop.stop()
