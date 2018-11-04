import logging
import threading

logger = logging.getLogger(__name__)


class PlotServer:
    def __init__(self, all_callback, pop_callback, path_callback, port=5001):
        self.all_callback = all_callback
        self.pop_callback = pop_callback
        self.path_callback = path_callback
        self.port = port

    def serve(self):
        logger.error('PlotServer.serve method not implemented')

    def serve_background(self):
        t = threading.Thread(target=self.serve)
        t.start()
