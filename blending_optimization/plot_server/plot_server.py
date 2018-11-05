import logging
import threading

from typing import Callable

logger = logging.getLogger(__name__)


class PlotServer:
    def __init__(self, all_callback: Callable, pop_callback: Callable, path_callback: Callable,
                 port: int = 5001) -> None:
        self.all_callback = all_callback
        self.pop_callback = pop_callback
        self.path_callback = path_callback
        self.port = port

    def serve(self) -> None:
        logger.error('PlotServer.serve method not implemented')

    def serve_background(self) -> None:
        t = threading.Thread(target=self.serve)
        t.start()
