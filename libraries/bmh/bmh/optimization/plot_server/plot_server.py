import logging
from threading import Thread

from typing import Callable, Optional


class PlotServer:
    def __init__(self, all_callback: Callable, pop_callback: Callable, path_callback: Callable,
                 port: int = 5001) -> None:
        self.all_callback = all_callback
        self.pop_callback = pop_callback
        self.path_callback = path_callback
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.thread: Optional[Thread] = None

    def serve(self) -> None:
        raise NotImplementedError()

    def serve_background(self) -> None:
        self.thread = Thread(target=self.serve)
        self.thread.start()

    def stop(self) -> None:
        raise NotImplementedError()

    def stop_background(self) -> None:
        if self.thread:
            self.stop()
            self.thread.join()
            self.thread = None
