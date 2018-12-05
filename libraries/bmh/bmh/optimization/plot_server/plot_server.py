import asyncio
import logging
from threading import Thread

from typing import Callable, Optional


def ensure_event_loop_exists():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


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

    def wrap_serve(self):
        ensure_event_loop_exists()
        self.serve()

    def serve_background(self) -> None:
        self.thread = Thread(target=self.wrap_serve)
        self.thread.start()

    def stop(self) -> None:
        raise NotImplementedError()

    def stop_background(self) -> None:
        if self.thread:
            self.stop()
            self.thread.join()
            self.thread = None
