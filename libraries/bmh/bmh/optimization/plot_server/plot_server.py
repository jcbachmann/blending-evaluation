import asyncio
import logging
from abc import ABC, abstractmethod
from threading import Thread

from typing import Optional, Dict, List


def ensure_event_loop_exists():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class PlotServerInterface(ABC):
    @abstractmethod
    def get_new_solutions(self, start: int) -> Dict[str, List[float]]:
        pass

    @abstractmethod
    def get_population(self) -> Dict[str, List[float]]:
        pass

    @abstractmethod
    def get_path(self, path_id: int) -> List[float]:
        pass


class PlotServer(ABC):
    def __init__(self, plot_server_interface: PlotServerInterface, port: int = 5001) -> None:
        self.plot_server_interface = plot_server_interface
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.thread: Optional[Thread] = None

    @abstractmethod
    def serve(self) -> None:
        pass

    def wrap_serve(self):
        ensure_event_loop_exists()
        self.serve()

    def serve_background(self) -> None:
        self.thread = Thread(target=self.wrap_serve)
        self.thread.start()

    @abstractmethod
    def stop(self) -> None:
        pass

    def stop_background(self) -> None:
        if self.thread:
            self.stop()
            self.thread.join()
            self.thread = None

    def reset(self) -> None:
        # Optional method for resetting the plot server state
        pass
