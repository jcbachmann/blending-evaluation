"""Independent work (optimization runs, trainings, sweep trials) in parallel processes, each with a fixed number of threads.

One process with many threads leaves most of the machine idle for this pipeline's work: an optimizer on a model is one Python thread, and
a small network does too little per step to keep many threads busy. Several processes with few threads each use the cores instead of
waiting on each other. The thread limits are set before the frameworks are imported, which is why the processes are started with `spawn`.
"""

import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")


def get_cpu_count() -> int:
    """The cores this process may use: its thread limit if it is a worker of `run_parallel`, otherwise all."""
    return get_thread_limit() or os.cpu_count() or 1


def get_threads_per_worker(workers: int) -> int:
    """An equal share of the cores for each of `workers` processes."""
    return max(1, get_cpu_count() // max(1, workers))


def limit_threads(threads: int) -> None:
    """Thread limits of the numeric libraries of this process, effective for the libraries imported afterwards."""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(threads)
    os.environ["BMH_ML_THREADS"] = str(threads)


def get_thread_limit() -> int | None:
    """The thread limit set by `limit_threads` for this process, if any: models use it instead of their own thread counts."""
    value = os.environ.get("BMH_ML_THREADS")
    return int(value) if value else None


def run_parallel(function: Callable, arguments: Iterable[tuple], workers: int, threads: int | None = None) -> list[Any]:
    """`function(*arguments)` for every tuple of arguments, in `workers` processes with `threads` threads each (default: an equal share
    of the cores). The results are in the order of the arguments. With one worker everything runs in this process."""
    arguments = list(arguments)
    workers = max(1, min(workers, len(arguments)))
    if workers == 1:
        return [function(*item) for item in arguments]
    threads = threads or get_threads_per_worker(workers)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=limit_threads, initargs=(threads,)) as executor:
        return list(executor.map(function, *zip(*arguments, strict=True)))
