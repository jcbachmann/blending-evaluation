import os

from bmh_ml.parallel import get_cpu_count, get_thread_limit, get_threads_per_worker, limit_threads, run_parallel


def describe_worker(value: int) -> tuple[int, int | None, str | None]:
    return value * 2, get_thread_limit(), os.environ.get("OMP_NUM_THREADS")


def test_the_results_come_in_the_order_of_the_arguments_and_the_workers_get_their_thread_limit():
    results = run_parallel(describe_worker, [(value,) for value in range(5)], workers=2, threads=3)

    assert [result[0] for result in results] == [0, 2, 4, 6, 8]
    assert all(result[1:] == (3, "3") for result in results)


def test_one_worker_runs_in_this_process():
    assert run_parallel(describe_worker, [(1,), (2,)], workers=1) == [describe_worker(1), describe_worker(2)]


def test_a_worker_counts_only_its_share_of_the_cores(monkeypatch):
    for variable in ("BMH_ML_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        monkeypatch.delenv(variable, raising=False)
    assert get_cpu_count() == os.cpu_count()

    limit_threads(2)

    assert get_cpu_count() == 2
    assert get_threads_per_worker(2) == 1
