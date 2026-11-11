"""Bounded process execution for independent experiment jobs.

Results arrive in completion order so the caller can checkpoint immediately.
The caller must aggregate them in its original model/fold order. This module
does not change estimator seeds, candidate grids, or experiment boundaries.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _optional_torch():
    """Load an available OpenMP runtime without requiring neural dependencies."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError):
        return None


def _initialize_worker():
    # On some macOS installs XGBoost needs PyTorch's bundled OpenMP runtime.
    torch = _optional_torch()
    if torch is not None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


@contextmanager
def _limited_worker_environment():
    previous = {name: os.environ.get(name) for name in _THREAD_ENVIRONMENT}
    try:
        # Set these before spawn, not only in its initializer: the new
        # interpreter can import numeric libraries before initializing a worker.
        os.environ.update({name: "1" for name in _THREAD_ENVIRONMENT})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _stop_executor(executor):
    """Do not leave expensive fitting jobs alive after caller interruption."""
    if hasattr(executor, "terminate_workers"):
        executor.terminate_workers()
    else:
        # Python 3.9-3.13 have no public ProcessPoolExecutor termination method.
        # Retain only this narrow compatibility access to worker handles.
        processes = list((getattr(executor, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def parallel_results(
    function: Callable[..., Any],
    jobs: Iterable[tuple[Any, tuple[Any, ...]]],
    workers: int,
) -> Iterator[tuple[Any, Any]]:
    """Yield ``(key, result)`` as independent jobs complete.

    ``jobs`` contains ``(key, positional_arguments)`` pairs. ``function`` must
    be a picklable top-level callable. With multiple workers, each spawned
    process uses one supported native/PyTorch compute thread. Python exceptions,
    interrupted iteration, and hard worker failures propagate to the caller;
    pending jobs are cancelled and running workers are stopped first.

    The parent optionally preloads PyTorch too, so deserializing an XGBoost
    estimator can find the same bundled OpenMP runtime as the worker. A core
    installation without PyTorch remains supported. Single-worker execution is
    direct; the experiment's own fit/predict thread limits still apply.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    items = list(jobs)
    if not items:
        return
    _optional_torch()
    if workers == 1:
        for key, arguments in items:
            yield key, function(*arguments)
        return

    with _limited_worker_environment():
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
        )
        futures = {}
        try:
            for key, arguments in items:
                futures[executor.submit(function, *arguments)] = key
            for future in as_completed(futures):
                yield futures[future], future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            _stop_executor(executor)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=True)
