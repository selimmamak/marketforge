import multiprocessing
import os
import time

import pytest

from marketforge.parallel import parallel_results


def _inspect_worker(value, delay=0):
    time.sleep(delay)
    return {
        "value": value * value,
        "pid": os.getpid(),
        "start_method": multiprocessing.get_start_method(),
        "thread_limits": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def _raise_worker_error():
    raise ValueError("worker failure must propagate")


def _raise_keyboard_interrupt():
    raise KeyboardInterrupt("worker interrupted")


def _sleep_then_return(seconds):
    time.sleep(seconds)
    return seconds


def test_process_results_are_complete_and_environment_is_restored(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS", raising=False)
    before = dict(os.environ)
    results = dict(parallel_results(
        _inspect_worker, [("a", (3,)), ("b", (4,)), ("c", (5,))], workers=2
    ))
    assert {key: value["value"] for key, value in results.items()} == {"a": 9, "b": 16, "c": 25}
    assert all(value["pid"] != os.getpid() for value in results.values())
    assert all(value["start_method"] == "spawn" for value in results.values())
    assert all(set(value["thread_limits"].values()) == {"1"} for value in results.values())
    assert dict(os.environ) == before


def test_fast_results_are_available_before_slow_first_submission():
    results = list(parallel_results(
        _inspect_worker, [("slow", (2, 1.5)), ("fast", (3, 0))], workers=2
    ))
    assert [key for key, _ in results] == ["fast", "slow"]


def test_worker_failure_propagates_and_restores_parent_environment(monkeypatch):
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="worker failure"):
        list(parallel_results(_raise_worker_error, [("failure", ())], workers=2))
    assert dict(os.environ) == before


def test_keyboard_interrupt_propagates_and_stops_other_workers():
    before = dict(os.environ)
    existing_pids = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(KeyboardInterrupt, match="worker interrupted"):
        list(parallel_results(_raise_keyboard_interrupt, [("interrupt", ())], workers=2))
    assert dict(os.environ) == before
    assert {process.pid for process in multiprocessing.active_children()} <= existing_pids


def test_closing_iteration_stops_pending_work_and_restores_environment():
    before = dict(os.environ)
    existing_pids = {process.pid for process in multiprocessing.active_children()}
    results = parallel_results(
        _sleep_then_return, [("fast", (0,)), ("pending", (30,))], workers=2
    )
    assert next(results)[0] == "fast"
    started = time.monotonic()
    results.close()
    assert time.monotonic() - started < 10
    assert dict(os.environ) == before
    assert {process.pid for process in multiprocessing.active_children()} <= existing_pids


def test_serial_mode_and_invalid_worker_counts():
    result = dict(parallel_results(_inspect_worker, [("x", (3,))], workers=1))["x"]
    assert result["value"] == 9
    assert result["pid"] == os.getpid()
    assert list(parallel_results(_inspect_worker, [], workers=2)) == []
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            list(parallel_results(_inspect_worker, [], workers=invalid))
