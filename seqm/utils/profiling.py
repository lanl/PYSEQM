import functools
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_RUNTIME_DIAGNOSTIC_RECORDER = ContextVar("pyseqm_runtime_diagnostics", default=None)


@contextmanager
def capture_runtime_diagnostics():
    """Capture lightweight solver events for internal performance benchmarks."""
    events = []
    token = _RUNTIME_DIAGNOSTIC_RECORDER.set(events)
    try:
        yield events
    finally:
        _RUNTIME_DIAGNOSTIC_RECORDER.reset(token)


def record_runtime_diagnostic(name, **values):
    """Record an event only when an internal benchmark enabled capture."""
    recorder = _RUNTIME_DIAGNOSTIC_RECORDER.get()
    if recorder is not None:
        recorder.append({"name": name, **values})


def attach_profile_range(message):
    def decorator(func):
        print("Decorating your function!", func)

        @functools.wraps(func)
        def inner(*args, **kwargs):
            print("Adding NVTX range", message)
            torch.cuda.nvtx.range_push(message)
            result = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            print("LEAVING NVTX RANGE", message)
            return result

        return inner

    return decorator
