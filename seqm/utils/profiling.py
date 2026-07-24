import functools

import torch


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
