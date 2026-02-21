import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


TRIL_IDX_4 = torch.tril_indices(4, 4, offset=0)
WEIGHT_10 = torch.tensor([1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=torch.float64)
K_IND_4 = torch.tensor([[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.long)

PACK_OFFSETS_4 = torch.tensor([0, 1, 5, 2, 6, 10, 3, 7, 11, 15], dtype=torch.int32)


def triton_eligible(P, w, failed_flag):
    return (
        TRITON_AVAILABLE
        and (not failed_flag)
        and P.is_cuda
        and w.is_cuda
        and P.dtype in (torch.float32, torch.float64)
    )


def get_sp_const_tensors(cache, device, dtype):
    key = (device, dtype)
    const = cache.get(key)
    if const is None:
        const = {
            "pack_off": PACK_OFFSETS_4.to(device=device),
            "pack_w": WEIGHT_10.to(device=device, dtype=dtype),
            "ind4_i32": K_IND_4.to(device=device, dtype=torch.int32),
        }
        cache[key] = const
    return const
