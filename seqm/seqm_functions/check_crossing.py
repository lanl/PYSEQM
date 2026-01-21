import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def hungarian_state_assignment_from_overlap(
    overlap: torch.Tensor,
    window: int = 2,
    window_penalty: int = 100000,
):
    """
    Hungarian state assignment from an overlap tensor.

    Parameters
    ----------
    overlap : torch.Tensor
        Overlap tensor with shape (B, N, N) or (N, N).
    window : int
        Allowed assignment band: |j - i| <= window.
    window_penalty : int
        Large positive cost for forbidden assignments.

    Returns
    -------
    iorden : torch.Tensor
        Shape (B, N). Mapping iorden[b, i] = j (reference i -> current j).
    cost : torch.Tensor
        Shape (B, N, N). Integer cost matrices used for Hungarian.
    score : torch.Tensor
        Shape (B, N, N). Integer similarity matrices before negation.
    """
    if not torch.is_tensor(overlap):
        raise TypeError("overlap must be a torch.Tensor")

    # Accept (N,N) by promoting to batch size 1
    if overlap.ndim == 2:
        overlap = overlap.unsqueeze(0)
    if overlap.ndim != 3 or overlap.shape[-1] != overlap.shape[-2]:
        raise ValueError(f"overlap must have shape (B,N,N) or (N,N); got {tuple(overlap.shape)}")

    B, N, _ = overlap.shape

    # Negate and square to avoid arbitrary sign flip 
    cost = -(overlap ** 2)

    # Apply window constraint: forbid |i-j| > window
    I = torch.arange(N, device=overlap.device).view(N, 1)
    J = torch.arange(N, device=overlap.device).view(1, N)
    mask_forbidden = (I - J).abs() > window
    cost[:, mask_forbidden] = int(window_penalty)

    # Run Hungarian on batch on cpu
    iorden_list = []
    cost_cpu = cost.detach().cpu().numpy()  # (B,N,N)
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost_cpu[b])

        order = np.argsort(row_ind)
        iorden_list.append(col_ind[order])

    iorden = torch.as_tensor(np.stack(iorden_list, axis=0), device=overlap.device, dtype=torch.int64) + 1 # ordering starting with index 1

    return iorden, cost

def ensure_batched_tdms(tdms: torch.Tensor) -> torch.Tensor:
    """Ensure tdms has shape (B, N, M, M). Accept (N,M,M) -> (1,N,M,M)."""
    if tdms.ndim == 3:
        return tdms.unsqueeze(0)
    if tdms.ndim != 4:
        raise ValueError(f"TDMs must have shape (B,N,M,M) or (N,M,M); got {tuple(tdms.shape)}")
    return tdms

def broadcast_reference_tdms(ref: torch.Tensor, B: int) -> torch.Tensor:
    """
    Accept reference as (N,M,M) or (1,N,M,M) or (B,N,M,M).
    Return (B,N,M,M) by broadcasting if needed.
    """
    ref = ensure_batched_tdms(ref)
    if ref.shape[0] == 1 and B > 1:
        ref = ref.expand(B, -1, -1, -1)  # broadcast across batch
    return ref
