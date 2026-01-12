import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def hungarian_state_assignment_from_overlap(
    scpr: torch.Tensor,
    window: int = 2,
    scale: float = 1e5,
    forbid_cost: int = 100000,
):
    """
    Hungarian state assignment from an overlap tensor.

    Parameters
    ----------
    scpr : torch.Tensor
        Overlap tensor with shape (B, N, N) or (N, N).
        Rows correspond to reference/old states, columns to current/new states.
    window : int
        Allowed assignment band: |j - i| <= window.
    scale : float
        Scaling used before integer conversion (NEXMD uses 1e5).
    forbid_cost : int
        Large positive cost for forbidden assignments.

    Returns
    -------
    iorden : torch.LongTensor
        Shape (B, N). Mapping iorden[b, i] = j (reference i -> current j).
    cost : torch.LongTensor
        Shape (B, N, N). Integer cost matrices used for Hungarian.
    score : torch.LongTensor
        Shape (B, N, N). Integer similarity matrices before negation.
    """
    if not torch.is_tensor(scpr):
        raise TypeError("scpr must be a torch.Tensor")

    # Accept (N,N) by promoting to batch size 1
    if scpr.ndim == 2:
        scpr = scpr.unsqueeze(0)
    if scpr.ndim != 3 or scpr.shape[-1] != scpr.shape[-2]:
        raise ValueError(f"scpr must have shape (B,N,N) or (N,N); got {tuple(scpr.shape)}")

    B, N, _ = scpr.shape

    # Build integer similarity like NEXMD: int(|overlap|^2 * 1e5)
    score = (scpr ** 2 * scale).to(dtype=torch.int64)

    # Convert to cost (minimize): cost = -score
    cost = -score

    # Apply window constraint: forbid |i-j| > window
    I = torch.arange(N, device=scpr.device).view(N, 1)
    J = torch.arange(N, device=scpr.device).view(1, N)
    mask_forbidden = (I - J).abs() > window
    cost[:, mask_forbidden] = int(forbid_cost)

    # Run Hungarian per batch (SciPy expects NumPy)
    iorden_list = []
    cost_cpu = cost.detach().cpu().numpy()  # (B,N,N)
    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost_cpu[b])
        # ensure row order corresponds to reference indices 0..N-1
        order = np.argsort(row_ind)
        iorden_list.append(col_ind[order])

    iorden = torch.as_tensor(np.stack(iorden_list, axis=0), device=scpr.device, dtype=torch.int64) + 1 # ordering starting with index 1

    return iorden, cost, score

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
# def hungarian_state_assignment_from_overlap(
#     scpr: np.ndarray,
#     window: int = 2,
#     scale: float = 1e5,
#     forbid_cost: int = 100000,
# ):
#     """
#     Python equivalent of NEXMD's APC/Hungarian state tracking from overlap matrix.

#     Parameters
#     ----------
#     scpr : (N, N) array
#         Overlap between "old/ref" TDM vectors (rows i) and "new/current" TDM vectors (cols j).
#         In NEXMD: scpr(i,j) = sum_k cmdqtold(k,i) * cmdqtnew(k,j)
#     window : int
#         Allowed assignment band: j must satisfy |j - i| <= window.
#         NEXMD uses window=2.
#     scale : float
#         Scaling used before integer conversion. NEXMD uses 1e5.
#     forbid_cost : int
#         The penalty magnitude used for forbidden assignments after negation.
#         NEXMD uses 1e5 (because valid scores are in [0,1e5]).

#     Returns
#     -------
#     iorden : (N,) int array
#         Mapping from old/ref index i -> new/current index j.
#     cost : (N, N) int array
#         Integer cost matrix passed to Hungarian algorithm.
#     score : (N, N) int array
#         Integer similarity matrix before negation.
#     """
#     scpr = np.asarray(scpr)
#     if scpr.ndim != 2 or scpr.shape[0] != scpr.shape[1]:
#         raise ValueError(f"scpr must be square (N,N); got {scpr.shape}")

#     N = scpr.shape[0]
#     score = (scpr ** 2 * scale).astype(np.int64)

#     # if (j < i-2) or (j > i+2): ascpr(i,j) = -1*100000
#     # Note: NEXMD sets this before the global negation, then negates everything.
#     # We'll directly set the final COST for forbidden entries to +forbid_cost,
#     # which is what the Fortran ends up with after negation.
#     cost = -score  # Equivalent of: ascpr = -ascpr

#     I, J = np.ogrid[:N, :N]
#     mask_forbidden = np.abs(I - J) > window
#     cost[mask_forbidden] = forbid_cost  # big positive cost -> never chosen

#     # --- Hungarian / assignment: minimize total cost
#     row_ind, col_ind = linear_sum_assignment(cost)

#     # Ensure mapping is ordered by row (old/ref state)
#     order = np.argsort(row_ind)
#     iorden = col_ind[order].astype(np.int64)

#     return iorden, cost, score
