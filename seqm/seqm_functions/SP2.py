import warnings

import torch

# SP2 convergence thresholds for different precisions
SP2_EPS_FLOAT32 = 1.0e-2  # float32 is harder to converge, use relaxed threshold
SP2_EPS_FLOAT64_MAX = 1.0e-3
SP2_EPS_FLOAT64_MIN = 1.0e-7
SP2_MAX_ITER = 100


def SP2(a, nocc, eps=1.0e-4, factor=2.0, max_iter=SP2_MAX_ITER):
    """
    Density matrices from batched Fock matrices `a` without diagonalization (no autograd support).
    a: (N, D, D) Fock matrices, don't need to be truncated
    nocc: (N,) number of occupied MOs
    factor: 1.0 or 2.0, return factor*X with tr = factor*nocc
    The result is a polynomial in `a`, so it commutes with `a`.
    """
    dtype = a.dtype
    single = dtype == torch.float32
    # float32: hard to converge to a small eps, use eps >= 1e-2 and stop when the error no longer improves
    # float64: the error keeps decreasing for many iterations, so use a small eps (<= 1e-3)
    # and stop when the errors of the current and last iterations are both <= eps
    eps = max(eps, SP2_EPS_FLOAT32) if single else min(max(eps, SP2_EPS_FLOAT64_MIN), SP2_EPS_FLOAT64_MAX)
    noccd = nocc.to(dtype)
    N, D, _ = a.shape

    # Gershgorin circle theorem estimate of the extreme eigenvalues
    aii = a.diagonal(dim1=1, dim2=2)
    ri = a.abs().sum(2) - aii.abs()
    h1 = (aii - ri).amin(1)
    hN = (aii + ri).amax(1)
    # scale so spectrum is in [0, 1], occupied (low energy) states near 1
    X = (torch.diag_embed(hN.unsqueeze(1).expand(N, D)) - a) / (hN - h1).reshape(-1, 1, 1)

    # converged molecules are written to `out` and dropped from the active batch;
    # idx maps active rows back to their position in the input batch
    out = torch.empty_like(a)
    idx = torch.arange(N, device=a.device)
    tr = X.diagonal(dim1=1, dim2=2).sum(1)
    # trace errors of the current, last, and second-to-last iterations
    e0 = (tr - noccd).abs()
    e1, e2 = e0, e0

    for _ in range(max_iter):
        X2 = X @ X
        tr2 = X2.diagonal(dim1=1, dim2=2).sum(1)
        # pick X^2 or 2X - X^2, whichever brings the trace closer to nocc
        low = (tr2 - noccd).abs() < (2.0 * tr - tr2 - noccd).abs()
        X = torch.where(low.reshape(-1, 1, 1), X2, 2.0 * X - X2)
        tr = torch.where(low, tr2, 2.0 * tr - tr2)
        e2, e1, e0 = e1, e0, (tr - noccd).abs()
        # NaN comparisons are all False and would otherwise loop forever
        if not torch.isfinite(e0).all():
            raise RuntimeError("SP2 produced non-finite values")
        done = (e0 < eps) & (e0 >= e2) if single else (e0 < eps) & (e1 < eps)
        if done.all():
            out[idx] = X
            return factor * out
        if done.any():
            out[idx[done]] = X[done]
            keep = ~done
            idx, X, tr, noccd, e0, e1, e2 = (
                idx[keep],
                X[keep],
                tr[keep],
                noccd[keep],
                e0[keep],
                e1[keep],
                e2[keep],
            )

    warnings.warn(f"SP2 did not converge in {max_iter} iterations (max trace error {e0.max().item():.2e})")
    out[idx] = X
    return factor * out
