import torch

from .diag import sym_eig_trunc, sym_eig_trunc1
from .pack import unpack


# @torch.jit.script
def Fermi_Q(H0, T, Nocc, nHeavy, nHydro, kB, scf_backward):
    """
    Fermi operator expansion, eigenapirs [QQ,e], and entropy S_Ent
    """
    max_iter = 64
    if H0.dtype == torch.float64:
        occ_tol = 1e-9
        entropy_eps = 1e-14
        tiny = 1e-30
    elif H0.dtype == torch.float32:
        occ_tol = 1e-5
        entropy_eps = 1e-7
        tiny = 1e-20
    else:
        raise RuntimeError("H0 must be float32 or float64")

    device, dtype = H0.device, H0.dtype

    if scf_backward >= 1:
        e, QQ = sym_eig_trunc1(H0, nHeavy, nHydro, Nocc, eig_only=True)
    else:
        e, QQ = sym_eig_trunc(H0, nHeavy, nHydro, Nocc, eig_only=True)

    # [B, M] eigenvalues (trim to match eigenvectors count)
    n_states = QQ.shape[-1]
    e = e[..., :n_states]

    beta = 1.0 / (kB * T)  # Temp in Kelvin
    norb = nHeavy * 4 + nHydro

    mu = (e.gather(1, Nocc.unsqueeze(0).T - 1) + e.gather(1, Nocc.unsqueeze(0).T)) / 2

    # Build per-molecule orbital mask (valid columns per molecule)
    ar = torch.arange(n_states, device=device).unsqueeze(0)  # [1,M]
    Occ_mask_bool = ar < norb.unsqueeze(1)  # [B,M]
    Occ_mask = Occ_mask_bool.to(dtype)  # [B,M], matches original dtype

    # Newton iterations for μ so that sum_i f_i(μ) == Nocc
    Fe_vec = None
    Nocc_f = Nocc.to(dtype)  # [B], for arithmetic
    for _ in range(max_iter):
        Fe_raw = torch.sigmoid(-beta * (e - mu))  # [B,M]
        Fe_vec = Fe_raw * Occ_mask  # apply mask once
        Occ_sum = Fe_vec.sum(dim=1)  # [B]
        dOcc = (beta * Fe_vec * (1.0 - Fe_vec)).sum(dim=1).clamp_min(tiny)  # [B]

        err = (Nocc_f - Occ_sum).abs()
        if bool((err <= occ_tol).all()):
            break

        delta = (Nocc_f - Occ_sum) / dOcc  # [B]
        mu = mu + delta.unsqueeze(1)  # [B,1]

    # Density matrix D0 (then unpack to block form), same semantics as original
    X = QQ * Fe_vec.unsqueeze(1)  # [B,norb_full,M]
    D0 = X @ QQ.transpose(1, 2)  # [B,norb_full,norb_full]
    D0 = 2.0 * unpack(D0, nHeavy.to(torch.long), nHydro.to(torch.long), H0.shape[-1])

    # Entropy: only where eps < f < 1-eps; avoid log(0)
    mask_S = (Fe_vec > entropy_eps) & ((1.0 - Fe_vec) > entropy_eps)  # [B,M]
    p_safe = Fe_vec.masked_fill(~mask_S, 0.5)
    S_terms = -kB * (p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
    S = (S_terms * mask_S.to(dtype)).sum(dim=1)  # [B]

    return D0, S, QQ, e, Fe_vec, mu, Occ_mask
