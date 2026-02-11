from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .rcis_batch import get_occ_virt, makeA_pi_batched


def elec_energy_excited_xl(
    mol, R: torch.Tensor, w, e_mo, xl_bomd_params: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute excited-state electronic energy and amplitudes with XL-ESMD approach

    Conventions / shapes (batched):
      - R:    (b, r, m, n)        (AO basis transition density for eta with r roots)

    Returns:
      - E:    (b, r) Excitation energy for r roots
      - X_AO: (b, r, m, n) amplitude in AO basis per r roots

    Do a Krylov-subspace Kernel construction (if xl_bomd_params has "max_rank"):
    """
    # --- Get MO windows / transforms ---
    nocc, nvirt, Cocc, Cvirt, ea_ei = get_occ_virt(mol, orbital_window=None, e_mo=e_mo)
    b = R.shape[0]
    r = R.shape[1]
    n = nocc * nvirt

    # --- Build eta = Xbar in MO (occ-virt) with r blocks ---
    with torch.no_grad():
        eta = torch.einsum("bmi,brmn,bna->bria", Cocc, R, Cvirt)

    # --- Define Coulomb-exchange integral function for amplitudes ---
    def G_apply(Y: torch.Tensor) -> torch.Tensor:
        # Y: (b,r,nocc,nvirt) -> G(Y): (b,r,nocc,nvirt)
        R_y = torch.einsum("bmi,bria,bna->brmn", Cocc, Y, Cvirt)
        G_ao = makeA_pi_batched(mol, R_y, w)  # expected (b,r,m,n)
        G_y = torch.einsum("bmi,brmn,bna->bria", Cocc, G_ao, Cvirt)
        return 2.0 * G_y

    Gx = G_apply(eta)  # (b,r,nocc,nvirt)

    eta_flat = eta.reshape(b, r, n)
    Gx_flat = Gx.reshape(b, r, n)
    ea_ei_flat = ea_ei.reshape(b, 1, n)

    # --- Solve for xi and omega ---
    with torch.no_grad():
        xi_flat, omega = solve_for_amplitude_omega(eta_flat, ea_ei_flat, Gx_flat)
        # xi_flat: (b,r,n), omega_br: (b,r)

    E1 = (xi_flat * xi_flat * ea_ei_flat).sum(dim=2)  # (b,r)
    E2 = ((2.0 * xi_flat - eta_flat) * Gx_flat).sum(dim=2)  # (b,r)
    E = E1 + E2  # (b,r)
    # --- Convert xi back to AO basis ---
    with torch.no_grad():
        xi = xi_flat.view(b, r, nocc, nvirt)
        xi_AO = torch.einsum("bmi,bria,bna->brmn", Cocc, xi, Cvirt)  # (b,r,m,n)

        # --- Compute dxi2dt2 via rank-m Krylov ---
        if xl_bomd_params is not None and "max_rank" in xl_bomd_params:
            with torch.no_grad():
                # precond = 1.0 / (omega.unsqueeze(-1) * (1.0 / ea_ei_flat) - 1.0 + 1e-8)  # (b,r,n)
                # precond = torch.ones_like(eta_flat)  # no preconditioning;
                precond = make_apply_precond_rank1(ea_ei_flat, eta_flat, xi_flat, omega)
                jvp_xi = make_jvp_xi(ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt)
                dxi2dt2_flat = compute_dxi2dt2_rankm(eta_flat, xi_flat, jvp_xi, xl_bomd_params, precond)
                # Convert to AO basis and store in mol for later use in BOMD
                mol.dxi2dt2 = torch.einsum(
                    "bmi,bria,bna->brmn", Cocc, dxi2dt2_flat.view(b, r, nocc, nvirt), Cvirt
                )

    return E, xi_AO


def make_apply_precond_rank1(
    ea_ei_flat: torch.Tensor,  # (b,1,n) diagonal of A (energy gaps)
    eta: torch.Tensor,  # (b,r,n)
    xi: torch.Tensor,  # (b,r,n)  (needed to build q)
    omega: torch.Tensor,  # (b,r)
    tau: float = 1e-8,  # damping for diagonal inversion
    eps: float = 1e-12,  # safety for divides / dot products
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns apply_prec(v) that computes y ≈ (J_f)^{-1} v using a cheap approximation:
        J_f ≈ D + u q^T
    with D diagonal, and (D + u q^T)^{-1} applied via Sherman–Morrison.

    Shapes:
      ea_ei_flat: (b,1,n)
      eta, xi:    (b,r,n)
      omega:      (b,r)
      v:          (b,r,n)  -> returns (b,r,n)
    """

    b, r, n = eta.shape

    # Precompute cheap pieces (all O(n))
    ainv = 1.0 / ea_ei_flat  # (b,1,n)
    # ainv = ainv.expand(b, r, n)                   # (b,r,n) broadcast over r

    u = ainv * eta  # (b,r,n) = A^{-1} eta
    d = torch.sum(eta * u, dim=2).clamp(min=eps)  # (b,r)   = eta^T A^{-1} eta

    # q ~ [-(xi-eta) - omega*A^{-1}eta] / d  (dropping K^T term)
    q = (-(xi - eta) - omega.unsqueeze(-1) * u) / d.unsqueeze(-1)  # (b,r,n)

    # D^{-1} diagonal: dinv_i = 1 / (omega/a_i - 1 + tau)
    dinv = 1.0 / (omega.unsqueeze(-1) * ainv - 1.0 + tau)  # (b,r,n)

    # Also precompute D^{-1}u for Sherman–Morrison
    w = dinv * u  # (b,r,n)
    qw = torch.sum(q * w, dim=2)  # (b,r)
    denom = 1.0 + qw
    # Avoid division by (near) zero while preserving sign
    denom = torch.sign(denom) * torch.clamp(denom.abs(), min=eps)  # (b,r)

    # Flatten br for easier indexing in apply_prec
    dinv = dinv.view(b * r, n)
    q = q.view(b * r, n)
    w = w.view(b * r, n)
    denom = denom.view(b * r)

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        z = dinv * v  # (B,n) = D^{-1} v
        qz = torch.sum(q * z, dim=1)  # (B)   = q^T D^{-1} v
        return z - w * (qz / denom).unsqueeze(-1)  # Sherman–Morrison

    return apply_prec


def solve_for_amplitude_omega(
    eta: torch.Tensor,  # (b,r,nov)  == eta
    ea_ei: torch.Tensor,  # (b,1,nov)    diagonal of A
    G: torch.Tensor,  # (b,r,nov)  == G(eta)
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve for xi and omega for each (b,r) block:
        D xi - omega*eta = -G(eta)
        (2xi - eta)^T eta = 1

    Returns:
        xi_flat:    (b,r,n) where n=nocc*nvirt
        omega: (b,r)
    """
    # Broadcast A's diagonal over r

    if torch.any(ea_ei < 1e-14):
        raise RuntimeError(
            "HOMO-LUMO gaps are small values; Kernel calculation for XL-ESMD will be unstable. Check inputs."
        )
    invA = 1.0 / ea_ei

    # L = 1 + ||eta||^2
    L = 1.0 + torch.sum(eta * eta, dim=2)  # (b,r)

    # A^{-1}(-G)
    Ainv_minusG = -invA * G  # (b,r,n)

    # S_inv = 1 / (2 * eta^T A^{-1} eta)
    den = torch.sum(invA * (eta * eta), dim=2).clamp(min=eps)  # (b,r)
    S_inv = 0.5 / den  # (b,r)

    # rhs2 = L - 2 * eta^T A^{-1}(-G) = 1+||eta||^2 + 2*eta^T A^{-1}G
    rhs2 = L - 2.0 * torch.sum(eta * Ainv_minusG, dim=2)  # (b,r)

    omega = rhs2 * S_inv  # (b,r)

    # X = A^{-1}(omega*eta - G)
    xi = (-G + eta * omega.unsqueeze(-1)) * invA  # (b,r,n)

    return xi, omega


# ----------------------------
# JVP for xi(eta) when G is linear
# ----------------------------
def make_jvp_xi(
    ea_ei: torch.Tensor,  # (b,1,nov)
    eta: torch.Tensor,  # (b,r,nov)
    xi: torch.Tensor,  # (b,r,nov)
    omega: torch.Tensor,  # (b,r)
    G_apply: Callable[[torch.Tensor], torch.Tensor],  # (b,r,i,a)->(b,r,i,a)
    nocc: int,
    nvirt: int,
    eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns jvp_xi(v) -> dxi_brn for the mapping xi(eta),
    assuming G is linear so d/dλ G(eta+λv)|0 = G(v).

    v shape: (b,r,n), returns (b,r,n)
    """
    invA_brn = 1.0 / ea_ei
    b, r, n = eta.shape
    i, a = nocc, nvirt

    def jvp_xi(v: torch.Tensor) -> torch.Tensor:
        # G(v)
        v_bria = v.reshape(b, r, i, a)
        Gv_bria = G_apply(v_bria)
        Gv = Gv_bria.reshape(b, r, n)

        # b_vec = omega*v - G(v)
        b_vec = omega.unsqueeze(-1) * v - Gv  # (b,r,n)

        # c = -2 (xi - eta)^T v
        dDS = xi - eta
        c = -2.0 * torch.sum(dDS * v, dim=2)  # (b,r)

        # eta^T A^{-1} eta
        eta_Ainv_eta = torch.sum(invA_brn * (eta * eta), dim=2).clamp(min=eps)  # (b,r)

        # eta^T A^{-1} b
        eta_Ainv_b = torch.sum(invA_brn * (eta * b_vec), dim=2)  # (b,r)

        # deltaOmega = (c - 2 eta^T A^{-1} b) / (2 eta^T A^{-1} eta)
        deltaOmega = (c - 2.0 * eta_Ainv_b) / (2.0 * eta_Ainv_eta)  # (b,r)

        # deltaXi = A^{-1}(b + eta*deltaOmega)
        deltaXi = invA_brn * (b_vec + eta * deltaOmega.unsqueeze(-1))  # (b,r,n)
        return deltaXi

    return jvp_xi


# ----------------------------
# Rank-m Krylov approximation for dxi2dt2 (not backprop-friendly)
# ----------------------------
def compute_dxi2dt2_rankm(
    eta_brn: torch.Tensor,  # (b,r,n)
    xi_brn: torch.Tensor,  # (b,r,n)
    jvp_xi: Callable[[torch.Tensor], torch.Tensor],  # v_brn -> dxi_brn
    xl_params: Dict,
    precond: torch.Tensor,  # preconditioner of shape (b,r,n) to improve convergence
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute dxi2dt2 ≈ -V_k alpha using a rank-m Krylov / Arnoldi-like process,
    where W_k spans Df[v] = (J_xi[v] - v), and we fit dDS = xi-eta in span(W).

    Returns:
      dxi2dt2_brn: (b,r,n)
    """
    Rank = int(xl_params["max_rank"])
    err_threshold = float(xl_params["err_threshold"])

    b, r, n = eta_brn.shape
    B = b * r

    # Flatten br into a single batch for linear algebra
    eta = eta_brn.reshape(B, n)
    xi = xi_brn.reshape(B, n)
    # K0 = precond.reshape(B, n)  # (B,n)
    K0 = precond

    # dDS = K0 * (xi - eta)  # (B,n)
    dDS = K0(xi - eta)  # (B,n)
    dDS_norm = torch.linalg.vector_norm(dDS, dim=1).clamp(min=eps)

    V = torch.zeros((B, n, Rank), dtype=xi.dtype, device=xi.device)
    W = torch.zeros((B, n, Rank), dtype=xi.dtype, device=xi.device)

    dW = dDS.clone()
    Error = torch.full((B,), 10.0, dtype=xi.dtype, device=xi.device)
    Rank_m = 0

    def vecnorm(u: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(u, dim=1)

    last_alpha = None

    for k in range(Rank):
        if torch.max(Error) <= err_threshold:
            break

        vk = dW  # (B,n)

        # Modified Gram-Schmidt (2 passes for improved numerical stability)
        if k > 0:
            Vprev = V[:, :, :k]  # (B,n,k)
            for _ in range(2):
                coeffs = torch.einsum("Bnk,Bn->Bk", Vprev, vk)  # (B,k)
                proj = torch.einsum("Bnk,Bk->Bn", Vprev, coeffs)  # (B,n)
                vk = vk - proj

        vk = vk / vecnorm(vk).clamp(min=eps).unsqueeze(-1)
        V[:, :, k] = vk

        # JVP expects (b,r,n)
        vk_brn = vk.reshape(b, r, n)
        dxi_brn = jvp_xi(vk_brn)  # (b,r,n)
        dxi = dxi_brn.reshape(B, n)  # (B,n)

        # wk = K0 * (dxi - vk)  # (B,n)
        wk = K0(dxi - vk)  # (B,n)
        W[:, :, k] = wk
        dW = wk
        Rank_m = k + 1

        # Least-squares in W-subspace: alpha = argmin ||Wk alpha - dDS||
        Wk = W[:, :, :Rank_m]  # (B,n,m)
        O = torch.einsum("Bnm,Bnl->Bml", Wk, Wk)  # (B,m,m)
        rhs = torch.einsum("Bnm,Bn->Bm", Wk, dDS)  # (B,m)

        alpha = torch.linalg.solve(O, rhs.unsqueeze(-1)).squeeze(-1)  # (B,m)
        last_alpha = alpha

        IdentRes = torch.einsum("Bnm,Bm->Bn", Wk, alpha)  # (B,n)
        Error = vecnorm(IdentRes - dDS) / dDS_norm
        print(f"Krylov rank {Rank_m}, max relative error in dDS fit: {torch.max(Error).item():.2e}")

    if Rank_m == 0:
        raise RuntimeError("Rank-m loop did not run; check max_rank and inputs.")

    # Compute final dxi2dt2 = -Vk alpha
    Vk = V[:, :, :Rank_m]
    alpha = last_alpha  # should exist
    dxi2dt2 = -torch.einsum("Bnm,Bm->Bn", Vk, alpha)  # (B,n)

    return dxi2dt2.reshape(b, r, n)


def sample_noisy_R_energy(
    mol,
    R,
    w,
    e_mo,
    n_steps=50,
    noise_scale=1e-5,
    cumulative=False,
    seed=None,
    return_data=False,
    plot=True,
    fit_order=2,
):
    """
    Generate noisy copies Rbar of R, compute total energy for each, and plot |R-Rbar| vs E.

    Args:
            mol: molecule object expected by elec_energy_excited_xl
            R: torch.Tensor, original transition-density tensor (same shape used by elec_energy_excited_xl)
            w, e_mo: arguments forwarded to elec_energy_excited_xl
            n_steps: number of noisy samples
            noise_scale: standard deviation of Gaussian noise added to R (absolute scale)
            cumulative: if True, noise is added cumulatively (Rbar <- Rbar + noise). If False, noise is added to original R each sample.
            seed: optional int seed for reproducibility
            return_data: if True, returns (deltas, energies)
            plot: if True, plots |R-Rbar| vs E using matplotlib
            fit_order: polynomial order to fit to (deltas, energies) for plotting (default 2)

    Returns:
            None or (deltas, energies) if return_data=True
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Work with detached copies to avoid grads / side-effects
    R = R.clone().detach()
    Rbar = R.clone().detach()
    deltas = []
    energies = []
    Rbase = torch.randn_like(R)
    for i in range(n_steps):
        noise = i * noise_scale * Rbase
        if cumulative:
            Rbar = Rbar + noise
        else:
            Rbar = R + noise

        with torch.no_grad():
            E, _ = elec_energy_excited_xl(mol, Rbar, w, e_mo)

        # Reduce E to a scalar for plotting: mean across batch if batched
        if torch.is_tensor(E):
            energy_scalar = float(E.mean().item())
        else:
            energy_scalar = float(np.asarray(E).mean())

        delta = float(torch.linalg.norm(R - Rbar).item())

        deltas.append(delta)
        energies.append(energy_scalar)
        # print(f"Step {i+1}/{n_steps}: |R-Rbar| = {delta:.6e}, E = {energy_scalar:.6e}")

    if plot:
        plt.figure()
        plt.plot(deltas, energies, marker="o", linestyle="-", label="data")
        plt.xlabel("|R - Rbar| (transition density error)")
        plt.ylabel("E")
        plt.title("Energy vs Error in transition density")

        # Fit an n-th order polynomial if there are enough points
        try:
            if len(deltas) > fit_order:
                x = np.array(deltas)
                y = np.array(energies)
                # compute polynomial coefficients (highest degree first)
                coeffs = np.polyfit(x, y, fit_order)
                print("Fitted polynomial coefficients (highest degree first):", coeffs)
                p = np.poly1d(coeffs)
                # sort for a smooth curve
                idx = np.argsort(x)
                xs = x[idx]
                ys = p(xs)
                plt.plot(xs, ys, color="red", linestyle="--", label=f"polynomial fit (order={fit_order})")
                plt.legend()
        except Exception:
            # don't fail plotting if fit fails; still show raw data
            pass

        plt.grid(True)
        plt.show()

    exit()

    if return_data:
        return deltas, energies
