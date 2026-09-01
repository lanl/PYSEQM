import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .excited_state_utils import get_occ_virt
from .hcore import (
    orthogonalized_overlap_from_matrices,
    overlap_between_geometries,
    overlap_matrix_current_geometry,
)
from .rcis_batch import _uniform_molecule_dimensions, makeA_pi_batched, packone_batch


def transport_cis_amplitudes(
    eta: torch.Tensor,
    occupied_overlap: torch.Tensor,
    virtual_overlap: torch.Tensor,
    *,
    polar_unitarize: bool = True,
) -> torch.Tensor:
    """Transport occupied--virtual amplitudes between two MO gauges.

    The leading dimensions of ``eta`` are arbitrary; its final two dimensions
    are occupied and virtual indices.  The overlap blocks map old MOs into the
    new gauge.  Keeping this algebra separate from the geometry-dependent
    overlap construction makes gauge transport directly testable.
    """
    if eta.shape[-2:] != (occupied_overlap.shape[-1], virtual_overlap.shape[-1]):
        raise ValueError("eta's final dimensions must match the occupied and virtual overlap blocks")

    S_oo = occupied_overlap
    S_vv = virtual_overlap
    if polar_unitarize:
        U_oo, _, Vh_oo = torch.linalg.svd(S_oo, full_matrices=False)
        U_vv, _, Vh_vv = torch.linalg.svd(S_vv, full_matrices=False)
        S_oo = U_oo @ Vh_oo
        S_vv = U_vv @ Vh_vv

    while S_oo.dim() < eta.dim():
        S_oo = S_oo.unsqueeze(-3)
        S_vv = S_vv.unsqueeze(-3)
    return S_oo @ eta @ S_vv.transpose(-1, -2)


def transport_mo_transition_amplitudes(
    mol, eta_prev, coords_prev, mos_prev, S_prev, *, polar_unitarize: bool = True
):
    """Transport CIS amplitudes from the previous to the current MO basis.

    ``eta_prev`` has shape ``(batch, roots, nov)`` or
    ``(batch, roots, nocc, nvirt)`` and is transformed independently for every
    root as ``S_oo @ eta_prev @ S_vv.T``.  ``S_prev`` is the packed AO overlap
    at the previous geometry.  The returned ``S_curr`` should be saved and
    passed as ``S_prev`` on the next step.  With ``polar_unitarize=True``,
    occupied and virtual overlap blocks are replaced by their nearest
    orthogonal matrices before transporting the amplitudes.
    """
    _, _, norb, nocc = _uniform_molecule_dimensions(mol)
    nvirt = norb - nocc
    nov = nocc * nvirt
    if eta_prev.dim() == 3 and eta_prev.shape[-1] == nov:
        eta = eta_prev.view(int(mol.nmol), eta_prev.shape[1], nocc, nvirt)
        flatten_output = True
    elif eta_prev.dim() == 4 and eta_prev.shape[-2:] == (nocc, nvirt):
        eta = eta_prev
        flatten_output = False
    else:
        raise ValueError("eta_prev must have shape (batch, roots, nov) or (batch, roots, nocc, nvirt)")

    pack_spec = (4 * mol.nHeavy[0], mol.nHydro[0], norb)
    coords_curr = mol.coordinates.detach()
    S_curr = packone_batch(overlap_matrix_current_geometry(mol), *pack_spec)
    S_cross = packone_batch(overlap_between_geometries(mol, coords_curr, coords_prev), *pack_spec)
    S_ao = orthogonalized_overlap_from_matrices(S_curr, S_cross, S_prev)
    C_curr = mol.molecular_orbitals
    C_prev = mos_prev
    S_mo = C_curr.transpose(1, 2) @ (S_ao @ C_prev)
    S_oo = S_mo[:, :nocc, :nocc]
    S_vv = S_mo[:, nocc:, nocc:]
    transported = transport_cis_amplitudes(eta, S_oo, S_vv, polar_unitarize=polar_unitarize)
    transported = transported.reshape_as(eta_prev) if flatten_output else transported
    return transported, S_curr


def get_exact_excited(mol, w, e_mo, R):
    nocc, nvirt, Cocc, Cvirt, ea_ei = get_occ_virt(mol, orbital_window=None, e_mo=e_mo)
    b = R.shape[0]
    r = R.shape[1]
    n = nocc * nvirt

    exact = mol.cis_amplitudes  # [:,mol.active_state]
    exact_e = mol.cis_energies  # [:,mol.active_state]
    torch.set_printoptions(precision=15)
    print("Exact CIS energies: ", exact_e)

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

    ea_ei_flat = ea_ei.reshape(b, 1, n)

    xl_bomd_params = {"max_rank": 3, "err_threshold": 1e-8}

    eta_flat = eta.reshape(b, r, n)
    print(
        f"Raw inital tdm diff (before occ-virt subspace projection of inital guess) is {torch.linalg.vector_norm(R - mol.transition_density_matrices, dim=(-2, -1))}"
    )

    print(
        "Before iterations: "
        f"diff = {torch.linalg.vector_norm(eta_flat - exact, dim=-1)}, "
        f"diff_tdm = {torch.linalg.vector_norm(torch.einsum('bmi,bria,bna->brmn', Cocc, eta_flat.view(b, r, nocc, nvirt), Cvirt) - mol.transition_density_matrices, dim=(-2, -1))}"
        "\n"
    )
    for iter in range(20):
        Gx = G_apply(eta_flat.view(b, r, nocc, nvirt))  # (b,r,nocc,nvirt)

        Gx_flat = Gx.reshape(b, r, n)

        # --- Solve for xi and omega ---
        with torch.no_grad():
            xi_flat, omega = solve_for_amplitude_omega(eta_flat, ea_ei_flat, Gx_flat)
            # xi_flat: (b,r,n), omega_br: (b,r)

        E1 = (xi_flat * xi_flat * ea_ei_flat).sum(dim=2)  # (b,r)
        E2 = ((2.0 * xi_flat - eta_flat) * Gx_flat).sum(dim=2)  # (b,r)
        E = E1 + E2  # (b,r)

        # --- Compute dxi2dt2 with the full old-Jacobian GMRES kernel ---
        # precond = make_apply_precond_rank1(ea_ei_flat, eta_flat, xi_flat, omega)
        precond = make_apply_precond_diagonal(ea_ei_flat, eta_flat, omega)
        jvp_xi = make_jvp_xi(ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt)
        dxi2dt2_flat = compute_dxi2dt2_rankm(eta_flat, xi_flat, jvp_xi, xl_bomd_params, precond)
        eta_flat = eta_flat + dxi2dt2_flat
        print(
            f"Iter {iter + 1}: E = {E.squeeze().cpu().numpy()}, diff = {torch.linalg.vector_norm(eta_flat - exact, dim=-1)}, "
            # f"diff_tdm = {torch.linalg.vector_norm(torch.einsum('bmi,bria,bna->brmn', Cocc, eta_flat.view(b, r, nocc, nvirt), Cvirt) - mol.transition_density_matrices, dim=(-2, -1))}"
        )
        if torch.all((E - exact_e).abs() < mol.seqm_parameters["excited_states"]["cis_tol"] * 10.0):
            print("Converged to exact energy!")
            break
        # # Convert to AO basis and store in mol for later use in BOMD
        # mol.dxi2dt2 = torch.einsum(
        #     "bmi,bria,bna->brmn", Cocc, dxi2dt2_flat.view(b, r, nocc, nvirt), Cvirt
        # )
    exit(0)


def elec_energy_excited_xl(
    # mol, R: torch.Tensor, w, e_mo, xl_E, xl_bomd_params: Optional[Dict] = None
    mol,
    R: torch.Tensor,
    w,
    e_mo,
    xl_bomd_params: Optional[Dict] = None,
    sequential_lock: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    MO_basis = R.shape == (b, r, n)
    if MO_basis:
        eta = R.reshape(b, r, nocc, nvirt)  # assume R is in MO basis; replace with above line for AO input
    else:
        with torch.no_grad():
            eta = torch.einsum("bmi,brmn,bna->bria", Cocc, R, Cvirt)

    # --- Define Coulomb-exchange integral function for amplitudes ---
    def G_apply(Y: torch.Tensor) -> torch.Tensor:
        # Y: (b,r,nocc,nvirt) -> G(Y): (b,r,nocc,nvirt)
        R_y = torch.einsum("bmi,bria,bna->brmn", Cocc, Y, Cvirt)
        G_ao = makeA_pi_batched(mol, R_y, w)  # expected (b,r,m,n)
        G_y = torch.einsum("bmi,brmn,bna->bria", Cocc, G_ao, Cvirt)
        return 2.0 * G_y

    eta_flat = eta.reshape(b, r, n)
    ea_ei_flat = ea_ei.reshape(b, 1, n)

    constraint_mode = "independent_linearized"
    if xl_bomd_params is not None:
        constraint_mode = str(xl_bomd_params.get("constraint_mode", constraint_mode)).lower()
    if sequential_lock:
        constraint_mode = "ordered_linearized"

    valid_modes = {"independent_linearized", "independent_exact", "ordered_linearized"}
    if constraint_mode not in valid_modes:
        raise ValueError(
            f"Unknown XL-ESMD constraint_mode={constraint_mode!r}; expected one of {sorted(valid_modes)}"
        )

    # --- Solve for xi and multipliers ---
    with torch.no_grad():
        # xi_flat: (b,r,n), omega_br: (b,r)
        if constraint_mode == "ordered_linearized":
            xi_flat, omega = solve_for_amplitudes_ordered(
                eta_flat, ea_ei_flat, lambda eta_s: G_apply(eta_s.view(b, r, nocc, nvirt)).reshape(b, r, n)
            )
        elif constraint_mode == "independent_exact":
            Gx_flat = G_apply(eta).reshape(b, r, n)
            omega_init = getattr(mol, "xlesmd_multipliers", None)
            if not (torch.is_tensor(omega_init) and omega_init.shape == (b, r)):
                omega_init = None
            xi_flat, omega = solve_for_amplitude_omega_newton(
                eta_flat, ea_ei_flat, Gx_flat, omega_init=omega_init
            )
        else:
            Gx_flat = G_apply(eta).reshape(b, r, n)
            xi_flat, omega = solve_for_amplitude_omega(eta_flat, ea_ei_flat, Gx_flat)
        mol.xlesmd_multipliers = omega.detach().clone()

    Gx_flat = G_apply(eta_flat.view(b, r, nocc, nvirt)).reshape(b, r, n)

    # if hasattr(mol, "omega_xl"):
    #     omega_init = mol.omega_xl
    # else:
    #     omega_init = mol.cis_energies
    # xi_flat, omega = solve_for_amplitude_omega_newton(eta_flat, ea_ei_flat, Gx_flat, omega_init)
    # mol.omega_xl = omega

    # xi_flat = refine_orthogonalize_delta3(xi_flat)

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
                # # The JVP/Krylov path below acts on each (molecule, root)
                # # Each (molecule, root) is propagated independently.
                # precond = make_apply_precond_diagonal(ea_ei_flat, eta_flat, omega)
                # jvp_xi = make_jvp_xi(ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt)
                # dxi2dt2_flat = compute_dxi2dt2_rankm(eta_flat, xi_flat, jvp_xi, xl_bomd_params, precond)

                if constraint_mode == "ordered_linearized":
                    jvp_xi = make_jvp_xi_ordered(ea_ei_flat, eta_flat, omega, G_apply, nocc, nvirt)
                elif constraint_mode == "independent_exact":
                    jvp_xi = make_jvp_xi_full_normalized(
                        ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt
                    )
                else:
                    jvp_xi = make_jvp_xi(ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt)

                # The diagonal orbital-gap/multiplier response is a cheap
                # approximation to (J_xi - I)^-1.  The GMRES operator is
                # (I-J_xi), so use its negative as a right preconditioner.
                preconditioner_name = str(xl_bomd_params.get("krylov_preconditioner", "none")).lower()
                preconditioner = None
                if preconditioner_name == "none":
                    pass
                elif preconditioner_name == "diagonal":
                    tau = float(xl_bomd_params.get("preconditioner_tau", 1.0e-5))
                    preconditioner_omega = (
                        torch.diagonal(omega, dim1=-2, dim2=-1) if omega.ndim == 3 else omega
                    )
                    kernel_inverse = make_apply_precond_diagonal(
                        ea_ei_flat, eta_flat, preconditioner_omega, tau=tau
                    )

                    def preconditioner(v):
                        return -kernel_inverse(v.reshape(b * r, n)).reshape(b, r, n)

                elif preconditioner_name == "rank1":
                    tau = float(xl_bomd_params.get("preconditioner_tau", 1.0e-5))
                    preconditioner_omega = (
                        torch.diagonal(omega, dim1=-2, dim2=-1) if omega.ndim == 3 else omega
                    )
                    kernel_inverse = make_apply_precond_rank1(
                        ea_ei_flat, eta_flat, xi_flat, preconditioner_omega, tau=tau
                    )

                    def preconditioner(v):
                        return -kernel_inverse(v.reshape(b * r, n)).reshape(b, r, n)

                else:
                    raise ValueError("krylov_preconditioner must be 'none', 'diagonal', or 'rank1'.")

                dxi2dt2_flat, krylov_info = compute_dxi2dt2_old_jacobian_gmres(
                    eta_brn=eta_flat,
                    xi_brn=xi_flat,
                    nu_br=omega,
                    ea_ei_flat=ea_ei_flat,
                    G_apply=G_apply,
                    nocc=nocc,
                    nvirt=nvirt,
                    xl_params=xl_bomd_params,
                    jvp_xi=jvp_xi,
                    preconditioner=preconditioner,
                    return_info=True,
                )
                mol.Krylov_Error = krylov_info["relative_residual"]

                # Convert to AO basis and store in mol for later use in BOMD
                if MO_basis:
                    mol.dxi2dt2 = dxi2dt2_flat
                else:
                    mol.dxi2dt2 = torch.einsum(
                        "bmi,bria,bna->brmn", Cocc, dxi2dt2_flat.view(b, r, nocc, nvirt), Cvirt
                    )
        else:
            mol.dxi2dt2 = None

        identity = torch.eye(r, dtype=eta_flat.dtype, device=eta_flat.device).unsqueeze(0)
        diagnostics = {
            "constraint_mode": constraint_mode,
            "fixed_point_residual": torch.linalg.vector_norm(xi_flat - eta_flat, dim=-1),
            "eta_orthogonality": torch.linalg.matrix_norm(eta_flat @ eta_flat.transpose(-1, -2) - identity),
            "xi_orthogonality": torch.linalg.matrix_norm(xi_flat @ xi_flat.transpose(-1, -2) - identity),
        }
        if constraint_mode == "independent_exact":
            diagnostics["minimum_pole_distance"] = torch.amin(
                torch.abs(ea_ei_flat - omega.unsqueeze(-1)), dim=-1
            )
        if xl_bomd_params is not None and "max_rank" in xl_bomd_params:
            diagnostics.update(
                krylov_rank=krylov_info["rank"],
                krylov_relative_residual=krylov_info["relative_residual"],
                krylov_converged=krylov_info["converged"],
                krylov_kernel_gain=krylov_info["kernel_gain"],
                krylov_kernel_gain_scale=krylov_info["kernel_gain_scale"],
                krylov_preconditioner=preconditioner_name,
            )
        mol.xlesmd_diagnostics = diagnostics
    if xl_bomd_params is not None and xl_bomd_params.get("verbose_xlesmd", False):
        dot_xi = torch.einsum("brn,bRn->brR", xi_flat, xi_flat)
        identity = torch.eye(r, dtype=xi_flat.dtype, device=xi_flat.device).unsqueeze(0)
        print("XL-ESMD constraint mode:", constraint_mode)
        print("Xi overlap matrix:\n", dot_xi)
        print("Xi orthogonality residual:", torch.linalg.norm(dot_xi - identity))
        if torch.is_tensor(getattr(mol, "cis_amplitudes", None)):
            print("Overlap of xi with previous xi", torch.sum(xi_flat * mol.cis_amplitudes, dim=2))

    return E, xi_AO, xi_flat


def refine_orthogonalize_delta3(Z: torch.Tensor) -> torch.Tensor:
    """
    One delta^3 iterative refinement step for orthogonalizing rows of Z.

    Z: shape (b, r, n), rows nearly orthonormal
    returns: shape (b, r, n), refined columns
    """

    k = Z.shape[1]
    I = torch.eye(k, dtype=Z.dtype, device=Z.device).unsqueeze(0)

    X = Z @ Z.transpose(-2, -1)
    # print(f"Overlap before orthogonalizing\n{X}")

    # delta^3 coefficients:
    # p(X) = 1.875 I - 1.25 X + 0.375 X^2
    P = 1.875 * I - 1.25 * X + 0.375 * (X @ X)

    # print(f"Norm of the orthogonalizing matrix is {torch.linalg.vector_norm(P, dim=(1, 2))}")
    orthoZ = P @ Z
    # print(f"Diff in eta before/after orthogonalizing:{torch.linalg.vector_norm(Z - orthoZ, dim=2)}")
    return orthoZ


def make_apply_precond_null() -> Callable[[torch.Tensor], torch.Tensor]:
    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        return v  # (B,n)

    return apply_prec


def make_apply_precond_diagonal(
    ea_ei_flat: torch.Tensor,  # (b,1,n) diagonal of A (energy gaps)
    eta: torch.Tensor,  # (b,r,n)
    omega: torch.Tensor,  # (b,r)
    tau: float = 1e-5,  # damping for diagonal inversion
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Shapes:
      ea_ei_flat: (b,1,n)
      eta, xi:    (b,r,n)
      omega:      (b,r)
      v:          (b,r,n)  -> returns (b,r,n)
    """

    b, r, n = eta.shape

    # Precompute cheap pieces (all O(n))
    ainv = 1.0 / ea_ei_flat  # (b,1,n)
    # D^{-1} diagonal: dinv_i = 1 / (omega/a_i - 1 + tau)
    dinv = 1.0 / (omega.unsqueeze(-1) * ainv - 1.0 + tau)  # (b,r,n)

    # Flatten br for easier indexing in apply_prec
    dinv = dinv.view(b * r, n)

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        return dinv * v  # (B,n) = D^{-1} v

    return apply_prec


def make_apply_precond_rank1(
    ea_ei_flat: torch.Tensor,  # (b,1,n) diagonal of A (energy gaps)
    eta: torch.Tensor,  # (b,r,n)
    xi: torch.Tensor,  # (b,r,n)  (needed to build q)
    omega: torch.Tensor,  # (b,r)
    tau: float = 1e-5,  # damping for diagonal inversion
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


def solve_for_amplitudes_sequentially_locked(
    eta: torch.Tensor,
    ea_ei: torch.Tensor,
    G_apply_one: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve XL-ESMD roots sequentially, locking each converged ``xi``.

    Root ``s`` is constrained to be Euclidean-orthogonal to all earlier roots.
    The auxiliary input is projected against the normalized locked roots before
    solving, and the constrained equations are solved through their small Schur
    complement rather than by projecting a finished solution afterward.
    """
    if torch.any(ea_ei < 1e-14):
        raise RuntimeError("HOMO-LUMO gaps are too small for a stable XL-ESMD kernel calculation.")

    b, r, n = eta.shape
    invD = 1.0 / ea_ei.expand(b, r, n)[:, 0]
    xi_out = torch.empty_like(eta)
    eta_out = torch.empty_like(eta)
    omega_out = torch.empty((b, r), dtype=eta.dtype, device=eta.device)
    locked = []

    for root in range(r):
        eta_root = eta[:, root]
        if locked:
            L = torch.stack(locked, dim=1)  # (b, nlocked, n), Euclidean-orthonormal rows
            overlap = torch.einsum("bln,bn->bl", L, eta_root)
            eta_root = eta_root - torch.einsum("bln,bl->bn", L, overlap)
        else:
            L = None

        eta_norm = torch.linalg.vector_norm(eta_root, dim=1)
        if torch.any(eta_norm < eps):
            raise RuntimeError(
                f"XL-ESMD sequential locking removed all of root {root}'s auxiliary amplitude."
            )

        G_root = G_apply_one(eta_root)
        a = torch.sum(eta_root * invD * eta_root, dim=1)
        g_eta = torch.sum(eta_root * invD * G_root, dim=1)
        q = 1.0 + torch.sum(eta_root * eta_root, dim=1)

        if L is None:
            omega_root = (q + 2.0 * g_eta) / (2.0 * a.clamp_min(eps))
            xi_root = invD * (-G_root + eta_root * omega_root.unsqueeze(-1))
        else:
            # D xi - omega eta + L lambda = -G,
            # 2 eta^T xi = 1 + eta^T eta, and L^T xi = 0.
            bvec = torch.einsum("bn,bln->bl", eta_root * invD, L)
            C = torch.einsum("bln,bmn->blm", L * invD.unsqueeze(1), L)
            g_L = torch.einsum("bln,bn->bl", L, invD * G_root)
            nlocked = L.shape[1]
            schur = torch.empty((b, nlocked + 1, nlocked + 1), dtype=eta.dtype, device=eta.device)
            schur[:, 0, 0] = 2.0 * a
            schur[:, 0, 1:] = -2.0 * bvec
            schur[:, 1:, 0] = -bvec
            schur[:, 1:, 1:] = C
            rhs = torch.cat([(q + 2.0 * g_eta).unsqueeze(1), -g_L], dim=1)
            solution = torch.linalg.solve(schur, rhs.unsqueeze(-1)).squeeze(-1)
            omega_root = solution[:, 0]
            lamb = solution[:, 1:]
            xi_root = invD * (
                -G_root + eta_root * omega_root.unsqueeze(-1) - torch.einsum("bln,bl->bn", L, lamb)
            )

        # Normalize only the locked copy: xi_root itself remains the exact
        # constrained XL solution used in the energy expression.
        lock_root = xi_root / torch.linalg.vector_norm(xi_root, dim=1, keepdim=True).clamp_min(eps)
        locked.append(lock_root)
        eta_out[:, root] = eta_root
        xi_out[:, root] = xi_root
        omega_out[:, root] = omega_root

    return xi_out, omega_out, eta_out


def solve_for_amplitudes_ordered(
    eta: torch.Tensor,
    ea_ei: torch.Tensor,
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Case 2a with ordered (lower-state) linearized constraints.

    State ``i`` obeys

        eta_j.T xi_i = 1/2 (delta_ij + eta_j.T eta_i),  j <= i.

    This removes the internal state-rotation gauge of the raw block solve while
    retaining only diagonal applications and small leading-block solves.
    Multipliers are returned in a lower-triangular ``(batch, roots, roots)``
    tensor, with row ``i`` holding the constraints used for state ``i``.
    """
    if eta.ndim != 3:
        raise ValueError("eta must have shape (batch, roots, excitations)")
    if torch.any(ea_ei <= 1e-14):
        raise RuntimeError("Positive occupied--virtual gaps are required for XL-ESMD.")

    b, r, n = eta.shape
    if ea_ei.shape[0] != b or ea_ei.shape[-1] != n:
        raise ValueError("ea_ei must have shape compatible with (batch, 1, excitations)")
    G = G_apply(eta)
    if G.shape != eta.shape:
        raise ValueError(f"G_apply must preserve eta's shape; got {G.shape} instead of {eta.shape}")

    invD = 1.0 / ea_ei
    xi = torch.empty_like(eta)
    multipliers = torch.zeros((b, r, r), dtype=eta.dtype, device=eta.device)

    for state in range(r):
        H = eta[:, : state + 1]
        eta_i = eta[:, state]
        G_i = G[:, state]
        S = torch.einsum("bjn,bkn->bjk", H, invD * H)
        gram_col = torch.einsum("bjn,bn->bj", H, eta_i)
        delta = torch.zeros_like(gram_col)
        delta[:, state] = 1.0
        c = 0.5 * (delta + gram_col)
        rhs = c + torch.einsum("bjn,bn->bj", H, invD[:, 0] * G_i)

        # A singular leading Gram block means the auxiliary states do not
        # define independent ordered constraints; silently regularizing it
        # would change the shadow functional.
        eval_min = torch.linalg.eigvalsh(S).amin(dim=-1)
        scale = torch.linalg.matrix_norm(S, ord=2).clamp_min(1.0)
        if torch.any(eval_min <= eps * scale):
            raise RuntimeError(f"Linearly dependent auxiliary states in ordered XL-ESMD state {state}.")

        lamb = torch.linalg.solve(S, rhs.unsqueeze(-1)).squeeze(-1)
        xi[:, state] = invD[:, 0] * (torch.einsum("bjn,bj->bn", H, lamb) - G_i)
        multipliers[:, state, : state + 1] = lamb

    return xi, multipliers


def make_jvp_xi_ordered(
    ea_ei: torch.Tensor,
    eta: torch.Tensor,
    multipliers: torch.Tensor,
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
    eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Analytic JVP of :func:`solve_for_amplitudes_ordered`."""
    b, r, n = eta.shape
    if n != nocc * nvirt:
        raise ValueError("The excitation dimension must equal nocc*nvirt.")
    if multipliers.shape != (b, r, r):
        raise ValueError(f"multipliers must have shape {(b, r, r)}")
    invD = 1.0 / ea_ei
    G_eta = G_apply(eta.reshape(b, r, nocc, nvirt)).reshape(b, r, n)

    def jvp_xi(v: torch.Tensor) -> torch.Tensor:
        if v.shape != eta.shape:
            raise ValueError(f"Expected v shape {eta.shape}, got {v.shape}")
        G_v = G_apply(v.reshape(b, r, nocc, nvirt)).reshape(b, r, n)
        dxi = torch.empty_like(v)

        for state in range(r):
            H = eta[:, : state + 1]
            V = v[:, : state + 1]
            eta_i = eta[:, state]
            v_i = v[:, state]
            G_i = G_eta[:, state]
            Gv_i = G_v[:, state]
            lamb = multipliers[:, state, : state + 1]

            S = torch.einsum("bjn,bkn->bjk", H, invD * H)
            dS = torch.einsum("bjn,bkn->bjk", V, invD * H)
            dS = dS + torch.einsum("bjn,bkn->bjk", H, invD * V)
            dc = 0.5 * (torch.einsum("bjn,bn->bj", V, eta_i) + torch.einsum("bjn,bn->bj", H, v_i))
            du = dc
            du = du + torch.einsum("bjn,bn->bj", V, invD[:, 0] * G_i)
            du = du + torch.einsum("bjn,bn->bj", H, invD[:, 0] * Gv_i)
            dlamb = torch.linalg.solve(S, (du - torch.einsum("bjk,bk->bj", dS, lamb)).unsqueeze(-1)).squeeze(
                -1
            )

            dxi[:, state] = invD[:, 0] * (
                torch.einsum("bjn,bj->bn", V, lamb) + torch.einsum("bjn,bj->bn", H, dlamb) - Gv_i
            )
        return dxi

    return jvp_xi


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
    jvp_xi: Callable[[torch.Tensor], torch.Tensor],
    xl_params: Dict,
    precond: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute dxi2dt2 using an independent rank-m Krylov approximation
    for every (molecule, root) block.

    We solve, approximately,

        (J_xi - I) Y = xi - eta

    in a preconditioned Krylov subspace and return

        dxi2dt2 = -Y.

    Unlike the previous implementation, Krylov convergence and breakdown are
    handled independently for every (molecule, root).  One converged or
    rank-deficient root therefore cannot make the batched solve singular.

    Shapes:
        eta_brn, xi_brn : (b, r, n)
        output          : (b, r, n)
    """
    Rank = int(xl_params["max_rank"])
    err_threshold = float(xl_params["err_threshold"])

    if Rank <= 0:
        return torch.zeros_like(eta_brn)

    b, r, n = eta_brn.shape
    B = b * r

    eta = eta_brn.reshape(B, n)
    xi = xi_brn.reshape(B, n)

    # Initial preconditioned residual / right-hand side.
    #
    # The existing algorithm fits
    #
    #     W alpha ~= dDS
    #
    # with
    #
    #     dDS = K0 (xi - eta),
    #     W    = K0 (J_xi - I) V,
    #
    # and finally returns -V alpha.
    dDS = precond(xi - eta)  # (B,n)

    # Use a dtype-aware floor.  eps=1e-12 is suitable for float64 but too
    # small to be a useful Krylov-breakdown threshold in float32.
    finfo = torch.finfo(xi.dtype)
    breakdown_tol = max(float(eps), 10.0 * float(finfo.eps))

    dDS_norm_raw = torch.linalg.vector_norm(dDS, dim=1)  # (B,)
    dDS_norm = dDS_norm_raw.clamp_min(breakdown_tol)

    # A root with essentially zero xi-eta already needs no correction.
    active = dDS_norm_raw > breakdown_tol

    V = torch.zeros((B, n, Rank), dtype=xi.dtype, device=xi.device)
    W = torch.zeros_like(V)

    dW = dDS.clone()

    # Relative residual for each independent (molecule, root).
    Error = torch.zeros((B,), dtype=xi.dtype, device=xi.device)
    Error[active] = float("inf")

    Rank_m = 0
    last_alpha = None

    def solve_gram_pinv(
        O: torch.Tensor,  # (B,m,m)
        rhs: torch.Tensor,  # (B,m)
    ) -> torch.Tensor:
        """
        Solve O alpha = rhs using the Moore-Penrose pseudoinverse
        of the small symmetric positive-semidefinite Gram matrix.

        This is the key difference from torch.linalg.solve(): a root whose
        Krylov columns are linearly dependent is allowed to be rank deficient
        without causing the entire batch to fail.
        """
        evals, evecs = torch.linalg.eigh(O)  # O = U diag(lambda) U^T

        # O is positive semidefinite in exact arithmetic.  Drop tiny and
        # negative roundoff eigenvalues.
        lam_max = evals.amax(dim=-1, keepdim=True).clamp_min(0.0)

        # Relative cutoff for numerical rank.
        rtol = max(float(eps), 10.0 * float(finfo.eps))
        cutoff = rtol * lam_max

        keep = evals > cutoff

        inv_evals = torch.zeros_like(evals)
        inv_evals[keep] = 1.0 / evals[keep]

        # alpha = U diag(lambda^+) U^T rhs
        rhs_eig = torch.einsum("Bmk,Bm->Bk", evecs, rhs)

        alpha = torch.einsum("Bmk,Bk->Bm", evecs, inv_evals * rhs_eig)

        return alpha

    for k in range(Rank):
        # All roots have either converged or experienced Krylov breakdown.
        if not torch.any(active):
            break

        vk = dW.clone()

        # Modified Gram-Schmidt, independently for every B=(molecule,root).
        if k > 0:
            Vprev = V[:, :, :k]  # (B,n,k)

            # Two passes improve orthogonality.
            for _ in range(2):
                coeffs = torch.einsum("Bnk,Bn->Bk", Vprev, vk)
                vk = vk - torch.einsum("Bnk,Bk->Bn", Vprev, coeffs)

        vknorm = torch.linalg.vector_norm(vk, dim=1)  # (B,)

        # IMPORTANT:
        # breakdown is per root, not global.
        can_expand = active & (vknorm > breakdown_tol)

        if not torch.any(can_expand):
            break

        # Do not divide inactive/broken roots by zero.
        vk_normalized = torch.zeros_like(vk)
        vk_normalized[can_expand] = vk[can_expand] / vknorm[can_expand].unsqueeze(-1)

        V[:, :, k] = vk_normalized

        # jvp_xi is linear in the trial direction.  Inactive roots therefore
        # receive a zero direction and produce zero JVP contribution.
        vk_brn = vk_normalized.reshape(b, r, n)
        dxi_brn = jvp_xi(vk_brn)
        dxi = dxi_brn.reshape(B, n)

        # W_k = K0 [(J_xi - I) V_k]
        wk = precond(dxi - vk_normalized)

        # Explicitly keep dead roots out of this column.
        wk[~can_expand] = 0.0

        W[:, :, k] = wk
        dW = wk

        Rank_m = k + 1

        # ------------------------------------------------------------
        # Least-squares fit
        #
        #       alpha = argmin ||W alpha - dDS||
        #
        # independently for every molecule/root.
        # ------------------------------------------------------------
        Wk = W[:, :, :Rank_m]  # (B,n,m)

        O = torch.einsum("Bnm,Bnl->Bml", Wk, Wk)

        rhs = torch.einsum("Bnm,Bn->Bm", Wk, dDS)

        # Rank-deficient-safe replacement for
        #
        #     torch.linalg.solve(O, rhs)
        #
        alpha = solve_gram_pinv(O, rhs)
        last_alpha = alpha

        fitted = torch.einsum("Bnm,Bm->Bn", Wk, alpha)

        Error = torch.linalg.vector_norm(fitted - dDS, dim=1) / dDS_norm

        # xi == eta -> exact zero correction, so define its error as zero.
        Error[dDS_norm_raw <= breakdown_tol] = 0.0

        # Each root continues only if:
        #
        #   1. it had a valid new Krylov direction, and
        #   2. it has not yet met its requested residual tolerance.
        #
        # A broken root freezes at its best solution so far while the other
        # roots continue building their own Krylov spaces.
        active = can_expand & (Error > err_threshold)

    # If all roots were already at xi == eta, or if no usable Krylov direction
    # existed at all, the physically sensible correction is zero.
    if Rank_m == 0 or last_alpha is None:
        return torch.zeros_like(eta_brn)

    Vk = V[:, :, :Rank_m]

    # Existing XL convention:
    #
    #     dxi2dt2 = -V alpha
    #
    dxi2dt2 = -torch.einsum("Bnm,Bm->Bn", Vk, last_alpha)

    # Optional diagnostic -- remove this for production if desired.
    print(
        f"Krylov rank used: {Rank_m}, "
        f"final max relative error: {torch.max(Error).item():.2e}, "
        f"converged/broken blocks: {(~active).sum().item()}/{B}"
    )

    return dxi2dt2.reshape(b, r, n)


def compute_dxi2dt2_projected_subspace(
    eta_brn: torch.Tensor,  # (b,r,n)
    xi_brn: torch.Tensor,  # (b,r,n)
    jvp_xi: Callable[[torch.Tensor], torch.Tensor],  # v_brn -> dxi_brn
    xl_params: Dict,
    precond: Callable[[torch.Tensor], torch.Tensor],
    mol,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Solve J[Y] = eta - xi with a projected, restarted subspace iteration.

    The Jacobian action is J[V] = J_xi[V] - V, while the preconditioner is only
    used to generate new search directions from the current residual.
    """
    max_space = int(xl_params["max_rank"])
    max_iter = int(xl_params.get("max_iter", max_space))
    err_threshold = float(xl_params["err_threshold"])

    b, r, n = eta_brn.shape
    B = b * r

    eta = eta_brn.reshape(B, n)
    xi = xi_brn.reshape(B, n)
    rhs = eta - xi

    sqrtn = math.sqrt(n)

    def rmsnorm(u: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(u, dim=1) / sqrtn

    def apply_jacobian(v: torch.Tensor) -> torch.Tensor:
        v_brn = v.reshape(b, r, n)
        return (jvp_xi(v_brn) - v_brn).reshape(B, n)

    def solve_projected(T: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        m = T.shape[-1]
        eye = torch.eye(m, dtype=T.dtype, device=T.device).unsqueeze(0).expand(T.shape[0], -1, -1)
        return torch.linalg.solve(T + eps * eye, g.unsqueeze(-1)).squeeze(-1)

    # mol.dxi2dt2 available from last step use that after converting to MO basis
    if hasattr(mol, "dxi2dt2"):
        nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol)
        t = torch.einsum("bmi,brmn,bna->bria", Cocc, mol.dxi2dt2, Cvirt).reshape(b, r, n).reshape(B, n)
    else:
        t = precond(rhs)
    # t_norm = rmsnorm(t)
    # active = t_norm > eps
    # if not torch.any(active):
    #     return torch.zeros_like(eta_brn)

    V = torch.zeros((B, max_space, n), dtype=eta.dtype, device=eta.device)
    W = torch.zeros_like(V)

    Y = torch.zeros((B, n), dtype=eta.dtype, device=eta.device)
    residual = rhs.clone()
    # res_norm = rmsnorm(residual)
    # if torch.max(res_norm) < err_threshold:
    #     return Y.reshape(b, r, n)

    raw_t_norm = torch.linalg.vector_norm(t, dim=1)
    v1 = torch.zeros_like(t)
    # v1[active] = t[active] / raw_t_norm[active].unsqueeze(-1)
    v1 = t / raw_t_norm.unsqueeze(-1)
    V[:, 0, :] = v1
    W[:, 0, :] = apply_jacobian(v1)
    m = 1

    for _ in range(max_iter):
        Vm = V[:, :m, :]
        Wm = W[:, :m, :]

        g = torch.einsum("Bmn,Bn->Bm", Vm, rhs)
        T = torch.einsum("Bmn,Bkn->Bmk", Vm, Wm)
        coeffs = solve_projected(T, g)

        Y = torch.einsum("Bmn,Bm->Bn", Vm, coeffs)
        JY = torch.einsum("Bmn,Bm->Bn", Wm, coeffs)
        residual = rhs - JY
        res_norm = rmsnorm(residual)

        # print error and convergence info
        print("Iter {}, residual norm: {:.2e}".format(m, torch.max(res_norm).item()))
        if torch.max(res_norm) < err_threshold:
            return Y.reshape(b, r, n)

        t = precond(residual)
        if m > 0:
            for _ in range(2):
                proj = torch.einsum("Bmn,Bn->Bm", Vm, t)
                t = t - torch.einsum("Bmn,Bm->Bn", Vm, proj)

        t_norm = rmsnorm(t)

        if not torch.any(t_norm > eps):
            return Y.reshape(b, r, n)

        raw_t_norm = torch.linalg.vector_norm(t, dim=1)
        t_next = torch.zeros_like(t)
        keep = t_norm > eps
        t_next[keep] = t[keep] / raw_t_norm[keep].unsqueeze(-1)

        if m < max_space:
            V[:, m, :] = t_next
            W[:, m, :] = apply_jacobian(t_next)
            m += 1
            continue

        # Restart from the current residual when the subspace is full.
        t_restart = precond(residual)
        y_norm = rmsnorm(Y)
        has_y = y_norm > eps
        if torch.any(has_y):
            raw_y_norm = torch.linalg.vector_norm(Y, dim=1)
            y_dir = torch.zeros_like(Y)
            y_dir[has_y] = Y[has_y] / raw_y_norm[has_y].unsqueeze(-1)
            t_restart = t_restart - torch.sum(t_restart * y_dir, dim=1, keepdim=True) * y_dir

        t_restart_norm = rmsnorm(t_restart)
        if not torch.any(t_restart_norm > eps):
            return Y.reshape(b, r, n)

        V.zero_()
        W.zero_()
        raw_t_restart_norm = torch.linalg.vector_norm(t_restart, dim=1)
        restart_keep = t_restart_norm > eps
        V[:, 0, :] = 0.0
        V[restart_keep, 0, :] = t_restart[restart_keep] / raw_t_restart_norm[restart_keep].unsqueeze(-1)
        W[:, 0, :] = apply_jacobian(V[:, 0, :])
        m = 1

    return Y.reshape(b, r, n)


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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError("Plotting requires matplotlib. Install it with `pip install matplotlib`.")
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


def solve_for_amplitude_omega_newton(
    eta_flat: torch.Tensor,  # (b,r,n), used to choose the continuous branch
    ea_ei_flat: torch.Tensor,  # (b,1,n) diagonal a_i
    Gx_flat: torch.Tensor,  # (b,r,n) g
    omega_init: Optional[torch.Tensor] = None,  # (b,r)
    *,
    pole_eps: float = 1e-6,
    tol: float = 1e-10,
    max_newton_iter: int = 50,
    max_backtrack: int = 20,
    max_step_frac_of_pole_dist: float = 0.8,
    fp_eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safeguarded stationary-branch solve for case 1b.

    ``omega_init`` should be the multiplier from the preceding geometry.  If
    it is omitted, the Rayleigh quotient of the supplied auxiliary amplitude
    selects the locally connected branch.  Steps may not cross a pole of
    ``D - omega I`` and failure to solve the secular equation is reported
    rather than hidden by normalizing a nonstationary vector afterward.
    """
    a = ea_ei_flat  # (b,1,n)
    g = Gx_flat  # (b,r,n)
    device, dtype = g.device, g.dtype
    b, r, n = g.shape

    if eta_flat.shape != g.shape:
        raise ValueError(f"eta_flat and Gx_flat must have the same shape; got {eta_flat.shape} and {g.shape}")

    # Initialize on the branch indicated by eta (or by the previous step).
    if omega_init is None:
        eta_norm2 = torch.sum(eta_flat.square(), dim=-1).clamp_min(fp_eps)
        omega = torch.sum(eta_flat * (a * eta_flat + g), dim=-1) / eta_norm2
    else:
        if omega_init.shape != (b, r):
            raise ValueError(f"omega_init must have shape {(b, r)}, got {omega_init.shape}")
        omega = omega_init.to(device=device, dtype=dtype).clone()

    def signed_floor(x: torch.Tensor, floor: float) -> torch.Tensor:
        sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return sign * x.abs().clamp_min(floor)

    # A Rayleigh estimate can land exactly on a diagonal pole.  Move it by a
    # small, deterministic amount while retaining the same local interval.
    pole_delta = a - omega.unsqueeze(-1)
    nearest_delta, nearest_idx = pole_delta.abs().min(dim=-1)
    near_pole = nearest_delta < pole_eps
    if torch.any(near_pole):
        nearest_signed = pole_delta.gather(-1, nearest_idx.unsqueeze(-1)).squeeze(-1)
        direction = torch.where(nearest_signed >= 0, -torch.ones_like(omega), torch.ones_like(omega))
        omega = torch.where(near_pole, omega + direction * (2.0 * pole_eps), omega)

    def f(om: torch.Tensor) -> torch.Tensor:  # (b,r)
        d = signed_floor(a - om.unsqueeze(-1), pole_eps)
        return (g.square() / d.square()).sum(dim=-1) - 1.0

    def fp(om: torch.Tensor) -> torch.Tensor:  # (b,r)
        d = signed_floor(a - om.unsqueeze(-1), pole_eps)
        return 2.0 * (g.square() / (d.square() * d)).sum(dim=-1)

    def pole_dist(om: torch.Tensor) -> torch.Tensor:  # (b,r)
        # min_i |a_i - om|
        return (a - om.unsqueeze(-1)).abs().min(dim=-1).values

    val = f(omega)
    for _ in range(max_newton_iter):
        converged = torch.abs(val) < tol
        if torch.all(converged):
            break

        der = fp(omega)
        if torch.any((der.abs() < fp_eps) & (~converged)):
            raise RuntimeError("The exact-normalization secular derivative vanished on the selected branch.")
        step = torch.where(converged, torch.zeros_like(val), val / signed_floor(der, fp_eps))

        # cap step so we don't run into poles in one update
        dist = pole_dist(omega)
        cap = (dist - pole_eps).clamp_min(0.0)
        cap = max_step_frac_of_pole_dist * cap
        step = torch.clamp(step, min=-cap, max=cap)

        # backtracking: accept only if improves and stays away from poles
        alpha = torch.ones_like(val)
        accepted = converged.clone()
        omega_prop = omega.clone()
        val_prop = val.clone()
        for _ in range(max_backtrack):
            trial = omega - alpha * step
            trial_val = f(trial)
            ok = converged | ((pole_dist(trial) >= pole_eps) & (torch.abs(trial_val) < torch.abs(val)))
            newly_accepted = ok & (~accepted)
            omega_prop = torch.where(newly_accepted, trial, omega_prop)
            val_prop = torch.where(newly_accepted, trial_val, val_prop)
            accepted |= ok
            if torch.all(accepted):
                break
            alpha = torch.where(accepted, alpha, 0.5 * alpha)

        if not torch.all(accepted):
            raise RuntimeError(
                "Could not take a pole-safe decreasing Newton step for exact-normalization XL-ESMD."
            )
        omega = omega_prop
        val = val_prop

    if not torch.all(torch.abs(val) < tol):
        raise RuntimeError(
            f"Exact-normalization XL-ESMD did not converge; max secular residual={torch.max(torch.abs(val)).item():.3e}."
        )

    # Recover the stationary amplitude.  Do not renormalize it: the secular
    # equation itself is the normalization condition.
    d = signed_floor(a - omega.unsqueeze(-1), pole_eps)
    xi = -g / d

    return xi, omega


def make_apply_precond_diag_full_normalized(
    a_diag: torch.Tensor,  # (b,1,n) diagonal of A
    eta: torch.Tensor,  # (b,r,n) (only for shape)
    omega: torch.Tensor,  # (b,r)
    Geta: Optional[torch.Tensor] = None,  # (b,r,n) = G(eta) if already available; else None
    tau: float = 1e-8,  # damping to avoid division by tiny denominators
    pole_eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns apply_prec(vB) -> (B,n), where vB is flattened (b*r,n).
    No G calls inside apply. If Geta is provided, uses it only to fit a scalar c.
    """

    b, r, n = eta.shape
    B = b * r

    # K_diag = a_i - omega
    K = (a_diag.expand(b, r, n) - omega.unsqueeze(-1)).reshape(B, n)  # (B,n)

    # Optional scalar fit c from already-computed Geta (no extra cost).
    if Geta is not None:
        etaB = eta.reshape(B, n)
        GetaB = Geta.reshape(B, n)
        num = torch.sum(etaB * GetaB, dim=1)  # (B,)
        den = torch.sum(etaB * etaB, dim=1).clamp_min(pole_eps)  # (B,)
        c = (num / den).unsqueeze(1)  # (B,1)
    else:
        c = 0.0

    # scale = -(K / (K + c + tau))
    denom = K + c + tau
    denom = torch.where(denom.abs() >= pole_eps, denom, denom.sign() * pole_eps)
    scale = -(K / denom)  # (B,n)

    def apply_prec(vB: torch.Tensor) -> torch.Tensor:
        return scale * vB

    return apply_prec


def make_jvp_xi_full_normalized(
    a_diag: torch.Tensor,  # (b,1,nov)  diagonal of A
    eta: torch.Tensor,  # (b,r,nov)
    xi: torch.Tensor,  # (b,r,nov)  solution at this eta (||xi||=1)
    omega: torch.Tensor,  # (b,r)      scalar Ω
    G_apply: Callable[[torch.Tensor], torch.Tensor],  # (b,r,i,a)->(b,r,i,a)
    nocc: int,
    nvirt: int,
    *,
    pole_eps: float = 1e-12,
    denom_eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    JVP for the mapping xi(eta) induced by:

        (A - Ω I) xi = - G(eta)     with ||xi||^2 = 1

    assuming G is linear so d/dλ G(eta+λv)|0 = G(v).

    Returns a function jvp_xi(v) that computes dxi for direction v.

    Shapes:
      v:  (b,r,nov)
      out:(b,r,nov)

    Notes:
      - A is diagonal, provided via a_diag.
      - This computes δΩ from the normalization constraint, then δxi from the
        linearized stationarity equation.
      - Be careful near poles where (a_diag - omega) ~ 0.
    """
    b, r, n = eta.shape
    i, a = nocc, nvirt
    assert n == i * a, "nov must equal nocc*nvirt"

    # K = A - Ω I  (diagonal)
    # We'll form invK safely: invK = 1/(K)
    k = a_diag - omega.unsqueeze(-1)  # (b,r,n) via broadcast of (b,1,n)
    k_sign = torch.where(k >= 0, torch.ones_like(k), -torch.ones_like(k))
    invK = 1.0 / torch.where(k.abs() >= pole_eps, k, k_sign * pole_eps)

    def jvp_xi(v: torch.Tensor) -> torch.Tensor:
        # 1) t = G(v)
        v_bria = v.reshape(b, r, i, a)
        Gv_bria = G_apply(v_bria)
        t = Gv_bria.reshape(b, r, n)  # (b,r,n)

        # 2) invK and invK_xi

        # 3) δΩ = (xi^T invK Gv) / (xi^T invK xi)
        #    numerator = xi^T (invK * t) == sum( xi * invK * t )
        num = torch.sum(xi * (invK * t), dim=2)  # (b,r)

        #    denom = xi^T invK xi == sum( xi * invK * xi )
        den = torch.sum(xi * (invK * xi), dim=2)  # (b,r)
        den_sign = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
        den = torch.where(den.abs() >= denom_eps, den, den_sign * denom_eps)

        deltaOmega = num / den  # (b,r)

        # 4) δxi = invK * (δΩ * xi - Gv)
        deltaXi = invK * (deltaOmega.unsqueeze(-1) * xi - t)  # (b,r,n)
        return deltaXi

    return jvp_xi


def compute_dxi2dt2_projected_minres(
    eta_brn: torch.Tensor,  # (b,r,n)
    xi_brn: torch.Tensor,  # (b,r,n)
    nu_br: torch.Tensor,  # (b,r), solved multiplier
    ea_ei_flat: torch.Tensor,  # (b,1,n) = D
    G_eta_flat: torch.Tensor,  # (b,r,n), cached G(eta)
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
    xl_params: Dict,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Projected MINRES/Lanczos solve for the XL electronic correction.

    Solve independently for every (molecule, root):

        (P K P) y = -P K eta

    where

        K = D + G - nu I
        P = I - eta eta^T / (eta^T eta)

    and return y = ddot(eta) / omega_XL^2.

    The XL Verlet recurrence supplies the omega_XL^2 dt^2 factor through
    kappa, so this function does NOT multiply by an XL frequency.

    Cost:
        - RHS: no new G_apply; uses cached G_eta_flat
        - each Krylov rank: exactly one batched G_apply
    """
    max_rank = int(xl_params["max_rank"])
    err_threshold = float(xl_params.get("err_threshold", 1e-6))

    if max_rank <= 0:
        return torch.zeros_like(eta_brn)

    b, r, n = eta_brn.shape
    B = b * r

    eta = eta_brn.reshape(B, n)
    nu = nu_br.reshape(B)

    D = ea_ei_flat.expand(b, r, n).reshape(B, n)
    G_eta = G_eta_flat.reshape(B, n)

    finfo = torch.finfo(eta.dtype)
    breakdown_tol = max(float(eps), 100.0 * float(finfo.eps))

    # ------------------------------------------------------------
    # Tangent-space projector
    #
    #     P v = v - eta (eta^T v)/(eta^T eta)
    # ------------------------------------------------------------
    eta_norm2 = torch.sum(eta * eta, dim=1).clamp_min(breakdown_tol)

    def project(v: torch.Tensor) -> torch.Tensor:
        coeff = torch.sum(eta * v, dim=1) / eta_norm2
        return v - eta * coeff.unsqueeze(-1)

    # ------------------------------------------------------------
    # Operator
    #
    #     A(v) = P (D + G - nu I) P v
    #
    # Exactly ONE expensive G_apply per call.
    # ------------------------------------------------------------
    def A_apply(v: torch.Tensor) -> torch.Tensor:
        pv = project(v)

        Gv = G_apply(pv.reshape(b, r, nocc, nvirt)).reshape(B, n)

        Kv = (D - nu.unsqueeze(-1)) * pv + Gv

        return project(Kv)

    # ------------------------------------------------------------
    # RHS
    #
    #     rhs = -P K eta
    #
    # G(eta) has already been computed in the electronic solve, so
    # there is NO additional G_apply here.
    # ------------------------------------------------------------
    K_eta = (D - nu.unsqueeze(-1)) * eta + G_eta
    rhs = -project(K_eta)

    # Equivalent near an exact shadow solve:
    #
    #     K eta = -D (xi - eta)
    #
    # so
    #
    #     rhs = P D (xi - eta).
    #
    # Using K_eta directly is preferable because G_eta is already cached
    # and it does not assume the stationarity equation is numerically exact.

    beta0 = torch.linalg.vector_norm(rhs, dim=1)
    beta0_safe = beta0.clamp_min(breakdown_tol)

    # Roots whose RHS is already zero need zero correction.
    live = beta0 > breakdown_tol

    if not torch.any(live):
        return torch.zeros_like(eta_brn)

    # ------------------------------------------------------------
    # Lanczos basis
    #
    # A V_m = V_{m+1} Tbar_m
    #
    # MINRES then minimizes
    #
    #     || beta e1 - Tbar_m y ||
    #
    # and x_m = V_m y.
    # ------------------------------------------------------------
    V = torch.zeros((B, n, max_rank), dtype=eta.dtype, device=eta.device)

    Tbar = torch.zeros((B, max_rank + 1, max_rank), dtype=eta.dtype, device=eta.device)

    small_rhs = torch.zeros((B, max_rank + 1), dtype=eta.dtype, device=eta.device)
    small_rhs[:, 0] = beta0

    q = torch.zeros_like(rhs)
    q[live] = rhs[live] / beta0[live].unsqueeze(-1)

    q_prev = torch.zeros_like(rhs)
    beta_prev = torch.zeros(B, dtype=eta.dtype, device=eta.device)

    solution = torch.zeros_like(rhs)

    relres = torch.zeros(B, dtype=eta.dtype, device=eta.device)
    relres[live] = float("inf")

    rank_used = 0

    for k in range(max_rank):
        if not torch.any(live):
            break

        iter_mask = live.clone()

        # Current Lanczos vector
        V[:, :, k] = q

        # One expensive G_apply here.
        z = A_apply(q)

        # Dead/converged roots must stay dead.
        z[~iter_mask] = 0.0

        # Three-term Lanczos recurrence
        if k > 0:
            z = z - (beta_prev * iter_mask).unsqueeze(-1) * q_prev

        alpha = torch.sum(q * z, dim=1)
        alpha = torch.where(iter_mask, alpha, torch.zeros_like(alpha))

        z = z - alpha.unsqueeze(-1) * q

        # Remove tiny numerical component parallel to eta.
        z = project(z)
        z[~iter_mask] = 0.0

        beta_next = torch.linalg.vector_norm(z, dim=1)
        beta_next = torch.where(iter_mask, beta_next, torch.zeros_like(beta_next))

        # Build the small symmetric Lanczos matrix.
        if k > 0:
            Tbar[:, k - 1, k] = torch.where(iter_mask, beta_prev, torch.zeros_like(beta_prev))

        Tbar[:, k, k] = alpha
        Tbar[:, k + 1, k] = beta_next

        rank_used = k + 1

        # --------------------------------------------------------
        # MINRES step:
        #
        #     min_y || beta e1 - Tbar y ||
        #
        # The matrices are tiny (max_rank x max_rank), so a
        # pseudoinverse is cheap and robust to per-root breakdown.
        # --------------------------------------------------------
        Tm = Tbar[:, : rank_used + 1, :rank_used]
        gm = small_rhs[:, : rank_used + 1]

        Tm_pinv = torch.linalg.pinv(Tm, rtol=100.0 * float(finfo.eps))

        y = torch.bmm(Tm_pinv, gm.unsqueeze(-1)).squeeze(-1)

        solution = torch.einsum("Bnm,Bm->Bn", V[:, :, :rank_used], y)

        # Cheap MINRES residual estimate: no new G_apply.
        small_residual = gm - torch.bmm(Tm, y.unsqueeze(-1)).squeeze(-1)

        relres = torch.linalg.vector_norm(small_residual, dim=1) / beta0_safe

        relres = torch.where(beta0 > breakdown_tol, relres, torch.zeros_like(relres))

        converged = relres <= err_threshold

        # Per-root Lanczos breakdown.
        can_expand = beta_next > breakdown_tol

        # Each root stops independently.
        live_next = iter_mask & (~converged) & can_expand

        q_next = torch.zeros_like(q)

        q_next[live_next] = z[live_next] / beta_next[live_next].unsqueeze(-1)

        q_prev = q
        q = q_next

        beta_prev = torch.where(live_next, beta_next, torch.zeros_like(beta_next))

        live = live_next

    if xl_params.get("verbose_krylov", False):
        print(
            f"Projected MINRES rank used: {rank_used}, "
            f"max estimated relative residual: "
            f"{torch.max(relres).item():.3e}"
        )

    # Numerical cleanup: enforce tangent-space result exactly.
    solution = project(solution)

    return solution.reshape(b, r, n)


def compute_dxi2dt2_old_jacobian_gmres(
    eta_brn: torch.Tensor,
    xi_brn: torch.Tensor,
    nu_br: torch.Tensor,
    ea_ei_flat: torch.Tensor,
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
    xl_params: Dict,
    eps: float = 1e-12,
    jvp_xi: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_info: bool = False,
):
    """Solve the original off-shell XL kernel equation with GMRES.

    The old Jacobian formulation is

        (I - J_xi) d = xi - eta,

    where ``J_xi`` is the derivative of the electronic map ``xi(eta)``.
    Unlike the projected MINRES equation, this retains the longitudinal
    component required when ``xi != eta``.  The operator is nonsymmetric away
    from self consistency, hence Arnoldi/GMRES rather than MINRES is used.

    Each (molecule, root) block is an independent GMRES solve.  Every Arnoldi
    iteration performs exactly one batched ``G_apply`` through ``jvp_xi``.
    A supplied preconditioner is applied on the right, so GMRES minimizes the
    true residual while using ``d = M^-1 y`` as its correction.
    """
    max_rank = int(xl_params["max_rank"])
    err_threshold = float(xl_params.get("err_threshold", 1e-6))

    if max_rank <= 0:
        solution = torch.zeros_like(eta_brn)
        info = {
            "rank": 0,
            "relative_residual": torch.ones(eta_brn.shape[:2], dtype=eta_brn.dtype, device=eta_brn.device),
            "converged": torch.zeros(eta_brn.shape[:2], dtype=torch.bool, device=eta_brn.device),
            "kernel_gain": torch.zeros(eta_brn.shape[:2], dtype=eta_brn.dtype, device=eta_brn.device),
            "kernel_gain_scale": torch.ones(eta_brn.shape[:2], dtype=eta_brn.dtype, device=eta_brn.device),
        }
        return (solution, info) if return_info else solution

    b, r, n = eta_brn.shape
    B = b * r
    dtype = eta_brn.dtype
    device = eta_brn.device
    finfo = torch.finfo(dtype)
    breakdown_tol = max(float(eps), 100.0 * float(finfo.eps))

    if jvp_xi is None:
        jvp_xi = make_jvp_xi(ea_ei_flat, eta_brn, xi_brn, nu_br, G_apply, nocc, nvirt, eps=eps)

    rhs = (xi_brn - eta_brn).reshape(B, n)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    rhs_norm_safe = rhs_norm.clamp_min(breakdown_tol)
    live = rhs_norm > breakdown_tol

    if not torch.any(live):
        solution = torch.zeros_like(eta_brn)
        info = {
            "rank": 0,
            "relative_residual": torch.zeros((b, r), dtype=dtype, device=device),
            "converged": torch.ones((b, r), dtype=torch.bool, device=device),
            "kernel_gain": torch.zeros((b, r), dtype=dtype, device=device),
            "kernel_gain_scale": torch.ones((b, r), dtype=dtype, device=device),
        }
        return (solution, info) if return_info else solution

    # Arnoldi relation: A V_m = V_{m+1} Hbar_m, A = I - J_xi.
    V = torch.zeros((B, n, max_rank + 1), dtype=dtype, device=device)
    Hbar = torch.zeros((B, max_rank + 1, max_rank), dtype=dtype, device=device)
    small_rhs = torch.zeros((B, max_rank + 1), dtype=dtype, device=device)
    small_rhs[:, 0] = rhs_norm
    V[live, :, 0] = rhs[live] / rhs_norm[live].unsqueeze(-1)

    solution = torch.zeros((B, n), dtype=dtype, device=device)
    relres = torch.zeros((B,), dtype=dtype, device=device)
    relres[live] = float("inf")
    rank_used = 0

    for k in range(max_rank):
        if not torch.any(live):
            break

        iter_mask = live.clone()
        vk = V[:, :, k].reshape(b, r, n)
        zk = preconditioner(vk) if preconditioner is not None else vk
        w = (zk - jvp_xi(zk)).reshape(B, n)
        w[~iter_mask] = 0.0

        # Two-pass modified Gram--Schmidt controls loss of orthogonality while
        # preserving the nonsymmetric Arnoldi relation used by GMRES.
        for _ in range(2):
            h = torch.einsum("Bni,Bn->Bi", V[:, :, : k + 1], w)
            Hbar[:, : k + 1, k] += h
            w = w - torch.einsum("Bni,Bi->Bn", V[:, :, : k + 1], h)

        h_next = torch.linalg.vector_norm(w, dim=1)
        h_next = torch.where(iter_mask, h_next, torch.zeros_like(h_next))
        Hbar[:, k + 1, k] = h_next

        can_expand = h_next > breakdown_tol
        V[can_expand, :, k + 1] = w[can_expand] / h_next[can_expand].unsqueeze(-1)
        rank_used = k + 1

        Hm = Hbar[:, : rank_used + 1, :rank_used]
        gm = small_rhs[:, : rank_used + 1]
        y = torch.bmm(torch.linalg.pinv(Hm, rtol=100.0 * float(finfo.eps)), gm.unsqueeze(-1)).squeeze(-1)
        solution = torch.einsum("Bni,Bi->Bn", V[:, :, :rank_used], y)

        small_residual = gm - torch.bmm(Hm, y.unsqueeze(-1)).squeeze(-1)
        relres = torch.linalg.vector_norm(small_residual, dim=1) / rhs_norm_safe
        relres = torch.where(rhs_norm > breakdown_tol, relres, torch.zeros_like(relres))

        converged = relres <= err_threshold
        live = iter_mask & (~converged) & can_expand

    if xl_params.get("verbose_krylov", False):
        print(
            f"Old-Jacobian GMRES rank used: {rank_used}, "
            f"max estimated relative residual: {torch.max(relres).item():.3e}"
        )

    solution = solution.reshape(b, r, n)
    if preconditioner is not None:
        solution = preconditioner(solution)

    # A nearly singular response map makes the exact inverse action much
    # larger than the electronic residual.  That is mathematically valid, but
    # can destabilize the finite-step XL oscillator far from the fixed point.
    # Treat ``kernel_max_amplification`` as a spectral/trust-region
    # regularizer: the unmodified value preserves the usual GMRES result,
    # while a positive value bounds ||d|| / ||xi-eta|| per root.
    correction_norm = torch.linalg.vector_norm(solution, dim=-1)
    gain = correction_norm / rhs_norm.reshape(b, r).clamp_min(breakdown_tol)
    gain_scale = torch.ones_like(gain)
    max_amplification = xl_params.get("kernel_max_amplification")
    if max_amplification is not None:
        max_amplification = float(max_amplification)
        if max_amplification <= 0.0:
            raise ValueError("kernel_max_amplification must be positive when supplied.")
        gain_scale = torch.clamp(max_amplification / gain.clamp_min(breakdown_tol), max=1.0)
        solution = solution * gain_scale.unsqueeze(-1)

    info = {
        "rank": rank_used,
        "relative_residual": relres.reshape(b, r),
        "converged": (relres <= err_threshold).reshape(b, r),
        "kernel_gain": gain,
        "kernel_gain_scale": gain_scale,
    }
    return (solution, info) if return_info else solution
