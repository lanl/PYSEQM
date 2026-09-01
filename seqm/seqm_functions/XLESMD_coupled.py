from typing import Callable, Dict, Optional, Tuple

import torch

from .excited_state_utils import get_occ_virt
from .rcis_batch import makeA_pi_batched

_COUPLED_MODE_ALIASES = {
    "2a": "coupled_linearized",
    "case_2a": "coupled_linearized",
    "coupled_linearized": "coupled_linearized",
    "2b": "coupled_exact",
    "case_2b": "coupled_exact",
    "coupled_exact": "coupled_exact",
}


def normalize_coupled_constraint_mode(mode: str) -> Optional[str]:
    """Return the canonical coupled mode name, or ``None`` for another mode."""
    return _COUPLED_MODE_ALIASES.get(str(mode).strip().lower())


def elec_energy_excited_xl_coupled(
    mol, R: torch.Tensor, w, e_mo, xl_bomd_params: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the raw coupled Case 2a or Case 2b shadow functional.

    These modes represent a *state subspace*.  The individual entries of the
    returned energy tensor are useful diagnostics, but only their sum is a
    rotation-invariant, variational nuclear potential.  ``Energy`` therefore
    adds the block sum, rather than selecting ``active_state``, when this
    function is used by XL-ESMD.
    """
    params = {} if xl_bomd_params is None else xl_bomd_params
    requested_mode = str(params.get("constraint_mode", "coupled_linearized"))
    constraint_mode = normalize_coupled_constraint_mode(requested_mode)
    if constraint_mode is None:
        raise ValueError(
            "The coupled XL-ESMD solver requires constraint_mode to be one of "
            f"{sorted(_COUPLED_MODE_ALIASES)}, got {requested_mode!r}."
        )

    nocc, nvirt, Cocc, Cvirt, ea_ei = get_occ_virt(mol, orbital_window=None, e_mo=e_mo)
    b = R.shape[0]
    r = R.shape[1]
    n = nocc * nvirt
    if r > n:
        raise ValueError("The coupled state block cannot have more roots than excitation dimensions.")

    MO_basis = R.shape == (b, r, n)
    if MO_basis:
        eta = R.reshape(b, r, nocc, nvirt)
    else:
        with torch.no_grad():
            eta = torch.einsum("bmi,brmn,bna->bria", Cocc, R, Cvirt)

    def G_apply(Y: torch.Tensor) -> torch.Tensor:
        R_y = torch.einsum("bmi,bria,bna->brmn", Cocc, Y, Cvirt)
        G_ao = makeA_pi_batched(mol, R_y, w)
        G_y = torch.einsum("bmi,brmn,bna->bria", Cocc, G_ao, Cvirt)
        return 2.0 * G_y

    eta_flat = eta.reshape(b, r, n)
    ea_ei_flat = ea_ei.reshape(b, 1, n)

    with torch.no_grad():
        Gx_solve = G_apply(eta).reshape(b, r, n)
        if constraint_mode == "coupled_linearized":
            xi_flat, lambda_ij = solve_for_amplitudes_coupled(eta_flat, ea_ei_flat, Gx_solve)
            solve_info = {
                "converged": True,
                "iterations": 1,
                "residual_norm": torch.linalg.vector_norm(
                    ea_ei_flat * xi_flat + Gx_solve - torch.einsum("bkn,bkl->bln", eta_flat, lambda_ij),
                    dim=(1, 2),
                ),
            }
            jvp_xi = make_jvp_xi_coupled(ea_ei_flat, eta_flat, xi_flat, lambda_ij, G_apply, nocc, nvirt)
        else:
            solve_tolerance = float(params.get("coupled_tolerance", 1.0e-10))
            xi_flat, lambda_ij, solve_info = solve_for_amplitudes_exact_orthonormal(
                eta_flat,
                ea_ei_flat,
                Gx_solve,
                initial_xi=eta_flat,
                tolerance=solve_tolerance,
                max_iter=int(params.get("coupled_max_iter", 200)),
            )
            if not solve_info["converged"]:
                residual = torch.max(solve_info["residual_norm"]).item()
                raise RuntimeError(
                    f"Case 2b exact coupled solve did not converge; maximum KKT residual is {residual:.3e}."
                )
            jvp_xi = make_jvp_xi_exact_orthonormal(ea_ei_flat, xi_flat, lambda_ij, G_apply, nocc, nvirt)

        preconditioner_name = "none"
        preconditioner = None
        if "max_rank" in params:
            # The cheap constraint-aware inverse below approximates
            # (J_xi - I)^-1.  GMRES solves (I - J_xi)d = rhs, hence the
            # required right preconditioner is its negative.
            preconditioner_name = str(params.get("coupled_krylov_preconditioner", "none")).lower()
            preconditioner_tau = float(params.get("coupled_preconditioner_tau", 1.0e-5))
            if preconditioner_name == "none":
                pass
            elif preconditioner_name == "lambda":
                kernel_inverse = make_apply_precond_lambda_coupled(
                    ea_ei_flat, lambda_ij, tau=preconditioner_tau
                )

                def preconditioner(v):
                    return -project_state_subspace_tangent(eta_flat, kernel_inverse(v))

            else:
                raise ValueError("coupled_krylov_preconditioner must be one of 'none' or 'lambda'.")
            dxi2dt2_flat, krylov_info = compute_dxi2dt2_coupled_gmres(
                eta_flat, xi_flat, jvp_xi, params, preconditioner=preconditioner, return_info=True
            )
        else:
            # K0 = -I baseline: retain only physical changes of the
            # orthonormal state subspace before applying the XL recurrence.
            dxi2dt2_flat = project_state_subspace_tangent(eta_flat, xi_flat - eta_flat)
            projected_rhs_norm = torch.linalg.vector_norm(dxi2dt2_flat, dim=(1, 2))
            krylov_info = {
                "rank": 0,
                "relative_residual": torch.where(
                    projected_rhs_norm > 0.0,
                    torch.ones_like(projected_rhs_norm),
                    torch.zeros_like(projected_rhs_norm),
                ),
                "converged": projected_rhs_norm == 0.0,
                "kernel_gain": torch.where(
                    projected_rhs_norm > 0.0,
                    torch.ones_like(projected_rhs_norm),
                    torch.zeros_like(projected_rhs_norm),
                ),
                "kernel_gain_scale": torch.ones_like(projected_rhs_norm),
            }

        mol.xlesmd_multipliers = lambda_ij.detach().clone()
        mol.dxi2dt2 = (
            dxi2dt2_flat
            if MO_basis
            else torch.einsum("bmi,bria,bna->brmn", Cocc, dxi2dt2_flat.view(b, r, nocc, nvirt), Cvirt)
        )
        xi = xi_flat.view(b, r, nocc, nvirt)
        xi_AO = torch.einsum("bmi,bria,bna->brmn", Cocc, xi, Cvirt)

        identity = torch.eye(r, dtype=eta_flat.dtype, device=eta_flat.device).expand(b, r, r)
        fixed_point = xi_flat - eta_flat
        projected_fixed_point = project_state_subspace_tangent(eta_flat, fixed_point)
        diagnostics = {
            "constraint_mode": constraint_mode,
            "block_energy": True,
            "fixed_point_residual": torch.linalg.vector_norm(fixed_point, dim=-1),
            "projected_fixed_point_residual": torch.linalg.vector_norm(projected_fixed_point, dim=(1, 2)),
            "vertical_fixed_point_residual": torch.linalg.vector_norm(
                fixed_point - projected_fixed_point, dim=(1, 2)
            ),
            "eta_orthogonality": torch.linalg.matrix_norm(
                eta_flat @ eta_flat.transpose(-1, -2) - identity, dim=(-2, -1)
            ),
            "xi_orthogonality": torch.linalg.matrix_norm(
                xi_flat @ xi_flat.transpose(-1, -2) - identity, dim=(-2, -1)
            ),
            "electronic_residual": solve_info["residual_norm"],
            "electronic_iterations": solve_info["iterations"],
            "electronic_converged": solve_info["converged"],
            "krylov_rank": krylov_info["rank"],
            "krylov_relative_residual": krylov_info["relative_residual"],
            "krylov_converged": krylov_info["converged"],
            "krylov_kernel_gain": krylov_info["kernel_gain"],
            "krylov_kernel_gain_scale": krylov_info["kernel_gain_scale"],
            "krylov_preconditioner": preconditioner_name,
        }
        mol.xlesmd_diagnostics = diagnostics

    # Rebuild G(eta) with autograd enabled.  xi is the exact stationary shadow
    # solution and deliberately remains detached: differentiating this partial
    # block functional gives the conservative shadow force.
    Gx_flat = G_apply(eta_flat.view(b, r, nocc, nvirt)).reshape(b, r, n)
    E1 = (xi_flat * xi_flat * ea_ei_flat).sum(dim=2)
    E2 = ((2.0 * xi_flat - eta_flat) * Gx_flat).sum(dim=2)
    E = E1 + E2

    if params.get("verbose_xlesmd", False):
        print("XL-ESMD constraint mode:", constraint_mode)
        print("Xi overlap matrix:\n", xi_flat @ xi_flat.transpose(-1, -2))
        print("Projected fixed-point residual:", diagnostics["projected_fixed_point_residual"])

    return E, xi_AO, xi_flat


def solve_for_amplitudes_coupled(
    eta: torch.Tensor,  # (b, r, n) == H, rows are states eta_k
    ea_ei: torch.Tensor,  # (b, 1, n) diagonal of A = Delta epsilon
    G: torch.Tensor,  # (b, r, n) == G(eta)
    eps: float = 1e-12,
    regularize_B: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Coupled multi-state solve for xi and Lambda.

    Solves, for each batch item,

        A X - H Lambda = -G

        2 H^T X = H^T H + I

    where

        H = [eta_1, ..., eta_r]
        X = [xi_1,  ..., xi_r]
        G = [G(eta_1), ..., G(eta_r)]

    In tensor layout here, states are stored along dim=1:

        eta[b, k, :] = eta_k
        xi[b,  l, :] = xi_l
        Lambda[b, k, l] = lambda_{k l}

    The solution is

        B = H^T A^{-1} H
        S = H^T H
        C = H^T A^{-1} G

        Lambda = 1/2 B^{-1} (S + I + 2 C)

        X = A^{-1}(H Lambda - G)

    Returns:
        xi:      (b, r, n)
        Lambda:  (b, r, r)
    """

    if torch.any(ea_ei < 1e-14):
        raise RuntimeError(
            "HOMO-LUMO gaps are small values; Kernel calculation for XL-ESMD will be unstable. Check inputs."
        )

    b, r, n = eta.shape

    if G.shape != eta.shape:
        raise ValueError(f"G must have shape {eta.shape}, got {G.shape}")

    if ea_ei.shape[0] != b or ea_ei.shape[-1] != n:
        raise ValueError(
            f"ea_ei must have shape compatible with (b, 1, n) = ({b}, 1, {n}), got {ea_ei.shape}"
        )

    invA = 1.0 / ea_ei  # (b, 1, n), broadcasts over r

    # S_{kl} = eta_k^T eta_l
    S = torch.einsum("bkn,bln->bkl", eta, eta)  # (b, r, r)

    # A^{-1} eta_l
    Ainv_eta = invA * eta  # (b, r, n)

    # B_{kl} = eta_k^T A^{-1} eta_l
    B = torch.einsum("bkn,bln->bkl", eta, Ainv_eta)  # (b, r, r)

    # A^{-1} G_l
    Ainv_G = invA * G  # (b, r, n)

    # C_{kl} = eta_k^T A^{-1} G_l
    C = torch.einsum("bkn,bln->bkl", eta, Ainv_G)  # (b, r, r)

    I = torch.eye(r, dtype=eta.dtype, device=eta.device).expand(b, r, r)

    # RHS = S + I + 2C
    rhs = S + I + 2.0 * C  # (b, r, r)

    if regularize_B:
        B_solve = B + eps * I
    else:
        B_solve = B

    # Lambda = 1/2 B^{-1} rhs
    Lambda = 0.5 * torch.linalg.solve(B_solve, rhs)  # (b, r, r)

    # H Lambda:
    # (H Lambda)_l = sum_k eta_k lambda_{k l}
    H_Lambda = torch.einsum("bkn,bkl->bln", eta, Lambda)  # (b, r, n)

    # X = A^{-1}(H Lambda - G)
    xi = invA * (H_Lambda - G)  # (b, r, n)

    return xi, Lambda


def _row_polar_orthonormalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Nearest row-orthonormal matrix under the Frobenius norm."""
    gram = x @ x.transpose(-1, -2)
    values, vectors = torch.linalg.eigh(gram)
    if torch.any(values <= eps):
        raise RuntimeError("Cannot orthonormalize a rank-deficient state block.")
    inv_sqrt = vectors @ torch.diag_embed(values.rsqrt()) @ vectors.transpose(-1, -2)
    return inv_sqrt @ x


def horizontal_project_state_block(eta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project a row-state block away from its internal rotational gauge.

    For a row-orthonormal reference ``q``, vertical directions have the form
    ``Omega @ q`` with ``Omega`` skew symmetric.  The polar factor of ``eta``
    supplies such a reference even when the propagated auxiliary block is
    only approximately orthonormal.
    """
    if eta.ndim != 3 or v.shape != eta.shape:
        raise ValueError("eta and v must have the same (batch, roots, excitations) shape")
    q = _row_polar_orthonormalize(eta)
    overlap = v @ q.transpose(-1, -2)
    skew = 0.5 * (overlap - overlap.transpose(-1, -2))
    return v - skew @ q


def project_state_subspace_tangent(eta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return the Grassmann tangent component of a row-state block.

    Coupled Cases 2a/2b evolve an unlabelled orthonormal state *subspace*.
    Components within its row space are not physical auxiliary directions:
    skew parts rotate the state labels and symmetric parts leave the Stiefel
    manifold.  The quotient tangent therefore satisfies ``v @ q.T = 0``.
    """
    if eta.ndim != 3 or v.shape != eta.shape:
        raise ValueError("eta and v must have the same (batch, roots, excitations) shape")
    q = _row_polar_orthonormalize(eta)
    return v - (v @ q.transpose(-1, -2)) @ q


def compute_dxi2dt2_coupled_gmres(
    eta: torch.Tensor,
    xi: torch.Tensor,
    jvp_xi: Callable[[torch.Tensor], torch.Tensor],
    xl_params: Dict,
    *,
    eps: float = 1.0e-12,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_info: bool = False,
):
    """Solve the projected coupled XL kernel equation with block GMRES.

    Each molecule is one vector of length ``roots * excitations``.  The solve
    is performed only in the Grassmann state-subspace tangent space,

    ``P_g (I - J_xi) P_g d = P_g (xi - eta)``,

    because raw Cases 2a/2b have rotational gauge directions and radial
    directions excluded by the row-orthonormal auxiliary constraint.
    Arnoldi products and orthogonalization use the Frobenius inner product
    over the complete state block.  When ``preconditioner`` is supplied it is
    used on the right: GMRES constructs ``d = M^-1 y`` and minimizes the true
    residual of ``P_g (I-J_xi) P_g M^-1 y = rhs``.
    """
    if eta.ndim != 3 or xi.shape != eta.shape:
        raise ValueError("eta and xi must have the same (batch, roots, excitations) shape")
    max_rank = int(xl_params["max_rank"])
    tolerance = float(xl_params.get("err_threshold", 1.0e-6))
    b, r, n = eta.shape
    dtype = eta.dtype
    device = eta.device

    if max_rank <= 0:
        info = {
            "rank": 0,
            "relative_residual": torch.ones((b,), dtype=dtype, device=device),
            "converged": torch.zeros((b,), dtype=torch.bool, device=device),
            "vertical_rhs_norm": torch.zeros((b,), dtype=dtype, device=device),
            "excluded_rhs_norm": torch.zeros((b,), dtype=dtype, device=device),
            "kernel_gain": torch.zeros((b,), dtype=dtype, device=device),
            "kernel_gain_scale": torch.ones((b,), dtype=dtype, device=device),
        }
        result = torch.zeros_like(eta)
        return (result, info) if return_info else result

    def project(v: torch.Tensor) -> torch.Tensor:
        return project_state_subspace_tangent(eta, v)

    raw_rhs = xi - eta
    rhs_block = project(raw_rhs)
    excluded_rhs_norm = torch.linalg.vector_norm(raw_rhs - rhs_block, dim=(1, 2))
    rhs = rhs_block.reshape(b, r * n)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    finfo = torch.finfo(dtype)
    breakdown_tolerance = max(float(eps), 100.0 * float(finfo.eps))
    rhs_norm_safe = rhs_norm.clamp_min(breakdown_tolerance)
    live = rhs_norm > breakdown_tolerance

    if not torch.any(live):
        info = {
            "rank": 0,
            "relative_residual": torch.zeros((b,), dtype=dtype, device=device),
            "converged": torch.ones((b,), dtype=torch.bool, device=device),
            "vertical_rhs_norm": excluded_rhs_norm,
            "excluded_rhs_norm": excluded_rhs_norm,
            "kernel_gain": torch.zeros((b,), dtype=dtype, device=device),
            "kernel_gain_scale": torch.ones((b,), dtype=dtype, device=device),
        }
        result = torch.zeros_like(eta)
        return (result, info) if return_info else result

    dimension = r * n
    V = torch.zeros((b, dimension, max_rank + 1), dtype=dtype, device=device)
    Hbar = torch.zeros((b, max_rank + 1, max_rank), dtype=dtype, device=device)
    small_rhs = torch.zeros((b, max_rank + 1), dtype=dtype, device=device)
    small_rhs[:, 0] = rhs_norm
    V[live, :, 0] = rhs[live] / rhs_norm[live, None]

    solution = torch.zeros((b, dimension), dtype=dtype, device=device)
    relative_residual = torch.zeros((b,), dtype=dtype, device=device)
    relative_residual[live] = float("inf")
    rank_used = 0

    for k in range(max_rank):
        if not torch.any(live):
            break
        iter_mask = live.clone()
        vk = V[:, :, k].reshape(b, r, n)
        vk = project(vk)
        zk = project(preconditioner(vk)) if preconditioner is not None else vk
        w = project(zk - jvp_xi(zk)).reshape(b, dimension)
        w[~iter_mask] = 0.0

        for _ in range(2):
            coefficients = torch.einsum("bdi,bd->bi", V[:, :, : k + 1], w)
            Hbar[:, : k + 1, k] += coefficients
            w = w - torch.einsum("bdi,bi->bd", V[:, :, : k + 1], coefficients)

        h_next = torch.linalg.vector_norm(w, dim=1)
        h_next = torch.where(iter_mask, h_next, torch.zeros_like(h_next))
        Hbar[:, k + 1, k] = h_next
        can_expand = h_next > breakdown_tolerance
        V[can_expand, :, k + 1] = w[can_expand] / h_next[can_expand, None]
        rank_used = k + 1

        Hm = Hbar[:, : rank_used + 1, :rank_used]
        gm = small_rhs[:, : rank_used + 1]
        y = torch.bmm(torch.linalg.pinv(Hm, rtol=100.0 * float(finfo.eps)), gm.unsqueeze(-1)).squeeze(-1)
        solution = torch.einsum("bdi,bi->bd", V[:, :, :rank_used], y)
        small_residual = gm - torch.bmm(Hm, y.unsqueeze(-1)).squeeze(-1)
        relative_residual = torch.linalg.vector_norm(small_residual, dim=1) / rhs_norm_safe
        relative_residual = torch.where(
            rhs_norm > breakdown_tolerance, relative_residual, torch.zeros_like(relative_residual)
        )
        converged = relative_residual <= tolerance
        live = iter_mask & (~converged) & can_expand

    result = solution.reshape(b, r, n)
    result = project(preconditioner(result)) if preconditioner is not None else project(result)

    # The projected operator can still contain very soft physical directions.
    # A converged inverse action along one of them may be much larger than the
    # fixed-point residual and destabilize finite-step XL propagation.  This
    # optional trust-region regularization bounds that gain without changing
    # the default dense/GMRES mathematical result used by validation tests.
    correction_norm = torch.linalg.vector_norm(result, dim=(1, 2))
    gain = correction_norm / rhs_norm.clamp_min(breakdown_tolerance)
    gain_scale = torch.ones_like(gain)
    max_amplification = xl_params.get("kernel_max_amplification")
    if max_amplification is not None:
        max_amplification = float(max_amplification)
        if max_amplification <= 0.0:
            raise ValueError("kernel_max_amplification must be positive when supplied.")
        gain_scale = torch.clamp(max_amplification / gain.clamp_min(breakdown_tolerance), max=1.0)
        result = result * gain_scale[:, None, None]

    info = {
        "rank": rank_used,
        "relative_residual": relative_residual,
        "converged": relative_residual <= tolerance,
        # Historical name retained for existing diagnostic consumers.  In the
        # subspace formulation this includes rotations and radial components.
        "vertical_rhs_norm": excluded_rhs_norm,
        "excluded_rhs_norm": excluded_rhs_norm,
        "kernel_gain": gain,
        "kernel_gain_scale": gain_scale,
    }
    if xl_params.get("verbose_krylov", False):
        print(
            f"Coupled projected GMRES rank used: {rank_used}, "
            f"max relative residual: {torch.max(relative_residual).item():.3e}"
        )
    return (result, info) if return_info else result


def _solve_exact_orthonormal_kkt(
    D: torch.Tensor,
    xi: torch.Tensor,
    Lambda: torch.Tensor,
    rhs_stationarity: torch.Tensor,
    rhs_constraint: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve the dense bordered tangent/KKT system used by case 2b."""
    b, r, n = xi.shape
    gamma_rows, gamma_cols = torch.triu_indices(r, r, device=xi.device)
    ngamma = gamma_rows.numel()
    gamma_basis = torch.zeros((ngamma, r, r), dtype=xi.dtype, device=xi.device)
    gamma_basis[torch.arange(ngamma, device=xi.device), gamma_rows, gamma_cols] = 1.0
    off_diagonal = gamma_rows != gamma_cols
    gamma_basis[
        torch.arange(ngamma, device=xi.device)[off_diagonal],
        gamma_cols[off_diagonal],
        gamma_rows[off_diagonal],
    ] = 1.0
    constraint_rows = [(i, j) for i in range(r) for j in range(i, r)]
    dimension = r * n + ngamma
    result = torch.empty_like(xi)

    if rhs_constraint is None:
        rhs_constraint = torch.zeros((b, r, r), dtype=xi.dtype, device=xi.device)

    for batch in range(b):
        matrix = torch.zeros((dimension, dimension), dtype=xi.dtype, device=xi.device)
        rhs = torch.zeros((dimension,), dtype=xi.dtype, device=xi.device)
        for i in range(r):
            for p in range(n):
                row = i * n + p
                for j in range(r):
                    matrix[row, j * n + p] = (D[batch, i, p] if i == j else 0.0) - Lambda[batch, i, j]
                matrix[row, r * n :] = -torch.einsum("gj,j->g", gamma_basis[:, i, :], xi[batch, :, p])
                rhs[row] = rhs_stationarity[batch, i, p]

        for offset, (i, j) in enumerate(constraint_rows):
            row = r * n + offset
            matrix[row, i * n : (i + 1) * n] += xi[batch, j]
            matrix[row, j * n : (j + 1) * n] += xi[batch, i]
            rhs[row] = rhs_constraint[batch, i, j]

        solution = torch.linalg.solve(matrix, rhs)
        result[batch] = solution[: r * n].reshape(r, n)
    return result


def solve_for_amplitudes_exact_orthonormal(
    eta: torch.Tensor,
    ea_ei: torch.Tensor,
    G: torch.Tensor,
    *,
    initial_xi: Optional[torch.Tensor] = None,
    tolerance: float = 1e-10,
    max_iter: int = 2000,
    initial_step: float = 1.0,
    min_step: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Solve case 2b on the row-Stiefel manifold.

    This comparison/reference implementation follows the stationary branch of
    the total block shadow functional with exact row orthonormality.  A branch
    can be a saddle, so the safeguarded Newton iteration reduces the KKT
    residual rather than assuming that every step lowers the objective.  The
    convergence diagnostics are returned because 2b is substantially more
    expensive and less robust than the closed-form linearized solvers.
    """
    if eta.ndim != 3 or G.shape != eta.shape:
        raise ValueError("eta and G must have shape (batch, roots, excitations)")
    b, r, n = eta.shape
    if ea_ei.shape[0] != b or ea_ei.shape[-1] != n:
        raise ValueError("ea_ei must have shape compatible with (batch, 1, excitations)")
    if r > n:
        raise ValueError("The number of states cannot exceed the excitation dimension.")

    x0 = eta if initial_xi is None else initial_xi
    if x0.shape != eta.shape:
        raise ValueError(f"initial_xi must have shape {eta.shape}")
    xi = _row_polar_orthonormalize(x0)
    D = ea_ei.expand(b, r, n)

    converged = torch.zeros(b, dtype=torch.bool, device=eta.device)
    residual_norm = torch.full((b,), float("inf"), dtype=eta.dtype, device=eta.device)
    iterations = 0

    for iteration in range(1, max_iter + 1):
        kkt_left = D * xi + G
        Lambda = 0.5 * (kkt_left @ xi.transpose(-1, -2) + xi @ kkt_left.transpose(-1, -2))
        residual = kkt_left - Lambda @ xi
        residual_norm = torch.linalg.vector_norm(residual, dim=(1, 2))
        converged = residual_norm <= tolerance
        iterations = iteration
        if torch.all(converged):
            break

        try:
            direction = _solve_exact_orthonormal_kkt(D, xi, Lambda, -residual)
        except torch.linalg.LinAlgError:
            direction = -residual
        step = torch.full_like(residual_norm, initial_step)
        accepted = converged.clone()
        trial = xi.clone()

        while torch.any((~accepted) & (step >= min_step)):
            candidate = _row_polar_orthonormalize(xi + step[:, None, None] * direction)
            candidate_left = D * candidate + G
            candidate_lambda = 0.5 * (
                candidate_left @ candidate.transpose(-1, -2) + candidate @ candidate_left.transpose(-1, -2)
            )
            candidate_residual = candidate_left - candidate_lambda @ candidate
            candidate_norm = torch.linalg.vector_norm(candidate_residual, dim=(1, 2))
            sufficient = candidate_norm < residual_norm
            newly_accepted = sufficient & (~accepted)
            trial = torch.where(newly_accepted[:, None, None], candidate, trial)
            accepted |= sufficient
            step = torch.where(accepted, step, 0.5 * step)

        if not torch.all(accepted):
            break
        xi = trial

    kkt_left = D * xi + G
    Lambda = 0.5 * (kkt_left @ xi.transpose(-1, -2) + xi @ kkt_left.transpose(-1, -2))
    residual = kkt_left - Lambda @ xi
    residual_norm = torch.linalg.vector_norm(residual, dim=(1, 2))
    converged = residual_norm <= tolerance
    info = {
        "converged": bool(torch.all(converged)),
        "converged_batch": converged,
        "iterations": iterations,
        "residual_norm": residual_norm,
    }
    return xi, Lambda, info


def make_jvp_xi_exact_orthonormal(
    ea_ei: torch.Tensor,
    xi: torch.Tensor,
    Lambda: torch.Tensor,
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Dense bordered-response JVP for the exact coupled case 2b.

    The bordered solve is suitable as a validation oracle.  Its cubic ambient
    cost is also a concrete reason not to select 2b for production XL dynamics.
    """
    b, r, n = xi.shape
    if n != nocc * nvirt:
        raise ValueError("The excitation dimension must equal nocc*nvirt.")
    if Lambda.shape != (b, r, r):
        raise ValueError(f"Lambda must have shape {(b, r, r)}")
    D = ea_ei.expand(b, r, n)

    def jvp(v: torch.Tensor) -> torch.Tensor:
        if v.shape != xi.shape:
            raise ValueError(f"Expected v shape {xi.shape}, got {v.shape}")
        Gv = G_apply(v.reshape(b, r, nocc, nvirt)).reshape(b, r, n)
        return _solve_exact_orthonormal_kkt(D, xi, Lambda, -Gv)

    return jvp


# ------------------------------------------------------------
# Null preconditioner for coupled state vectors
# ------------------------------------------------------------
def make_apply_precond_null_coupled() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Identity preconditioner.

    Input/output:
        v: (b, r, n)
    """

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        return v

    return apply_prec

# ------------------------------------------------------------
# Coupled Lambda preconditioner
# ------------------------------------------------------------
def make_apply_precond_lambda_coupled(
    ea_ei_flat: torch.Tensor,  # (b, 1, n), diagonal of A = Delta epsilon
    Lambda: torch.Tensor,  # (b, r, r), Lambda[k,l] = lambda_{kl}
    tau: float = 1e-5,
    eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Coupled cheap approximate inverse for

        J0[dH] = A^{-1} dH Lambda - dH.

    With damping, we use

        J0_tau[dH] = A^{-1} dH Lambda - (1 - tau) dH.

    For each excitation index p with a_p = Delta epsilon_p,

        y_p [Lambda / a_p - (1 - tau) I] = v_p

    equivalently

        y_p [Lambda - (1 - tau) a_p I] = a_p v_p.

    This includes interstate coupling through Lambda.

    Input/output:
        v: (b, r, n)

    Returns:
        y: (b, r, n)
    """

    b, r, _ = Lambda.shape
    n = ea_ei_flat.shape[-1]

    if ea_ei_flat.shape[0] != b:
        raise ValueError(f"Batch mismatch: ea_ei_flat has batch {ea_ei_flat.shape[0]}, Lambda has batch {b}")

    if ea_ei_flat.shape[-1] != n:
        raise ValueError("Invalid ea_ei_flat shape.")

    I = torch.eye(r, dtype=Lambda.dtype, device=Lambda.device)  # (r, r)

    # a: (b, n)
    a = ea_ei_flat.squeeze(1)

    # M[b,p,:,:] = Lambda[b,:,:] - (1 - tau) a[b,p] I
    M = Lambda[:, None, :, :] - (1.0 - tau) * a[:, :, None, None] * I[None, None, :, :]  # (b, n, r, r)

    # Optional tiny diagonal safety shift.
    M = M + eps * I[None, None, :, :]

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        if v.shape != (b, r, n):
            raise ValueError(f"Expected v shape {(b, r, n)}, got {v.shape}")

        # Work row-by-row in excitation index:
        # v_perm[b,p,k] = v[b,k,p]
        v_perm = v.permute(0, 2, 1)  # (b, n, r)

        rhs = a[:, :, None] * v_perm  # (b, n, r)

        # Need y_p M_p = rhs_p.
        # torch.linalg.solve solves M_p^T y_p^T = rhs_p^T.
        y_perm = torch.linalg.solve(M.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)  # (b, n, r)

        y = y_perm.permute(0, 2, 1)  # (b, r, n)

        return y

    return apply_prec


def make_jvp_xi_coupled(
    ea_ei: torch.Tensor,  # (b, 1, n)
    eta: torch.Tensor,  # (b, r, n)
    xi: torch.Tensor,  # (b, r, n)
    Lambda: torch.Tensor,  # (b, r, r)
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
    eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Exact JVP for the coupled map

        eta -> xi(eta)

    assuming G is linear:

        dG = G[v] = Gamma v.

    Shapes:
        eta:    (b, r, n)
        xi:     (b, r, n)
        Lambda: (b, r, r)
        v:      (b, r, n)

    Returns:
        jvp_xi(v): (b, r, n)
    """

    b, r, n = eta.shape
    i, a = nocc, nvirt

    if n != i * a:
        raise ValueError(f"n must equal nocc*nvirt. Got n={n}, nocc*nvirt={i * a}.")

    if xi.shape != eta.shape:
        raise ValueError(f"xi must have shape {eta.shape}, got {xi.shape}")

    if Lambda.shape != (b, r, r):
        raise ValueError(f"Lambda must have shape {(b, r, r)}, got {Lambda.shape}")

    invA = 1.0 / ea_ei  # (b, 1, n)

    # A^{-1} eta_l
    Ainv_eta = invA * eta  # (b, r, n)

    # B_{kl} = eta_k^T A^{-1} eta_l
    Bmat = torch.einsum("bkn,bln->bkl", eta, Ainv_eta)  # (b, r, r)

    # Small regularization in case Bmat is nearly singular
    I = torch.eye(r, dtype=eta.dtype, device=eta.device).expand(b, r, r)
    Bmat_solve = Bmat + eps * I

    def jvp_xi(v: torch.Tensor) -> torch.Tensor:
        if v.shape != eta.shape:
            raise ValueError(f"Expected v shape {eta.shape}, got {v.shape}")

        # G[v]
        v_bria = v.reshape(b, r, i, a)
        Gv_bria = G_apply(v_bria)
        Gv = Gv_bria.reshape(b, r, n)

        # b_vec_l = sum_k v_k Lambda_{kl} - G[v]_l
        #
        # Shape:
        #   v:      (b, k, n)
        #   Lambda: (b, k, l)
        #   b_vec:  (b, l, n)
        b_vec = torch.einsum("bkn,bkl->bln", v, Lambda) - Gv  # (b, r, n)

        # lower_{kl} = eta_k^T v_l + v_k^T (eta_l - 2 xi_l)
        lower = torch.einsum("bkn,bln->bkl", eta, v) + torch.einsum(
            "bkn,bln->bkl", v, eta - 2.0 * xi
        )  # (b, r, r)

        # D_{kl} = eta_k^T A^{-1} b_vec_l
        Ainv_b = invA * b_vec  # (b, r, n)
        Dmat = torch.einsum("bkn,bln->bkl", eta, Ainv_b)  # (b, r, r)

        # 2 B deltaLambda = lower - 2 D
        rhs = lower - 2.0 * Dmat  # (b, r, r)

        delta_Lambda = 0.5 * torch.linalg.solve(Bmat_solve, rhs)  # (b, r, r)

        # H deltaLambda
        H_delta_Lambda = torch.einsum("bkn,bkl->bln", eta, delta_Lambda)  # (b, r, n)

        # deltaX = A^{-1}(b_vec + H deltaLambda)
        delta_xi = invA * (b_vec + H_delta_Lambda)  # (b, r, n)

        return delta_xi

    return jvp_xi


def compute_dxi2dt2_rankm_coupled(
    eta_brn: torch.Tensor,  # (b, r, n)
    xi_brn: torch.Tensor,  # (b, r, n)
    jvp_xi: Callable[[torch.Tensor], torch.Tensor],
    xl_params: Dict,
    precond: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-12,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Coupled rank-m Krylov approximation.

    This solves approximately, in a Krylov subspace,

        (J_xi - I) delta = xi - eta

    and returns

        dxi2dt2 = -delta.

    Important:
        Unlike the old state-by-state version, this treats each batch item
        as one coupled vector of length r*n.

    Shapes:
        eta_brn: (b, r, n)
        xi_brn:  (b, r, n)

    Returns:
        dxi2dt2_brn: (b, r, n)
    """

    Rank = int(xl_params["max_rank"])
    err_threshold = float(xl_params["err_threshold"])

    b, r, n = eta_brn.shape
    rn = r * n

    # Initial residual R = xi - eta
    R_brn = xi_brn - eta_brn  # (b, r, n)

    # Preconditioned residual
    dDS_brn = precond(R_brn)  # (b, r, n)
    dDS = dDS_brn.reshape(b, rn)  # (b, r*n)

    dDS_norm = torch.linalg.vector_norm(dDS, dim=1).clamp(min=eps)

    V = torch.zeros((b, rn, Rank), dtype=xi_brn.dtype, device=xi_brn.device)
    W = torch.zeros((b, rn, Rank), dtype=xi_brn.dtype, device=xi_brn.device)

    dW = dDS.clone()
    Error = torch.full((b,), 10.0, dtype=xi_brn.dtype, device=xi_brn.device)

    Rank_m = 0
    last_alpha: Optional[torch.Tensor] = None

    def vecnorm(u: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(u, dim=1)

    for k in range(Rank):
        if torch.max(Error) <= err_threshold:
            break

        vk = dW.clone()  # (b, r*n)

        # Modified Gram-Schmidt, two passes
        if k > 0:
            Vprev = V[:, :, :k]  # (b, r*n, k)
            for _ in range(2):
                coeffs = torch.einsum("Bnk,Bn->Bk", Vprev, vk)
                proj = torch.einsum("Bnk,Bk->Bn", Vprev, coeffs)
                vk = vk - proj

        vknorm = vecnorm(vk)

        if torch.all(vknorm <= eps):
            break

        vk = vk / vknorm.clamp(min=eps).unsqueeze(-1)
        V[:, :, k] = vk

        # Convert Krylov vector back to coupled state tensor
        vk_brn = vk.reshape(b, r, n)

        # Exact JVP: delta xi
        dxi_brn = jvp_xi(vk_brn)  # (b, r, n)

        # Jacobian of residual map: J_F[v] = J_xi[v] - v
        wk_raw_brn = dxi_brn - vk_brn  # (b, r, n)

        # Apply preconditioner
        wk_brn = precond(wk_raw_brn)  # (b, r, n)

        wk = wk_brn.reshape(b, rn)  # (b, r*n)

        W[:, :, k] = wk
        dW = wk
        Rank_m = k + 1

        # Least-squares fit:
        #
        #   W_k alpha ~= dDS
        #
        Wk = W[:, :, :Rank_m]  # (b, r*n, m)

        O = torch.einsum("Bnm,Bnl->Bml", Wk, Wk)  # (b, m, m)
        rhs = torch.einsum("Bnm,Bn->Bm", Wk, dDS)  # (b, m)

        # Small diagonal regularization for numerical safety
        m = Rank_m
        Im = torch.eye(m, dtype=xi_brn.dtype, device=xi_brn.device).expand(b, m, m)
        O = O + eps * Im

        alpha = torch.linalg.solve(O, rhs.unsqueeze(-1)).squeeze(-1)  # (b, m)
        last_alpha = alpha

        IdentRes = torch.einsum("Bnm,Bm->Bn", Wk, alpha)  # (b, r*n)

        Error = vecnorm(IdentRes - dDS) / dDS_norm

    if verbose:
        print(
            f"Coupled Krylov rank used: {Rank_m}, "
            f"final max relative error in dDS fit: {torch.max(Error).item():.2e}"
        )

    if Rank_m == 0:
        if verbose:
            print("Rank-m loop did not run; returning preconditioned residual fallback.")
        return dDS_brn

    Vk = V[:, :, :Rank_m]  # (b, r*n, m)
    alpha = last_alpha

    # delta = Vk alpha
    delta = torch.einsum("Bnm,Bm->Bn", Vk, alpha)  # (b, r*n)

    # Same sign convention as your old code:
    # dxi2dt2 = -delta
    dxi2dt2 = -delta.reshape(b, r, n)

    return dxi2dt2


def make_apply_precond_constraint_lowrank_coupled(
    ea_ei_flat: torch.Tensor,  # (b, 1, n)
    eta: torch.Tensor,  # (b, r, n) == H
    xi: torch.Tensor,  # (b, r, n) == X
    Lambda: torch.Tensor,  # (b, r, r)
    tau: float = 1e-5,
    eps: float = 1e-12,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Low-rank / block-rank constraint preconditioner for the coupled multi-state problem.

    Approximates the inverse of the kernel-free Jacobian

        J_c[V] = A^{-1} V Lambda + A^{-1} H deltaLambda_0[V] - V,

    where

        deltaLambda_0[V]
        =
        1/2 B^{-1}
        [
            H^T V
            + V^T(H - 2X)
            - 2 H^T A^{-1} V Lambda
        ]

    and

        B = H^T A^{-1} H.

    This uses Woodbury around the simpler base operator

        J_0[V] = A^{-1} V Lambda - V.

    Input/output:
        v: (b, r, n)

    Returns:
        approximately J_c^{-1}[v], shape (b, r, n)
    """

    b, r, n = eta.shape

    if xi.shape != eta.shape:
        raise ValueError(f"xi must have shape {eta.shape}, got {xi.shape}")

    if Lambda.shape != (b, r, r):
        raise ValueError(f"Lambda must have shape {(b, r, r)}, got {Lambda.shape}")

    if ea_ei_flat.shape[0] != b or ea_ei_flat.shape[-1] != n:
        raise ValueError(f"ea_ei_flat must be compatible with {(b, 1, n)}, got {ea_ei_flat.shape}")

    dtype = eta.dtype
    device = eta.device

    invA = 1.0 / ea_ei_flat  # (b, 1, n)
    a_diag = ea_ei_flat.squeeze(1)  # (b, n)

    I_r = torch.eye(r, dtype=dtype, device=device)
    I_brr = I_r.expand(b, r, r)

    # Y = A^{-1} H
    Y = invA * eta  # (b, r, n)

    # B = H^T A^{-1} H
    Bmat = torch.einsum("bkn,bln->bkl", eta, Y)  # (b, r, r)
    Bmat = Bmat + eps * I_brr

    # ------------------------------------------------------------------
    # Base inverse J_0^{-1}
    #
    # J_0[V] = A^{-1} V Lambda - V.
    #
    # For each excitation p:
    #
    #   y_p [Lambda - (1 - tau) a_p I] = a_p v_p
    #
    # ------------------------------------------------------------------

    M = Lambda[:, None, :, :] - (1.0 - tau) * a_diag[:, :, None, None] * I_r[None, None, :, :]  # (b, n, r, r)

    M = M + eps * I_r[None, None, :, :]

    def apply_J0_inv(v: torch.Tensor) -> torch.Tensor:
        if v.shape != (b, r, n):
            raise ValueError(f"Expected v shape {(b, r, n)}, got {v.shape}")

        v_perm = v.permute(0, 2, 1)  # (b, n, r)
        rhs = a_diag[:, :, None] * v_perm  # (b, n, r)

        y_perm = torch.linalg.solve(M.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)  # (b, n, r)

        return y_perm.permute(0, 2, 1)  # (b, r, n)

    # ------------------------------------------------------------------
    # C[V] = deltaLambda_0[V], with G[V] dropped
    # ------------------------------------------------------------------

    def apply_C(v: torch.Tensor) -> torch.Tensor:
        """
        C[v] = deltaLambda_0[v].

        Input:
            v: (b, r, n)

        Output:
            deltaLambda_0: (b, r, r)
        """

        # q = V Lambda
        q = torch.einsum("bkn,bkl->bln", v, Lambda)  # (b, r, n)

        # H^T V
        HtV = torch.einsum("bkn,bln->bkl", eta, v)  # (b, r, r)

        # V^T (H - 2X)
        Vt_Hm2X = torch.einsum("bkn,bln->bkl", v, eta - 2.0 * xi)  # (b, r, r)

        # H^T A^{-1} q
        Ainv_q = invA * q
        Ht_Ainv_q = torch.einsum("bkn,bln->bkl", eta, Ainv_q)  # (b, r, r)

        rhs = HtV + Vt_Hm2X - 2.0 * Ht_Ainv_q  # (b, r, r)

        deltaLambda0 = 0.5 * torch.linalg.solve(Bmat, rhs)  # (b, r, r)

        return deltaLambda0

    # ------------------------------------------------------------------
    # U[M] = A^{-1} H M = Y M
    # ------------------------------------------------------------------

    def apply_U(mat: torch.Tensor) -> torch.Tensor:
        """
        Input:
            mat: (b, r, r)

        Output:
            Y mat: (b, r, n)
        """

        return torch.einsum("bkn,bkl->bln", Y, mat)

    # ------------------------------------------------------------------
    # Build the small Woodbury matrix:
    #
    #   K = I + C J0^{-1} U
    #
    # K has shape (b, r*r, r*r).
    # ------------------------------------------------------------------

    rr = r * r

    basis = torch.eye(rr, dtype=dtype, device=device).reshape(rr, r, r)
    basis = basis.unsqueeze(0).expand(b, rr, r, r)  # (b, rr, r, r)

    Kcols = []

    for j in range(rr):
        E_j = basis[:, j, :, :]  # (b, r, r)
        U_Ej = apply_U(E_j)  # (b, r, n)
        J0inv_U_Ej = apply_J0_inv(U_Ej)  # (b, r, n)
        C_J0inv_U_Ej = apply_C(J0inv_U_Ej)  # (b, r, r)
        Kcols.append(C_J0inv_U_Ej.reshape(b, rr))

    K = torch.stack(Kcols, dim=2)  # (b, rr, rr)

    I_rr = torch.eye(rr, dtype=dtype, device=device).expand(b, rr, rr)
    K = K + I_rr + eps * I_rr

    # ------------------------------------------------------------------
    # Final Woodbury inverse:
    #
    #   J_c^{-1} v
    #   =
    #   z - J0^{-1} U K^{-1} C z
    #
    # where z = J0^{-1} v.
    # ------------------------------------------------------------------

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        if v.shape != (b, r, n):
            raise ValueError(f"Expected v shape {(b, r, n)}, got {v.shape}")

        # z = J0^{-1} v
        z = apply_J0_inv(v)  # (b, r, n)

        # c = C z
        c = apply_C(z).reshape(b, rr)  # (b, r*r)

        # beta = K^{-1} c
        beta = torch.linalg.solve(K, c.unsqueeze(-1)).squeeze(-1)  # (b, r*r)
        beta_mat = beta.reshape(b, r, r)  # (b, r, r)

        # correction = J0^{-1} U beta
        U_beta = apply_U(beta_mat)  # (b, r, n)
        correction = apply_J0_inv(U_beta)  # (b, r, n)

        return z - correction

    return apply_prec
