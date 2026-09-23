import math
from typing import Callable, Dict, Optional, Tuple

import torch

from .excited_state_utils import get_occ_virt
from .hcore import (
    orthogonalized_overlap_from_matrices,
    overlap_between_geometries,
    overlap_matrix_current_geometry,
)
from .rcis_batch import _uniform_molecule_dimensions, makeA_pi_batched, packone_batch


def _hermitize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def augmented_cis_residual(
    eta: torch.Tensor, multiplier: torch.Tensor, hamiltonian_apply: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual of the full-Hermitian augmented XL-CIS eigenproblem.

    State vectors are rows of ``eta`` with shape ``(batch, roots, nov)`` and
    ``multiplier`` is the propagated Hermitian ``(batch, roots, roots)``
    matrix.  This is intentionally independent from all legacy XL modes.
    """
    if eta.ndim != 3 or multiplier.shape != eta.shape[:2] + (eta.shape[1],):
        raise ValueError("Expected eta=(batch, roots, nov), multiplier=(batch, roots, roots).")
    h_eta = hamiltonian_apply(eta)
    if h_eta.shape != eta.shape:
        raise ValueError("hamiltonian_apply must preserve eta's shape.")
    identity = torch.eye(eta.shape[1], dtype=eta.dtype, device=eta.device).unsqueeze(0)
    # The minus sign and factor 1/2 make the bordered Newton/KKT operator
    # self-adjoint when the multiplier block is equipped with its Frobenius
    # product.  This is deliberately distinct from the legacy XL residuals.
    return h_eta - multiplier @ eta, 0.5 * (identity - _hermitize(eta @ eta.conj().transpose(-1, -2)))


def make_augmented_cis_jvp(
    eta: torch.Tensor, multiplier: torch.Tensor, hamiltonian_apply: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    """Analytic JVP of the full-multiplier augmented CIS residual."""
    if multiplier.shape != eta.shape[:2] + (eta.shape[1],):
        raise ValueError("Augmented CIS multiplier has incompatible shape.")

    def jvp(direction: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_eta, delta_multiplier = direction
        if delta_eta.shape != eta.shape or delta_multiplier.shape != multiplier.shape:
            raise ValueError("Augmented CIS JVP direction has incompatible shapes.")
        delta_f_eta = hamiltonian_apply(delta_eta) - multiplier @ delta_eta - delta_multiplier @ eta
        delta_f_s = -0.5 * (
            delta_eta @ eta.conj().transpose(-1, -2) + eta @ delta_eta.conj().transpose(-1, -2)
        )
        # Do not symmetrize the JVP: for Hermitian directions its second block
        # is already Hermitian, and retaining the exact formula is essential
        # for the self-adjoint KKT identity.
        return delta_f_eta, delta_f_s

    return jvp


def make_augmented_cis_gauge_projector(
    eta: torch.Tensor, multiplier: torch.Tensor
) -> Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
    """Frozen Frobenius-orthogonal projector off root-rotation gauge modes.

    For every antisymmetric elementary matrix ``A_ij`` this explicitly builds
    ``g_ij = (A_ij eta, A_ij Lambda - Lambda A_ij)`` and applies
    ``I - G (G.T G)^+ G.T``.  Keeping the projector frozen for one electronic
    solve is what makes ``P J P`` symmetric.
    """
    b, r, _ = eta.shape
    if multiplier.shape != (b, r, r):
        raise ValueError("Augmented CIS gauge projector received incompatible Lambda.")
    gauge_eta = []
    gauge_lambda = []
    for i in range(1, r):
        for j in range(i):
            generator = torch.zeros((r, r), dtype=eta.dtype, device=eta.device)
            generator[i, j] = 1.0
            generator[j, i] = -1.0
            generator = generator.unsqueeze(0)
            gauge_eta.append(generator @ eta)
            gauge_lambda.append(generator @ multiplier - multiplier @ generator)
    if not gauge_eta:

        def identity_project(
            direction: Tuple[torch.Tensor, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return direction[0], _hermitize(direction[1])

        return identity_project

    g_eta = torch.stack(gauge_eta, dim=1)
    g_lambda = torch.stack(gauge_lambda, dim=1)
    gram = torch.einsum("Bqri,Bpri->Bqp", g_eta.conj(), g_eta) + torch.einsum(
        "Bqij,Bpij->Bqp", g_lambda.conj(), g_lambda
    )
    gram_inv = torch.linalg.pinv(_hermitize(gram), rtol=100.0 * torch.finfo(eta.real.dtype).eps)

    def project(direction: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_eta, delta_multiplier = direction
        overlap = torch.einsum("Bqri,Bri->Bq", g_eta.conj(), delta_eta) + torch.einsum(
            "Bqij,Bij->Bq", g_lambda.conj(), delta_multiplier
        )
        coefficients = torch.bmm(gram_inv, overlap.unsqueeze(-1)).squeeze(-1)
        return (
            delta_eta - torch.einsum("Bq,Bqri->Bri", coefficients, g_eta),
            _hermitize(delta_multiplier - torch.einsum("Bq,Bqij->Bij", coefficients, g_lambda)),
        )

    return project


def augmented_svec(matrix: torch.Tensor) -> torch.Tensor:
    """Pack a real symmetric matrix in the Frobenius-orthonormal svec basis."""
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("svec requires square matrices.")
    r = matrix.shape[-1]
    rows, columns = torch.tril_indices(r, r, offset=-1, device=matrix.device)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    # sqrt(2), rather than 2 sqrt(2), is the orthonormal convention:
    # <svec(A), svec(B)> = tr(A.T B) for symmetric A and B.
    off_diagonal = math.sqrt(2.0) * matrix[..., rows, columns]
    return torch.cat((diagonal, off_diagonal), dim=-1)


def augmented_smat(vector: torch.Tensor, roots: int) -> torch.Tensor:
    """Inverse of :func:`augmented_svec` for real symmetric matrices."""
    size = roots * (roots + 1) // 2
    if vector.shape[-1] != size:
        raise ValueError("svec length is incompatible with the requested root count.")
    result = torch.zeros(vector.shape[:-1] + (roots, roots), dtype=vector.dtype, device=vector.device)
    diagonal = torch.arange(roots, device=vector.device)
    rows, columns = torch.tril_indices(roots, roots, offset=-1, device=vector.device)
    result[..., diagonal, diagonal] = vector[..., :roots]
    off_diagonal = vector[..., roots:] / math.sqrt(2.0)
    result[..., rows, columns] = off_diagonal
    result[..., columns, rows] = off_diagonal
    return result


def make_apply_precond_augmented_cis_symmetric(
    ea_ei_flat: torch.Tensor,
    eta: torch.Tensor,
    multiplier: torch.Tensor,
    tau: float = 1.0e-5,
    schur_relative_floor: float = 1.0e-10,
    use_schur: bool = True,
) -> Tuple[
    Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]
]:
    """Return the SPD block ``M^{-1/2}`` for the augmented KKT MINRES solve."""
    b, r, n = eta.shape
    if ea_ei_flat.shape != (b, 1, n) or multiplier.shape != (b, r, r):
        raise ValueError("Incompatible full-multiplier augmented CIS preconditioner inputs.")
    eigenvalues, eigenvectors = torch.linalg.eigh(_hermitize(multiplier))
    weights = torch.sqrt((ea_ei_flat - eigenvalues.unsqueeze(-1)).square() + float(tau) ** 2)

    def amplitude_inverse(value: torch.Tensor) -> torch.Tensor:
        rotated = eigenvectors.conj().transpose(-1, -2) @ value
        return eigenvectors @ (rotated / weights)

    def amplitude_inverse_half(value: torch.Tensor) -> torch.Tensor:
        rotated = eigenvectors.conj().transpose(-1, -2) @ value
        return eigenvectors @ (rotated / torch.sqrt(weights))

    sdim = r * (r + 1) // 2
    if use_schur:
        columns = []
        for column in range(sdim):
            basis_vector = torch.zeros((b, sdim), dtype=eta.dtype, device=eta.device)
            basis_vector[:, column] = 1.0
            basis = augmented_smat(basis_vector, r)
            response = _hermitize(amplitude_inverse(basis @ eta) @ eta.conj().transpose(-1, -2))
            columns.append(augmented_svec(response))
        schur = torch.stack(columns, dim=-1)
        schur = _hermitize(schur)
        schur_values, schur_vectors = torch.linalg.eigh(schur)
        scale = schur_values.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(eta.real.dtype).eps)
        floor = torch.maximum(
            scale * float(schur_relative_floor), torch.full_like(scale, torch.finfo(eta.real.dtype).eps)
        )
        schur_values_safe = torch.maximum(schur_values, floor)
    else:
        schur_values_safe = torch.ones((b, sdim), dtype=eta.dtype, device=eta.device)
        schur_vectors = torch.eye(sdim, dtype=eta.dtype, device=eta.device).unsqueeze(0).expand(b, -1, -1)

    def multiplier_inverse_half(value: torch.Tensor) -> torch.Tensor:
        packed = augmented_svec(_hermitize(value))
        rotated = torch.bmm(schur_vectors.conj().transpose(-1, -2), packed.unsqueeze(-1)).squeeze(-1)
        result = torch.bmm(schur_vectors, (rotated / torch.sqrt(schur_values_safe)).unsqueeze(-1)).squeeze(-1)
        return augmented_smat(result, r)

    def apply_inverse_half(direction: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return amplitude_inverse_half(direction[0]), _hermitize(multiplier_inverse_half(direction[1]))

    return apply_inverse_half, {
        "lambda_schur_eigenvalue": schur_values_safe,
        "lambda_schur_enabled": torch.as_tensor(use_schur, dtype=torch.bool, device=eta.device),
    }


def make_augmented_cis_two_sided_operator(
    eta: torch.Tensor,
    multiplier: torch.Tensor,
    hamiltonian_apply: Callable[[torch.Tensor], torch.Tensor],
    xl_params: Dict,
    *,
    ea_ei_flat: Optional[torch.Tensor] = None,
    preconditioner: Optional[
        Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
) -> Tuple[
    Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor],
    Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
    Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
    Dict[str, torch.Tensor],
]:
    """Build the frozen ``M^{-1/2} P J P M^{-1/2}`` MINRES system.

    The returned operator and right hand side use the natural amplitude plus
    Frobenius product for the symmetric multiplier block.  This small helper
    is deliberately public so the symmetry of the *actual* Krylov operator
    can be tested without duplicating the solver construction.
    """
    f_eta, f_s = augmented_cis_residual(eta, multiplier, hamiltonian_apply)
    jvp = make_augmented_cis_jvp(eta, multiplier, hamiltonian_apply)
    project = make_augmented_cis_gauge_projector(eta, multiplier)
    preconditioner_info: Dict[str, torch.Tensor] = {}
    if preconditioner is None and ea_ei_flat is not None:
        preconditioner, preconditioner_info = make_apply_precond_augmented_cis_symmetric(
            ea_ei_flat,
            eta,
            multiplier,
            tau=float(xl_params.get("preconditioner_tau", 1.0e-5)),
            schur_relative_floor=float(xl_params.get("preconditioner_schur_relative_floor", 1.0e-10)),
            use_schur=bool(xl_params.get("augmented_schur_preconditioner", True)),
        )

    def mhalf(direction: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return preconditioner(direction) if preconditioner is not None else direction

    def operator(direction: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Exactly M^-1/2 P J P M^-1/2.  P remains frozen over the solve.
        return mhalf(project(jvp(project(mhalf(direction)))))

    rhs = mhalf(project((-f_eta, -f_s)))
    return operator, rhs, project, mhalf, preconditioner_info


def solve_augmented_cis_newton_minres(
    eta: torch.Tensor,
    multiplier: torch.Tensor,
    hamiltonian_apply: Callable[[torch.Tensor], torch.Tensor],
    xl_params: Dict,
    *,
    ea_ei_flat: Optional[torch.Tensor] = None,
    preconditioner: Optional[
        Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Gauge-projected, symmetrically preconditioned MINRES correction."""
    operator_pair, rhs_pair, project, mhalf, preconditioner_info = make_augmented_cis_two_sided_operator(
        eta, multiplier, hamiltonian_apply, xl_params, ea_ei_flat=ea_ei_flat, preconditioner=preconditioner
    )
    b, r, n = eta.shape
    sdim = r * (r + 1) // 2
    dim, max_rank = r * n + sdim, min(int(xl_params.get("max_rank", 0)), r * n + sdim)
    tolerance = float(xl_params.get("err_threshold", 1.0e-6))
    dtype, device = eta.dtype, eta.device
    breakdown = max(float(eps), 100.0 * float(torch.finfo(eta.real.dtype).eps))

    def pack(pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        first, second = pair
        return torch.cat((first.reshape(b, r * n), augmented_svec(_hermitize(second))), dim=-1)

    def unpack(vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return vector[:, : r * n].reshape(b, r, n), augmented_smat(vector[:, r * n :], r)

    def operator(vector: torch.Tensor) -> torch.Tensor:
        return pack(operator_pair(unpack(vector)))

    rhs = pack(rhs_pair)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1)
    rhs_safe = rhs_norm.clamp_min(breakdown)
    live = rhs_norm > breakdown
    if max_rank <= 0 or not torch.any(live):
        return (
            torch.zeros_like(eta),
            torch.zeros_like(multiplier),
            {
                "rank": torch.zeros((), dtype=torch.long, device=device),
                "relative_residual": torch.where(
                    live, torch.ones_like(rhs_norm.real), torch.zeros_like(rhs_norm.real)
                ),
                "converged": ~live,
            },
        )
    V = torch.zeros((b, dim, max_rank + 1), dtype=dtype, device=device)
    alpha = torch.zeros((b, max_rank), dtype=dtype, device=device)
    beta = torch.zeros((b, max_rank), dtype=dtype, device=device)
    V[live, :, 0] = rhs[live] / rhs_norm[live, None]
    solution_y = torch.zeros((b, dim), dtype=dtype, device=device)
    relative_residual = torch.full((b,), float("inf"), dtype=eta.real.dtype, device=device)
    rank_used = 0
    for k in range(max_rank):
        if not torch.any(live):
            break
        w = operator(V[:, :, k])
        if k:
            w = w - beta[:, k - 1, None] * V[:, :, k - 1]
        alpha[:, k] = torch.einsum("Bi,Bi->B", V[:, :, k].conj(), w)
        w = w - alpha[:, k, None] * V[:, :, k]
        beta[:, k] = torch.linalg.vector_norm(w, dim=-1)
        can_expand = beta[:, k] > breakdown
        V[can_expand, :, k + 1] = w[can_expand] / beta[can_expand, k, None]
        rank_used = k + 1
        lanczos = torch.zeros((b, rank_used + 1, rank_used), dtype=dtype, device=device)
        for column in range(rank_used):
            lanczos[:, column, column] = alpha[:, column]
            lanczos[:, column + 1, column] = beta[:, column]
            if column:
                lanczos[:, column - 1, column] = beta[:, column - 1]
        rhs_small = torch.zeros((b, rank_used + 1), dtype=dtype, device=device)
        rhs_small[:, 0] = rhs_norm
        coefficients = torch.bmm(torch.linalg.pinv(lanczos), rhs_small.unsqueeze(-1)).squeeze(-1)
        solution_y = torch.einsum("Bni,Bi->Bn", V[:, :, :rank_used], coefficients)
        residual = rhs_small - torch.bmm(lanczos, coefficients.unsqueeze(-1)).squeeze(-1)
        relative_residual = torch.linalg.vector_norm(residual, dim=-1).real / rhs_safe.real
        live = (relative_residual > tolerance) & can_expand
    correction = project(mhalf(unpack(solution_y)))
    info = {
        "rank": torch.as_tensor(rank_used, dtype=torch.long, device=device),
        "relative_residual": relative_residual,
        "converged": relative_residual <= tolerance,
        "solver": "minres",
    }
    info.update(preconditioner_info)
    return correction[0], _hermitize(correction[1]), info


def solve_augmented_cis_newton_gmres(*args, **kwargs):
    """Backward-compatible entry point; the augmented solve is now MINRES."""
    return solve_augmented_cis_newton_minres(*args, **kwargs)


def augmented_cis_adiabatic_readout(
    eta: torch.Tensor, hamiltonian_apply: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return adiabatic energies/vectors without changing propagated eta."""
    gram = _hermitize(eta @ eta.conj().transpose(-1, -2))
    values, vectors = torch.linalg.eigh(gram)
    floor = torch.finfo(eta.real.dtype).eps * torch.linalg.matrix_norm(gram, ord=2).clamp_min(1.0)
    if torch.any(values <= floor[:, None]):
        raise RuntimeError("Augmented XL-CIS propagated subspace is rank deficient.")
    inv_sqrt = vectors @ torch.diag_embed(values.rsqrt()) @ vectors.conj().transpose(-1, -2)
    q = inv_sqrt @ eta
    h_sub = _hermitize(q @ hamiltonian_apply(q).conj().transpose(-1, -2))
    energies, vectors = torch.linalg.eigh(h_sub)
    return energies.real, vectors.conj().transpose(-1, -2) @ q, q


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

    symmetric_modes = {"symmetric_linearized", "symmetric_propagated_gauge"}
    valid_modes = {
        "independent_linearized",
        "independent_exact",
        "ordered_linearized",
        "symmetric_linearized",
        "symmetric_propagated_gauge",
        "augmented_eigenproblem",
    }
    if constraint_mode not in valid_modes:
        raise ValueError(
            f"Unknown XL-ESMD constraint_mode={constraint_mode!r}; expected one of {sorted(valid_modes)}"
        )

    if constraint_mode == "augmented_eigenproblem":
        # Separate XL-CIS formulation: (eta, Lambda) is propagated directly
        # in its smooth frame.  There is no xi(eta) solve or adiabatic rotation
        # of any XL coordinate/history in this branch.
        def hamiltonian_apply(flat: torch.Tensor) -> torch.Tensor:
            if flat.shape != eta_flat.shape:
                raise ValueError(f"Expected CIS block shape {eta_flat.shape}, got {flat.shape}")
            response = G_apply(flat.view(b, r, nocc, nvirt)).reshape(b, r, n)
            return ea_ei_flat * flat + response

        multiplier = getattr(mol, "xlesmd_augmented_lambda", None)
        if not (torch.is_tensor(multiplier) and multiplier.shape == (b, r, r)):
            seed_energies = getattr(mol, "cis_energies", None)
            if torch.is_tensor(seed_energies) and seed_energies.shape == (b, r):
                multiplier = torch.diag_embed(seed_energies.to(dtype=eta_flat.dtype, device=eta_flat.device))
            else:
                # A Rayleigh block is only a fallback for a direct electronic
                # call without normal XL-ESMD initialization.
                multiplier = _hermitize(
                    eta_flat @ hamiltonian_apply(eta_flat).conj().transpose(-1, -2)
                ).detach()
        else:
            multiplier = _hermitize(multiplier.to(dtype=eta_flat.dtype, device=eta_flat.device))

        # Readout only: Q and Xad determine physical PES quantities without
        # altering eta, Lambda, or their smooth-gauge Verlet histories.
        E, x_ad, q = augmented_cis_adiabatic_readout(eta_flat, hamiltonian_apply)

        with torch.no_grad():
            f_eta, f_s = augmented_cis_residual(eta_flat, multiplier, hamiltonian_apply)
            if xl_bomd_params is not None and "max_rank" in xl_bomd_params:
                deta2dt2, dlambda2dt2, krylov_info = solve_augmented_cis_newton_minres(
                    eta_flat, multiplier, hamiltonian_apply, xl_bomd_params, ea_ei_flat=ea_ei_flat
                )
                mol.Krylov_Error = krylov_info["relative_residual"]
            else:
                deta2dt2, dlambda2dt2 = torch.zeros_like(eta_flat), torch.zeros_like(multiplier)
                krylov_info = None
            mol.dxi2dt2 = (
                deta2dt2
                if MO_basis
                else torch.einsum("bmi,bria,bna->brmn", Cocc, deta2dt2.view(b, r, nocc, nvirt), Cvirt)
            )
            mol.xlesmd_augmented_deta2dt2 = deta2dt2.detach().clone()
            mol.xlesmd_augmented_dlambda2dt2 = dlambda2dt2.detach().clone()
            mol.xlesmd_augmented_lambda = multiplier.detach().clone()
            mol.xlesmd_augmented_eta_gauge = eta_flat.detach().clone()
            mol.xlesmd_augmented_xad = x_ad.detach().clone()
            diagnostics = {
                "constraint_mode": constraint_mode,
                "augmented_residual": torch.sqrt(
                    torch.sum(f_eta.conj() * f_eta, dim=-1).real
                    + torch.sum(f_s.conj() * f_s, dim=(-2, -1)).real
                ),
                "augmented_multiplier": multiplier.detach().clone(),
                "augmented_subspace_eigenvalue": E.detach().clone(),
                "eta_orthogonality": torch.linalg.matrix_norm(f_s),
            }
            if krylov_info is not None:
                diagnostics.update(
                    krylov_rank=krylov_info["rank"],
                    krylov_relative_residual=krylov_info["relative_residual"],
                    krylov_converged=krylov_info["converged"],
                    krylov_preconditioner="augmented_symmetric_spd_block",
                    krylov_root_mode="gauge_projected_augmented",
                )
            mol.xlesmd_diagnostics = diagnostics
            eta_AO = torch.einsum("bmi,bria,bna->brmn", Cocc, x_ad.view(b, r, nocc, nvirt), Cvirt)
        return E, eta_AO, x_ad

    # Symmetric XL keeps one geometry-differentiable G(eta) evaluation for its
    # shadow eigenvalues; all electronic-kernel uses receive its detached copy.
    G_energy = None
    if constraint_mode in symmetric_modes:
        eta_energy = eta_flat.detach()
        G_energy = G_apply(eta_energy.view(b, r, nocc, nvirt)).reshape(b, r, n)
        Gx_flat = G_energy.detach()

    # ``G(eta)`` and the constraint Gram factors are geometry-local invariants.
    # Reuse them for the shadow solve and all JVP evaluations.
    with torch.no_grad():
        if G_energy is None:
            Gx_flat = G_apply(eta).reshape(b, r, n)
        ordered_gram_factors = None
        symmetric_gram_factor = None
        if constraint_mode == "ordered_linearized":
            ordered_gram_factors = _ordered_gram_cholesky(eta_flat, 1.0 / ea_ei_flat)
        elif constraint_mode in symmetric_modes:
            symmetric_gram_factor = _symmetric_gram_cholesky(eta_flat, 1.0 / ea_ei_flat)

        # --- Solve for xi and multipliers ---
        # xi_flat: (b,r,n), omega_br: (b,r)
        if constraint_mode == "ordered_linearized":
            xi_flat, omega = solve_for_amplitudes_ordered(
                eta_flat, ea_ei_flat, G_eta=Gx_flat, gram_factors=ordered_gram_factors
            )
        elif constraint_mode in symmetric_modes:
            xi_flat, omega = solve_for_amplitudes_symmetric(
                eta_flat, ea_ei_flat, G_eta=Gx_flat, gram_factor=symmetric_gram_factor
            )
        elif constraint_mode == "independent_exact":
            omega_init = getattr(mol, "xlesmd_multipliers", None)
            if not (torch.is_tensor(omega_init) and omega_init.shape == (b, r)):
                omega_init = None
            xi_flat, omega = solve_for_amplitude_omega_newton(
                eta_flat, ea_ei_flat, Gx_flat, omega_init=omega_init
            )
        else:
            xi_flat, omega = solve_for_amplitude_omega(eta_flat, ea_ei_flat, Gx_flat)
        mol.xlesmd_multipliers = omega.detach().clone()

    # if hasattr(mol, "omega_xl"):
    #     omega_init = mol.omega_xl
    # else:
    #     omega_init = mol.cis_energies
    # xi_flat, omega = solve_for_amplitude_omega_newton(eta_flat, ea_ei_flat, Gx_flat, omega_init)
    # mol.omega_xl = omega

    # xi_flat = refine_orthogonalize_delta3(xi_flat)

    # The legacy formulations retain their individual row shadow energies.
    # Symmetric constraints instead describe a gauge block.  The ordinary
    # symmetric mode obtains physical labelled states from shadow-Hamiltonian
    # eigenpairs; the propagated-gauge variant deliberately keeps row labels.
    xi_gauge_flat = xi_flat
    if constraint_mode not in symmetric_modes:
        E1 = (xi_flat * xi_flat * ea_ei_flat).sum(dim=2)  # (b,r)
        E2 = ((2.0 * xi_flat - eta_flat) * Gx_flat).sum(dim=2)  # (b,r)
        E = E1 + E2  # (b,r)
    # --- Convert xi back to AO basis ---
    with torch.no_grad():
        xi = xi_gauge_flat.view(b, r, nocc, nvirt)
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
                    jvp_xi = make_jvp_xi_ordered(
                        ea_ei_flat,
                        eta_flat,
                        omega,
                        G_apply,
                        nocc,
                        nvirt,
                        G_eta=Gx_flat,
                        gram_factors=ordered_gram_factors,
                    )
                elif constraint_mode in symmetric_modes:
                    jvp_xi = make_jvp_xi_symmetric(
                        ea_ei_flat,
                        eta_flat,
                        omega,
                        G_apply,
                        nocc,
                        nvirt,
                        G_eta=Gx_flat,
                        gram_factor=symmetric_gram_factor,
                    )
                elif constraint_mode == "independent_exact":
                    jvp_xi = make_jvp_xi_full_normalized(
                        ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt
                    )
                else:
                    jvp_xi = make_jvp_xi(ea_ei_flat, eta_flat, xi_flat, omega, G_apply, nocc, nvirt)

                default_preconditioner = {
                    "ordered_linearized": "ordered_lowrank",
                    "symmetric_linearized": "symmetric_lowrank",
                    "symmetric_propagated_gauge": "symmetric_lowrank",
                }.get(constraint_mode, "none")
                preconditioner_name = str(
                    xl_bomd_params.get("krylov_preconditioner", default_preconditioner)
                ).lower()
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

                elif preconditioner_name == "ordered_lowrank":
                    if constraint_mode != "ordered_linearized":
                        raise ValueError(
                            "ordered_lowrank is only available with constraint_mode='ordered_linearized'."
                        )
                    tau = float(xl_bomd_params.get("preconditioner_tau", 1.0e-5))
                    kernel_inverse = make_apply_precond_ordered_lowrank(
                        ea_ei_flat, eta_flat, omega, Gx_flat, tau=tau, gram_factors=ordered_gram_factors
                    )

                    def preconditioner(v):
                        return -kernel_inverse(v)

                elif preconditioner_name == "symmetric_lowrank":
                    if constraint_mode not in symmetric_modes:
                        raise ValueError(
                            "symmetric_lowrank is only available with a symmetric constraint mode."
                        )
                    tau = float(xl_bomd_params.get("preconditioner_tau", 1.0e-5))
                    kernel_inverse = make_apply_precond_symmetric_lowrank(
                        ea_ei_flat, eta_flat, omega, Gx_flat, tau=tau, gram_factor=symmetric_gram_factor
                    )

                    def preconditioner(v):
                        return -kernel_inverse(v)

                else:
                    raise ValueError(
                        "krylov_preconditioner must be 'none', 'diagonal', 'rank1', "
                        "'ordered_lowrank', or 'symmetric_lowrank'."
                    )

                krylov_params = dict(xl_bomd_params)
                if constraint_mode in symmetric_modes:
                    # Full-root constraints couple every state, so solve the
                    # entire block in one GMRES system regardless of a stale
                    # independent-root setting in the input parameters.
                    krylov_params["krylov_root_mode"] = "coupled"
                else:
                    krylov_params.setdefault(
                        "krylov_root_mode",
                        "coupled" if constraint_mode == "ordered_linearized" else "independent",
                    )
                dxi2dt2_flat, krylov_info = compute_dxi2dt2_old_jacobian_gmres(
                    eta_brn=eta_flat,
                    xi_brn=xi_flat,
                    nu_br=omega,
                    ea_ei_flat=ea_ei_flat,
                    G_apply=G_apply,
                    nocc=nocc,
                    nvirt=nvirt,
                    xl_params=krylov_params,
                    jvp_xi=jvp_xi,
                    preconditioner=preconditioner,
                    return_info=True,
                    constraint_mode=constraint_mode,
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
        if constraint_mode in symmetric_modes:
            mol.xlesmd_xi_gauge = xi_gauge_flat.detach().clone()
            diagnostics["symmetric_constraint_residual"] = torch.linalg.matrix_norm(
                2.0 * eta_flat @ xi_gauge_flat.transpose(-1, -2)
                - eta_flat @ eta_flat.transpose(-1, -2)
                - identity
            )
            project_gauge = _make_symmetric_rotation_projector(eta_flat)
            correction = (
                dxi2dt2_flat
                if xl_bomd_params is not None and "max_rank" in xl_bomd_params
                else torch.zeros_like(eta_flat)
            )
            diagnostics["symmetric_qtd_norm"] = torch.linalg.vector_norm(
                (correction - project_gauge(correction)).reshape(b, r * n), dim=-1
            )
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
                krylov_root_mode=krylov_info["root_mode"],
            )
            for key in (
                "unregularized_kernel_gain",
                "rotational_regularized",
                "rotational_pair_softness",
                "rotational_pair_candidate",
                "rotational_pair_shift",
                "rotational_max_gain",
            ):
                if key in krylov_info:
                    diagnostics[f"krylov_{key}"] = krylov_info[key]
            if "history" in krylov_info:
                diagnostics["krylov_history"] = krylov_info["history"]
        mol.xlesmd_diagnostics = diagnostics

    if constraint_mode == "symmetric_linearized":
        # This is the nuclear partial derivative: eta and xi_gauge are fixed
        # electronic variables, while D and G(eta) retain their full explicit
        # geometry dependence.  The eigenvalues are the physical PESs.
        xi_gauge_energy = xi_gauge_flat.detach()
        H_shadow = _symmetric_shadow_hamiltonian(xi_gauge_energy, eta_energy, ea_ei_flat, G_energy)
        E, U = torch.linalg.eigh(H_shadow)
        xi_phys_flat = U.transpose(-1, -2) @ xi_gauge_energy
        eta_phys_flat = U.transpose(-1, -2) @ eta_energy
        xi_phys = xi_phys_flat.view(b, r, nocc, nvirt)
        xi_AO = torch.einsum("bmi,bria,bna->brmn", Cocc, xi_phys, Cvirt)
        with torch.no_grad():
            mol.xlesmd_eta_phys = eta_phys_flat.detach().clone()
            mol.xlesmd_diagnostics["symmetric_shadow_eigenvalues"] = E.detach().clone()
            mol.xlesmd_diagnostics["symmetric_shadow_hamiltonian"] = H_shadow.detach().clone()
        xi_flat = xi_phys_flat
    elif constraint_mode == "symmetric_propagated_gauge":
        # These are continuously propagated-gauge row PESs.  A different
        # initial common root rotation may give different individual rows;
        # no shadow-Hamiltonian diagonalization or frame rotation is applied.
        xi_gauge_energy = xi_gauge_flat.detach()
        E = (xi_gauge_energy * xi_gauge_energy * ea_ei_flat).sum(dim=2)
        E = E + ((2.0 * xi_gauge_energy - eta_energy) * G_energy).sum(dim=2)
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
    denom_sign = torch.where(denom >= 0.0, torch.ones_like(denom), -torch.ones_like(denom))
    denom = denom_sign * torch.clamp(denom.abs(), min=eps)  # (b,r)

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


def make_apply_precond_ordered_lowrank(
    ea_ei_flat: torch.Tensor,
    eta: torch.Tensor,
    multipliers: torch.Tensor,
    G_eta: torch.Tensor,
    tau: float = 1e-5,
    *,
    gram_factors: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Approximate ``(J_xi - I)^-1`` for ordered constraints, omitting only ``G(v)``.

    The gap term is lower triangular in root index and the complete multiplier
    response is a rank ``roots * (roots + 1) / 2`` Woodbury update.
    """
    b, r, n = eta.shape
    if multipliers.shape != (b, r, r) or G_eta.shape != eta.shape:
        raise ValueError("Ordered preconditioner inputs have incompatible shapes.")
    invD = 1.0 / ea_ei_flat[:, 0]
    if gram_factors is None:
        gram_factors = _ordered_gram_cholesky(eta, invD.unsqueeze(1))
    starts = [j * (j + 1) // 2 for j in range(r)]
    nlambda = r * (r + 1) // 2

    def apply_b_inverse(v: torch.Tensor) -> torch.Tensor:
        rows = []
        for j in range(r):
            value = v[:, j]
            for i, previous in enumerate(rows):
                value = value - (multipliers[:, j, i].unsqueeze(-1) * invD) * previous
            diagonal = multipliers[:, j, j].unsqueeze(-1) * invD - 1.0 - tau
            rows.append(value / diagonal)
        return torch.stack(rows, dim=1)

    def apply_u(beta: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(eta)
        for j in range(r):
            start = starts[j]
            result[:, j] = invD * torch.einsum("bjn,bj->bn", eta[:, : j + 1], beta[:, start : start + j + 1])
        return result

    def apply_l(v: torch.Tensor) -> torch.Tensor:
        result = torch.empty((b, nlambda), dtype=eta.dtype, device=eta.device)
        for j in range(r):
            H, V = eta[:, : j + 1], v[:, : j + 1]
            lam = multipliers[:, j, : j + 1]
            dS = torch.einsum("bjn,bkn->bjk", V, invD * H)
            dS = dS + torch.einsum("bjn,bkn->bjk", H, invD * V)
            dc = 0.5 * (torch.einsum("bjn,bn->bj", V, eta[:, j]) + torch.einsum("bjn,bn->bj", H, v[:, j]))
            du = dc + torch.einsum("bjn,bn->bj", V, invD * G_eta[:, j])
            start = starts[j]
            result[:, start : start + j + 1] = torch.cholesky_solve(
                (du - torch.einsum("bjk,bk->bj", dS, lam)).unsqueeze(-1), gram_factors[j]
            ).squeeze(-1)
        return result

    # Form I + L B^-1 U once; its dimension is only the number of multipliers.
    columns = []
    for column in range(nlambda):
        basis = torch.zeros((b, nlambda), dtype=eta.dtype, device=eta.device)
        basis[:, column] = 1.0
        columns.append(apply_l(apply_b_inverse(apply_u(basis))))
    woodbury = torch.eye(nlambda, dtype=eta.dtype, device=eta.device).unsqueeze(0)
    woodbury = woodbury + torch.stack(columns, dim=-1)
    woodbury_inverse = torch.linalg.pinv(woodbury, rtol=100.0 * torch.finfo(eta.dtype).eps)

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        z = apply_b_inverse(v)
        beta = torch.bmm(woodbury_inverse, apply_l(z).unsqueeze(-1)).squeeze(-1)
        return z - apply_b_inverse(apply_u(beta))

    return apply_prec


def make_apply_precond_symmetric_lowrank(
    ea_ei_flat: torch.Tensor,
    eta: torch.Tensor,
    multipliers: torch.Tensor,
    G_eta: torch.Tensor,
    tau: float = 1e-5,
    *,
    gram_factor: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Approximate ``(J_xi - I)^-1`` for full symmetric constraints.

    As in the ordered preconditioner, this retains the complete multiplier
    response and omits only ``G(v)``.  Its base block is dense in root index,
    unlike the lower-triangular ordered-constraint block.
    """
    b, r, n = eta.shape
    if multipliers.shape != (b, r, r) or G_eta.shape != eta.shape:
        raise ValueError("Symmetric preconditioner inputs have incompatible shapes.")
    invD = 1.0 / ea_ei_flat[:, 0]
    if gram_factor is None:
        gram_factor = _symmetric_gram_cholesky(eta, invD.unsqueeze(1))
    if gram_factor.shape != (b, r, r):
        raise ValueError(f"gram_factor must have shape {(b, r, r)}")

    identity = torch.eye(r, dtype=eta.dtype, device=eta.device)
    base = invD[:, :, None, None] * multipliers[:, None] - identity
    base = base - tau * identity

    def apply_b_inverse(v: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(base, v.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)

    def apply_u(beta: torch.Tensor) -> torch.Tensor:
        return invD.unsqueeze(1) * torch.einsum("bjn,bij->bin", eta, beta)

    def apply_l(v: torch.Tensor) -> torch.Tensor:
        dS = torch.einsum("bjn,bkn->bjk", v, invD.unsqueeze(1) * eta)
        dS = dS + torch.einsum("bjn,bkn->bjk", eta, invD.unsqueeze(1) * v)
        dc = 0.5 * (torch.einsum("bjn,bin->bji", v, eta) + torch.einsum("bjn,bin->bji", eta, v))
        du = dc + torch.einsum("bjn,bin->bji", v, invD.unsqueeze(1) * G_eta)
        lamb = multipliers.transpose(1, 2)
        return torch.cholesky_solve(du - torch.einsum("bjk,bki->bji", dS, lamb), gram_factor).transpose(1, 2)

    nlambda = r * r
    columns = []
    for column in range(nlambda):
        basis = torch.zeros((b, r, r), dtype=eta.dtype, device=eta.device)
        basis.reshape(b, nlambda)[:, column] = 1.0
        columns.append(apply_l(apply_b_inverse(apply_u(basis))).reshape(b, nlambda))
    woodbury = torch.eye(nlambda, dtype=eta.dtype, device=eta.device).unsqueeze(0)
    woodbury = woodbury + torch.stack(columns, dim=-1)
    woodbury_inverse = torch.linalg.pinv(woodbury, rtol=100.0 * torch.finfo(eta.dtype).eps)

    def apply_prec(v: torch.Tensor) -> torch.Tensor:
        z = apply_b_inverse(v)
        beta = torch.bmm(woodbury_inverse, apply_l(z).reshape(b, nlambda, 1)).reshape(b, r, r)
        return z - apply_b_inverse(apply_u(beta))

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
    G_apply: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    eps: float = 1e-12,
    *,
    G_eta: Optional[torch.Tensor] = None,
    gram_factors: Optional[Tuple[torch.Tensor, ...]] = None,
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
    if G_eta is None:
        if G_apply is None:
            raise ValueError("G_apply is required when G_eta is not supplied.")
        G_eta = G_apply(eta)
    G = G_eta
    if G.shape != eta.shape:
        raise ValueError(f"G_eta must have shape {eta.shape}; got {G.shape}")

    invD = 1.0 / ea_ei
    if gram_factors is None:
        gram_factors = _ordered_gram_cholesky(eta, invD)
    if len(gram_factors) != r:
        raise ValueError("gram_factors must contain one leading Gram factor per root.")
    xi = torch.empty_like(eta)
    multipliers = torch.zeros((b, r, r), dtype=eta.dtype, device=eta.device)

    for state in range(r):
        H = eta[:, : state + 1]
        eta_i = eta[:, state]
        G_i = G[:, state]
        gram_col = torch.einsum("bjn,bn->bj", H, eta_i)
        delta = torch.zeros_like(gram_col)
        delta[:, state] = 1.0
        c = 0.5 * (delta + gram_col)
        rhs = c + torch.einsum("bjn,bn->bj", H, invD[:, 0] * G_i)

        lamb = torch.cholesky_solve(rhs.unsqueeze(-1), gram_factors[state]).squeeze(-1)
        xi[:, state] = invD[:, 0] * (torch.einsum("bjn,bj->bn", H, lamb) - G_i)
        multipliers[:, state, : state + 1] = lamb

    return xi, multipliers


def _ordered_gram_cholesky(eta: torch.Tensor, invD: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Factor the leading ordered-constraint Gram matrices once per geometry."""
    factors = []
    for state in range(eta.shape[1]):
        H = eta[:, : state + 1]
        gram = torch.einsum("bjn,bkn->bjk", H, invD * H)
        factor, info = torch.linalg.cholesky_ex(gram)
        if torch.any(info > 0):
            raise RuntimeError(f"Linearly dependent auxiliary states in ordered XL-ESMD state {state}.")
        factors.append(factor)
    return tuple(factors)


def solve_for_amplitudes_symmetric(
    eta: torch.Tensor,
    ea_ei: torch.Tensor,
    G_apply: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    eps: float = 1e-12,
    *,
    G_eta: Optional[torch.Tensor] = None,
    gram_factor: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the full symmetric linearized constraints for every root pair.

    Each state ``i`` uses all auxiliary states as constraints:

        eta_j.T xi_i = 1/2 (delta_ij + eta_j.T eta_i).

    The returned ``(batch, roots, roots)`` multiplier matrix has row ``i``
    equal to the full multiplier vector for state ``i``.
    """
    if eta.ndim != 3:
        raise ValueError("eta must have shape (batch, roots, excitations)")
    if torch.any(ea_ei <= 1e-14):
        raise RuntimeError("Positive occupied--virtual gaps are required for XL-ESMD.")

    b, r, n = eta.shape
    if ea_ei.shape[0] != b or ea_ei.shape[-1] != n:
        raise ValueError("ea_ei must have shape compatible with (batch, 1, excitations)")
    if G_eta is None:
        if G_apply is None:
            raise ValueError("G_apply is required when G_eta is not supplied.")
        G_eta = G_apply(eta)
    if G_eta.shape != eta.shape:
        raise ValueError(f"G_eta must have shape {eta.shape}; got {G_eta.shape}")

    invD = 1.0 / ea_ei
    if gram_factor is None:
        gram_factor = _symmetric_gram_cholesky(eta, invD)
    if gram_factor.shape != (b, r, r):
        raise ValueError(f"gram_factor must have shape {(b, r, r)}")

    overlap = torch.einsum("bjn,bin->bji", eta, eta)
    identity = torch.eye(r, dtype=eta.dtype, device=eta.device).unsqueeze(0)
    c = 0.5 * (identity + overlap)
    rhs = c + torch.einsum("bjn,bin->bji", eta, invD * G_eta)
    multipliers = torch.cholesky_solve(rhs, gram_factor).transpose(1, 2)
    xi = invD * (torch.einsum("bjn,bij->bin", eta, multipliers) - G_eta)
    return xi, multipliers


def _symmetric_gram_cholesky(eta: torch.Tensor, invD: torch.Tensor) -> torch.Tensor:
    """Factor the common full-root Gram matrix for symmetric XL-ESMD."""
    gram = torch.einsum("bjn,bkn->bjk", eta, invD * eta)
    factor, info = torch.linalg.cholesky_ex(gram)
    if torch.any(info > 0):
        raise RuntimeError("Linearly dependent auxiliary states in symmetric XL-ESMD.")
    return factor


def _symmetric_shadow_hamiltonian(
    xi_gauge: torch.Tensor, eta: torch.Tensor, ea_ei: torch.Tensor, G_eta: torch.Tensor
) -> torch.Tensor:
    """Return the gauge-covariant symmetric shadow Hamiltonian."""
    hamiltonian = (xi_gauge * ea_ei) @ xi_gauge.transpose(-1, -2)
    hamiltonian = hamiltonian + xi_gauge @ G_eta.transpose(-1, -2)
    hamiltonian = hamiltonian + G_eta @ xi_gauge.transpose(-1, -2)
    hamiltonian = hamiltonian - 0.5 * (eta @ G_eta.transpose(-1, -2) + G_eta @ eta.transpose(-1, -2))
    return 0.5 * (hamiltonian + hamiltonian.transpose(-1, -2))


def _make_symmetric_rotation_projector(eta: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Project away ``A @ eta`` by a cached antisymmetric Sylvester solve."""
    gram = eta @ eta.transpose(-1, -2)
    values, vectors = torch.linalg.eigh(gram)
    minimum = torch.finfo(eta.dtype).eps * torch.linalg.matrix_norm(gram, ord=2).clamp_min(1.0)
    if torch.any(values <= minimum[:, None]):
        raise RuntimeError("Symmetric XL-ESMD auxiliary states are rank deficient.")
    denominator = values.unsqueeze(-1) + values.unsqueeze(-2)

    def project(v: torch.Tensor) -> torch.Tensor:
        C = v @ eta.transpose(-1, -2)
        rhs = vectors.transpose(-1, -2) @ (C - C.transpose(-1, -2)) @ vectors
        A = vectors @ (rhs / denominator) @ vectors.transpose(-1, -2)
        A = 0.5 * (A - A.transpose(-1, -2))
        return v - A @ eta

    return project


def make_jvp_xi_ordered(
    ea_ei: torch.Tensor,
    eta: torch.Tensor,
    multipliers: torch.Tensor,
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
    eps: float = 1e-12,
    *,
    G_eta: Optional[torch.Tensor] = None,
    gram_factors: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Analytic JVP of :func:`solve_for_amplitudes_ordered`."""
    b, r, n = eta.shape
    if n != nocc * nvirt:
        raise ValueError("The excitation dimension must equal nocc*nvirt.")
    if multipliers.shape != (b, r, r):
        raise ValueError(f"multipliers must have shape {(b, r, r)}")
    invD = 1.0 / ea_ei
    if G_eta is None:
        G_eta = G_apply(eta.reshape(b, r, nocc, nvirt)).reshape(b, r, n)
    if G_eta.shape != eta.shape:
        raise ValueError(f"G_eta must have shape {eta.shape}; got {G_eta.shape}")
    if gram_factors is None:
        gram_factors = _ordered_gram_cholesky(eta, invD)
    if len(gram_factors) != r:
        raise ValueError("gram_factors must contain one leading Gram factor per root.")

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

            dS = torch.einsum("bjn,bkn->bjk", V, invD * H)
            dS = dS + torch.einsum("bjn,bkn->bjk", H, invD * V)
            dc = 0.5 * (torch.einsum("bjn,bn->bj", V, eta_i) + torch.einsum("bjn,bn->bj", H, v_i))
            du = dc
            du = du + torch.einsum("bjn,bn->bj", V, invD[:, 0] * G_i)
            du = du + torch.einsum("bjn,bn->bj", H, invD[:, 0] * Gv_i)
            dlamb = torch.cholesky_solve(
                (du - torch.einsum("bjk,bk->bj", dS, lamb)).unsqueeze(-1), gram_factors[state]
            ).squeeze(-1)

            dxi[:, state] = invD[:, 0] * (
                torch.einsum("bjn,bj->bn", V, lamb) + torch.einsum("bjn,bj->bn", H, dlamb) - Gv_i
            )
        return dxi

    return jvp_xi


def make_jvp_xi_symmetric(
    ea_ei: torch.Tensor,
    eta: torch.Tensor,
    multipliers: torch.Tensor,
    G_apply: Callable[[torch.Tensor], torch.Tensor],
    nocc: int,
    nvirt: int,
    eps: float = 1e-12,
    *,
    G_eta: Optional[torch.Tensor] = None,
    gram_factor: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Analytic JVP of :func:`solve_for_amplitudes_symmetric`."""
    b, r, n = eta.shape
    if n != nocc * nvirt:
        raise ValueError("The excitation dimension must equal nocc*nvirt.")
    if multipliers.shape != (b, r, r):
        raise ValueError(f"multipliers must have shape {(b, r, r)}")
    invD = 1.0 / ea_ei
    if G_eta is None:
        G_eta = G_apply(eta.reshape(b, r, nocc, nvirt)).reshape(b, r, n)
    if G_eta.shape != eta.shape:
        raise ValueError(f"G_eta must have shape {eta.shape}; got {G_eta.shape}")
    if gram_factor is None:
        gram_factor = _symmetric_gram_cholesky(eta, invD)
    if gram_factor.shape != (b, r, r):
        raise ValueError(f"gram_factor must have shape {(b, r, r)}")

    def jvp_xi(v: torch.Tensor) -> torch.Tensor:
        if v.shape != eta.shape:
            raise ValueError(f"Expected v shape {eta.shape}, got {v.shape}")
        G_v = G_apply(v.reshape(b, r, nocc, nvirt)).reshape(b, r, n)
        dS = torch.einsum("bjn,bkn->bjk", v, invD * eta)
        dS = dS + torch.einsum("bjn,bkn->bjk", eta, invD * v)
        dc = 0.5 * (torch.einsum("bjn,bin->bji", v, eta) + torch.einsum("bjn,bin->bji", eta, v))
        du = dc
        du = du + torch.einsum("bjn,bin->bji", v, invD * G_eta)
        du = du + torch.einsum("bjn,bin->bji", eta, invD * G_v)
        lamb = multipliers.transpose(1, 2)
        dlamb = torch.cholesky_solve(du - torch.einsum("bjk,bki->bji", dS, lamb), gram_factor).transpose(1, 2)
        return invD * (
            torch.einsum("bjn,bij->bin", v, multipliers) + torch.einsum("bjn,bij->bin", eta, dlamb) - G_v
        )

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
    constraint_mode: Optional[str] = None,
):
    """Solve (I - J_xi) d = xi - eta with right-preconditioned GMRES.

    Ordered roots screen near-degenerate pair rotations before GMRES.  The
    symmetric formulation instead removes its exact rotational gauge span and
    solves directly on the quotient space.
    """
    b, r, n = eta_brn.shape
    dtype, device = eta_brn.dtype, eta_brn.device
    max_rank = int(xl_params["max_rank"])
    err_threshold = float(xl_params.get("err_threshold", 1e-6))
    root_mode = str(xl_params.get("krylov_root_mode", "independent")).lower()
    if root_mode not in {"independent", "coupled"}:
        raise ValueError("krylov_root_mode must be 'independent' or 'coupled'.")
    coupled = root_mode == "coupled"
    constraint_mode = str(xl_params.get("constraint_mode", constraint_mode or "")).lower()
    ordered = constraint_mode == "ordered_linearized"
    symmetric = constraint_mode in {"symmetric_linearized", "symmetric_propagated_gauge"}
    if (ordered or symmetric) and not coupled:
        raise ValueError(f"{constraint_mode} requires krylov_root_mode='coupled'.")

    info_shape = (b,) if coupled else (b, r)
    jacobian_regularization = float(xl_params.get("jacobian_regularization", 0.0))
    if jacobian_regularization < 0.0:
        raise ValueError("jacobian_regularization must be nonnegative.")

    auto_rot = bool(xl_params.get("rotational_regularization", ordered and coupled))
    rot_max_gain = float(xl_params.get("rotational_max_gain", 20.0))
    rot_rel_gap_tol = float(xl_params.get("rotational_candidate_relative_gap", 0.02))
    rot_mix_tol = float(xl_params.get("rotational_candidate_mixing", 0.10))
    if rot_max_gain <= 1.0:
        raise ValueError("rotational_max_gain must be > 1.")
    if rot_rel_gap_tol < 0.0 or rot_mix_tol < 0.0:
        raise ValueError("rotational candidate thresholds must be nonnegative.")
    rot_soft_tol = 1.0 / rot_max_gain

    if max_rank <= 0:
        solution = torch.zeros_like(eta_brn)
        info = {
            "rank": 0,
            "relative_residual": torch.ones(info_shape, dtype=dtype, device=device),
            "converged": torch.zeros(info_shape, dtype=torch.bool, device=device),
            "kernel_gain": torch.zeros(info_shape, dtype=dtype, device=device),
            "kernel_gain_scale": torch.ones(info_shape, dtype=dtype, device=device),
            "root_mode": root_mode,
            "rotational_regularized": torch.zeros((b,), dtype=torch.bool, device=device),
        }
        return (solution, info) if return_info else solution

    B, N = (b, r * n) if coupled else (b * r, n)
    finfo = torch.finfo(dtype)
    breakdown_tol = max(float(eps), 100.0 * float(finfo.eps))

    if jvp_xi is None:
        jvp_xi = make_jvp_xi(ea_ei_flat, eta_brn, xi_brn, nu_br, G_apply, nocc, nvirt, eps=eps)

    if symmetric:
        project = _make_symmetric_rotation_projector(eta_brn)

    else:

        def project(v_brn: torch.Tensor) -> torch.Tensor:
            return v_brn

    rhs = project(xi_brn - eta_brn).reshape(B, N)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=1)
    rhs_norm_safe = rhs_norm.clamp_min(breakdown_tol)

    # ------------------------------------------------------------
    # Before GMRES, identify potentially dangerous ordered-state rotations.
    # The multiplier/gap screen is free; only candidate pairs receive a JVP.
    # ------------------------------------------------------------
    rot_Q = None
    rot_Qgram_inv = None
    rot_mu = None
    rot_regularized = torch.zeros((b,), dtype=torch.bool, device=device)
    rot_pair_indices = []
    rot_pair_candidate = None
    rot_pair_softness = None
    rot_pair_rayleigh = None
    rot_pair_shift = None

    if auto_rot and ordered and coupled and r > 1:
        diag = torch.diagonal(nu_br, dim1=-2, dim2=-1)  # (b,r)
        q_columns = []
        candidate_columns = []
        softness_columns = []
        rayleigh_columns = []
        shift_columns = []
        soft_columns = []

        for i in range(1, r):
            for j in range(i):
                rot_pair_indices.append((i, j))

                gap = torch.abs(diag[:, i] - diag[:, j])
                scale = torch.maximum(diag[:, i].abs(), diag[:, j].abs()).clamp_min(breakdown_tol)
                rel_gap = gap / scale
                gap_floor = 1.0e-8 * scale
                mixing = torch.abs(nu_br[:, i, j]) / torch.maximum(gap, gap_floor)
                candidate = (rel_gap < rot_rel_gap_tol) | (mixing > rot_mix_tol)

                q = torch.zeros_like(eta_brn)
                q[:, i, :] = eta_brn[:, j, :]
                q[:, j, :] = -eta_brn[:, i, :]
                qnorm = torch.linalg.vector_norm(q.reshape(b, N), dim=1)
                valid = qnorm > breakdown_tol
                candidate = candidate & valid

                q_unit = torch.zeros_like(q)
                q_unit[valid] = q[valid] / qnorm[valid, None, None]

                softness = torch.full((b,), float("inf"), dtype=dtype, device=device)
                rayleigh = torch.zeros((b,), dtype=dtype, device=device)
                shift = torch.zeros((b,), dtype=dtype, device=device)
                soft = torch.zeros((b,), dtype=torch.bool, device=device)

                if torch.any(candidate):
                    # One exact probe for this candidate pair.  Noncandidate
                    # molecules are zeroed, so the batched JVP costs one call.
                    q_probe = q_unit * candidate[:, None, None]
                    Kq = q_probe - jvp_xi(q_probe)  # unregularized K = I - J_xi
                    Kqf = Kq.reshape(b, N)
                    qf = q_probe.reshape(b, N)

                    softness_now = torch.linalg.vector_norm(Kqf, dim=1)
                    rayleigh_now = torch.sum(qf * Kqf, dim=1)
                    softness = torch.where(candidate, softness_now, softness)
                    rayleigh = torch.where(candidate, rayleigh_now, rayleigh)
                    soft = candidate & (softness < rot_soft_tol)

                    # Move the signed Rayleigh value away from zero without
                    # changing its sign.  This avoids a positive shift making
                    # a small negative eigenvalue even closer to zero.
                    sign = torch.where(rayleigh >= 0.0, torch.ones_like(rayleigh), -torch.ones_like(rayleigh))
                    shift = torch.where(soft, sign * rot_soft_tol - rayleigh, shift)

                q_columns.append(q_unit.reshape(b, N))
                candidate_columns.append(candidate)
                softness_columns.append(softness)
                rayleigh_columns.append(rayleigh)
                shift_columns.append(shift)
                soft_columns.append(soft)

        if q_columns:
            Qall = torch.stack(q_columns, dim=-1)  # (b,N,npair)
            rot_pair_candidate = torch.stack(candidate_columns, dim=-1)
            rot_pair_softness = torch.stack(softness_columns, dim=-1)
            rot_pair_rayleigh = torch.stack(rayleigh_columns, dim=-1)
            rot_pair_shift = torch.stack(shift_columns, dim=-1)
            soft_mask = torch.stack(soft_columns, dim=-1)
            rot_regularized = torch.any(soft_mask, dim=-1)

            if torch.any(rot_regularized):
                # Use basis coordinates rather than assuming pair-rotation
                # vectors are exactly orthogonal away from self consistency.
                rot_Q = Qall * soft_mask[:, None, :]
                gram = torch.einsum("bNk,bNl->bkl", rot_Q, rot_Q)
                rot_Qgram_inv = torch.linalg.pinv(gram, rtol=100.0 * float(finfo.eps))
                rot_mu = rot_pair_shift * soft_mask

    def rotational_correction(v_brn: torch.Tensor) -> torch.Tensor:
        if rot_Q is None:
            return torch.zeros_like(v_brn)
        vf = v_brn.reshape(b, N)
        coeff = torch.einsum("bNk,bN->bk", rot_Q, vf)
        coeff = torch.bmm(rot_Qgram_inv, coeff.unsqueeze(-1)).squeeze(-1)
        coeff = rot_mu * coeff
        return torch.einsum("bNk,bk->bN", rot_Q, coeff).reshape(b, r, n)

    # ------------------------------------------------------------
    # Single right-preconditioned GMRES solve.
    # ------------------------------------------------------------
    live = rhs_norm > breakdown_tol
    if not torch.any(live):
        solution = torch.zeros_like(eta_brn)
        info = {
            "rank": 0,
            "relative_residual": torch.zeros(info_shape, dtype=dtype, device=device),
            "converged": torch.ones(info_shape, dtype=torch.bool, device=device),
            "kernel_gain": torch.zeros(info_shape, dtype=dtype, device=device),
            "kernel_gain_scale": torch.ones(info_shape, dtype=dtype, device=device),
            "root_mode": root_mode,
            "rotational_regularized": rot_regularized,
        }
        return (solution, info) if return_info else solution

    V = torch.zeros((B, N, max_rank + 1), dtype=dtype, device=device)
    Hbar = torch.zeros((B, max_rank + 1, max_rank), dtype=dtype, device=device)
    small_rhs = torch.zeros((B, max_rank + 1), dtype=dtype, device=device)
    small_rhs[:, 0] = rhs_norm
    V[live, :, 0] = rhs[live] / rhs_norm[live].unsqueeze(-1)

    solution_y = torch.zeros((B, N), dtype=dtype, device=device)
    relres = torch.zeros((B,), dtype=dtype, device=device)
    relres[live] = float("inf")
    rank_used = 0

    for k in range(max_rank):
        if not torch.any(live):
            break

        iter_mask = live.clone()
        vk = V[:, :, k].reshape(b, r, n)
        zk = preconditioner(vk) if preconditioner is not None else vk
        zk = project(zk)

        w_brn = (1.0 + jacobian_regularization) * zk - jvp_xi(zk)
        if rot_Q is not None:
            w_brn = w_brn + rotational_correction(zk)
        w_brn = project(w_brn)
        w = w_brn.reshape(B, N)
        w[~iter_mask] = 0.0

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
        solution_y = torch.einsum("Bni,Bi->Bn", V[:, :, :rank_used], y)

        small_residual = gm - torch.bmm(Hm, y.unsqueeze(-1)).squeeze(-1)
        relres = torch.linalg.vector_norm(small_residual, dim=1) / rhs_norm_safe
        relres = torch.where(rhs_norm > breakdown_tol, relres, torch.zeros_like(relres))

        converged = relres <= err_threshold
        live = iter_mask & (~converged) & can_expand

    solution = solution_y.reshape(b, r, n)
    if preconditioner is not None:
        solution = preconditioner(solution)
    solution = project(solution)

    correction_norm = torch.linalg.vector_norm(solution.reshape(B, N), dim=-1)
    gain = correction_norm / rhs_norm_safe
    gain = gain.reshape(b) if coupled else gain.reshape(b, r)
    gain_scale = torch.ones_like(gain)

    # Optional emergency cap; normally leave unset when selective rotation
    # regularization is enabled.
    max_amplification = xl_params.get("kernel_max_amplification")
    if max_amplification is not None:
        max_amplification = float(max_amplification)
        if max_amplification <= 0.0:
            raise ValueError("kernel_max_amplification must be positive when supplied.")
        gain_scale = torch.clamp(max_amplification / gain.clamp_min(breakdown_tol), max=1.0)
        solution = solution * (gain_scale[:, None, None] if coupled else gain_scale[..., None])

    if xl_params.get("verbose_krylov", False):
        print(
            f"Old-Jacobian GMRES rank used: {rank_used}, "
            f"max relative residual: {torch.max(relres).item():.3e}, "
            f"max kernel gain: {torch.max(gain).item():.2e}, "
            f"rot-reg: {int(rot_regularized.sum())}/{b}"
        )

    info = {
        "rank": rank_used,
        "relative_residual": relres.reshape(b) if coupled else relres.reshape(b, r),
        "converged": (relres <= err_threshold).reshape(b)
        if coupled
        else (relres <= err_threshold).reshape(b, r),
        "kernel_gain": gain,
        "kernel_gain_scale": gain_scale,
        "rotational_regularized": rot_regularized,
        "rotational_max_gain": rot_max_gain,
        "root_mode": root_mode,
    }
    if symmetric:
        info["quotient_gauge_overlap"] = torch.linalg.vector_norm(
            (solution - project(solution)).reshape(b, N), dim=-1
        )
    if rot_pair_candidate is not None:
        info["rotational_pair_indices"] = rot_pair_indices
        info["rotational_pair_candidate"] = rot_pair_candidate
        info["rotational_pair_softness"] = rot_pair_softness
        info["rotational_pair_rayleigh"] = rot_pair_rayleigh
        info["rotational_pair_shift"] = rot_pair_shift

    return (solution, info) if return_info else solution
