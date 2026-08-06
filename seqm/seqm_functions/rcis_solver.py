"""Implicit RCIS gradients and their matrix-free adjoint solve."""

import torch

from .excited_state_utils import get_occ_virt

_DEGENERACY_ATOL = 1.0e-6
_DEGENERACY_RTOL = 1.0e-6
_ADJOINT_MAX_ITER = 200
_PARAMETER_NAMES = ("g_ss", "g_pp", "g_sp", "g_p2", "h_sp")


def rcis_needs_grad(*tensors):
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def rcis_inputs(mol, e_mo, w):
    parameters = mol.parameters
    return (mol.molecular_orbitals, e_mo, w, *(parameters[name] for name in _PARAMETER_NAMES))


def make_rcis_response_builder(mol, matrix_vector_product, *, orbital_window=None, cache_chunk_plan=False):
    """Bind a native RCIS matrix action to the differentiable adjoint inputs."""

    def build_response(molecular_orbitals, e_mo, w, *parameter_values):
        _, _, Cocc, Cvirt, energy_differences = get_occ_virt(
            mol, orbital_window=orbital_window, e_mo=e_mo, molecular_orbitals=molecular_orbitals
        )
        parameters = dict(zip(_PARAMETER_NAMES, parameter_values))
        action_kwargs = {"parameters": parameters}
        if cache_chunk_plan:
            action_kwargs["chunk_plan_cache"] = {}

        def apply(vectors):
            return matrix_vector_product(mol, vectors, w, energy_differences, Cocc, Cvirt, **action_kwargs)

        return apply

    return build_response


def run_native_rcis(native_solver, response_builder, root_tol, *inputs):
    """Run a native Davidson primal and attach its implicit VJP when required."""

    with torch.no_grad():
        evals, amplitudes, nroots_target, precond_diag, valid_mask, max_subspace, converged = native_solver()
    unstable = rcis_instability_mask(evals, nroots_target)
    converged = converged & ~unstable
    if rcis_needs_grad(*inputs):
        evals, amplitudes = attach_implicit_rcis(
            evals,
            amplitudes,
            nroots_target,
            valid_mask,
            precond_diag,
            max_subspace,
            response_builder,
            root_tol,
            *inputs,
        )
    return evals, amplitudes, nroots_target, converged, unstable


def attach_implicit_rcis(
    evals,
    amplitudes,
    nroots_target,
    valid_mask,
    precond_diag,
    max_subspace_size,
    response_builder,
    root_tol,
    *inputs,
):
    """Attach an implicit VJP to roots produced by a native RCIS Davidson solve."""

    return _ImplicitRCISBackward.apply(
        response_builder,
        float(root_tol),
        int(max_subspace_size),
        evals,
        amplitudes,
        nroots_target,
        valid_mask,
        precond_diag,
        *inputs,
    )


class _ImplicitRCISBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        response_builder,
        root_tol,
        max_subspace_size,
        evals,
        amplitudes,
        nroots_target,
        valid_mask,
        precond_diag,
        *inputs,
    ):
        ctx.response_builder = response_builder
        ctx.root_tol = root_tol
        ctx.max_subspace_size = max_subspace_size
        ctx.save_for_backward(evals, amplitudes, nroots_target, valid_mask, precond_diag, *inputs)
        return evals, amplitudes

    @staticmethod
    def backward(ctx, grad_evals, grad_amplitudes):
        evals, amplitudes, nroots_target, valid_mask, precond_diag, *inputs = ctx.saved_tensors
        active_roots = _active_root_mask(nroots_target)
        grad_evals = _zero_if_none(grad_evals, evals) * active_roots
        grad_amplitudes = _zero_if_none(grad_amplitudes, amplitudes)
        grad_amplitudes = grad_amplitudes * active_roots.unsqueeze(-1) * valid_mask.unsqueeze(1)

        detached_inputs, grad_inputs = [], []
        for tensor in inputs:
            detached = tensor.detach()
            if tensor.requires_grad:
                detached.requires_grad_(True)
                grad_inputs.append(detached)
            detached_inputs.append(detached)

        with torch.enable_grad():
            apply = ctx.response_builder(*detached_inputs)
            with torch.no_grad():
                adjoint = _solve_adjoint_batch(
                    apply,
                    ctx.max_subspace_size,
                    grad_amplitudes,
                    amplitudes,
                    evals,
                    nroots_target,
                    valid_mask,
                    precond_diag,
                    ctx.root_tol,
                )

            cotangent = grad_evals.unsqueeze(-1) * amplitudes - adjoint
            cotangent = cotangent * active_roots.unsqueeze(-1) * valid_mask.unsqueeze(1)
            root_indices, selected_lanes = _selected_root_indices(cotangent.abs().any(dim=-1))
            if root_indices is None:
                gradients = (None,) * len(grad_inputs)
            else:
                cotangent = _gather_root_vectors(cotangent, root_indices, selected_lanes)
                trial_vectors = _gather_root_vectors(amplitudes.detach(), root_indices, selected_lanes)
                scalar = torch.sum(cotangent * apply(trial_vectors))
                gradients = torch.autograd.grad(scalar, grad_inputs, allow_unused=True)

        grad_iter = iter(gradients)
        input_grads = [next(grad_iter) if tensor.requires_grad else None for tensor in inputs]
        return (None, None, None, None, None, None, None, None, *input_grads)


def _solve_adjoint_batch(
    apply,
    max_subspace_size,
    grad_amplitudes,
    amplitudes,
    evals,
    nroots_target,
    valid_mask,
    diagonal,
    root_tol,
):
    """Solve all requested CIS adjoints in one packed Davidson iteration loop."""

    nmol, nroots, nov = amplitudes.shape
    active_roots = _active_root_mask(nroots_target)
    selected = active_roots & grad_amplitudes.abs().any(dim=-1)
    adjoint = torch.zeros_like(amplitudes)
    root_indices, selected_lanes = _selected_root_indices(selected)
    if root_indices is None:
        return adjoint

    protected, counts = _protected_root_bases(
        amplitudes, evals, nroots_target, root_indices, valid_mask, root_tol, selected_lanes
    )
    nselected_roots = root_indices.shape[1]
    nsystems = nmol * nselected_roots
    protected = protected.reshape(nsystems, nroots, nov)
    counts = counts.reshape(nsystems)
    valid = valid_mask[:, None, :].expand(-1, nselected_roots, -1).reshape(nsystems, nov)
    rhs = _gather_root_vectors(grad_amplitudes, root_indices, selected_lanes).reshape(nsystems, nov)
    rhs = _project(rhs, protected, valid)
    omega = torch.gather(evals, 1, root_indices).reshape(nsystems)
    diagonal = diagonal[:, None, :].expand(-1, nselected_roots, -1).reshape(nsystems, nov)
    selected = selected_lanes.reshape(nsystems)
    rhs_norm = torch.linalg.vector_norm(rhs, ord=torch.inf, dim=1)

    capacity = max_subspace_size
    V = torch.zeros(nsystems, capacity, nov, dtype=rhs.dtype, device=rhs.device)
    BV = torch.zeros_like(V)
    vend = torch.zeros(nsystems, dtype=torch.long, device=rhs.device)
    vstart = torch.zeros_like(vend)
    done = ~selected | (rhs_norm == 0)
    shift = diagonal - omega.unsqueeze(1)

    for local_idx in torch.nonzero(~done, as_tuple=False).squeeze(1).tolist():
        basis = protected[local_idx, : counts[local_idx]]
        initial = _orthogonalize(
            _precondition(rhs[local_idx], shift[local_idx], valid[local_idx]), basis, valid[local_idx]
        )
        if initial is None:
            initial = _orthogonalize(rhs[local_idx], basis, valid[local_idx])
        if initial is None:
            done[local_idx] = True
        else:
            V[local_idx, 0] = initial
            vend[local_idx] = 1

    best = torch.zeros_like(rhs)
    tolerance = max(root_tol, 10.0 * torch.finfo(rhs.dtype).eps)
    stalled = torch.zeros_like(done)
    for _ in range(_ADJOINT_MAX_ITER):
        if done.all():
            break

        packed_vectors, system_idx, absolute_idx, mask, packed_idx = _gather_packed_adjoint_subspace(
            V, vstart, vend, done, nmol, nselected_roots
        )
        if packed_vectors is None:
            raise RuntimeError("RCIS adjoint Davidson has no vectors left to evaluate.")
        width = packed_vectors.shape[2]
        nslots = packed_vectors.shape[1]
        raw_action = apply(packed_vectors.reshape(nmol, nslots * width, nov))
        raw_action = raw_action.reshape(nmol, nslots, width, nov)[packed_idx[:, 0], packed_idx[:, 1]]
        new_vectors = V[system_idx[:, None], absolute_idx.clamp(max=capacity - 1)] * mask.unsqueeze(-1)
        projected = _project(new_vectors, protected[system_idx], valid[system_idx])
        action = _project(
            raw_action - omega[system_idx, None, None] * projected, protected[system_idx], valid[system_idx]
        )
        BV[system_idx[:, None], absolute_idx.clamp(max=capacity - 1)] = action * mask.unsqueeze(-1)

        vend_max = int(vend.max().item())
        coefficients = torch.zeros(nsystems, vend_max, dtype=rhs.dtype, device=rhs.device)
        for size in torch.unique(vend[~done]).tolist():
            group = torch.nonzero(~done & (vend == size), as_tuple=False).squeeze(1)
            hamiltonian = torch.einsum("bno,bro->bnr", V[group, :size], BV[group, :size])
            small_rhs = torch.einsum("bno,bo->bn", V[group, :size], rhs[group])
            coefficients[group, :size] = _solve_projected_system(hamiltonian, small_rhs)

        solution = torch.einsum("bn,bno->bo", coefficients, V[:, :vend_max])
        residual = _project(
            rhs - torch.einsum("bn,bno->bo", coefficients, BV[:, :vend_max]), protected, valid
        )
        best[~done] = solution[~done]
        residual_norm = torch.linalg.vector_norm(residual, ord=torch.inf, dim=1)
        converged = residual_norm <= tolerance * torch.clamp(rhs_norm, min=1.0)
        done |= converged

        for local_idx in torch.nonzero(~done & (vend >= capacity), as_tuple=False).squeeze(1).tolist():
            basis = protected[local_idx, : counts[local_idx]]
            restart = _orthogonalize(solution[local_idx], basis, valid[local_idx])
            correction_basis = basis if restart is None else torch.cat((basis, restart.unsqueeze(0)))
            correction = _orthogonalize(
                _precondition(residual[local_idx], shift[local_idx], valid[local_idx]),
                correction_basis,
                valid[local_idx],
            )
            V[local_idx].zero_()
            BV[local_idx].zero_()
            vend[local_idx] = 0
            vstart[local_idx] = 0
            if restart is None or correction is None or capacity < 2:
                done[local_idx] = True
                stalled[local_idx] = True
                continue
            V[local_idx, 0] = restart
            V[local_idx, 1] = correction
            vend[local_idx] = 2

        for local_idx in torch.nonzero(~done & (vend < capacity), as_tuple=False).squeeze(1).tolist():
            basis = torch.cat((protected[local_idx, : counts[local_idx]], V[local_idx, : vend[local_idx]]))
            correction = _orthogonalize(
                _precondition(residual[local_idx], shift[local_idx], valid[local_idx]),
                basis,
                valid[local_idx],
            )
            if correction is None:
                done[local_idx] = True
                stalled[local_idx] = True
            else:
                vstart[local_idx] = vend[local_idx]
                V[local_idx, vend[local_idx]] = correction
                vend[local_idx] += 1

        if stalled.any():
            systems = torch.nonzero(stalled, as_tuple=False).squeeze(1).tolist()
            raise RuntimeError(f"RCIS adjoint Davidson stagnated for systems {systems}.")
    else:
        raise RuntimeError("RCIS adjoint Davidson did not converge.")

    molecule_idx, lane_idx = torch.nonzero(selected_lanes, as_tuple=True)
    adjoint[molecule_idx, root_indices[molecule_idx, lane_idx]] = best.reshape(nmol, nselected_roots, nov)[
        molecule_idx, lane_idx
    ]
    return adjoint


def _gather_packed_adjoint_subspace(V, vstart, vend, done, nmol, nroots):
    """Pack only active root updates for each molecule into the operator axis."""

    nsystems, capacity, nov = V.shape
    delta = vend - vstart
    delta = delta.masked_fill(done, 0)
    active = delta > 0
    if not active.any():
        return None, None, None, None, None

    width = int(delta[active].max().item())
    active = active.view(nmol, nroots)
    nslots = int(active.sum(dim=1).max().item())
    system_idx = torch.nonzero(active.reshape(-1), as_tuple=False).squeeze(1)
    molecule_idx = torch.div(system_idx, nroots, rounding_mode="floor")
    root_idx = system_idx.remainder(nroots)
    packed_slot = active.cumsum(dim=1)[molecule_idx, root_idx] - 1

    relative_idx = torch.arange(width, device=V.device).unsqueeze(0)
    absolute_idx = relative_idx + vstart[system_idx].unsqueeze(1)
    mask = relative_idx < delta[system_idx].unsqueeze(1)
    new_vectors = V[system_idx[:, None], absolute_idx.clamp(max=capacity - 1)] * mask.unsqueeze(-1)
    packed = torch.zeros(nmol, nslots, width, nov, dtype=V.dtype, device=V.device)
    packed[molecule_idx, packed_slot] = new_vectors
    packed_idx = torch.stack((molecule_idx, packed_slot), dim=1)
    return packed, system_idx, absolute_idx, mask, packed_idx


def _selected_root_indices(selected):
    """Compact root selection independently for every molecule in the batch."""

    counts = selected.sum(dim=1)
    if not counts.any():
        return None, None

    nlanes = int(counts.max().item())
    root_indices = torch.argsort((~selected).to(torch.int64), dim=1, stable=True)[:, :nlanes]
    selected_lanes = torch.arange(nlanes, device=selected.device).unsqueeze(0) < counts.unsqueeze(1)
    return root_indices, selected_lanes


def _gather_root_vectors(vectors, root_indices, selected_lanes):
    """Gather selected root rows and zero padding introduced by batch compaction."""

    gathered = torch.gather(vectors, 1, root_indices.unsqueeze(-1).expand(-1, -1, vectors.shape[-1]))
    return gathered * selected_lanes.unsqueeze(-1)


def _protected_root_bases(
    amplitudes, evals, nroots_target, root_indices, valid_mask, root_tol, selected_lanes=None
):
    """Build orthonormal bases for each selected root's degenerate subspace."""

    nmol, nroots, nov = amplitudes.shape
    if root_indices.dim() == 1:
        root_indices = root_indices.expand(nmol, -1)
    nselected_roots = root_indices.shape[1]
    if selected_lanes is None:
        selected_lanes = torch.ones(nmol, nselected_roots, dtype=torch.bool, device=amplitudes.device)
    root_index = torch.arange(nroots, device=amplitudes.device)
    active = root_index.unsqueeze(0) < nroots_target.unsqueeze(1)
    scale = (evals.abs() * active).amax(dim=1, keepdim=True)
    threshold = max(_DEGENERACY_ATOL, 10.0 * root_tol) + _DEGENERACY_RTOL * scale
    target_evals = torch.gather(evals, 1, root_indices)
    cluster = (
        selected_lanes.unsqueeze(-1)
        & active[:, None, :]
        & (torch.abs(evals[:, None, :] - target_evals[:, :, None]) <= threshold[:, :, None])
    )

    order = torch.argsort((~cluster).to(torch.int64), dim=2, stable=True)
    candidates = amplitudes[:, None, :, :].expand(-1, nselected_roots, -1, -1)
    protected = torch.gather(candidates, 2, order.unsqueeze(-1).expand(-1, -1, -1, nov))
    counts = cluster.sum(dim=2)
    keep = torch.arange(nroots, device=amplitudes.device).view(1, 1, -1) < counts.unsqueeze(-1)
    protected = protected * keep.unsqueeze(-1) * valid_mask[:, None, None, :]
    protected, counts = _orthonormalize_protected_basis(
        protected.reshape(nmol * nselected_roots, nroots, nov),
        counts.reshape(-1),
        valid_mask[:, None, :].expand(-1, nselected_roots, -1).reshape(nmol * nselected_roots, nov),
    )
    return protected.reshape(nmol, nselected_roots, nroots, nov), counts.reshape(nmol, nselected_roots)


def _orthonormalize_protected_basis(protected, counts, valid_mask):
    """Reorthogonalize protected eigenvectors before using them as a projector."""

    out = torch.zeros_like(protected)
    new_counts = torch.zeros_like(counts)
    rank_scale = 10.0 * max(protected.shape[-2:]) * torch.finfo(protected.dtype).eps
    for local_idx in range(protected.shape[0]):
        basis = protected[local_idx, : counts[local_idx]] * valid_mask[local_idx]
        if not basis.abs().any():
            continue
        _, singular_values, right_vectors = torch.linalg.svd(basis, full_matrices=False)
        rank = int((singular_values > rank_scale * singular_values[0]).sum().item())
        out[local_idx, :rank] = right_vectors[:rank] * valid_mask[local_idx]
        new_counts[local_idx] = rank
    return out, new_counts


def _solve_projected_system(hamiltonian, rhs):
    """Solve tiny symmetric Galerkin systems, tolerating near-singular clusters."""

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.transpose(-1, -2))
    try:
        return torch.linalg.solve(hamiltonian, rhs.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(hamiltonian, rhs.unsqueeze(-1)).solution.squeeze(-1)


def _project(vectors, basis, valid_mask):
    scale = valid_mask if vectors.dim() == 2 else valid_mask.unsqueeze(1)
    out = vectors * scale
    if basis.shape[1] == 0:
        return out
    for _ in range(2):
        if vectors.dim() == 2:
            out -= torch.einsum("bp,bpn->bn", torch.einsum("bpn,bn->bp", basis, out), basis)
        else:
            out -= torch.einsum("bmp,bpn->bmn", torch.einsum("bpn,bmn->bmp", basis, out), basis)
    return out * scale


def _orthogonalize(vector, basis, valid_mask, tol=1.0e-12):
    out = vector * valid_mask
    if basis.numel():
        for _ in range(2):
            out -= torch.einsum("p,pn->n", basis @ out, basis)
    norm = torch.linalg.vector_norm(out)
    return None if norm <= tol else out / norm


def _precondition(residual, diagonal, valid_mask):
    floor = 1.0e-8
    safe_diagonal = torch.where(diagonal.abs() < floor, torch.where(diagonal < 0, -floor, floor), diagonal)
    return torch.where(valid_mask, residual / safe_diagonal, torch.zeros_like(residual))


def _active_root_mask(nroots_target):
    return (
        torch.arange(int(nroots_target.max().item()), device=nroots_target.device).unsqueeze(0)
        < nroots_target[:, None]
    )


def rcis_instability_mask(evals, nroots_target):
    """Return the per-molecule negative-root mask without synchronizing to Python."""
    active = _active_root_mask(nroots_target)
    return (active & ((evals < 0.0) | ~torch.isfinite(evals))).any(dim=1)


def validate_rcis_stability(evals, nroots_target):
    """Reject materially negative CIS roots from an unstable SCF stationary point."""

    active = _active_root_mask(nroots_target)
    unstable = active & ((evals < 0.0) | ~torch.isfinite(evals))
    if unstable.any():
        mol_idx, root_idx = torch.nonzero(unstable, as_tuple=True)
        values = evals[mol_idx, root_idx]
        details = ", ".join(
            f"mol {int(mol)} root {int(root)}: {value:.3e}"
            for mol, root, value in zip(mol_idx.tolist(), root_idx.tolist(), values.tolist())
        )
        raise RuntimeError(
            "RCIS stability failure: negative excitation root(s) indicate an unstable SCF stationary point; "
            + details
        )


def _zero_if_none(gradient, reference):
    return torch.zeros_like(reference) if gradient is None else gradient
