import torch

from .cg_solver import conjugate_gradient_batch
from .orthogonalization import orthogonalize_to_current_subspace

DEFAULT_RCIS_DEGENERACY_TOL = 1.0e-6
DEFAULT_RCIS_ADJOINT_REG = 1.0e-10
DEFAULT_RCIS_ADJOINT_MAXITER = 200
DEFAULT_RCIS_DENSE_ADJOINT_THRESHOLD = 20
DEFAULT_RCIS_ADJOINT_SOLVER = "cg_squared"


def canonicalize_eigenvectors(vectors, active_root_mask=None):
    if vectors.numel() == 0:
        return vectors

    pivot = torch.argmax(vectors.abs(), dim=-1, keepdim=True)
    pivot_val = vectors.gather(-1, pivot)
    sign = torch.where(pivot_val < 0.0, -torch.ones_like(pivot_val), torch.ones_like(pivot_val))
    out = vectors * sign
    if active_root_mask is not None:
        out = out * active_root_mask.unsqueeze(-1)
    return out


def build_dense_cis_matrix(problem, chunk_size=None):
    nmol = problem["nmol"]
    nov = problem["nov"]
    eye = torch.eye(nov, dtype=problem["dtype"], device=problem["device"])
    dense = torch.zeros(nmol, nov, nov, dtype=problem["dtype"], device=problem["device"])
    step = nov if chunk_size is None else max(1, int(chunk_size))

    for start in range(0, nov, step):
        end = min(start + step, nov)
        vec = eye[start:end].unsqueeze(0).expand(nmol, -1, -1)
        avec = problem["apply"](vec)
        dense[:, :, start:end] = avec.transpose(1, 2)

    return dense


def davidson_solve(problem, nroots, root_tol, best_guess_from_prev=True, init_amplitude_guess=None):
    device = problem["device"]
    dtype = problem["dtype"]
    nmol = problem["nmol"]
    nov = problem["nov"]
    max_subspace = problem["max_subspace_size"]
    apply = problem["apply"]
    initialize_subspace = problem["initialize_subspace"]
    orthogonalize = problem["orthogonalize"]
    precond_diag = problem["precond_diag"]
    max_iter = int(problem.get("max_iter", 100))
    canonicalize = problem.get("canonicalize", False)

    V = torch.zeros(nmol, max_subspace, nov, device=device, dtype=dtype)
    HV = torch.zeros_like(V)

    vend, nroots_target, vector_tol = initialize_subspace(
        V=V,
        nroots=nroots,
        max_subspace_size=max_subspace,
        root_tol=root_tol,
        best_guess_from_prev=best_guess_from_prev,
        init_amplitude_guess=init_amplitude_guess,
    )

    vstart = torch.zeros(nmol, dtype=torch.long, device=device)
    done = torch.zeros(nmol, dtype=torch.bool, device=device)
    nroots_max = int(nroots_target.max().item())
    active_root_mask = torch.arange(nroots_max, device=device).unsqueeze(0) < nroots_target.unsqueeze(1)

    evals = torch.zeros(nmol, nroots_max, dtype=dtype, device=device)
    amplitudes_out = torch.zeros(nmol, nroots_max, nov, dtype=dtype, device=device)
    n_collapses = torch.zeros_like(vstart)
    n_iters = torch.zeros_like(vstart)

    davidson_iter = 0
    while davidson_iter <= max_iter:
        delta = vend - vstart
        max_v = int(delta.max().item())
        rel_idx = torch.arange(max_v, device=device).unsqueeze(0)
        abs_idx = rel_idx + vstart.unsqueeze(1)
        mask = rel_idx < delta.unsqueeze(1)
        batch_idx = torch.arange(nmol, device=device).unsqueeze(1).expand(-1, max_v)

        V_batched = torch.zeros(nmol, max_v, nov, dtype=dtype, device=device)
        V_batched[mask] = V[batch_idx[mask], abs_idx[mask], :]

        HV_batch = apply(V_batched)
        HV[batch_idx[mask], abs_idx[mask], :] = HV_batch[mask]

        vend_max = int(vend.max().item())
        H = torch.einsum("bno,bro->bnr", V[:, :vend_max], HV[:, :vend_max])

        davidson_iter += 1

        zero_pad = vend_max - vend
        e_vec_n = _get_subspace_eig(H, nroots_target, nroots_max, zero_pad, evals, done)

        amplitudes = torch.einsum("bvr,bvo->bro", e_vec_n, V[:, :vend_max, :])
        residual = torch.einsum("bvr,bvo->bro", e_vec_n, HV[:, :vend_max, :]) - amplitudes * evals.unsqueeze(
            2
        )
        residual = residual * active_root_mask.unsqueeze(-1)
        resid_norm = torch.linalg.vector_norm(residual, dim=2, ord=torch.inf)
        roots_not_converged = (resid_norm > root_tol) & active_root_mask

        mol_converged = roots_not_converged.sum(dim=1) == 0
        done_this_loop = (~done) & mol_converged
        done[done_this_loop] = True
        n_iters[done_this_loop] = davidson_iter
        amplitudes_out[done_this_loop] = amplitudes[done_this_loop]

        collapse_condition = (roots_not_converged.sum(dim=1) + vend > max_subspace) & (max_subspace != nov)
        collapse_mask = (~done) & (~mol_converged) & collapse_condition
        if collapse_mask.any():
            if davidson_iter == 1:
                raise RuntimeError("Insufficient memory to perform even a single Davidson expansion step")

            V[collapse_mask] = 0.0
            collapse_targets = nroots_target[collapse_mask]
            kmax = int(collapse_targets.max().item())
            root_mask = torch.arange(kmax, device=device).unsqueeze(0) < collapse_targets.unsqueeze(1)

            amp_block = amplitudes[collapse_mask, :kmax, :]
            hv_block = torch.einsum(
                "bvk,bvo->bko", e_vec_n[collapse_mask, :, :kmax], HV[collapse_mask, :vend_max, :]
            )

            V_c = V[collapse_mask, :kmax, :]
            HV_c = HV[collapse_mask, :kmax, :]
            mask3 = root_mask.unsqueeze(-1)
            V_c[mask3.expand_as(V_c)] = amp_block[mask3.expand_as(amp_block)]
            HV_c[mask3.expand_as(HV_c)] = hv_block[mask3.expand_as(hv_block)]
            V[collapse_mask, :kmax, :] = V_c
            HV[collapse_mask, :kmax, :] = HV_c
            vend[collapse_mask] = collapse_targets
            vstart[collapse_mask] = 0
            n_collapses[collapse_mask] += 1

        orthogonalize_mask = (~done) & (~mol_converged)
        for i in torch.nonzero(orthogonalize_mask, as_tuple=False).squeeze(1):
            unconverged = roots_not_converged[i]
            if not unconverged.any():
                continue

            denom = evals[i, unconverged].unsqueeze(1) - precond_diag[i].unsqueeze(0)
            newsubspace = residual[i, unconverged, :] / denom
            vstart[i] = vend[i]
            vend[i] = orthogonalize(V[i], newsubspace, int(vend[i].item()), vector_tol[i])
            if vend[i] - vstart[i] == 0:
                done[i] = True
                amplitudes_out[i] = amplitudes[i]
                n_iters[i] = davidson_iter

        if done.all():
            break

        if davidson_iter > max_iter:
            raise RuntimeError("Maximum Davidson iterations reached but RCIS roots have not converged")

    if canonicalize:
        amplitudes_out = canonicalize_eigenvectors(amplitudes_out, active_root_mask)
    info = {
        "nroots_target": nroots_target,
        "active_root_mask": active_root_mask,
        "n_iters": n_iters,
        "n_collapses": n_collapses,
    }
    return evals, amplitudes_out, info


def solve_rcis(
    spec,
    molecular_orbitals,
    e_mo,
    w,
    gss,
    gpp,
    gsp,
    gp2,
    hsp,
    nroots,
    root_tol,
    best_guess_from_prev=True,
    init_amplitude_guess=None,
):
    inputs = (molecular_orbitals, e_mo, w, gss, gpp, gsp, gp2, hsp)
    needs_grad = any(torch.is_tensor(x) and x.requires_grad for x in inputs)
    differentiable = torch.is_grad_enabled() and needs_grad and spec.get("differentiable", True)

    if differentiable:
        spec_with_phase = dict(spec)
        spec_with_phase["canonicalize"] = True
        evals, amplitudes, nroots_target = _DifferentiableRCIS.apply(
            spec_with_phase,
            int(nroots),
            float(root_tol),
            bool(best_guess_from_prev),
            init_amplitude_guess,
            *inputs,
        )
        info = {
            "nroots_target": nroots_target,
            "active_root_mask": torch.arange(
                int(nroots_target.max().item()), device=nroots_target.device
            ).unsqueeze(0)
            < nroots_target.unsqueeze(1),
            "n_iters": None,
            "n_collapses": None,
        }
        return evals, amplitudes, info

    problem = spec["build_problem"](*inputs)
    if "canonicalize" in spec:
        problem["canonicalize"] = spec["canonicalize"]
    evals, amplitudes, info = davidson_solve(
        problem,
        nroots=nroots,
        root_tol=root_tol,
        best_guess_from_prev=best_guess_from_prev,
        init_amplitude_guess=init_amplitude_guess,
    )
    return evals, amplitudes, info


class _DifferentiableRCIS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        spec,
        nroots,
        root_tol,
        best_guess_from_prev,
        init_amplitude_guess,
        molecular_orbitals,
        e_mo,
        w,
        gss,
        gpp,
        gsp,
        gp2,
        hsp,
    ):
        problem = spec["build_problem"](molecular_orbitals, e_mo, w, gss, gpp, gsp, gp2, hsp)
        if "canonicalize" in spec:
            problem["canonicalize"] = spec["canonicalize"]
        evals, amplitudes, info = davidson_solve(
            problem,
            nroots=nroots,
            root_tol=root_tol,
            best_guess_from_prev=best_guess_from_prev,
            init_amplitude_guess=init_amplitude_guess,
        )

        ctx.spec = spec
        ctx.root_tol = root_tol
        ctx.nroots = nroots
        ctx.best_guess_from_prev = best_guess_from_prev
        ctx.init_amplitude_guess = init_amplitude_guess
        ctx.save_for_backward(
            amplitudes,
            evals,
            info["nroots_target"],
            info["active_root_mask"],
            problem["valid_transition_mask"],
            problem["precond_diag"],
            molecular_orbitals,
            e_mo,
            w,
            gss,
            gpp,
            gsp,
            gp2,
            hsp,
        )
        return evals, amplitudes, info["nroots_target"]

    @staticmethod
    def backward(ctx, grad_evals, grad_amplitudes, grad_nroots_target):
        del grad_nroots_target
        (
            amplitudes,
            evals,
            nroots_target,
            active_root_mask,
            valid_transition_mask,
            precond_diag,
            molecular_orbitals,
            e_mo,
            w,
            gss,
            gpp,
            gsp,
            gp2,
            hsp,
        ) = ctx.saved_tensors

        grad_evals = _default_like(grad_evals, evals) * active_root_mask
        grad_amplitudes = _default_like(grad_amplitudes, amplitudes)
        grad_amplitudes = (
            grad_amplitudes * active_root_mask.unsqueeze(-1) * valid_transition_mask.unsqueeze(1)
        )

        inputs = (molecular_orbitals, e_mo, w, gss, gpp, gsp, gp2, hsp)
        detached_inputs = []
        grad_inputs = []
        for tensor in inputs:
            if tensor.requires_grad:
                detached = tensor.detach().requires_grad_(True)
                detached_inputs.append(detached)
                grad_inputs.append(detached)
            else:
                detached_inputs.append(tensor.detach())

        with torch.enable_grad():
            problem = ctx.spec["build_problem"](*detached_inputs)
            for key in (
                "adjoint_solver",
                "adjoint_max_subspace",
                "adjoint_precond_min_abs",
                "adjoint_collapse",
                "adjoint_dense_regularization",
            ):
                if key in ctx.spec:
                    problem[key] = ctx.spec[key]

            adjoint = torch.zeros_like(amplitudes)
            dense_A = None
            with torch.no_grad():
                adjoint_solver = str(ctx.spec.get("adjoint_solver", DEFAULT_RCIS_ADJOINT_SOLVER)).lower()
                degeneracy_tol = float(ctx.spec.get("degeneracy_tol", DEFAULT_RCIS_DEGENERACY_TOL))
                adjoint_tol = float(ctx.spec.get("adjoint_tol", max(ctx.root_tol, 1.0e-8)))
                adjoint_max_iter = int(ctx.spec.get("adjoint_max_iter", DEFAULT_RCIS_ADJOINT_MAXITER))
                adjoint_reg = float(ctx.spec.get("adjoint_regularization", DEFAULT_RCIS_ADJOINT_REG))
                dense_threshold = int(
                    ctx.spec.get("dense_adjoint_threshold", DEFAULT_RCIS_DENSE_ADJOINT_THRESHOLD)
                )
                if problem["nov"] <= dense_threshold:
                    dense_A = build_dense_cis_matrix(problem)
                nroots_max = active_root_mask.shape[1]
                for root_idx in range(nroots_max):
                    root_mask = active_root_mask[:, root_idx]
                    if not root_mask.any():
                        continue

                    grad_mask = torch.any(grad_amplitudes[:, root_idx].abs() > 0.0, dim=1)
                    solve_mask = root_mask & grad_mask
                    if not solve_mask.any():
                        continue

                    mol_indices = torch.nonzero(solve_mask, as_tuple=False).squeeze(1)
                    rhs = grad_amplitudes[mol_indices, root_idx]
                    omega = evals[mol_indices, root_idx]
                    valid_mask = valid_transition_mask[mol_indices]
                    diag = precond_diag[mol_indices]

                    protected_list = []
                    protected_counts = []
                    for local_idx, mol_idx in enumerate(mol_indices.tolist()):
                        cluster_mask = _degeneracy_cluster(
                            evals[mol_idx, : nroots_target[mol_idx]], root_idx, tol=degeneracy_tol
                        )
                        protected = amplitudes[mol_idx, : nroots_target[mol_idx]][cluster_mask]
                        protected = _orthonormalize_protected_basis(
                            protected, valid_mask[local_idx], tol=1.0e-12
                        )
                        protected_list.append(protected)
                        protected_counts.append(protected.shape[0])

                    protected_counts = torch.tensor(protected_counts, device=rhs.device, dtype=torch.long)
                    max_protected = int(protected_counts.max().item()) if protected_counts.numel() > 0 else 0
                    protected_basis = rhs.new_zeros((mol_indices.numel(), max_protected, rhs.shape[1]))
                    for local_idx, protected in enumerate(protected_list):
                        if protected.shape[0] > 0:
                            protected_basis[local_idx, : protected.shape[0]] = protected

                    if dense_A is not None:
                        for local_idx, mol_idx in enumerate(mol_indices.tolist()):
                            adjoint[mol_idx, root_idx] = _solve_adjoint_vector_dense(
                                dense_A=dense_A[mol_idx],
                                rhs=_project_out(
                                    rhs[local_idx], protected_list[local_idx], valid_mask[local_idx]
                                ),
                                omega=omega[local_idx],
                                protected_basis=protected_list[local_idx],
                                valid_mask=valid_mask[local_idx],
                                tol=adjoint_tol,
                            )
                        continue

                    if adjoint_solver == "davidson":
                        adjoint_root = _solve_adjoint_vectors_davidson(
                            problem=problem,
                            mol_indices=mol_indices,
                            rhs=rhs,
                            omega=omega,
                            protected_basis=protected_basis,
                            protected_counts=protected_counts,
                            valid_mask=valid_mask,
                            diag=diag,
                            tol=adjoint_tol,
                            max_iter=adjoint_max_iter,
                            max_subspace=int(
                                problem.get("adjoint_max_subspace", problem["max_subspace_size"])
                            ),
                            precond_min_abs=float(problem.get("adjoint_precond_min_abs", 1.0e-8)),
                            collapse=bool(problem.get("adjoint_collapse", True)),
                            dense_regularization=float(problem.get("adjoint_dense_regularization", 0.0)),
                        )
                    else:
                        adjoint_root = _solve_adjoint_vectors_cg_squared(
                            problem=problem,
                            mol_indices=mol_indices,
                            rhs=rhs,
                            omega=omega,
                            protected_basis=protected_basis,
                            valid_mask=valid_mask,
                            diag=diag,
                            tol=adjoint_tol,
                            max_iter=adjoint_max_iter,
                            reg=adjoint_reg,
                        )
                    adjoint[mol_indices, root_idx] = adjoint_root

            cotangent = grad_evals.unsqueeze(-1) * amplitudes - adjoint
            cotangent = cotangent * active_root_mask.unsqueeze(-1) * valid_transition_mask.unsqueeze(1)

            Ax = problem["apply"](amplitudes.detach())
            scalar = torch.sum(cotangent * Ax)
            grads = torch.autograd.grad(scalar, grad_inputs, allow_unused=True)

        grad_map = {}
        grad_iter = iter(grads)
        for idx, tensor in enumerate(inputs):
            grad_map[idx] = next(grad_iter) if tensor.requires_grad else None

        return (
            None,
            None,
            None,
            None,
            None,
            grad_map[0],
            grad_map[1],
            grad_map[2],
            grad_map[3],
            grad_map[4],
            grad_map[5],
            grad_map[6],
            grad_map[7],
        )


def _default_like(grad, ref):
    if grad is None:
        return torch.zeros_like(ref)
    return grad


def _get_subspace_eig(H, nroots_target, nroots_max, zero_pad, evals, done):
    r_eval, r_evec = torch.linalg.eigh(H[~done])
    nmol, subspace = H.shape[0], H.shape[1]
    e_vec_n = torch.zeros(nmol, subspace, nroots_max, device=H.device, dtype=H.dtype)
    active_indices = torch.nonzero(~done, as_tuple=False).squeeze(1)
    if active_indices.numel() == 0:
        return e_vec_n

    gather_cols = zero_pad[active_indices].unsqueeze(1) + torch.arange(nroots_max, device=H.device).unsqueeze(
        0
    )
    gathered_evals = torch.gather(r_eval, 1, gather_cols)
    gathered_evecs = torch.gather(r_evec, 2, gather_cols.unsqueeze(1).expand(-1, subspace, -1))
    evals[active_indices, :nroots_max] = gathered_evals
    e_vec_n[active_indices] = gathered_evecs

    return e_vec_n


def _degeneracy_cluster(evals, root_idx, tol):
    ref = evals[root_idx]
    return torch.abs(evals - ref) <= tol


def _project_out(v, basis, valid_mask):
    out = v * valid_mask
    if basis.numel() == 0:
        return out
    for _ in range(2):
        coeff = torch.einsum("dn,n->d", basis, out)
        out = out - torch.einsum("d,dn->n", coeff, basis)
    return out * valid_mask


def _project_out_batch(v, basis, valid_mask):
    if v.dim() == 2:
        out = v * valid_mask
        if basis.numel() == 0:
            return out
        coeff = torch.einsum("bpn,bn->bp", basis, out)
        out = out - torch.einsum("bp,bpn->bn", coeff, basis)
        return out * valid_mask

    out = v * valid_mask.unsqueeze(1)
    if basis.numel() == 0:
        return out
    coeff = torch.einsum("bpn,bmn->bmp", basis, out)
    out = out - torch.einsum("bmp,bpn->bmn", coeff, basis)
    return out * valid_mask.unsqueeze(1)


def _safe_precondition(residual, diag_shift, valid_mask, min_abs=1.0e-8):
    denom = diag_shift.clone()
    small = denom.abs() < min_abs
    denom = torch.where(
        small, min_abs * torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom)), denom
    )
    out = residual / denom
    return torch.where(valid_mask, out, torch.zeros_like(out))


def _orthogonalize_one(v, basis, valid_mask, tol=1.0e-12):
    vec = (v * valid_mask).unsqueeze(0)
    if basis is None or basis.numel() == 0:
        norm = torch.linalg.vector_norm(vec[0])
        if norm <= tol:
            return None
        return vec[0] / norm

    workspace = torch.zeros(basis.shape[0] + 1, basis.shape[1], dtype=basis.dtype, device=basis.device)
    workspace[: basis.shape[0]] = basis
    vend = orthogonalize_to_current_subspace(workspace, vec, basis.shape[0], tol)
    if vend == basis.shape[0]:
        return None
    return workspace[basis.shape[0]]


def _orthonormalize_protected_basis(protected_basis, valid_mask, tol=1.0e-12):
    if protected_basis.numel() == 0:
        return protected_basis

    vecs = []
    for i in range(protected_basis.shape[0]):
        vec = protected_basis[i] * valid_mask
        if vecs:
            basis = torch.stack(vecs, dim=0)
            vec = _project_out(vec, basis, valid_mask)
        norm = torch.linalg.vector_norm(vec)
        if norm > tol:
            vecs.append(vec / norm)

    if not vecs:
        return protected_basis.new_zeros((0, protected_basis.shape[-1]))
    return torch.stack(vecs, dim=0)


def _solve_adjoint_vectors_cg_squared(
    problem, mol_indices, rhs, omega, protected_basis, valid_mask, diag, tol, max_iter, reg
):
    rhs = _project_out_batch(rhs, protected_basis, valid_mask)
    if not torch.any(rhs.abs() > 0.0):
        return torch.zeros_like(rhs)

    def apply_B(v):
        vec = _project_out_batch(v, protected_basis, valid_mask)
        Av = _apply_selected(problem, mol_indices, vec)
        return _project_out_batch(Av - omega.unsqueeze(1) * vec, protected_basis, valid_mask)

    def apply_B2(v):
        vec = _project_out_batch(v, protected_basis, valid_mask)
        return apply_B(apply_B(vec)) + reg * vec

    diag_shift = ((diag - omega.unsqueeze(1)) ** 2 + reg).clamp_min(1.0e-12)
    diag_shift = torch.where(valid_mask, diag_shift, torch.ones_like(diag_shift))

    z = conjugate_gradient_batch(apply_B2, rhs, M_diag=diag_shift, max_iter=max_iter, tol=tol)
    return apply_B(z)


def _solve_adjoint_vectors_davidson(
    problem,
    mol_indices,
    rhs,
    omega,
    protected_basis,
    protected_counts,
    valid_mask,
    diag,
    tol,
    max_iter,
    max_subspace,
    precond_min_abs,
    collapse,
    dense_regularization,
):
    batch_size, nov = rhs.shape
    rhs = _project_out_batch(rhs, protected_basis, valid_mask)
    if not torch.any(rhs.abs() > 0.0):
        return torch.zeros_like(rhs)

    diag_shift = torch.where(valid_mask, diag - omega.unsqueeze(1), torch.ones_like(diag))
    subspace_capacity = max_subspace if collapse else nov

    def project_batch(local_batch_idx, vecs):
        return _project_out_batch(vecs, protected_basis[local_batch_idx], valid_mask[local_batch_idx])

    def apply_projected_batch(local_batch_idx, vecs):
        if local_batch_idx.numel() == 0:
            return vecs
        vecs_proj = project_batch(local_batch_idx, vecs)
        Av = _apply_selected(problem, mol_indices[local_batch_idx], vecs_proj)
        if vecs_proj.dim() == 3:
            shifted = Av - omega[local_batch_idx].view(-1, 1, 1) * vecs_proj
        else:
            shifted = Av - omega[local_batch_idx].unsqueeze(1) * vecs_proj
        return project_batch(local_batch_idx, shifted)

    b_norm = torch.linalg.vector_norm(rhs, ord=torch.inf, dim=1)
    done = b_norm <= tol
    best_z = torch.zeros_like(rhs)
    best_res_norm = b_norm.clone()
    V = torch.zeros(batch_size, subspace_capacity, nov, dtype=rhs.dtype, device=rhs.device)
    BV = torch.zeros_like(V)
    vend = torch.zeros(batch_size, dtype=torch.long, device=rhs.device)
    vstart = torch.zeros_like(vend)

    for i in torch.nonzero(~done, as_tuple=False).squeeze(1).tolist():
        basis = protected_basis[i, : int(protected_counts[i].item())]
        g0 = _safe_precondition(rhs[i], diag_shift[i], valid_mask[i], min_abs=precond_min_abs)
        v0 = _orthogonalize_one(g0, basis, valid_mask[i], tol=1.0e-12)
        if v0 is None:
            v0 = _orthogonalize_one(rhs[i], basis, valid_mask[i], tol=1.0e-12)
        if v0 is None:
            done[i] = True
            continue
        V[i, 0] = v0
        vend[i] = 1

    eye_cache = {}
    davidson_iter = 0
    while davidson_iter <= max_iter:
        if done.all():
            return best_z

        delta = vend - vstart
        max_v = int(delta.max().item())
        rel_idx = torch.arange(max_v, device=rhs.device).unsqueeze(0)
        abs_idx = rel_idx + vstart.unsqueeze(1)
        update_mask = rel_idx < delta.unsqueeze(1)
        batch_idx = torch.arange(batch_size, device=rhs.device).unsqueeze(1).expand(-1, max_v)

        V_batched = torch.zeros(batch_size, max_v, nov, dtype=rhs.dtype, device=rhs.device)
        V_batched[update_mask] = V[batch_idx[update_mask], abs_idx[update_mask], :]
        BV_batch = apply_projected_batch(torch.arange(batch_size, device=rhs.device), V_batched)
        BV[batch_idx[update_mask], abs_idx[update_mask], :] = BV_batch[update_mask]

        vend_max = int(vend.max().item())
        S = torch.einsum("bno,bro->bnr", V[:, :vend_max], BV[:, :vend_max])
        S = 0.5 * (S + S.transpose(1, 2))
        small_rhs = torch.einsum("bno,bo->bn", V[:, :vend_max], rhs)

        davidson_iter += 1

        coeff = torch.zeros(batch_size, vend_max, dtype=rhs.dtype, device=rhs.device)
        active_idx = torch.nonzero(~done, as_tuple=False).squeeze(1)
        active_sizes = vend[active_idx]
        for size in torch.unique(active_sizes).tolist():
            if size == 0:
                continue
            size_mask = active_sizes == size
            group_idx = active_idx[size_mask]
            S_block = S[group_idx, :size, :size]
            rhs_block = small_rhs[group_idx, :size]
            if dense_regularization != 0.0:
                if size not in eye_cache:
                    eye_cache[size] = torch.eye(size, dtype=rhs.dtype, device=rhs.device)
                S_block = S_block + dense_regularization * eye_cache[size].unsqueeze(0)
            try:
                coeff_block = torch.linalg.solve(S_block, rhs_block.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                coeff_block = torch.linalg.lstsq(S_block, rhs_block.unsqueeze(-1)).solution.squeeze(-1)
            coeff[group_idx, :size] = coeff_block

        z = torch.einsum("bn,bno->bo", coeff, V[:, :vend_max])
        Bz = torch.einsum("bn,bno->bo", coeff, BV[:, :vend_max])
        residual = _project_out_batch(rhs - Bz, protected_basis, valid_mask)
        res_norm = torch.linalg.vector_norm(residual, ord=torch.inf, dim=1)

        active_before = ~done
        best_z[active_before] = z[active_before]
        best_res_norm[active_before] = res_norm[active_before]

        converged = active_before & (res_norm <= tol * torch.maximum(torch.ones_like(b_norm), b_norm))
        done[converged] = True

        collapse_mask = (~done) & collapse & (vend >= subspace_capacity)
        for i in torch.nonzero(collapse_mask, as_tuple=False).squeeze(1).tolist():
            basis = protected_basis[i, : int(protected_counts[i].item())]
            z_normed = _orthogonalize_one(
                _project_out(z[i], basis, valid_mask[i]), basis, valid_mask[i], tol=1.0e-12
            )
            r_prec = _safe_precondition(residual[i], diag_shift[i], valid_mask[i], min_abs=precond_min_abs)
            r_prec = _project_out(r_prec, basis, valid_mask[i])
            if z_normed is None:
                z_normed = _orthogonalize_one(rhs[i], basis, valid_mask[i], tol=1.0e-12)

            V[i] = 0.0
            BV[i] = 0.0
            vend[i] = 0
            vstart[i] = 0

            if z_normed is not None:
                V[i, 0] = z_normed
                vend[i] = 1

            basis_for_r = basis if vend[i] == 0 else torch.cat([basis, V[i, : vend[i]]], dim=0)
            r_new = _orthogonalize_one(r_prec, basis_for_r, valid_mask[i], tol=1.0e-12)
            if r_new is not None:
                V[i, vend[i]] = r_new
                vend[i] += 1

            if vend[i] == 0:
                done[i] = True

        orthogonalize_mask = (~done) & (~collapse_mask)
        for i in torch.nonzero(orthogonalize_mask, as_tuple=False).squeeze(1).tolist():
            if vend[i] >= subspace_capacity:
                done[i] = True
                continue

            correction = _safe_precondition(
                residual[i], diag_shift[i], valid_mask[i], min_abs=precond_min_abs
            )
            current_basis = V[i, : vend[i]]
            basis = protected_basis[i, : int(protected_counts[i].item())]
            if basis.numel() > 0:
                current_basis = torch.cat([basis, current_basis], dim=0)
            new_v = _orthogonalize_one(correction, current_basis, valid_mask[i], tol=1.0e-12)
            if new_v is None:
                done[i] = True
                continue
            vstart[i] = vend[i]
            V[i, vend[i]] = new_v
            vend[i] += 1

    raise RuntimeError(
        f"Davidson RCIS adjoint did not converge in {max_iter} iterations (resid={best_res_norm})"
    )


def _apply_selected(problem, mol_indices, vecs):
    squeeze = False
    if vecs.dim() == 2:
        vecs = vecs.unsqueeze(1)
        squeeze = True
    batch = torch.zeros(problem["nmol"], vecs.shape[1], problem["nov"], dtype=vecs.dtype, device=vecs.device)
    batch[mol_indices] = vecs
    out = problem["apply"](batch)[mol_indices]
    return out[:, 0] if squeeze else out


def _solve_adjoint_vector_dense(dense_A, rhs, omega, protected_basis, valid_mask, tol):
    idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    A = dense_A.index_select(0, idx).index_select(1, idx)
    rhs_v = rhs.index_select(0, idx)
    if protected_basis.numel() == 0:
        basis_v = protected_basis.new_zeros((0, idx.numel()))
    else:
        basis_v = protected_basis.index_select(1, idx)

    eye = torch.eye(idx.numel(), dtype=A.dtype, device=A.device)
    if basis_v.numel() == 0:
        projector = eye
    else:
        projector = eye - basis_v.transpose(0, 1) @ basis_v
    B = projector @ (A - omega * eye) @ projector

    evals, evecs = torch.linalg.eigh(B)
    coeff = evecs.transpose(0, 1) @ rhs_v
    inv = torch.zeros_like(evals)
    keep = torch.abs(evals) > max(tol, 1.0e-12)
    inv[keep] = evals[keep].reciprocal()
    y_v = evecs @ (inv * coeff)

    y = torch.zeros_like(rhs)
    y.index_copy_(0, idx, y_v)
    return y
