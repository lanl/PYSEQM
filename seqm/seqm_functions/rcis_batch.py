import math

import torch

from seqm.dynamics.active_state import active_state_tensor
from seqm.utils.profiling import record_runtime_diagnostic
from seqm.utils.torch_compile import optional_compile_function

from .constants import a0
from .dipole import calc_dipole_matrix
from .fock import UPPER_IDX0_4, UPPER_IDX1_4, WEIGHT_10, K_ind_4, _cached_index, _cached_tensor

# from seqm.seqm_functions.pack import packone, unpackone

_makeA_pi_batched_dispatch = None
_makeA_pi_symm_batch_dispatch = None
_ao_transition_density_dispatch = None
_mo_fock_action_dispatch = None
_relaxed_rhs_dispatch = None
_relaxed_finish_dispatch = None


def _uniform_molecule_dimensions(mol):
    """Return immutable uniform-molecule dimensions without repeated CUDA syncs."""
    cached = getattr(mol, "_rcis_uniform_dimensions", None)
    if cached is None:
        cached = (
            int(mol.nHeavy[0].item()),
            int(mol.nHydro[0].item()),
            int(mol.norb[0].item()),
            int(mol.nocc[0].item()),
        )
        mol._rcis_uniform_dimensions = cached
    return cached


def enable_rcis_compile(mode=None, **options):
    """Compile the repeatedly profitable CIS Davidson tensor contractions."""
    global _makeA_pi_batched_dispatch
    global _ao_transition_density_dispatch
    global _mo_fock_action_dispatch

    compile_options = dict(options)
    if mode is not None:
        compile_options["mode"] = mode

    _makeA_pi_batched_dispatch = optional_compile_function(
        _makeA_pi_batched_kernel, compile_options=compile_options, label="rcis.makeA_pi_batched"
    )
    _ao_transition_density_dispatch = optional_compile_function(
        _ao_transition_density_kernel, compile_options=compile_options, label="rcis.ao_transition_density"
    )
    _mo_fock_action_dispatch = optional_compile_function(
        _mo_fock_action_kernel, compile_options=compile_options, label="rcis.mo_fock_action"
    )


def rcis_batch(
    mol,
    w,
    e_mo,
    nroots,
    root_tol,
    best_guess_from_prev=True,
    init_amplitude_guess=None,
    orbital_window=None,
    save_tdm=False,
    compute_transition_properties=True,
):
    """Calculate the restricted Configuration Interaction Single (RCIS) excitation energies and amplitudes
       using davidson diagonalization
       This function is called when all the molecules in the batch are the same

    :param mol: Molecule Orbital Coefficients
    :param w: 2-electron integrals
    :param e_mo: Orbital energies
    :param nroots: Number of CIS states requested
    :param best_guess_from_prev: When running MD, you might want to use the amplitudes from previous step as guess, and fix the descrepancy b/w
                                 molecular orbital signs from previous and this step. Leave it alone when doing XL-ESMD
    :param save_tdm: save transition density matrices; this option will be used for XL-ESMD
    :param orbital_window: tuple (n,m) where n orbitals below the HOMO and m orbitals above LUMO are included in the active space
    :returns:

    """

    device = w.device
    dtype = w.dtype

    norb_batch, nocc_batch, nmol = mol.norb, mol.nocc, mol.nmol
    if nmol > 1 and (
        not torch.all(norb_batch == norb_batch[0]) or not torch.all(nocc_batch == nocc_batch[0])
    ):
        raise ValueError("All molecules in the batch must have the same number of orbitals and electrons")

    nocc, nvirt, Cocc, Cvirt, ea_ei = get_occ_virt(mol, orbital_window, e_mo)

    nov = nocc * nvirt
    if nroots > nov:
        raise Exception(f"Maximum number of roots for this molecule is {nov}.")

    # Energy differences
    # ea_ei = e_mo[:, virt_idx].unsqueeze(1) - e_mo[:, occ_idx].unsqueeze(2)

    # Precompute energy differences (ea_ei) and form the approximate diagonal of the Hamiltonian (approxH)
    # ea_ei contains the list of orbital energy difference between the virtual and occupied orbitals
    approxH = ea_ei.view(-1, nov)

    maxSubspacesize = getMaxSubspacesize(dtype, device, nov, nroots=nroots, nmol=nmol)  # TODO: User-defined

    V = torch.zeros(nmol, maxSubspacesize, nov, device=device, dtype=dtype)
    HV = torch.clone(V)

    vector_tol = root_tol * 0.01 * math.sqrt(nov)  # Vectors whose norm is smaller than this will be discarded

    if init_amplitude_guess is None:
        nstart, nroots = make_guess(approxH, nroots, maxSubspacesize, V, nmol, nov)
    else:
        if best_guess_from_prev:
            make_best_guess_from_previous_amplitudes(mol, init_amplitude_guess, V, nocc)
        else:
            if (
                init_amplitude_guess.shape[-1] == norb_batch[0]
            ):  # initial amplitude guess provided in AO basis, i.e. transition density matrices
                V[:, :nroots] = torch.einsum(
                    "bmi,brmn,bna->bria", Cocc, init_amplitude_guess[:, :nroots], Cvirt
                ).flatten(start_dim=-2)
                # need to orthonormalize, use the orthonormalization procedure below
                # Normalize first vector
                V[:, 0] /= torch.linalg.vector_norm(V[:, 0], dim=1, keepdim=True)
                if nroots > 1:
                    for i in range(nmol):
                        n_new = orthogonalize_to_current_subspace(V[i], V[i, 1:nroots], 1, vector_tol)
                        if n_new < nroots:
                            raise RuntimeError("Some roots were lost while orthogonalizing, cannot proceed")
            else:  # initial amplitude guess provided in MO basis, assume orthonormalized
                V[:, :nroots] = init_amplitude_guess[:, :nroots]
        nstart = nroots

        # # fix signs of the Molecular Orbitals by looking at the MOs from the previous step.
        # # This fails when orbitals are degenerate and switch order
        # mol.molecular_orbitals *= torch.sign((torch.einsum('Nmp,Nmp->Np',mol.molecular_orbitals,mol.old_mos))).unsqueeze(1)

    max_iter = int(mol.seqm_parameters.get("excited_states", {}).get("max_iter", 200))
    davidson_iter = 0
    vstart = torch.zeros(nmol, dtype=torch.long, device=device)
    vend = torch.full((nmol,), nstart, dtype=torch.long, device=device)
    done = torch.zeros(nmol, dtype=torch.bool, device=device)

    # TODO: Test if orthogonal or nonorthogonal version is more efficient
    nonorthogonal = False  # TODO: User-defined/fixed

    # C = mol.molecular_orbitals
    # Cocc = C[:,:,occ_idx]
    # Cvirt = C[:,:,virt_idx]

    e_val_n = torch.empty(nmol, nroots, dtype=dtype, device=device)
    amplitude_store = torch.empty(nmol, nroots, nov, dtype=dtype, device=device)

    n_collapses = torch.zeros_like(vstart)
    n_iters = torch.zeros_like(vstart)
    collapse_events = 0
    chunk_plan_cache = {}
    mol_idx = torch.arange(nmol, device=device)
    subspace_idx = torch.arange(maxSubspacesize, device=device)
    # header = f"{'Iteration':>10} | {'States Found':^15} | {'Total Error':>15}"
    # print("-" * len(header))
    # print(header)
    # print("-" * len(header))

    while davidson_iter <= max_iter:  # Davidson loop
        # Determine current subspace dimensions per molecule
        delta = vend - vstart
        delta[done] = 0
        max_v = int(delta.max().item())
        if max_v == 0:
            break
        rel_idx = subspace_idx[:max_v].unsqueeze(0)  # (1, max_v)
        abs_idx = rel_idx + vstart.unsqueeze(1)  # (nmol, max_v)
        mask = rel_idx < delta.unsqueeze(1)  # (nmol, max_v)
        batch_idx = mol_idx.unsqueeze(1).expand(-1, max_v)

        # Gather current subspace vectors into V_batched. If every active molecule
        # has max_v new vectors, avoid zero-padding and boolean scatter.
        dense_gather = nmol == 1 or bool(torch.all(mask).item())
        if dense_gather:
            V_batched = V[batch_idx, abs_idx, :]
        else:
            V_batched = torch.zeros(nmol, max_v, nov, dtype=dtype, device=device)
            V_batched[mask] = V[batch_idx[mask], abs_idx[mask], :]

        # Compute the matrix-vector product in the current subspace
        HV_batch = matrix_vector_product_batched(
            mol, V_batched, w, ea_ei, Cocc, Cvirt, chunk_plan_cache=chunk_plan_cache
        )
        if dense_gather:
            HV[batch_idx, abs_idx, :] = HV_batch
        else:
            HV[batch_idx[mask], abs_idx[mask], :] = HV_batch[mask]

        # Make H by multiplying V.T * HV
        vend_max = int(torch.max(vend).item())
        active_mask = ~done
        H = torch.empty(nmol, vend_max, vend_max, dtype=dtype, device=device)
        H[active_mask] = torch.einsum("bno,bro->bnr", V[active_mask, :vend_max], HV[active_mask, :vend_max])

        davidson_iter = davidson_iter + 1
        n_iters[active_mask] = davidson_iter

        # Diagonalize the subspace hamiltonian
        e_vec_n = get_subspace_eig_batched(H, nroots, vend, e_val_n, done, nonorthogonal)

        # Compute CIS amplitudes and the residual
        amplitudes = torch.einsum("bvr,bvo->bro", e_vec_n, V[:, :vend_max, :])
        residual = torch.einsum("bvr,bvo->bro", e_vec_n, HV[:, :vend_max, :])
        residual.addcmul_(amplitudes, e_val_n.unsqueeze(2), value=-1.0)
        # resid_norm = torch.norm(residual,dim=2)
        resid_norm = torch.linalg.vector_norm(residual, dim=2, ord=torch.inf)
        roots_not_converged = resid_norm > root_tol

        # Mark molecules with all roots converged and store amplitudes
        mol_converged = roots_not_converged.sum(dim=1) == 0
        done_this_loop = (~done) & mol_converged
        done[done_this_loop] = True
        n_iters[done_this_loop] = davidson_iter
        amplitude_store[done_this_loop] = amplitudes[done_this_loop]

        # Collapse the subspace for those molecules whose subspace will exceed maxSubspacesize
        collapse_condition = (roots_not_converged.sum(dim=1) + vend > maxSubspacesize) & (
            maxSubspacesize != nov
        )
        collapse_mask = (~done) & (~mol_converged) & collapse_condition
        if collapse_mask.sum() > 0:
            collapse_events += 1
            if davidson_iter == 1:
                raise Exception(
                    "Insufficient memory to perform even a single iteration of subspace expansion"
                )

            V[collapse_mask] = 0
            V[collapse_mask, :nroots, :] = amplitudes[collapse_mask]
            HV[collapse_mask, :nroots, :] = torch.einsum(
                "bvr,bvo->bro", e_vec_n[collapse_mask], HV[collapse_mask, :vend_max, :]
            )
            HV[collapse_mask, nroots:] = 0
            vstart[collapse_mask] = 0
            vend[collapse_mask] = nroots
            n_collapses[collapse_mask] += 1

        # Orthogonalize the residual vectors for molecules
        orthogonalize_mask = (~done) & (~mol_converged)  # & (~collapse_condition)
        mols_to_ortho = torch.nonzero(orthogonalize_mask).squeeze(1)
        for i in mols_to_ortho:
            newsubspace = residual[i, roots_not_converged[i], :] / (
                e_val_n[i, roots_not_converged[i]].unsqueeze(1) - approxH[i].unsqueeze(0)
            )

            vstart[i] = vend[i]
            # The original 'V' vector is passed by reference to the 'orthogonalize_to_current_subspace' function.
            # This means changes inside the function will directly modify 'V[i]'
            vend[i] = orthogonalize_to_current_subspace(V[i], newsubspace, vend[i], vector_tol)
            if vend[i] - vstart[i] == 0:
                done[i] = True
                amplitude_store[i] = amplitudes[i]
                n_iters[i] = davidson_iter

            # if davidson_iter % 5 == 0:
            # print(f"davidson_iteration {davidson_iter:2}: Found {nroots-roots_left}/{nroots} states, Total Error: {torch.sum(resid_norm[i]):.4e}")
            # states_found = f'{nroots-roots_left:3d}/{nroots:3d}'
            # print(f"{davidson_iter:10d} | {states_found:^15} | {total_error[i]:15.4e}")

        if torch.all(done):
            break
        if davidson_iter > max_iter:
            n_converged = torch.where(
                done, torch.full_like(n_iters, nroots), (~roots_not_converged).sum(dim=1)
            )
            info = []
            for j in range(nmol):
                missing = roots_not_converged[j] & ~done[j]
                max_err = resid_norm[j, missing].max().item() if missing.any() else 0.0
                errs = ", ".join(f"{x:.2e}" for x in resid_norm[j].detach().cpu().tolist())
                info.append(
                    f"mol {j}: {n_converged[j].item()}/{nroots} roots converged, "
                    f"max remaining error {max_err:.3e}, subspace {vend[j].item()}/{maxSubspacesize}, "
                    f"errors [{errs}], iters {n_iters[j].item()}, collapses {n_collapses[j].item()}"
                )
            raise Exception("Maximum iterations reached but roots have not converged; " + "; ".join(info))

    # print("-" * len(header))
    # print("\nCIS excited states:")
    # for j in range(nmol):
    #     if nmol>1: print(f"\nMolecule {j+1}")
    #     print(f"Number of davidson iterations: {n_iters[j]}, number of subspace collapses: {n_collapses[j]}")
    #     for i, energy in enumerate(e_val_n[j], start=1):
    #         print(f"State {i:3d}: {energy:.15f} eV")
    # print("")

    record_runtime_diagnostic("davidson", iterations=davidson_iter, collapse_events=collapse_events)

    # Post CIS analysis
    if mol.verbose:
        print(f"Number of davidson iterations: {n_iters}, number of subspace collapses: {n_collapses}")
    rcis_analysis(
        mol,
        e_val_n,
        amplitude_store,
        nroots,
        orbital_window=orbital_window,
        save_tdm=save_tdm,
        compute_transition_properties=compute_transition_properties,
    )

    return e_val_n, amplitude_store


def matrix_vector_product_batched(mol, V, w, ea_ei, Cocc, Cvirt, makeB=False, chunk_plan_cache=None):
    # C: Molecule Orbital Coefficients
    nmol, nNewRoots, _ = V.shape

    nocc = Cocc.shape[2]
    nvirt = Cvirt.shape[2]

    Via = V.view(nmol, nNewRoots, nocc, nvirt)

    # I often run out of memory because I calculate \sum_ia (ia||jb)V_ia in makeA_pi_batched for all the roots in V_ia
    # at once. To avoid this, we first estimate the peak memory usage and then chunk over nNewRoots.
    # TODO: Also chunk over the nmol dimension
    chunk_key = int(nNewRoots)
    if chunk_plan_cache is None:
        need_to_chunk, chunk_size = getMemUse(V.dtype, V.device, mol, nNewRoots)
    else:
        plan = chunk_plan_cache.get(chunk_key)
        if plan is None:
            plan = getMemUse(V.dtype, V.device, mol, nNewRoots)
            chunk_plan_cache[chunk_key] = plan
        need_to_chunk, chunk_size = plan
    make_density = _ao_transition_density_dispatch or _ao_transition_density_kernel
    make_post = _mo_fock_action_dispatch or _mo_fock_action_kernel

    if not need_to_chunk:
        P_xi = make_density(Cocc, Via, Cvirt)
        F0 = makeA_pi_batched(mol, P_xi, w)
        result = make_post(Via, F0, ea_ei, Cocc, Cvirt, bool(makeB), torch.is_grad_enabled())
        if makeB:
            A, B = result
        else:
            A = result
    else:
        A = torch.empty(nmol, nNewRoots, nocc, nvirt, device=V.device, dtype=V.dtype)
        if makeB:
            B = torch.empty_like(A)
        for start in range(0, nNewRoots, chunk_size):
            end = min(start + chunk_size, nNewRoots)
            Via_chunk = Via[:, start:end]
            P_xi = make_density(Cocc, Via_chunk, Cvirt)
            F0 = makeA_pi_batched(mol, P_xi, w)
            if makeB:
                A[:, start:end], B[:, start:end] = make_post(
                    Via_chunk, F0, ea_ei, Cocc, Cvirt, True, torch.is_grad_enabled()
                )
            else:
                A[:, start:end] = make_post(Via_chunk, F0, ea_ei, Cocc, Cvirt, False, torch.is_grad_enabled())
    A = A.reshape(nmol, nNewRoots, -1)

    if makeB:
        B = B.reshape(nmol, nNewRoots, -1)
        return A, B

    return A


def _ao_transition_density_kernel(Cocc, Via, Cvirt, symmetrize: bool = False):
    density = torch.einsum("bmi,bria,bna->brmn", Cocc, Via, Cvirt)
    if symmetrize:
        density = density + density.transpose(-1, -2)
    return density


def _mo_fock_action_kernel(Via, F0, ea_ei, Cocc, Cvirt, makeB: bool, grad_enabled: bool):
    A = torch.einsum("bmi,brmn,bna->bria", Cocc, F0, Cvirt) * 2.0
    if grad_enabled:
        A = A + Via * ea_ei.unsqueeze(1)
    else:
        A.addcmul_(Via, ea_ei.unsqueeze(1))
    if not makeB:
        return A
    B = torch.einsum("bmi,brnm,bna->bria", Cocc, F0, Cvirt) * 2.0
    return A, B


def _rcis_constant_tensors(dtype, device):
    tri_i = _cached_index(UPPER_IDX0_4, device)
    tri_j = _cached_index(UPPER_IDX1_4, device)
    weight = _cached_tensor(WEIGHT_10, device, dtype).reshape((-1, 10))
    ind = _cached_index(K_ind_4, device)
    return tri_i, tri_j, weight, ind


def makeA_pi_batched(mol, P_xi, w_, allSymmetric=False):
    r"""
    Given amplitudes in the AO basis, calculate \sum_jb (\mu\nu||jb)X_jb.
    """
    npairs_per_mol = (int(mol.molsize) * (int(mol.molsize) - 1)) // 2
    nmol = int(mol.nmol)
    molsize = int(mol.molsize)
    nHeavy, nHydro, norb, _ = _uniform_molecule_dimensions(mol)

    dispatch = _makeA_pi_batched_dispatch or _makeA_pi_batched_kernel
    nnewRoots = P_xi.shape[1]
    P0 = unpackone_batch(P_xi.reshape(nmol * nnewRoots, norb, norb), 4 * nHeavy, nHydro, molsize * 4).view(
        nmol, nnewRoots, 4 * molsize, 4 * molsize
    )
    F0 = dispatch(
        P0,
        w_.view(nmol, npairs_per_mol, 10, 10),
        mol.mask[:npairs_per_mol],
        mol.maskd[:molsize],
        mol.mask_l[:npairs_per_mol],
        mol.idxi[:npairs_per_mol],
        mol.idxj[:npairs_per_mol],
        mol.parameters["g_ss"].view(nmol, -1),
        mol.parameters["g_sp"].view(nmol, -1),
        mol.parameters["g_pp"].view(nmol, -1),
        mol.parameters["g_p2"].view(nmol, -1),
        mol.parameters["h_sp"].view(nmol, -1),
        molsize,
        nmol,
        bool(allSymmetric),
    )
    F0 = packone_batch(F0, 4 * nHeavy, nHydro, norb).view(nmol, nnewRoots, norb, norb)
    return F0.view(nmol, nnewRoots, norb, norb)


def _makeA_pi_batched_kernel(
    P0,
    w,
    mask,
    maskd,
    mask_l,
    idxi,
    idxj,
    gss,
    gsp,
    gpp,
    gp2,
    hsp,
    molsize: int,
    nmol: int,
    allSymmetric: bool = False,
):
    device = P0.device
    dtype = P0.dtype
    nnewRoots = P0.shape[1]
    F = _makeA_pi_symm_batch_kernel(
        P0, w, mask, maskd, mask_l, idxi, idxj, gss, gsp, gpp, gp2, hsp, molsize, nmol
    )

    if not allSymmetric:
        P0_blocks = P0.reshape(nmol, nnewRoots, molsize, 4, molsize, 4).transpose(3, 4)
        P_anti = 0.5 * (P0_blocks - P0_blocks.transpose(2, 3).transpose(4, 5))
        P_anti = P_anti.reshape(nmol, nnewRoots, molsize * molsize, 4, 4)
        del P0_blocks, P0

        _, _, _, ind = _rcis_constant_tensors(dtype, device)
        sumK = torch.empty(nmol, nnewRoots, w.shape[1], 4, 4, dtype=dtype, device=device)
        Pp = P_anti[:, :, mask]
        for i in range(4):
            for j in range(4):
                sumK[..., i, j] = -0.5 * torch.sum(
                    Pp * w[..., ind[i], :][..., :, ind[j]].unsqueeze(1), dim=(3, 4)
                )
        F.index_add_(2, mask, sumK)
        F[:, :, mask_l] -= sumK.transpose(3, 4)
        del Pp
        del sumK

        F2e1c = torch.zeros(nmol, nnewRoots, maskd.shape[0], 4, 4, device=device, dtype=dtype)
        for i in range(1, 4):
            F2e1c[..., 0, i] = P_anti[..., maskd, 0, i] * (0.5 * hsp - 0.5 * gsp).unsqueeze(1)
        for i, j in [(1, 2), (1, 3), (2, 3)]:
            F2e1c[..., i, j] = P_anti[..., maskd, i, j] * (0.25 * gpp - 0.75 * gp2).unsqueeze(1)

        F2e1c.add_(F2e1c.triu(1).transpose(3, 4), alpha=-1.0)
        F[:, :, maskd] += F2e1c
        del P_anti
        del F2e1c

    F0 = (
        F.reshape(nmol, nnewRoots, molsize, molsize, 4, 4)
        .transpose(3, 4)
        .reshape(nmol * nnewRoots, 4 * molsize, 4 * molsize)
    )
    del F
    return F0


def makeA_pi_symm_batch(mol, P0, w):
    molsize = int(mol.molsize)
    nmol = int(mol.nmol)
    npairs_per_mol = (molsize * (molsize - 1)) // 2

    dispatch = _makeA_pi_symm_batch_dispatch or _makeA_pi_symm_batch_kernel
    return dispatch(
        P0,
        w,
        mol.mask[:npairs_per_mol],
        mol.maskd[:molsize],
        mol.mask_l[:npairs_per_mol],
        mol.idxi[:npairs_per_mol],
        mol.idxj[:npairs_per_mol],
        mol.parameters["g_ss"].view(nmol, -1),
        mol.parameters["g_sp"].view(nmol, -1),
        mol.parameters["g_pp"].view(nmol, -1),
        mol.parameters["g_p2"].view(nmol, -1),
        mol.parameters["h_sp"].view(nmol, -1),
        molsize,
        nmol,
    )


def _makeA_pi_symm_batch_kernel(
    P0, w, mask, maskd, mask_l, idxi, idxj, gss, gsp, gpp, gp2, hsp, molsize: int, nmol: int
):
    nnewRoots = P0.shape[1]
    dtype = P0.dtype
    device = P0.device

    P0_blocks = P0.reshape(nmol, nnewRoots, molsize, 4, molsize, 4).transpose(3, 4)
    P = 0.5 * (P0_blocks + P0_blocks.transpose(2, 3).transpose(4, 5))
    P = P.reshape(nmol, nnewRoots, molsize * molsize, 4, 4)
    del P0_blocks
    F = torch.zeros_like(P)
    # print_memory_usage("After P_symm, and Fock_symm")

    # Calculate Coulomb contribution J
    tri_i, tri_j, weight, ind = _rcis_constant_tensors(dtype, device)

    grad_enabled = torch.is_grad_enabled()

    Fdiag = torch.zeros(nmol, nnewRoots, maskd.shape[0], 4, 4, dtype=dtype, device=device)

    PA = P[:, :, maskd[idxi]][..., tri_i, tri_j] * weight
    sumA = torch.zeros(nmol, nnewRoots, w.shape[1], 4, 4, dtype=dtype, device=device)
    sumA[..., tri_i, tri_j] = torch.einsum("nrps,npsS->nrpS", PA, w)
    del PA
    Fdiag.index_add_(2, idxj, sumA)

    if grad_enabled:
        sum_shape = sumA.shape
        del sumA
        sumB = torch.zeros(sum_shape, dtype=dtype, device=device)
    else:
        sumB = sumA
        sumB.zero_()

    PB = P[:, :, maskd[idxj]][..., tri_i, tri_j] * weight
    sumB[..., tri_i, tri_j] = torch.einsum("nrpS,npsS->nrps", PB, w)
    del PB
    Fdiag.index_add_(2, idxi, sumB)

    if grad_enabled:
        sum_shape = sumB.shape
        del sumB
        sumK = torch.empty(sum_shape, dtype=dtype, device=device)
    else:
        sumK = sumB
    Pp = P[:, :, mask]
    for i in range(4):
        for j in range(4):
            sumK[..., i, j] = -0.5 * torch.einsum("nrpsS,npsS->nrp", Pp, w[..., ind[i], :][..., :, ind[j]])
    F[:, :, mask] = sumK
    F[:, :, mask_l] = sumK.transpose(3, 4)
    del Pp

    Pdiag = P[:, :, maskd]
    Pptot_diag = Pdiag[..., 1, 1] + Pdiag[..., 2, 2] + Pdiag[..., 3, 3]

    del sumK

    Fdiag[..., 0, 0] += 0.5 * Pdiag[..., 0, 0] * gss.unsqueeze(1) + Pptot_diag * (gsp - 0.5 * hsp).unsqueeze(
        1
    )
    for i in range(1, 4):
        # (p,p)
        Fdiag[..., i, i] += (
            Pdiag[..., 0, 0] * (gsp - 0.5 * hsp).unsqueeze(1)
            + 0.5 * Pdiag[..., i, i] * gpp.unsqueeze(1)
            + (Pptot_diag - Pdiag[..., i, i]) * (1.25 * gp2 - 0.25 * gpp).unsqueeze(1)
        )
        # (s,p) = (p,s) upper triangle
        Fdiag[..., 0, i] += Pdiag[..., 0, i] * (1.5 * hsp - 0.5 * gsp).unsqueeze(1)
    # (p,p*)
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        Fdiag[..., i, j] += Pdiag[..., i, j] * (0.75 * gpp - 1.25 * gp2).unsqueeze(1)
    del Pdiag, Pptot_diag

    Fdiag.add_(Fdiag.triu(1).transpose(3, 4))
    F[:, :, maskd] = Fdiag
    # F[:,:,maskd] += F2e1c
    # F[:,:,maskd] += F[:,:,maskd].triu(1).transpose(3,4)

    return F


def orthogonalize_to_current_subspace(V, newsubspace, vend, tol):
    """
    Block orthogonalize newsubspace against V[:vend], then append an
    orthonormal basis for the surviving new directions.

    V, newsubspace have row vectors.
    """

    V_old = V[:vend]

    # Project the whole new block against the existing subspace.
    W = newsubspace

    W = W - (W @ V_old.T) @ V_old
    # Reorthogonalization for numerical stability of Gram-Schmidt
    W = W - (W @ V_old.T) @ V_old

    # Drop rows that do not survive projection against the old subspace.
    row_norms = torch.linalg.vector_norm(W, dim=1)
    W = W[row_norms > tol]

    if W.shape[0] == 0:
        return vend

    # Robustly get an orthonormal basis for the row span of W. QR factorization without pivoting is problematic.
    # Vh rows are orthonormal right singular vectors, i.e. basis vectors
    # in the same ambient space as rows of V.
    _, s, Vh = torch.linalg.svd(W, full_matrices=False)

    keep = s > tol
    Q_new = Vh[keep]

    if Q_new.shape[0] == 0:
        return vend

    # Optional but useful: clean up against old V again in finite precision.
    overlap = Q_new @ V_old.T
    max_leak = overlap.abs().max()
    orth_atol = 1e-13 if Q_new.dtype == torch.float64 else 1e-6

    if max_leak > orth_atol:
        Q_new -= (Q_new @ V[:vend].T) @ V[:vend]
        Q_new -= (Q_new @ V[:vend].T) @ V[:vend]
        # Re-orthonormalize after cleanup.
        _, s2, Vh2 = torch.linalg.svd(Q_new, full_matrices=False)
        Q_new = Vh2[s2 > tol]

    k = Q_new.shape[0]

    if vend + k > V.shape[0]:
        raise ValueError(f"Not enough space in V: need {vend + k} rows, but V only has {V.shape[0]} rows.")

    V[vend : vend + k] = Q_new
    return vend + k


import psutil  # to get the memory size


def getMaxSubspacesize(
    dtype, device, nov, nroots, nmol=1, num_big_matrices=2, memory_fraction=0.4, retry_memory_fraction=0.65
):
    """Calculate the maximum size of the subspace dimension
    based on available memory. The full subspace size is nov
    """

    device = device.type

    bytes_per_element = torch.finfo(dtype).bits // 8  # Bytes per element

    def _candidate(frac):
        # Get available memory
        if device == "cpu":
            available_memory = psutil.virtual_memory().available
        elif device == "cuda":
            available_memory, _ = torch.cuda.mem_get_info(device)
        else:
            raise ValueError("Unsupported device. Use 'cpu' or 'cuda'.")
        usable_memory = available_memory * frac
        n_calculated = int(usable_memory // (nov * nmol * bytes_per_element * num_big_matrices))
        return max(1, min(n_calculated, nov))

    maxSubspacesize = _candidate(memory_fraction)
    min_required_subspace = min(nov, 3 * nroots)
    if maxSubspacesize >= min_required_subspace:
        return maxSubspacesize

    if device == "cuda":
        torch.cuda.empty_cache()
        maxSubspacesize = _candidate(memory_fraction)

    if maxSubspacesize >= min_required_subspace:
        return maxSubspacesize

    maxSubspacesize_retry = _candidate(retry_memory_fraction)
    if maxSubspacesize_retry >= min_required_subspace:
        return maxSubspacesize_retry

    raise RuntimeError(
        "Unable to allocate a Davidson subspace large enough for the requested roots. "
        f"Requested at least 3*nroots={3 * nroots}, got {maxSubspacesize} at memory_fraction={memory_fraction:.2f} "
        f"and {maxSubspacesize_retry} at memory_fraction={retry_memory_fraction:.2f} "
        f"with nov={nov}, nmol={nmol}, num_big_matrices={num_big_matrices}."
    )


def get_subspace_eig_batched(H, nroots, vend, e_val_n, done, nonorthogonal):
    if nonorthogonal:
        raise NotImplementedError("Non-orthogonal davidson not yet implemented")
        # # Need to solve the generalized eigenvalue problem
        # # Method as described in Appendix section 1 of J. Chem. Phys. 144, 174105 (2016) https://doi.org/10.1063/1.4947245
        #
        # S = torch.einsum('ro,no->rn',V[:vend],V[:vend])
        # # S = 0.5*(S+S.T) # symmetrize for numerical stability
        # # Step 1: Calculate D^(-1/2)
        # D = torch.diag(S)  # Extract the diagonal elements of S
        # D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
        #
        # # Step 2: Compute D^(-1/2) S D^(-1/2) this is done to reduce the condition number of S
        # S_tilde = torch.einsum('ab,bc,cd->ad',D_inv_sqrt,S,D_inv_sqrt) # D_inv_sqrt @ S @ D_inv_sqrt
        #
        # # Step 3: Cholesky decomposition of S_tilde
        # L = torch.linalg.cholesky(S_tilde)
        # L_inv_D_inv_sqrt = torch.linalg.solve_triangular(L,D_inv_sqrt,upper=False)
        # D_inv_sqrt_L_inv_T = L_inv_D_inv_sqrt.T
        #
        # # Step 4: Compute the modified A matrix
        # A_tilde = torch.einsum('ab,bc,cd->ad',L_inv_D_inv_sqrt,H,D_inv_sqrt_L_inv_T)
        # # A_tilde = torch.linalg.inv(L) @ (D_inv_sqrt @ H @ D_inv_sqrt) @ torch.linalg.inv(L).T
        #
        # # Step 5: Solve the standard eigenvalue problem A_tilde X = X λ
        # r_eval, X = torch.linalg.eigh(A_tilde)
        #
        # # Step 6: Transform the eigenvectors back to the original problem
        # r_evec = D_inv_sqrt_L_inv_T @ X[:,:nroots]

    else:
        nmol, subspacesize = H.shape[0], H.shape[1]
        e_vec_n = torch.zeros(nmol, subspacesize, nroots, device=H.device, dtype=H.dtype)
        if nmol == 1:
            r_eval, r_evec = torch.linalg.eigh(H)
            e_val_n[0] = r_eval[0, :nroots]
            e_vec_n[0] = r_evec[0, :, :nroots]
            return e_vec_n

        active_indices = torch.nonzero(~done, as_tuple=False).squeeze(1)

        for v in torch.unique(vend[active_indices]):
            v_int = int(v.item())
            group = active_indices[vend[active_indices] == v]
            r_eval, r_evec = torch.linalg.eigh(H[group, :v_int, :v_int])
            e_val_n[group] = r_eval[:, :nroots]
            e_vec_n[group, :v_int] = r_evec[:, :, :nroots]
        return e_vec_n


def unpackone_batch(x0, nho, nHydro, size):
    x = torch.zeros((x0.shape[0], size, size), dtype=x0.dtype, device=x0.device)
    x[:, :nho, :nho] = x0[:, :nho, :nho]
    x[:, :nho, nho : (nho + 4 * nHydro) : 4] = x0[:, :nho, nho : (nho + nHydro)]
    x[:, nho : (nho + 4 * nHydro) : 4, nho : (nho + 4 * nHydro) : 4] = x0[
        :, nho : (nho + nHydro), nho : (nho + nHydro)
    ]
    x[:, nho : (nho + 4 * nHydro) : 4, :nho] = x0[:, nho : (nho + nHydro), :nho]
    return x


def packone_batch(x, nho, nHydro, norb):
    x0 = torch.empty((x.shape[0], norb, norb), dtype=x.dtype, device=x.device)
    x0[:, :nho, :nho] = x[:, :nho, :nho]
    x0[:, :nho, nho : (nho + nHydro)] = x[:, :nho, nho : (nho + 4 * nHydro) : 4]
    x0[:, nho : (nho + nHydro), nho : (nho + nHydro)] = x[
        :, nho : (nho + 4 * nHydro) : 4, nho : (nho + 4 * nHydro) : 4
    ]
    x0[:, nho : (nho + nHydro), :nho] = x[:, nho : (nho + 4 * nHydro) : 4, :nho]
    return x0


def print_memory_usage(step_description, device=0):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"[{step_description}]")
    print(f"  Allocated Memory: {allocated:.2f} MB")
    print(f"  Reserved Memory: {reserved:.2f} MB")
    print(f"  Max Allocated Memory: {max_allocated:.2f} MB")
    print(f"  Max Reserved Memory: {max_reserved:.2f} MB\n")


def _store_tdm_by_mode(mol, R, tdm_mode):
    if tdm_mode == "diag":
        mol.transition_density_matrices = torch.diagonal(R, dim1=-2, dim2=-1).clone()
        return
    if tdm_mode != "full":
        raise ValueError("transition_density_matrices_mode only supports 'full' and 'diag'.")
    mol.transition_density_matrices = R.clone()


def pack_dipole_matrix(mol, dipole_mat):
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]
    return packone_batch(
        dipole_mat.view(3 * mol.nmol, 4 * mol.molsize, 4 * mol.molsize), 4 * nHeavy, nHydro, norb
    ).view(mol.nmol, 3, norb, norb)


def _resolve_tdm_mode(mol):
    exc_cfg = mol.seqm_parameters["excited_states"]
    tdm_mode = str(exc_cfg.get("transition_density_matrices_mode", "full")).strip().lower()
    # XL-BOMD propagation requires full in-memory TDM.
    if bool(exc_cfg.get("save_tdm_xlbomd", False)):
        return "full"
    return tdm_mode


def rcis_analysis(
    mol,
    excitation_energies,
    amplitudes,
    nroots,
    rpa=False,
    orbital_window=None,
    save_tdm=False,
    compute_transition_properties=True,
):
    if not (compute_transition_properties or save_tdm or mol.verbose):
        mol.transition_dipole = None
        mol.oscillator_strength = None
        return

    if not (
        mol.verbose
        or save_tdm
        or torch.any(active_state_tensor(mol.active_state, int(mol.nmol), mol.coordinates.device) > 0)
    ):
        return

    tdm_mode = _resolve_tdm_mode(mol)

    dipole_mat = calc_dipole_matrix(mol) if (compute_transition_properties or mol.verbose) else None
    transition_dipole, oscillator_strength = calc_transition_dipoles(
        mol,
        amplitudes,
        excitation_energies,
        nroots,
        dipole_mat,
        rpa,
        orbital_window,
        save_tdm,
        compute_transition_properties=compute_transition_properties,
        tdm_mode=tdm_mode,
    )
    if mol.verbose:
        print_rcis_analysis(excitation_energies, transition_dipole, oscillator_strength)
    mol.transition_dipole, mol.oscillator_strength = transition_dipole, oscillator_strength


def calc_transition_dipoles(
    mol,
    amplitudes,
    excitation_energies,
    nroots,
    dipole_mat,
    rpa=False,
    orbital_window=None,
    save_tdm=False,
    compute_transition_properties=True,
    tdm_mode="full",
):
    nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol, orbital_window)

    if rpa:
        amp_ia_X = amplitudes[0].view(mol.nmol, nroots, nocc, nvirt)
        amp_ia_Y = amplitudes[1].view(mol.nmol, nroots, nocc, nvirt)
    else:
        amp_ia_X = amplitudes.view(mol.nmol, nroots, nocc, nvirt)

    # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a
    R = torch.einsum("bmi,bria,bna->brmn", Cocc, amp_ia_X, Cvirt)
    if rpa:
        R += torch.einsum("bma,bria,bni->brmn", Cvirt, amp_ia_Y, Cocc)

    do_transition_props = bool(compute_transition_properties or mol.verbose)
    if save_tdm:
        _store_tdm_by_mode(mol, R, tdm_mode)
    if not do_transition_props:
        return None, None

    dipole_mat_packed = pack_dipole_matrix(mol, dipole_mat)
    # Transition dipole in AU as calculated in NEXMD
    transition_dipole = torch.einsum("brmn,bdmn->brd", R, dipole_mat_packed) * math.sqrt(2.0) / a0
    hartree = 27.2113962  # value used in NEXMD
    oscillator_strength = (
        2.0 / 3.0 * excitation_energies / hartree * torch.square(transition_dipole).sum(dim=2)
    )
    return transition_dipole, oscillator_strength


def print_rcis_analysis(excitation_energies, transition_dipole, oscillator_strength):
    print(f"Number of excited states: {excitation_energies.shape[1]}\n")
    print("Excitation energies E (eV), Transition dipoles d (au), and Oscillator strengths f (au)")
    row_format = "{:<10}   {:>10}   {:>10}   {:>10}      {:<10}"

    # Print header
    print(row_format.format("E", "d x", "d y", "d z", "f"))
    print("-" * 65)

    nmol = excitation_energies.shape[0]
    # Loop over molecules and states using enumerate and zip
    for mol_idx, (mol_energy, mol_dipole, mol_strength) in enumerate(
        zip(excitation_energies, transition_dipole, oscillator_strength), start=1
    ):
        if nmol > 1:
            print(f"Molecule {mol_idx}:")
        for energy_val, dipole_vals, strength_val in zip(mol_energy, mol_dipole, mol_strength):
            # Convert single-value tensors to Python floats
            e = energy_val.item()
            dx, dy, dz = dipole_vals.tolist()
            s = strength_val.item()
            print(row_format.format(f"{e:.6f}", f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}", f"{s:.6f}"))
        print("")


def getMemUse(dtype, device, mol, nroots=1):
    r"""
    Estimate the peak memory usage while calculating  \sum_ia (ia||jb)V_ia
    (contracting the roots with the two-e integrals)
    We approximate the largest intermediate allocations (P, F, PA, PB, suma, sumA etc. in makeA_pi_symm_batch())
    via a factor ~ 200 * nmol * nnewRoots * (molsize^2) * bytes_per_element.
    If that estimate exceeds available memory, we return need_to_chunk=True
    and compute a chunk_size to avoid out-of-memory issues.

    Returns:
        need_to_chunk (bool): Whether we must chunk the computation.
        chunk_size (int or None): Suggested chunk size if need_to_chunk=True, else None.
    """

    dev_type = device.type
    if dev_type == "cpu":
        available_memory = psutil.virtual_memory().available
    elif dev_type == "cuda":
        available_memory, _ = torch.cuda.mem_get_info(device)
    else:
        raise ValueError("Unsupported device type. Use 'cpu' or 'cuda'.")

    bytes_per_element = torch.finfo(dtype).bits // 8
    # 200 is an approximate factor from analyzing the shape and count
    # of all major tensors in makeA_pi_symm_batch. The factor seems to perform well while benchmarking
    mem_per_root = 200.0 * mol.nmol * (mol.molsize**2) * bytes_per_element
    total_mem_estimate = mem_per_root * nroots

    # print(f"Available: {available_memory/1073741824} GiB; Max memory used will be {total_mem_estimate/1073741824} GiB, per root: {mem_per_root/1073741824} GiB")
    need_to_chunk = total_mem_estimate > available_memory
    chunk_size = nroots

    if need_to_chunk:
        # Ensure at least 1, up to nroots
        chunk_size = max(1, min(nroots, int(available_memory // mem_per_root)))

    return need_to_chunk, chunk_size


def make_guess(ea_ei, nroots, maxSubspacesize, V, nmol, nov):
    # Make the davidson guess vectors
    sorted_ediff, sortedidx = torch.sort(
        ea_ei, stable=True, descending=False
    )  # stable to preserve the order of degenerate orbitals

    nroots_expand = nroots
    # If the last chosen root was degenerate in ea_ei, then expand the subspace to include all the degenerate roots
    while nroots_expand < len(sorted_ediff[0]) and torch.all(
        (sorted_ediff[:, nroots_expand] - sorted_ediff[:, nroots_expand - 1]) < 1e-4
    ):
        nroots_expand += 1
    if nroots_expand > nroots:
        print(
            f"Increasing the number of states calculated from {nroots} to {nroots_expand} because of orbital degeneracies"
        )
        nroots = nroots_expand

    extra_subspace = min(7, nov - nroots)
    # if after the extra_subspace i dont have enough space for subspace expansion then i shouldnt use extra_subspace
    extra_subspace = (
        extra_subspace
        if 2 * nroots + extra_subspace < maxSubspacesize
        else max(0, maxSubspacesize - 2 * nroots)
    )
    nstart = nroots + extra_subspace
    V[
        torch.arange(nmol, device=V.device).unsqueeze(1),
        torch.arange(nstart, device=V.device),
        sortedidx[:, :nstart],
    ] = 1.0

    return nstart, nroots


def calc_cis_energy(mol, w, e_mo, amplitude, F, P, rpa=False, orbital_window=None):
    norb_batch, nocc_batch = mol.norb, mol.nocc
    if not torch.all(norb_batch == norb_batch[0]) or not torch.all(nocc_batch == nocc_batch[0]):
        raise ValueError("All molecules in the batch must have the same number of orbitals and electrons")

    # for near-degenerate use the formulation where E_cis is expressed in atomic orbital basis only
    norb, nocc = norb_batch[0], nocc_batch[0]
    if ((e_mo[:norb, 1:] - e_mo[:norb, :-1]) < 1e-4).any() and not rpa:
        return calc_cis_energy_from_density(mol, w, F, P, amplitude, rpa, orbital_window)

    if orbital_window is not None:
        n_below, m_above = orbital_window
        occ_idx = torch.arange(nocc - n_below, nocc)
        virt_idx = torch.arange(nocc, nocc + m_above)
    else:
        occ_idx = torch.arange(nocc)
        virt_idx = torch.arange(nocc, norb)
    ea_ei = e_mo[:, virt_idx].unsqueeze(1) - e_mo[:, occ_idx].unsqueeze(2)

    C = mol.molecular_orbitals
    Cocc = C[:, :, occ_idx]
    Cvirt = C[:, :, virt_idx]

    if not rpa:  # CIS: w = XAX
        HV = matrix_vector_product_batched(mol, amplitude.unsqueeze(1), w, ea_ei, Cocc, Cvirt)
        E_cis = torch.linalg.vecdot(amplitude, HV.squeeze(1))
    else:  # RPA
        #  w = (X Y)(A B)(X) = X(AX+BY) + Y(BX+AY)
        #           (B A)(Y)
        X = amplitude[0, ...]
        Y = amplitude[1, ...]
        AX, BX = matrix_vector_product_batched(mol, X.unsqueeze(1), w, ea_ei, Cocc, Cvirt, makeB=True)
        AY, BY = matrix_vector_product_batched(mol, Y.unsqueeze(1), w, ea_ei, Cocc, Cvirt, makeB=True)
        E_cis = torch.linalg.vecdot(X, (AX + BY).squeeze(1)) + torch.linalg.vecdot(Y, (BX + AY).squeeze(1))

    # For calculating excitation energy gradient with backprop
    # L = E_cis.sum()
    # L.backward(create_graph=False,retain_graph=True)
    # force = mol.coordinates.grad.clone()
    # with torch.no_grad(): mol.coordinates.grad.zero_()
    # torch.set_printoptions(precision=15)
    # print(f'E_cis is {E_cis}')
    # print(f'Grad CIS from backprop is\n{force}')

    return E_cis


def calc_cis_energy_from_density(mol, w, F, P, amplitude, rpa=False, orbital_window=None):
    if rpa:
        raise NotImplementedError

    nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol, orbital_window=orbital_window)
    with torch.no_grad():
        R = torch.einsum("bmi,bia,bna->bmn", Cocc, amplitude.view(-1, nocc, nvirt), Cvirt)
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]
    D = packone_batch(P, 4 * nHeavy, nHydro, norb)  # occ subspace projector/density matrix
    Q = torch.eye(D.shape[1], dtype=D.dtype, device=D.device).unsqueeze(0) - D  # virtual subspace projector

    R = D @ R @ Q / 2.0  # project to occ-virt subspace
    F0 = makeA_pi_batched(mol, R.unsqueeze(1), w).squeeze(1) * 2.0
    F_ = packone_batch(F, 4 * nHeavy, nHydro, norb)
    F0 += R @ F_ - F_ @ R
    F0 = D @ F0 @ Q / 2.0  # project to occ-virt subspace
    E_cis = (R * F0).sum(dim=(1, 2))

    return E_cis


def make_best_guess_from_previous_amplitudes(mol, V_old, V, nocc):
    nroots = V_old.shape[1]
    V_sq = (V_old * V_old).reshape(mol.nmol, nroots, nocc, -1)
    occ_mo_overlap = mol.molecular_orbitals[:, :, :nocc].transpose(1, 2) @ mol.old_mos[:, :, :nocc]
    occ_weight = V_sq.sum(dim=(1, 3)) / nroots
    occ_mo_overlap *= occ_weight.unsqueeze(1)
    try:
        U, _, Vt = torch.linalg.svd(occ_mo_overlap)
        rot_occ_transpose = U @ Vt

        virt_mo_overlap = mol.molecular_orbitals[:, :, nocc:].transpose(1, 2) @ mol.old_mos[:, :, nocc:]
        virt_weight = V_sq.sum(dim=(1, 2)) / nroots
        virt_mo_overlap *= virt_weight.unsqueeze(1)
        U, _, Vt = torch.linalg.svd(virt_mo_overlap)
        rot_virt = (U @ Vt).transpose(1, 2)
        V[:, :nroots] = torch.einsum(
            "Nji,Nria,Nab->Nrjb", rot_occ_transpose, V_old.view(mol.nmol, nroots, nocc, -1), rot_virt
        ).reshape(mol.nmol, nroots, -1)
    except RuntimeError:
        V[:, :nroots] = V_old


def get_occ_virt(mol, orbital_window=None, e_mo=None):
    C = mol.molecular_orbitals  # (nmol, nbasis, norb_i)
    nocc_b, norb_b = mol.nocc, mol.norb  # (nmol,)
    nvirt_b = norb_b - nocc_b
    nmol, nbasis = C.shape[:2]
    device, dtype = C.device, C.dtype

    uniform = nmol == 1 or (torch.all(nocc_b == nocc_b[0]) and torch.all(norb_b == norb_b[0]))
    if uniform:
        _, _, norb, nocc = _uniform_molecule_dimensions(mol)
        if orbital_window is not None:
            n_below, m_above = map(int, orbital_window)
            if not (0 <= n_below <= nocc and 0 <= m_above <= norb - nocc):
                raise ValueError("orbital_window out of bounds.")
            occ_idx = torch.arange(nocc - n_below, nocc, device=device)
            virt_idx = torch.arange(nocc, nocc + m_above, device=device)
        else:
            occ_idx = torch.arange(nocc, device=device)
            virt_idx = torch.arange(nocc, norb, device=device)
        Cocc, Cvirt = C[:, :, occ_idx], C[:, :, virt_idx]

        if e_mo is not None:
            ea_ei = e_mo[:, virt_idx].unsqueeze(1) - e_mo[:, occ_idx].unsqueeze(2)
            return occ_idx.numel(), virt_idx.numel(), Cocc, Cvirt, ea_ei
        return occ_idx.numel(), virt_idx.numel(), Cocc, Cvirt

    if orbital_window is not None:
        raise ValueError("orbital_window requires uniform nocc/norb across the batch.")

    nocc_max = int(nocc_b.max().item())
    nvirt_max = int(nvirt_b.max().item())
    Cocc = torch.zeros((nmol, nbasis, nocc_max), device=device, dtype=dtype)
    Cvirt = torch.zeros((nmol, nbasis, nvirt_max), device=device, dtype=dtype)

    for i in range(nmol):
        nocc_i, norb_i = int(nocc_b[i].item()), int(norb_b[i].item())
        Cocc[i, :, :nocc_i] = C[i, :, :nocc_i]
        Cvirt[i, :, : nvirt_b[i]] = C[i, :, nocc_i:norb_i]

    if e_mo is not None:
        ea_ei = torch.zeros(nmol, nocc_max, nvirt_max, device=device, dtype=dtype)
        for i in range(nmol):
            nocc_i, norb_i = int(nocc_b[i].item()), int(norb_b[i].item())
            ea_ei[i, :nocc_i, : nvirt_b[i]] = e_mo[i, nocc_i:norb_i].unsqueeze(0) - e_mo[
                i, :nocc_i
            ].unsqueeze(1)
        return nocc_max, nvirt_max, Cocc, Cvirt, ea_ei

    return nocc_max, nvirt_max, Cocc, Cvirt


def make_A_times_zvector_batched(mol, z, w, ea_ei, Cocc, Cvirt):
    nmol = int(mol.nmol)
    _, _, norb, nocc = _uniform_molecule_dimensions(mol)
    nvirt = norb - nocc

    nroots = z.shape[0] // nmol
    Via = z.reshape(nmol, nroots, nocc, nvirt)
    make_density = _ao_transition_density_dispatch or _ao_transition_density_kernel
    make_post = _mo_fock_action_dispatch or _mo_fock_action_kernel
    P_xi = make_density(Cocc, Via, Cvirt, True)

    F0 = makeA_pi_batched(mol, P_xi, w, allSymmetric=True)
    A = make_post(Via, F0, ea_ei, Cocc, Cvirt, False, torch.is_grad_enabled())

    return A.reshape(nmol * nroots, nocc * nvirt)


from seqm.seqm_functions.cg_solver import conjugate_gradient_batch


def make_cis_densities(
    mol,
    do_transition_denisty,
    do_difference_density,
    do_relaxed_density,
    orbital_window=None,
    w=None,
    e_mo=None,
    zvec_tolerance=1e-6,
    rpa=False,
):
    active_states = active_state_tensor(mol.active_state, int(mol.nmol), mol.cis_amplitudes.device)
    if torch.any(active_states <= 0):
        raise ValueError("Active states must be >0 for CIS density construction.")
    state_idx = active_states - 1
    nmol = int(mol.nmol)
    if mol.cis_amplitudes.dim() == 4:
        idx = state_idx.view(1, nmol, 1, 1).expand(2, -1, 1, mol.cis_amplitudes.shape[-1])
        amp = mol.cis_amplitudes.gather(2, idx).squeeze(2)
    else:
        idx = state_idx.view(nmol, 1, 1).expand(-1, 1, mol.cis_amplitudes.shape[-1])
        amp = mol.cis_amplitudes.gather(1, idx).squeeze(1)
    nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol, orbital_window=orbital_window)
    nmol, norb = Cocc.shape[:2]

    if rpa:
        amp_ia_X = amp[0].view(mol.nmol, nocc, nvirt)
        amp_ia_Y = amp[1].view(mol.nmol, nocc, nvirt)
    else:
        amp_ia_X = amp.view(mol.nmol, nocc, nvirt)

    cis_densities = {}
    if not rpa and (do_difference_density or do_relaxed_density):
        # Keep eager: outputs are reused after compiled calls, and CUDA Graph
        # output buffers would require extra clones.
        R, B, B_virt, B_occ = _cis_density_kernel(Cocc, Cvirt, amp_ia_X)
        if do_transition_denisty or do_relaxed_density:
            cis_densities["transition_density"] = R
        if do_difference_density:
            cis_densities["difference_density"] = B
    elif do_transition_denisty or do_relaxed_density:
        # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a
        R = torch.einsum("bmi,bia,bna->bmn", Cocc, amp_ia_X, Cvirt)
        if rpa:
            R += torch.einsum("bma,bia,bni->bmn", Cvirt, amp_ia_Y, Cocc)
        cis_densities["transition_density"] = R

    if do_difference_density:
        if rpa:
            B_virt = torch.einsum("Nma,Nia->Nmi", Cvirt, amp_ia_X)
            B_occ = torch.einsum("Nmi,Nia->Nma", Cocc, amp_ia_X)
            B = torch.einsum("Nmi,Nni->Nmn", B_virt, B_virt) - torch.einsum("Nmi,Nni->Nmn", B_occ, B_occ)
            B_virt_Y = torch.einsum("Nma,Nia->Nmi", Cvirt, amp_ia_Y)
            B_occ_Y = torch.einsum("Nmi,Nia->Nma", Cocc, amp_ia_Y)
            B += torch.einsum("Nmi,Nni->Nmn", B_virt_Y, B_virt_Y) - torch.einsum(
                "Nmi,Nni->Nmn", B_occ_Y, B_occ_Y
            )
            cis_densities["difference_density"] = B

        if do_relaxed_density:
            # Calculate z-vector
            # make RHS of the CPSCF equation:
            B_pi = makeA_pi_batched(mol, B.unsqueeze(1), w, allSymmetric=True).squeeze(1) * 2.0
            R_pi = makeA_pi_batched(mol, R.unsqueeze(1), w).squeeze(1) * 2.0
            make_rhs = _relaxed_rhs_dispatch or _relaxed_rhs_kernel
            RHS = make_rhs(Cocc, Cvirt, B_pi, R_pi, B_virt, B_occ)

            if rpa:
                RHS -= torch.einsum("Nni,Nnm,Nma->Nia", B_virt_Y, R_pi, Cvirt)
                RHS += torch.einsum("Nni,Nnm,Nma->Nia", Cocc, R_pi, B_occ_Y)

            del B_occ, B_virt
            RHS = RHS.reshape(nmol, nocc * nvirt)
            ea_ei = e_mo[:, nocc:norb].unsqueeze(1) - e_mo[:, :nocc].unsqueeze(2)
            ea_flat = ea_ei.reshape(nmol, nocc * nvirt)
            rhs0 = RHS / ea_flat

            def applyA(z):
                return make_A_times_zvector_batched(mol, z, w, ea_ei, Cocc, Cvirt)

            zvec = conjugate_gradient_batch(applyA, RHS, ea_flat, tol=zvec_tolerance, x0=rhs0)

            make_relaxed = _relaxed_finish_dispatch or _relaxed_finish_kernel
            cis_densities["relaxed_difference_density"] = make_relaxed(
                Cocc, Cvirt, zvec, B, nmol, nocc, nvirt
            )

    return cis_densities


def _cis_density_kernel(Cocc, Cvirt, amp_ia):
    R = torch.einsum("bmi,bia,bna->bmn", Cocc, amp_ia, Cvirt)
    B_virt = torch.einsum("Nma,Nia->Nmi", Cvirt, amp_ia)
    B_occ = torch.einsum("Nmi,Nia->Nma", Cocc, amp_ia)
    B = torch.einsum("Nmi,Nni->Nmn", B_virt, B_virt) - torch.einsum("Nmi,Nni->Nmn", B_occ, B_occ)
    return R, B, B_virt, B_occ


def _relaxed_rhs_kernel(Cocc, Cvirt, B_pi, R_pi, B_virt, B_occ):
    RHS = -torch.einsum("Nni,Nmn,Nma->Nia", Cocc, B_pi, Cvirt)
    RHS -= torch.einsum("Nni,Nmn,Nma->Nia", B_virt, R_pi, Cvirt)
    RHS += torch.einsum("Nni,Nmn,Nma->Nia", Cocc, R_pi, B_occ)
    return RHS


def _relaxed_finish_kernel(Cocc, Cvirt, zvec, B, nmol: int, nocc: int, nvirt: int):
    z_ao = torch.einsum("Nmi,Nia,Nna->Nmn", Cocc, zvec.view(nmol, nocc, nvirt), Cvirt)
    return B + z_ao + z_ao.transpose(1, 2)


# Function to verify the linearization of energy (w.r.t. density, transition density) for XL-BOMD
def cis_energy_from_transition_density(mol, F, R, w, D, Hcore):
    F0 = makeA_pi_batched(mol, R.unsqueeze(1), w).squeeze(1) * 2.0
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]
    F_ = packone_batch(F, 4 * nHeavy, nHydro, norb).squeeze(1)

    F0 -= F_ @ R - R @ F_
    E = (R * F0).sum(dim=(1, 2))
    print(f"CIS Energy is {E}")

    from seqm.seqm_functions.energy import elec_energy, elec_energy_xl

    ground = False
    # ground = False
    norm = True
    err = []
    molecule = mol
    from seqm.seqm_functions.hcore import hcore

    M, *rest = hcore(molecule)
    W = torch.tensor([0], device=molecule.nocc.device)
    from seqm.seqm_functions.fock import fock

    ener = []
    scale = 1e-5
    if ground:
        noise = torch.randn_like(D)
    else:
        noise = torch.randn_like(R)
    for i in range(0, 100, 5):
        if ground:
            P = D + noise * scale * i
            F_ = fock(
                molecule.nmol,
                molecule.molsize,
                P,
                M,
                molecule.maskd,
                molecule.mask,
                molecule.idxi,
                molecule.idxj,
                w,
                W,
                molecule.parameters["g_ss"],
                molecule.parameters["g_pp"],
                molecule.parameters["g_sp"],
                molecule.parameters["g_p2"],
                molecule.parameters["h_sp"],
                molecule.method,
                molecule.parameters["s_orb_exp_tail"],
                molecule.parameters["p_orb_exp_tail"],
                molecule.parameters["d_orb_exp_tail"],
                molecule.Z,
                molecule.parameters["F0SD"],
                molecule.parameters["G2SD"],
            )

            Ei = elec_energy_xl(D, P, F_, Hcore)
            if i == 0:
                print(f"XL E is {Ei[0].item()}\nActual E is{elec_energy(D, F, Hcore)[0].item()} ")
        elif norm:
            Q = R + noise * scale * i
            Ei = (Q * (2.0 * R - Q)).sum(dim=(1, 2))
            print(f"Norm is {Ei[0].item()}")
        else:
            Q = R + noise * scale * i
            Ei = linearlized_cis_energy(mol, F, R, Q, w)
        err.append(i)
        ener.append(Ei[0].item())
    exit()


def linearlized_cis_energy(mol, F, R, Q, w):
    nHeavy = mol.nHeavy[0]
    nHydro = mol.nHydro[0]
    norb = mol.norb[0]
    F_ = packone_batch(F, 4 * nHeavy, nHydro, norb).squeeze(1)

    F1 = -(F_ @ R - R @ F_)
    E1 = (R * F1).sum(dim=(1, 2))

    G = makeA_pi_batched(mol, Q.unsqueeze(1), w).squeeze(1) * 2.0
    E2alt = ((2.0 * R - Q) * G).sum(dim=(1, 2))

    E = E1 + E2alt

    print(f"CIS Energy is {E}")
    return E
