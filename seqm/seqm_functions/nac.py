import torch

from seqm.seqm_functions.anal_grad import (
    omx_fd,
    omx_threebody_ortho_grad,
    overlap_der_finiteDiff,
    w_der,
    w_derivative_numerical,
)
from seqm.seqm_functions.cg_solver import conjugate_gradient_batch
from seqm.seqm_functions.fock import (
    EMAT_SCALE_4,
    UPPER_IDX0_4,
    UPPER_IDX1_4,
    WEIGHT_10,
    K_ind_4,
    _cached_index,
    _cached_tensor,
)
from seqm.seqm_functions.rcis_batch import (
    _uniform_molecule_dimensions,
    get_occ_virt,
    make_A_times_zvector_batched,
    makeA_pi_batched,
    unpackone_batch,
)
from seqm.utils.torch_compile import optional_compile_function

from .constants import a0
from .omx_utils import OMX_METHODS, get_orbital_zetas

_contract_nac_density_dispatch = None
_contract_mixed_transition_terms_dispatch = None
_pair_response_rhs_dispatch = None


def enable_nac_compile(mode=None, **options):
    """Compile tensor contractions used by CIS nonadiabatic coupling vectors."""
    global _contract_nac_density_dispatch
    global _contract_mixed_transition_terms_dispatch
    global _pair_response_rhs_dispatch

    compile_options = dict(options)
    if mode is not None:
        compile_options["mode"] = mode

    _contract_nac_density_dispatch = optional_compile_function(
        _contract_nac_density_kernel, compile_options=compile_options, label="nac.contract_density"
    )
    _contract_mixed_transition_terms_dispatch = optional_compile_function(
        _contract_mixed_transition_terms_kernel,
        compile_options=compile_options,
        label="nac.contract_mixed_transition",
    )
    _pair_response_rhs_dispatch = optional_compile_function(
        _pair_response_rhs_kernel, compile_options=compile_options, label="nac.pair_response_rhs"
    )


def _state_pair_tensors(state_pairs, device):
    state_pairs = list(state_pairs)
    if len(state_pairs) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    pair_tensor = torch.as_tensor(state_pairs, dtype=torch.long, device=device)
    if pair_tensor.dim() != 2 or pair_tensor.shape[1] != 2:
        raise ValueError("state_pairs must be an iterable of (state1, state2) pairs.")
    return pair_tensor[:, 0] - 1, pair_tensor[:, 1] - 1


def _build_nac_derivative_operators(mol, P0, ri, riXH, dtype, device, return_w_x=False):
    npairs = mol.rij.shape[0]
    ortho_cache = None
    if mol.method in OMX_METHODS:
        overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
        Xij = mol.xij * mol.rij.unsqueeze(1) * a0
        w_x = torch.zeros(npairs, 3, 10, 10, dtype=dtype, device=device)
        e1b_x, e2a_x, _, _, ortho_cache = omx_fd(
            mol, overlap_x, w_x, Xij, mol.ni, mol.nj, mol.idxi, mol.idxj, mol.method, return_ortho_cache=True
        )
    else:
        overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
        Xij = mol.xij * mol.rij.unsqueeze(1) * a0
        w_x = torch.zeros(npairs, 3, 10, 10, dtype=dtype, device=device)
        zetas, zetap = get_orbital_zetas(mol.parameters, mol.method)
        zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)), dim=1)
        overlap_der_finiteDiff(
            overlap_x,
            mol.idxi,
            mol.idxj,
            mol.rij,
            Xij,
            mol.parameters["beta"],
            mol.ni,
            mol.nj,
            zeta,
            mol.const.qn_int,
        )
        if riXH is not None and ri is not None:
            e1b_x, e2a_x = w_der(
                mol.const,
                mol.Z,
                mol.const.tore,
                mol.ni,
                mol.nj,
                w_x,
                mol.rij,
                mol.xij,
                Xij,
                mol.idxi,
                mol.idxj,
                mol.parameters["g_ss"],
                mol.parameters["g_pp"],
                mol.parameters["g_p2"],
                mol.parameters["h_sp"],
                zetas,
                zetap,
                riXH,
                ri,
            )
        else:
            e1b_x, e2a_x = w_derivative_numerical(mol, Xij, w_x)

    # The following logic to form the coulomb and exchange integrals by contracting the two-electron integrals
    # with the density matrix has been cribbed from fock.py.
    ind = _cached_index(K_ind_4, device)
    overlap_KAB_x = overlap_x
    P = (
        P0.reshape(mol.nmol, mol.molsize, 4, mol.molsize, 4)
        .transpose(2, 3)
        .reshape(mol.nmol * mol.molsize * mol.molsize, 4, 4)
    )
    Pp = P[mol.mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            overlap_KAB_x[..., i, j] -= torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    weight = _cached_tensor(WEIGHT_10, device, dtype).reshape((-1, 10))
    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)
    PA = (P[mol.maskd[mol.idxi]][..., idx0, idx1] * weight).unsqueeze(-1)
    PB = (P[mol.maskd[mol.idxj]][..., idx0, idx1] * weight).unsqueeze(-2)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)
    sumA = torch.zeros_like(overlap_KAB_x)
    sumA[..., idx0, idx1] = suma
    e2a_x.add_(sumA)

    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)
    sumB = torch.zeros_like(overlap_KAB_x)
    sumB[..., idx0, idx1] = sumb
    e1b_x.add_(sumB)

    scale_emat = _cached_tensor(EMAT_SCALE_4, device, dtype)
    e1b_x *= scale_emat
    e2a_x *= scale_emat
    if return_w_x:
        return overlap_KAB_x, e1b_x, e2a_x, None, ortho_cache, w_x
    return overlap_KAB_x, e1b_x, e2a_x, None, ortho_cache


def _contract_nac_density_batch(
    mol, B, B0, overlap_KAB_x, e1b_x, e2a_x, _unused_p0_ortho_grad, ortho_cache, nmol, molsize
):
    dispatch = _contract_nac_density_dispatch or _contract_nac_density_kernel
    nac_cis = dispatch(
        B,
        overlap_KAB_x,
        e1b_x,
        e2a_x,
        mol.mask,
        mol.maskd[mol.idxi],
        mol.maskd[mol.idxj],
        mol.idxi,
        mol.idxj,
        nmol,
        molsize,
    )
    if ortho_cache is not None:
        nac_cis += omx_threebody_ortho_grad(
            mol,
            B0,
            ortho_cache["S_x"],
            ortho_cache["B_x"],
            ortho_cache["pair_core_semi_x"],
            ortho_cache=ortho_cache,
            unrestricted=False,
        )
    return nac_cis


def _contract_nac_density_kernel(
    B, overlap_KAB_x, e1b_x, e2a_x, mask, maskd_idxi, maskd_idxj, idxi, idxj, nmol: int, molsize: int
):
    pair_grad = torch.einsum("pbxy,pcxy->pbc", B[mask], overlap_KAB_x)
    pair_grad.add_(
        torch.einsum("pbxy,pcxy->pbc", B[maskd_idxj], e2a_x)
        + torch.einsum("pbxy,pcxy->pbc", B[maskd_idxi], e1b_x)
    )

    nac_cis = torch.zeros(nmol * molsize, pair_grad.shape[1], 3, dtype=B.dtype, device=B.device)
    nac_cis.index_add_(0, idxi, pair_grad)
    nac_cis.index_add_(0, idxj, pair_grad, alpha=-1.0)
    return nac_cis.view(nmol, molsize, pair_grad.shape[1], 3).permute(0, 2, 1, 3)


def _build_pair_response_density_batch(
    mol, w, e_mo, Cocc, Cvirt, amp_i, amp_j, Bij_symm, zvec_tolerance, pair_response_cache=None
):
    nmol, nbatch, nocc, nvirt = amp_i.shape
    if pair_response_cache is None:
        RI = torch.einsum("Nmi,Nbia,Nna->Nbmn", Cocc, amp_i, Cvirt)
        RJ = torch.einsum("Nmi,Nbia,Nna->Nbmn", Cocc, amp_j, Cvirt)
        Bv_i = torch.einsum("Nma,Nbia->Nbmi", Cvirt, amp_i)
        Bv_j = torch.einsum("Nma,Nbia->Nbmi", Cvirt, amp_j)
        Bo_i = torch.einsum("Nmi,Nbia->Nbma", Cocc, amp_i)
        Bo_j = torch.einsum("Nmi,Nbia->Nbma", Cocc, amp_j)
    else:
        RI, RJ, Bv_i, Bv_j, Bo_i, Bo_j = pair_response_cache
    pair_pi = makeA_pi_batched(mol, torch.cat((Bij_symm, RI, RJ), dim=1), w)
    BIJ_pi = pair_pi[:, :nbatch] * 2.0
    RI_pi = pair_pi[:, nbatch : 2 * nbatch]
    RJ_pi = pair_pi[:, 2 * nbatch :]

    make_rhs = _pair_response_rhs_dispatch or _pair_response_rhs_kernel
    rhs = make_rhs(Cocc, Cvirt, BIJ_pi, RI_pi, RJ_pi, Bv_i, Bv_j, Bo_i, Bo_j)
    ea_ei = e_mo[:, nocc : nocc + nvirt].unsqueeze(1) - e_mo[:, :nocc].unsqueeze(2)
    rhs_flat = rhs.reshape(nmol * nbatch, nocc * nvirt)
    ea_flat = ea_ei.repeat_interleave(nbatch, dim=0).reshape(nmol * nbatch, nocc * nvirt)
    z0_flat = rhs_flat / ea_flat

    def applyA(z):
        return make_A_times_zvector_batched(mol, z, w, ea_ei, Cocc, Cvirt)

    zvec = conjugate_gradient_batch(applyA, rhs_flat, ea_flat, tol=zvec_tolerance, x0=z0_flat)
    z_ao = torch.einsum("Nmi,Nbia,Nna->Nbmn", Cocc, zvec.view(nmol, nbatch, nocc, nvirt), Cvirt)
    Dij = Bij_symm + z_ao + z_ao.transpose(-1, -2)
    return Dij, RI, RJ


def _pair_response_rhs_kernel(Cocc, Cvirt, BIJ_pi, RI_pi, RJ_pi, Bv_i, Bv_j, Bo_i, Bo_j):
    rhs = -torch.einsum("Nni,Nbmn,Nma->Nbia", Cocc, BIJ_pi, Cvirt)
    rhs -= torch.einsum("Nbni,Nbmn,Nma->Nbia", Bv_i, RJ_pi, Cvirt)
    rhs -= torch.einsum("Nbni,Nbmn,Nma->Nbia", Bv_j, RI_pi, Cvirt)
    rhs += torch.einsum("Nni,Nbmn,Nbma->Nbia", Cocc, RI_pi, Bo_j)
    rhs += torch.einsum("Nni,Nbmn,Nbma->Nbia", Cocc, RJ_pi, Bo_i)
    return rhs


def _contract_mixed_transition_terms(mol, RI0, RJ0, w_x, dtype, device):
    molsize = int(mol.molsize)
    nmol = int(mol.nmol)
    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)
    weight = _cached_tensor(WEIGHT_10, device, dtype).reshape(1, 1, 10)
    scale_emat = _cached_tensor(EMAT_SCALE_4, device, dtype)
    ind = _cached_index(K_ind_4, device)

    dispatch = _contract_mixed_transition_terms_dispatch or _contract_mixed_transition_terms_kernel
    return dispatch(
        RI0,
        RJ0,
        w_x,
        mol.mask,
        mol.maskd,
        mol.idxi,
        mol.idxj,
        idx0,
        idx1,
        weight,
        scale_emat,
        ind,
        nmol,
        molsize,
    )


def _ao4_nac(T, nmol: int, nbatch: int, molsize: int):
    return (
        T.reshape(nmol, nbatch, molsize, 4, molsize, 4)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(nmol * molsize * molsize, nbatch, 4, 4)
    )


def _mixed_transition_component(
    left0,
    right0,
    w_x,
    mask,
    maskd,
    idxi,
    idxj,
    idx0,
    idx1,
    weight,
    scale_emat,
    ind,
    nmol: int,
    molsize: int,
    include_coulomb: bool,
):
    nbatch = left0.shape[1]
    Rl = _ao4_nac(left0, nmol, nbatch, molsize)
    Rr = _ao4_nac(right0, nmol, nbatch, molsize)
    pair_grad = torch.zeros(idxi.shape[0], nbatch, 3, dtype=left0.dtype, device=left0.device)

    if include_coulomb:
        Rr_diag = Rr[maskd]
        PA = (Rr_diag[idxi][..., idx0, idx1] * weight).unsqueeze(-1)
        PB = (Rr_diag[idxj][..., idx0, idx1] * weight).unsqueeze(-2)

        J_x_2a = torch.zeros((idxi.shape[0], nbatch, 3, 4, 4), dtype=left0.dtype, device=left0.device)
        J_x_1b = torch.zeros_like(J_x_2a)
        J_x_2a[..., idx0, idx1] = torch.sum(PA.unsqueeze(2) * w_x.unsqueeze(1), dim=3)
        J_x_1b[..., idx0, idx1] = torch.sum(PB.unsqueeze(2) * w_x.unsqueeze(1), dim=4)
        scale = scale_emat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        J_x_2a *= scale
        J_x_1b *= scale
        pair_grad.add_(
            (Rl[maskd[idxj]].unsqueeze(2) * J_x_2a).sum(dim=(3, 4))
            + (Rl[maskd[idxi]].unsqueeze(2) * J_x_1b).sum(dim=(3, 4))
        )

    overlap_rx = torch.zeros((idxi.shape[0], nbatch, 3, 4, 4), dtype=left0.dtype, device=left0.device)
    Pp = Rr[mask].unsqueeze(2)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :].unsqueeze(1)
        for j in range(4):
            overlap_rx[..., i, j] = -0.5 * torch.sum(Pp * w_x_i[..., :, ind[j]], dim=(3, 4))
    pair_grad.add_((2.0 * Rl[mask].unsqueeze(2) * overlap_rx).sum(dim=(3, 4)))
    return pair_grad


def _contract_mixed_transition_terms_kernel(
    RI0, RJ0, w_x, mask, maskd, idxi, idxj, idx0, idx1, weight, scale_emat, ind, nmol: int, molsize: int
):
    nbatch = RI0.shape[1]
    RI_symm = 0.5 * (RI0 + RI0.transpose(-1, -2))
    RJ_symm = 0.5 * (RJ0 + RJ0.transpose(-1, -2))
    RI_antisymm = 0.5 * (RI0 - RI0.transpose(-1, -2))
    RJ_antisymm = 0.5 * (RJ0 - RJ0.transpose(-1, -2))

    pair_grad = 2.0 * _mixed_transition_component(
        RI_symm,
        RJ_symm,
        w_x,
        mask,
        maskd,
        idxi,
        idxj,
        idx0,
        idx1,
        weight,
        scale_emat,
        ind,
        nmol,
        molsize,
        True,
    )
    pair_grad += 2.0 * _mixed_transition_component(
        RI_antisymm,
        RJ_antisymm,
        w_x,
        mask,
        maskd,
        idxi,
        idxj,
        idx0,
        idx1,
        weight,
        scale_emat,
        ind,
        nmol,
        molsize,
        False,
    )

    nac_cis = torch.zeros(nmol * molsize, nbatch, 3, dtype=RI0.dtype, device=RI0.device)
    nac_cis.index_add_(0, idxi, pair_grad)
    nac_cis.index_add_(0, idxj, pair_grad, alpha=-1.0)
    return nac_cis.view(nmol, molsize, nbatch, 3).permute(0, 2, 1, 3)


def calc_nac(
    mol,
    amp,
    e_exc,
    P0,
    ri,
    riXH,
    state_pairs,
    rpa=False,
    pair_batch_size=4,
    include_response_terms=True,
    w=None,
    e_mo=None,
    zvec_tolerance=1e-6,
    **kwargs,
):
    """
    amp: tensor of CIS amplitudes of shape [nmol, nroots, nov].
    state_pairs: iterable of 1-based (state1, state2) pairs.

    Returns a tensor with shape [nmol, len(state_pairs), molsize, 3], ordered like state_pairs.
    """
    if "full_nac" in kwargs:
        include_response_terms = kwargs.pop("full_nac")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
    if rpa:
        raise NotImplementedError(
            "Nonadiabatic coupling vecotrs not yet implemented for RPA. Use CIS instead."
        )
    if include_response_terms and (w is None or e_mo is None):
        raise ValueError("include_response_terms=True requires w and e_mo.")
    device = amp.device
    dtype = amp.dtype
    nmol = int(mol.nmol)
    molsize = int(mol.molsize)
    state_i, state_j = _state_pair_tensors(state_pairs, device)
    pair_sign = torch.where(state_i <= state_j, 1.0, -1.0).to(dtype)
    state_i, state_j = torch.minimum(state_i, state_j), torch.maximum(state_i, state_j)
    n_state_pairs = int(state_i.numel())
    if n_state_pairs == 0:
        return torch.empty((nmol, 0, molsize, 3), dtype=dtype, device=device)

    # CIS unrelaxed density:
    # B = \sum_iab C_\mu a * t_ai * t_bi * C_\nu b - \sum_ija C_\mu i * t_ai * t_aj * C_\nu j
    if include_response_terms:
        nocc, nvirt, Cocc, Cvirt = get_occ_virt(mol)
    else:
        _, _, norb, nocc = _uniform_molecule_dimensions(mol)
        nvirt = norb - nocc
        Cocc = mol.molecular_orbitals[:, :, :nocc]
        Cvirt = mol.molecular_orbitals[:, :, nocc : nocc + nvirt]
    nroots = amp.shape[1]
    amp_ia = amp.view(nmol, nroots, nocc, nvirt)
    if include_response_terms:
        overlap_KAB_x, e1b_x, e2a_x, p0_ortho_grad, ortho_cache, w_x = _build_nac_derivative_operators(
            mol, P0, ri, riXH, dtype, device, return_w_x=True
        )
    else:
        overlap_KAB_x, e1b_x, e2a_x, p0_ortho_grad, ortho_cache = _build_nac_derivative_operators(
            mol, P0, ri, riXH, dtype, device
        )
        w_x = None

    nHeavy, nHydro, _, _ = _uniform_molecule_dimensions(mol)
    size_full = molsize * 4
    pair_batch_size = max(1, int(pair_batch_size))
    nac_cis = torch.empty((nmol, n_state_pairs, molsize, 3), dtype=dtype, device=device)

    for start in range(0, n_state_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, n_state_pairs)
        i_chunk = state_i[start:stop]
        j_chunk = state_j[start:stop]
        nbatch = int(i_chunk.numel())

        amp_i = amp_ia.index_select(1, i_chunk)
        amp_j = amp_ia.index_select(1, j_chunk)
        v_i = torch.einsum("Nma,Nbia->Nbmi", Cvirt, amp_i)
        v_j = torch.einsum("Nma,Nbia->Nbmi", Cvirt, amp_j)
        o_i = torch.einsum("Nmi,Nbia->Nbma", Cocc, amp_i)
        o_j = torch.einsum("Nmi,Nbia->Nbma", Cocc, amp_j)
        Bij_chunk = torch.matmul(v_i, v_j.transpose(-1, -2)) - torch.matmul(o_i, o_j.transpose(-1, -2))
        Bij_chunk = 0.5 * (Bij_chunk + Bij_chunk.transpose(-1, -2))

        density0 = Bij_chunk
        RI0 = RJ0 = None
        if include_response_terms:
            RI = torch.einsum("Nmi,Nbia,Nna->Nbmn", Cocc, amp_i, Cvirt)
            RJ = torch.einsum("Nmi,Nbia,Nna->Nbmn", Cocc, amp_j, Cvirt)
            density0, RI0, RJ0 = _build_pair_response_density_batch(
                mol,
                w,
                e_mo,
                Cocc,
                Cvirt,
                amp_i,
                amp_j,
                Bij_chunk,
                zvec_tolerance,
                pair_response_cache=(RI, RJ, v_i, v_j, o_i, o_j),
            )

        B0 = unpackone_batch(
            density0.reshape(nmol * nbatch, density0.shape[2], density0.shape[3]),
            4 * nHeavy,
            nHydro,
            size_full,
        )
        B = (
            B0.reshape(nmol, nbatch, molsize, 4, molsize, 4)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(nmol * molsize * molsize, nbatch, 4, 4)
        )
        nac_cis[:, start:stop] = _contract_nac_density_batch(
            mol,
            B,
            B0.view(nmol, nbatch, size_full, size_full),
            overlap_KAB_x,
            e1b_x,
            e2a_x,
            p0_ortho_grad,
            ortho_cache,
            nmol,
            molsize,
        )
        if include_response_terms:
            nac_cis[:, start:stop] += _contract_mixed_transition_terms(
                mol,
                unpackone_batch(
                    RI0.reshape(nmol * nbatch, RI0.shape[2], RI0.shape[3]), 4 * nHeavy, nHydro, size_full
                ).view(nmol, nbatch, size_full, size_full),
                unpackone_batch(
                    RJ0.reshape(nmol * nbatch, RJ0.shape[2], RJ0.shape[3]), 4 * nHeavy, nHydro, size_full
                ).view(nmol, nbatch, size_full, size_full),
                w_x,
                dtype,
                device,
            )

    denom = e_exc[:, state_j] - e_exc[:, state_i]
    nac_cis = nac_cis / denom[:, :, None, None]
    # Preserve the sign correction for callers that pass reversed state pairs.
    nac_cis = nac_cis * pair_sign.view(1, n_state_pairs, 1, 1)
    return nac_cis
