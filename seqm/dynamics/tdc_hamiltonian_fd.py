import torch

from seqm.seqm_functions.anal_grad import _build_omx_ortho_fd_cache, omx_threebody_ortho_grad
from seqm.seqm_functions.constants import a0, overlap_cutoff
from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from seqm.seqm_functions.fock import (
    EMAT_SCALE_4,
    UPPER_IDX0_4,
    UPPER_IDX1_4,
    WEIGHT_10,
    K_ind_4,
    _cached_index,
    _cached_tensor,
)
from seqm.seqm_functions.nac import _build_pair_response_density_batch
from seqm.seqm_functions.om2_hcore import build_omx_pair_context
from seqm.seqm_functions.omx_utils import OMX_METHODS, get_orbital_zetas
from seqm.seqm_functions.rcis_batch import unpackone_batch
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as TETCI
from seqm.utils.torch_compile import optional_compile_function

_prepare_directional_pair_ops_dispatch = None
_contract_pair_density_directional_dispatch = None
_contract_mixed_transition_directional_dispatch = None


def enable_tdc_hamiltonian_fd_compile(mode=None, **options):
    """Compile tensor contractions used by Hamiltonian finite-difference TD-NAC."""
    global _prepare_directional_pair_ops_dispatch
    global _contract_pair_density_directional_dispatch
    global _contract_mixed_transition_directional_dispatch

    compile_options = dict(options)
    if mode is not None:
        compile_options["mode"] = mode

    _prepare_directional_pair_ops_dispatch = optional_compile_function(
        _prepare_pair_operators_directional_kernel,
        compile_options=compile_options,
        label="tdc_fd.prepare_pair_ops",
    )
    _contract_pair_density_directional_dispatch = optional_compile_function(
        _contract_pair_density_directional_kernel,
        compile_options=compile_options,
        label="tdc_fd.contract_density",
    )
    _contract_mixed_transition_directional_dispatch = optional_compile_function(
        _contract_mixed_transition_directional_kernel,
        compile_options=compile_options,
        label="tdc_fd.contract_mixed_transition",
    )


def build_fd_displaced_geometries(
    molecule, vel_old, acc_old, dtnact, damp=None, langevin_c1=None, langevin_c2=None
):
    R = molecule.coordinates.detach()
    vel_eff = vel_old.clone()
    if damp is not None:
        if langevin_c1 is None or langevin_c2 is None:
            raise RuntimeError("Langevin TD-NAC requires langevin_c1 and langevin_c2.")
        noise = torch.randn_like(vel_old)
        vel_eff = langevin_c1 * vel_eff + langevin_c2 * noise

    # Technically R+dtnact = R_old + vel_old*(dtmd+dtnact) + 0.5*acc_old*(dtmd+dtnact)**2
    # = R_old + vel_old*dtmd + 0.5*acc_old*dtmd**2 + vel_old*dtnact + acc_old*dtmd*dtnact + 0.5*acc_old*dtnact**2
    # = R + vel_old*dtnact + acc_old*dtmd*dtnact + 0.5*acc_old*dtnact**2.
    # But NEXMD strangely ignores acc_old*dtmd*dtnact and keeps 0.5*acc_old*dtnact**2 term, so we do the same for consistency.
    vel_eff = vel_eff + 0.5 * acc_old * dtnact
    disp = vel_eff * dtnact
    return R + disp, R - disp


def _pair_geometry_from_coords(mol, coords):
    flat = coords.reshape(-1, 3)
    Xij = flat[mol.idxj] - flat[mol.idxi]  # match Parser orientation (mol.xij)
    dist = torch.linalg.norm(Xij, dim=1)
    if torch.any(dist <= 0.0):
        raise RuntimeError("Encountered zero interatomic distance in TD-NAC finite difference.")
    xij = Xij / dist.unsqueeze(1)
    rij = dist / a0
    return xij, rij


def _directional_omx_derivatives(mol, xij_plus, rij_plus, xij_minus, rij_minus, dtnact):
    npairs = xij_plus.shape[0]
    dtype = xij_plus.dtype
    device = xij_plus.device

    if npairs == 0:
        empty_mat = torch.zeros((0, 4, 4), dtype=dtype, device=device)
        empty_vec = torch.zeros((0, 3), dtype=dtype, device=device)
        empty_cs = torch.zeros((0, 4, 2), dtype=dtype, device=device)
        return empty_mat.clone(), empty_vec, empty_mat.clone(), empty_mat.clone(), empty_mat.clone(), empty_cs

    idxi_ = torch.cat((mol.idxi, mol.idxi), dim=0)
    idxj_ = torch.cat((mol.idxj, mol.idxj), dim=0)
    ni_ = torch.cat((mol.ni, mol.ni), dim=0)
    nj_ = torch.cat((mol.nj, mol.nj), dim=0)
    xij_ = torch.cat((xij_plus, xij_minus), dim=0)
    rij_ = torch.cat((rij_plus, rij_minus), dim=0)

    # Mirror the OMx finite-difference path used by omx_fd in anal_grad.py.
    ctx = build_omx_pair_context(
        mol, method=mol.method, idxi=idxi_, idxj=idxj_, ni=ni_, nj=nj_, xij=xij_, rij=rij_
    )
    pair = ctx["pair"]
    pair_resonance_t = (ctx["pair_resonance"][:npairs] - ctx["pair_resonance"][npairs:]) / (2.0 * dtnact)
    w_t = (pair["w"][:npairs] - pair["w"][npairs:]) / (2.0 * dtnact)
    e1b_t = (pair["e1b"][:npairs] - pair["e1b"][npairs:]) / (2.0 * dtnact)
    e2a_t = (pair["e2a"][:npairs] - pair["e2a"][npairs:]) / (2.0 * dtnact)
    return pair_resonance_t, w_t, e1b_t, e2a_t


def _directional_overlap_derivative(mol, xij_plus, rij_plus, xij_minus, rij_minus, dtnact):
    npairs = xij_plus.shape[0]
    dtype = xij_plus.dtype
    device = xij_plus.device

    zeta_s, zeta_p = get_orbital_zetas(mol.parameters, mol.method)
    zeta = torch.cat((zeta_s.unsqueeze(1), zeta_p.unsqueeze(1)), dim=1)
    beta = mol.parameters["beta"]

    di_plus = torch.zeros((npairs, 4, 4), dtype=dtype, device=device)
    di_minus = torch.zeros_like(di_plus)

    mask_plus = rij_plus <= overlap_cutoff
    mask_minus = rij_minus <= overlap_cutoff
    if mask_plus.any():
        di_plus[mask_plus] = diatom_overlap_matrix_PM6_SP(
            mol.ni[mask_plus],
            mol.nj[mask_plus],
            xij_plus[mask_plus],
            rij_plus[mask_plus],
            zeta[mol.idxi][mask_plus],
            zeta[mol.idxj][mask_plus],
            mol.const.qn_int,
        )
    if mask_minus.any():
        di_minus[mask_minus] = diatom_overlap_matrix_PM6_SP(
            mol.ni[mask_minus],
            mol.nj[mask_minus],
            xij_minus[mask_minus],
            rij_minus[mask_minus],
            zeta[mol.idxi][mask_minus],
            zeta[mol.idxj][mask_minus],
            mol.const.qn_int,
        )

    overlap_t = (di_plus - di_minus) / (2.0 * dtnact)
    overlap_t[..., 0, 0] *= beta[mol.idxi, 0] + beta[mol.idxj, 0]
    overlap_t[..., 0, 1:] *= beta[mol.idxi, 0:1] + beta[mol.idxj, 1:2]
    overlap_t[..., 1:, 0] *= beta[mol.idxi, 1:2] + beta[mol.idxj, 0:1]
    overlap_t[..., 1:, 1:] *= beta[mol.idxi, 1:2, None] + beta[mol.idxj, 1:2, None]
    return overlap_t


def _directional_tetci_derivative(mol, xij_plus, rij_plus, xij_minus, rij_minus, dtnact):
    npairs = xij_plus.shape[0]

    zeta_s, zeta_p = get_orbital_zetas(mol.parameters, mol.method)
    rep = lambda x: torch.cat([x, x], dim=0)
    ni_ = rep(mol.ni)
    nj_ = rep(mol.nj)
    Z_ = rep(mol.Z)
    idxi_ = rep(mol.idxi)
    idxj_ = rep(mol.idxj)
    zeta_s_ = rep(zeta_s)
    zeta_p_ = rep(zeta_p)
    g_ss_ = rep(mol.parameters["g_ss"])
    g_pp_ = rep(mol.parameters["g_pp"])
    g_p2_ = rep(mol.parameters["g_p2"])
    h_sp_ = rep(mol.parameters["h_sp"])
    rho_core_ = rep(mol.parameters["rho_core"])

    rij_ = torch.cat([rij_plus, rij_minus], dim=0)
    xij_ = torch.cat([xij_plus, xij_minus], dim=0)

    w_, e1b_, e2a_, _, _, _, _ = TETCI(
        mol.const,
        idxi_,
        idxj_,
        ni_,
        nj_,
        xij_,
        rij_,
        Z_,
        zeta_s_,
        zeta_p_,
        None,
        None,
        None,
        None,
        g_ss_,
        g_pp_,
        g_p2_,
        h_sp_,
        None,
        None,
        rho_core_,
        None,
        None,
        mol.method,
    )

    w_t = (w_[:npairs] - w_[npairs:]) / (2.0 * dtnact)
    e1b_t = (e1b_[:npairs] - e1b_[npairs:]) / (2.0 * dtnact)
    e2a_t = (e2a_[:npairs] - e2a_[npairs:]) / (2.0 * dtnact)
    return w_t, e1b_t, e2a_t


def _prepare_pair_operators_for_directional_nac(mol, P, overlap_t, w_t, e1b_t, e2a_t):
    device = P.device
    dtype = P.dtype
    ind = _cached_index(K_ind_4, device)
    weight = _cached_tensor(WEIGHT_10, device, dtype).view(1, 10)
    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)
    scale_emat = _cached_tensor(EMAT_SCALE_4, device, dtype)

    dispatch = _prepare_directional_pair_ops_dispatch or _prepare_pair_operators_directional_kernel
    return dispatch(
        P,
        overlap_t,
        w_t,
        e1b_t,
        e2a_t,
        mol.mask,
        mol.maskd,
        mol.idxi,
        mol.idxj,
        ind,
        idx0,
        idx1,
        weight,
        scale_emat,
    )


def _prepare_pair_operators_directional_kernel(
    P, overlap_t, w_t, e1b_t, e2a_t, mask, maskd, idxi, idxj, ind, idx0, idx1, weight, scale_emat
):
    overlap_eff = overlap_t.clone()
    P_offdiag = P[mask]
    for i in range(4):
        w_i = w_t[..., ind[i], :]
        for j in range(4):
            overlap_eff[..., i, j].sub_(torch.sum(P_offdiag * (w_i[..., :, ind[j]]), dim=(1, 2)))

    PA = P[maskd[idxi]][..., idx0, idx1] * weight
    PB = P[maskd[idxj]][..., idx0, idx1] * weight
    suma = torch.einsum("pi,pij->pj", PA, w_t)
    sumb = torch.einsum("pj,pij->pi", PB, w_t)

    e2a_eff = e2a_t.clone()
    e1b_eff = e1b_t.clone()
    for k in range(idx0.numel()):
        e2a_eff[..., idx0[k], idx1[k]].add_(suma[..., k])
        e1b_eff[..., idx0[k], idx1[k]].add_(sumb[..., k])

    e1b_eff.mul_(scale_emat)
    e2a_eff.mul_(scale_emat)
    return overlap_eff, e1b_eff, e2a_eff


def _contract_pair_density_directional_batch(mol, B, overlap_eff, e1b_eff, e2a_eff, nmol):
    # B: (nmol*molsize*molsize, n_state_pairs, 4, 4)
    dispatch = _contract_pair_density_directional_dispatch or _contract_pair_density_directional_kernel
    return dispatch(
        B,
        overlap_eff,
        e1b_eff,
        e2a_eff,
        mol.mask,
        mol.maskd[mol.idxi],
        mol.maskd[mol.idxj],
        mol.pair_molid,
        int(nmol),
    )


def _contract_pair_density_directional_kernel(
    B, overlap_eff, e1b_eff, e2a_eff, mask, maskd_idxi, maskd_idxj, pair_molid, nmol: int
):
    B_offdiag = B[mask]
    B_diag_j = B[maskd_idxj]
    B_diag_i = B[maskd_idxi]
    pair_val = (B_offdiag * overlap_eff[:, None, :, :]).sum(dim=(2, 3))
    pair_val = pair_val + (B_diag_j * e2a_eff[:, None, :, :]).sum(dim=(2, 3))
    pair_val = pair_val + (B_diag_i * e1b_eff[:, None, :, :]).sum(dim=(2, 3))

    out = torch.zeros((nmol, pair_val.shape[1]), dtype=B.dtype, device=B.device)
    out.index_add_(0, pair_molid, pair_val)
    return out


def _contract_mixed_transition_terms_directional_batch(mol, RI0, RJ0, w_t, dtype, device):
    nmol, nbatch, _, _ = RI0.shape
    molsize = int(mol.molsize)
    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)
    weight = _cached_tensor(WEIGHT_10, device, dtype).reshape(1, 1, 10)
    scale_emat = _cached_tensor(EMAT_SCALE_4, device, dtype)
    ind = _cached_index(K_ind_4, device)

    dispatch = (
        _contract_mixed_transition_directional_dispatch or _contract_mixed_transition_directional_kernel
    )
    return dispatch(
        RI0,
        RJ0,
        w_t,
        mol.mask,
        mol.maskd,
        mol.idxi,
        mol.idxj,
        mol.pair_molid,
        idx0,
        idx1,
        weight,
        scale_emat,
        ind,
        nmol,
        nbatch,
        molsize,
    )


def _ao4_directional(T, nmol: int, nbatch: int, molsize: int):
    return (
        T.reshape(nmol, nbatch, molsize, 4, molsize, 4)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(nmol * molsize * molsize, nbatch, 4, 4)
    )


def _directional_mixed_component(
    left0,
    right0,
    w_t,
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
    nbatch: int,
    molsize: int,
    include_coulomb: bool,
):
    Rl = _ao4_directional(left0, nmol, nbatch, molsize)
    Rr = _ao4_directional(right0, nmol, nbatch, molsize)
    pair_val = torch.zeros((idxi.shape[0], nbatch), dtype=left0.dtype, device=left0.device)
    if include_coulomb:
        Rr_diag = Rr[maskd]
        PA = (Rr_diag[idxi][..., idx0, idx1] * weight).unsqueeze(-1)
        PB = (Rr_diag[idxj][..., idx0, idx1] * weight).unsqueeze(-2)

        J_t_2a = torch.zeros((idxi.shape[0], nbatch, 4, 4), dtype=left0.dtype, device=left0.device)
        J_t_1b = torch.zeros_like(J_t_2a)
        J_t_2a[..., idx0, idx1] = torch.sum(PA * w_t.unsqueeze(1), dim=2)
        J_t_1b[..., idx0, idx1] = torch.sum(PB * w_t.unsqueeze(1), dim=3)
        J_t_2a *= scale_emat.unsqueeze(0).unsqueeze(0)
        J_t_1b *= scale_emat.unsqueeze(0).unsqueeze(0)
        pair_val.add_((Rl[maskd[idxj]] * J_t_2a).sum(dim=(2, 3)) + (Rl[maskd[idxi]] * J_t_1b).sum(dim=(2, 3)))

    overlap_rt = torch.zeros((idxi.shape[0], nbatch, 4, 4), dtype=left0.dtype, device=left0.device)
    Pp = Rr[mask]
    for i in range(4):
        w_i = w_t[..., ind[i], :]
        for j in range(4):
            overlap_rt[..., i, j] = -0.5 * torch.sum(Pp * w_i[:, None, :, ind[j]], dim=(2, 3))
    pair_val.add_((2.0 * Rl[mask] * overlap_rt).sum(dim=(2, 3)))
    return pair_val


def _contract_mixed_transition_directional_kernel(
    RI0,
    RJ0,
    w_t,
    mask,
    maskd,
    idxi,
    idxj,
    pair_molid,
    idx0,
    idx1,
    weight,
    scale_emat,
    ind,
    nmol: int,
    nbatch: int,
    molsize: int,
):
    RI_symm = 0.5 * (RI0 + RI0.transpose(-1, -2))
    RJ_symm = 0.5 * (RJ0 + RJ0.transpose(-1, -2))
    RI_antisymm = 0.5 * (RI0 - RI0.transpose(-1, -2))
    RJ_antisymm = 0.5 * (RJ0 - RJ0.transpose(-1, -2))
    pair_val = 2.0 * _directional_mixed_component(
        RI_symm,
        RJ_symm,
        w_t,
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
        nbatch,
        molsize,
        True,
    )
    pair_val += 2.0 * _directional_mixed_component(
        RI_antisymm,
        RJ_antisymm,
        w_t,
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
        nbatch,
        molsize,
        False,
    )

    out = torch.zeros((nmol, nbatch), dtype=RI0.dtype, device=RI0.device)
    out.index_add_(0, pair_molid, pair_val)
    return out


def compute_tdc_hamiltonian_fd(
    nad,
    molecule,
    cache_new,
    learned_parameters,
    vel_old,
    acc_old,
    include_response_terms=False,
    validate=True,
):
    ref_amp = cache_new["cis_amp"]
    ref_energies = cache_new["energies"]
    if validate:
        if molecule.method == "PM6":
            raise NotImplementedError("hamiltonian_fd TD-NAC is not implemented for PM6.")
        if molecule.nocc.dim() != 1:
            raise NotImplementedError(
                "hamiltonian_fd TD-NAC currently supports restricted closed-shell only."
            )
        if ref_amp.dim() != 3:
            raise NotImplementedError(
                "hamiltonian_fd TD-NAC currently supports CIS amplitudes only, not RPA."
            )
    if include_response_terms and (
        getattr(molecule, "w", None) is None or getattr(molecule, "e_mo", None) is None
    ):
        raise ValueError("include_response_terms=True requires molecule.w and molecule.e_mo.")

    dtnact = nad._dtnact

    mos_ref = molecule.molecular_orbitals
    ortho_cache = None

    R_plus, R_minus = build_fd_displaced_geometries(
        molecule,
        vel_old,
        acc_old,
        dtnact,
        damp=nad.damp,
        langevin_c1=getattr(nad, "langevin_c1", None),
        langevin_c2=getattr(nad, "langevin_c2", None),
    )

    xij_plus, rij_plus = _pair_geometry_from_coords(molecule, R_plus)
    xij_minus, rij_minus = _pair_geometry_from_coords(molecule, R_minus)
    if molecule.method in OMX_METHODS:
        pair_resonance_t, w_t, e1b_t, e2a_t = _directional_omx_derivatives(
            molecule, xij_plus, rij_plus, xij_minus, rij_minus, dtnact
        )
        overlap_t = 2.0 * pair_resonance_t
        if molecule.method in {"OM2", "OM3"}:
            Xij = molecule.xij * molecule.rij.unsqueeze(1) * a0
            ortho_cache = _build_omx_ortho_fd_cache(
                molecule,
                Xij,
                molecule.ni,
                molecule.nj,
                molecule.idxi,
                molecule.idxj,
                molecule.method,
                mos_ref.dtype,
                mos_ref.device,
            )
            vel_eff = (R_plus - R_minus) / (2.0 * dtnact)

    else:
        overlap_t = _directional_overlap_derivative(
            molecule, xij_plus, rij_plus, xij_minus, rij_minus, dtnact
        )
        w_t, e1b_t, e2a_t = _directional_tetci_derivative(
            molecule, xij_plus, rij_plus, xij_minus, rij_minus, dtnact
        )

    nmol = int(molecule.nmol)
    molsize = int(molecule.molsize)
    nocc = int(molecule.nocc[0].item())
    nstates = int(ref_amp.shape[1])
    nov = int(ref_amp.shape[2])
    if nocc <= 0 or nov % nocc != 0:
        raise ValueError(f"Inconsistent CIS dimensions: nocc={nocc}, nov={nov}")
    if ref_energies.shape[1] < nstates:
        raise ValueError("Energies tensor has fewer states than cis_amp.")
    nvirt = nov // nocc
    if int(molecule.norb[0].item()) < (nocc + nvirt):
        raise ValueError("Not enough orbitals to match CIS amplitude dimensions.")

    P = (
        molecule.dm.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )
    overlap_eff, e1b_eff, e2a_eff = _prepare_pair_operators_for_directional_nac(
        molecule, P, overlap_t, w_t, e1b_t, e2a_t
    )

    amp = ref_amp.view(nmol, nstates, nocc, nvirt)
    Cocc = mos_ref[:, :, :nocc]
    Cvirt = mos_ref[:, :, nocc : (nocc + nvirt)]
    Bvirt = torch.einsum("bma,bria->brmi", Cvirt, amp)
    Bocc = torch.einsum("bmi,bria->brma", Cocc, amp)

    state_i, state_j = torch.triu_indices(nstates, nstates, offset=1, device=ref_amp.device)
    if state_i.numel() == 0:
        raise RuntimeError("At least two excited states are required for TD-NAC.")
    n_state_pairs = int(state_i.numel())
    nHeavy = int(molecule.nHeavy[0].item())
    nHydro = int(molecule.nHydro[0].item())
    size_full = molsize * 4
    pair_batch_size = nstates
    dot_h_upper = torch.zeros((nmol, n_state_pairs), dtype=ref_amp.dtype, device=ref_amp.device)
    for start in range(0, n_state_pairs, pair_batch_size):
        stop = min(start + pair_batch_size, n_state_pairs)
        i_chunk = state_i[start:stop]
        j_chunk = state_j[start:stop]
        nbatch = int(i_chunk.numel())

        amp_i = amp.index_select(1, i_chunk)
        amp_j = amp.index_select(1, j_chunk)
        v_i = Bvirt.index_select(1, i_chunk)
        v_j = Bvirt.index_select(1, j_chunk)
        o_i = Bocc.index_select(1, i_chunk)
        o_j = Bocc.index_select(1, j_chunk)
        Bij_chunk = torch.matmul(v_i, v_j.transpose(-1, -2)) - torch.matmul(o_i, o_j.transpose(-1, -2))
        Bij_chunk = 0.5 * (Bij_chunk + Bij_chunk.transpose(-1, -2))
        density0 = Bij_chunk
        RI0 = RJ0 = None
        if include_response_terms:
            RI = torch.einsum("bmi,bria,bna->brmn", Cocc, amp_i, Cvirt)
            RJ = torch.einsum("bmi,bria,bna->brmn", Cocc, amp_j, Cvirt)
            density0, RI0, RJ0 = _build_pair_response_density_batch(
                molecule,
                molecule.w,
                molecule.e_mo,
                Cocc,
                Cvirt,
                amp_i,
                amp_j,
                Bij_chunk,
                molecule.seqm_parameters["excited_states"]["tolerance"],
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

        dot_h_upper[:, start:stop] = _contract_pair_density_directional_batch(
            molecule, B, overlap_eff, e1b_eff, e2a_eff, nmol
        )
        if ortho_cache is not None:
            dot_h_upper[:, start:stop].add_(
                omx_threebody_ortho_grad(
                    molecule,
                    B0.view(nmol, nbatch, size_full, size_full),
                    ortho_cache["S_x"],
                    ortho_cache["B_x"],
                    ortho_cache["pair_core_semi_x"],
                    ortho_cache=ortho_cache,
                    unrestricted=False,
                    vel_eff=vel_eff[:, :molsize],
                )
            )
        if include_response_terms:
            dot_h_upper[:, start:stop].add_(
                _contract_mixed_transition_terms_directional_batch(
                    molecule,
                    unpackone_batch(
                        RI0.reshape(nmol * nbatch, RI0.shape[2], RI0.shape[3]), 4 * nHeavy, nHydro, size_full
                    ).view(nmol, nbatch, size_full, size_full),
                    unpackone_batch(
                        RJ0.reshape(nmol * nbatch, RJ0.shape[2], RJ0.shape[3]), 4 * nHeavy, nHydro, size_full
                    ).view(nmol, nbatch, size_full, size_full),
                    w_t,
                    ref_amp.dtype,
                    ref_amp.device,
                )
            )

    denom_upper = ref_energies[:, state_j] - ref_energies[:, state_i]
    # if torch.any(denom_upper.abs() < 1e-12):
    #     raise RuntimeError(
    #         "Small energy gap encountered in TD-NAC finite difference, leading to numerical instability."
    #     )
    nac_upper = dot_h_upper / denom_upper

    nac_dt = torch.zeros((nmol, nstates, nstates), dtype=ref_amp.dtype, device=ref_amp.device)
    nac_dt[:, state_i, state_j] = nac_upper
    nac_dt[:, state_j, state_i] = -nac_upper
    return nac_dt
