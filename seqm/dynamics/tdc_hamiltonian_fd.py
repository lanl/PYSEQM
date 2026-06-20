import torch

from seqm.seqm_functions.anal_grad import _build_omx_ortho_cache, omx_threebody_ortho_grad
from seqm.seqm_functions.anal_grad import delta as omx_fd_delta
from seqm.seqm_functions.constants import a0, overlap_cutoff
from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from seqm.seqm_functions.om2_hcore import build_omx_pair_context
from seqm.seqm_functions.omx_utils import OMX_METHODS, get_orbital_zetas
from seqm.seqm_functions.rcis_batch import unpackone_batch
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as TETCI


def build_fd_displaced_geometries(
    molecule, vel_old, acc_old, dtnact, dtmd, damp=None, langevin_c1=None, langevin_c2=None
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
    ind = torch.tensor(
        [[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.int64, device=device
    )
    weight = torch.tensor(
        [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device
    ).view(1, 10)
    idx0 = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=torch.int64, device=device)
    idx1 = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int64, device=device)
    scale_emat = torch.tensor(
        [[1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )

    overlap_eff = overlap_t
    P_offdiag = P[mol.mask]
    for i in range(4):
        w_i = w_t[..., ind[i], :]
        for j in range(4):
            overlap_eff[..., i, j].sub_(torch.sum(P_offdiag * (w_i[..., :, ind[j]]), dim=(1, 2)))

    PA = P[mol.maskd[mol.idxi]][..., idx0, idx1] * weight
    PB = P[mol.maskd[mol.idxj]][..., idx0, idx1] * weight
    suma = torch.einsum("pi,pij->pj", PA, w_t)
    sumb = torch.einsum("pj,pij->pi", PB, w_t)

    e2a_eff = e2a_t
    e1b_eff = e1b_t
    for k in range(idx0.numel()):
        e2a_eff[..., idx0[k], idx1[k]].add_(suma[..., k])
        e1b_eff[..., idx0[k], idx1[k]].add_(sumb[..., k])

    e1b_eff.mul_(scale_emat)
    e2a_eff.mul_(scale_emat)
    return overlap_eff, e1b_eff, e2a_eff


def _contract_pair_density_directional_batch(mol, B, overlap_eff, e1b_eff, e2a_eff, nmol):
    # B: (nmol*molsize*molsize, n_state_pairs, 4, 4)
    B_offdiag = B[mol.mask]
    B_diag_j = B[mol.maskd[mol.idxj]]
    B_diag_i = B[mol.maskd[mol.idxi]]

    pair_val = (B_offdiag * overlap_eff[:, None, :, :]).sum(dim=(2, 3))
    pair_val = pair_val + (B_diag_j * e2a_eff[:, None, :, :]).sum(dim=(2, 3))
    pair_val = pair_val + (B_diag_i * e1b_eff[:, None, :, :]).sum(dim=(2, 3))

    out = torch.zeros((nmol, pair_val.shape[1]), dtype=B.dtype, device=B.device)
    out.index_add_(0, mol.pair_molid, pair_val)
    return out


def _project_omx_orthogonalization_velocity(molecule, density, vel_eff, ortho_cache):
    if molecule.method not in {"OM2", "OM3"}:
        return None
    if ortho_cache is None:
        raise RuntimeError("OMx orthogonalization projection requires a prebuilt ortho_cache.")

    cache = ortho_cache
    npairs = molecule.rij.shape[0]
    dtype = density.dtype
    device = density.device
    rep = lambda x: torch.cat((x, x), dim=0)
    idxi_ = rep(molecule.idxi)
    idxj_ = rep(molecule.idxj)
    ni_ = rep(molecule.ni)
    nj_ = rep(molecule.nj)

    Xij = molecule.xij * molecule.rij.unsqueeze(1) * a0
    S_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
    B_x = torch.zeros_like(S_x)
    pair_core_semi_x = None
    if molecule.method == "OM2":
        pair_core_semi_x = torch.zeros((npairs, 3, 4, 2), dtype=dtype, device=device)

    # These are Cartesian x/y/z derivatives. The directionally projected
    # pair_overlap_t / pair_core_semi_t from _directional_omx_derivatives
    # are not interchangeable here.
    for coord in range(3):
        Xij[:, coord] -= omx_fd_delta
        rij_plus = torch.linalg.norm(Xij, dim=1)
        xij_plus = Xij / rij_plus.unsqueeze(1)
        rij_plus = rij_plus / a0

        Xij[:, coord] += 2.0 * omx_fd_delta
        rij_minus = torch.linalg.norm(Xij, dim=1)
        xij_minus = Xij / rij_minus.unsqueeze(1)
        rij_minus = rij_minus / a0

        xij_ = torch.cat((xij_plus, xij_minus), dim=0)
        rij_ = torch.cat((rij_plus, rij_minus), dim=0)
        ctx = build_omx_pair_context(
            molecule, method=molecule.method, idxi=idxi_, idxj=idxj_, ni=ni_, nj=nj_, xij=xij_, rij=rij_
        )
        pair_resonance_t = (ctx["pair_resonance"][:npairs] - ctx["pair_resonance"][npairs:]) / (
            2.0 * omx_fd_delta
        )
        pair_overlap_t = (ctx["pair_overlap"][:npairs] - ctx["pair_overlap"][npairs:]) / (2.0 * omx_fd_delta)
        pair_core_semi_t = None
        if pair_core_semi_x is not None:
            pair_core_semi_t = (ctx["pair"]["core_semi"][:npairs] - ctx["pair"]["core_semi"][npairs:]) / (
                2.0 * omx_fd_delta
            )
        Xij[:, coord] -= omx_fd_delta

        B_x[:, coord] = pair_resonance_t
        S_x[:, coord] = pair_overlap_t
        if pair_core_semi_x is not None:
            pair_core_semi_x[:, coord] = pair_core_semi_t

    p0_ortho = omx_threebody_ortho_grad(
        molecule,
        density,
        S_x,
        B_x,
        pair_core_semi_x,
        ortho_cache=cache,
        unrestricted=False,
        vel_eff=vel_eff[:, : molecule.molsize],
    )
    cache = {**cache, "S_x": S_x, "B_x": B_x, "pair_core_semi_x": pair_core_semi_x}
    if p0_ortho.dim() == 1:
        p0_ortho = p0_ortho.unsqueeze(1)
    return p0_ortho, cache


def compute_tdc_hamiltonian_fd(nad, molecule, cache_new, learned_parameters, vel_old, acc_old, validate=True):
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
            raise NotImplementedError("hamiltonian_fd TD-NAC currently supports CIS amplitudes only.")

    dtnact = nad._dtnact

    dm_ref = molecule.dm
    mos_ref = molecule.molecular_orbitals
    p0_ortho = None
    ortho_cache = None

    R_plus, R_minus = build_fd_displaced_geometries(
        molecule,
        vel_old,
        acc_old,
        dtnact,
        nad.timestep,
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
            ortho_cache = _build_omx_ortho_cache(molecule)
            vel_eff = (R_plus - R_minus) / (2.0 * dtnact)
            p0_ortho, ortho_cache = _project_omx_orthogonalization_velocity(
                molecule, dm_ref, vel_eff, ortho_cache=ortho_cache
            )

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

    P = dm_ref.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
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

        v_i = Bvirt.index_select(1, i_chunk)
        v_j = Bvirt.index_select(1, j_chunk)
        o_i = Bocc.index_select(1, i_chunk)
        o_j = Bocc.index_select(1, j_chunk)
        Bij_chunk = torch.matmul(v_i, v_j.transpose(-1, -2)) - torch.matmul(o_i, o_j.transpose(-1, -2))
        Bij_chunk = 0.5 * (Bij_chunk + Bij_chunk.transpose(-1, -2))

        B0 = unpackone_batch(
            Bij_chunk.reshape(nmol * nbatch, Bij_chunk.shape[2], Bij_chunk.shape[3]),
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
            dot_h_upper[:, start:stop].add_(p0_ortho)
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
