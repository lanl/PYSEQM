import torch

from seqm.seqm_functions.constants import a0, overlap_cutoff
from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from seqm.seqm_functions.rcis_batch import unpackone_batch
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as TETCI


def build_fd_displaced_geometries(
    molecule, vel_old, acc_old, dtnact, dtmd, damp=None, langevin_c1=None, langevin_c2=None
):
    R = molecule.coordinates.detach()
    if damp is not None:
        if langevin_c1 is None or langevin_c2 is None:
            raise RuntimeError("Langevin TD-NAC requires langevin_c1 and langevin_c2.")
        noise = torch.randn_like(vel_old)
        vel_old = langevin_c1 * vel_old + langevin_c2 * noise

    # Technically R+dtnact = R_old + vel_old*(dtmd+dtnact) + 0.5*acc_old*(dtmd+dtnact)**2
    # = R_old + vel_old*dtmd + 0.5*acc_old*dtmd**2 + vel_old*dtnact + acc_old*dtmd*dtnact + 0.5*acc_old*dtnact**2
    # = R + vel_old*dtnact + acc_old*dtmd*dtnact + 0.5*acc_old*dtnact**2.
    # But NEXMD strangely ignores acc_old*dtmd*dtnact and keeps 0.5*acc_old*dtnact**2 term, so we do the same for consistency.
    vel_old += 0.5 * acc_old * dtnact
    disp = vel_old * dtnact
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


def _directional_overlap_derivative(mol, xij_plus, rij_plus, xij_minus, rij_minus, dtnact):
    npairs = xij_plus.shape[0]
    dtype = xij_plus.dtype
    device = xij_plus.device

    zeta = torch.cat((mol.parameters["zeta_s"].unsqueeze(1), mol.parameters["zeta_p"].unsqueeze(1)), dim=1)
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

    rep = lambda x: torch.cat([x, x], dim=0)
    ni_ = rep(mol.ni)
    nj_ = rep(mol.nj)
    Z_ = rep(mol.Z)
    idxi_ = rep(mol.idxi)
    idxj_ = rep(mol.idxj)
    zeta_s_ = rep(mol.parameters["zeta_s"])
    zeta_p_ = rep(mol.parameters["zeta_p"])
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


def compute_tdc_hamiltonian_fd(nad, molecule, cache_new, learned_parameters, vel_old, acc_old):
    if molecule.method == "PM6":
        raise NotImplementedError("hamiltonian_fd TD-NAC is not implemented for PM6.")
    if molecule.nocc.dim() != 1:
        raise NotImplementedError("hamiltonian_fd TD-NAC currently supports restricted closed-shell only.")

    ref_amp = cache_new.get("cis_amp")
    ref_energies = cache_new.get("energies")
    if ref_amp is None or ref_energies is None:
        raise RuntimeError("cache_new must contain 'cis_amp' and 'energies'.")
    if ref_amp.dim() != 3:
        raise NotImplementedError("hamiltonian_fd TD-NAC currently supports CIS amplitudes only.")

    dm_ref = molecule.dm
    mos_ref = molecule.molecular_orbitals

    dtnact = nad._dtnact

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
    overlap_t = _directional_overlap_derivative(molecule, xij_plus, rij_plus, xij_minus, rij_minus, dtnact)
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
    dot_h_upper = torch.empty((nmol, n_state_pairs), dtype=ref_amp.dtype, device=ref_amp.device)

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
