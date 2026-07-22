import torch

from seqm.seqm_functions.anal_grad import (
    core_core_der,
    omx_fd,
    overlap_der_finiteDiff,
    w_der,
    w_derivative_numerical,
)
from seqm.seqm_functions.fock import (
    EMAT_SCALE_4,
    UPPER_IDX0_4,
    UPPER_IDX1_4,
    WEIGHT_10,
    K_ind_4,
    _cached_index,
    _cached_tensor,
)
from seqm.seqm_functions.rcis_batch import _uniform_molecule_dimensions, make_cis_densities, unpackone_batch
from seqm.utils.torch_compile import optional_compile_function

from .constants import a0, ev
from .dispersion_am1_fs1 import dEdisp_dr
from .omx_utils import get_orbital_zetas

_rcis_grad_contract_dispatch = None


def enable_rcis_grad_compile(mode=None, **options):
    """Compile tensor contractions used after CIS derivative integrals are built."""
    global _rcis_grad_contract_dispatch

    compile_options = dict(options)
    if mode is not None:
        compile_options["mode"] = mode

    _rcis_grad_contract_dispatch = optional_compile_function(
        _rcis_grad_contract_kernel, compile_options=compile_options, label="rcis_grad.contract"
    )


def rcis_grad_batch(
    mol,
    w,
    e_mo,
    riXH,
    ri,
    P0,
    zvec_tolerance,
    gam,
    method,
    parnuc,
    rpa=False,
    include_ground_state=False,
    orbital_window=None,
    calculate_dipole=False,
):
    """
    amp: tensor of CIS amplitudes of shape [b,nov]. For each of the b molecules, the CIS amplitues of the
         state for which the gradient is required has to be selected and put together into the amp tensor
    """
    molsize = mol.molsize
    nHeavy, nHydro, _, _ = _uniform_molecule_dimensions(mol)
    cis_densities = make_cis_densities(
        mol,
        do_transition_denisty=True,
        do_difference_density=True,
        do_relaxed_density=True,
        orbital_window=orbital_window,
        w=w,
        e_mo=e_mo,
        zvec_tolerance=zvec_tolerance,
        rpa=rpa,
    )
    if calculate_dipole:
        make_cis_state_dipole(
            mol, cis_densities["difference_density"], cis_densities["relaxed_difference_density"], P0
        )
    # B0 = torch.stack([ unpackone(dens_BR[i,0], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    B0 = unpackone_batch(cis_densities["relaxed_difference_density"], 4 * nHeavy, nHydro, molsize * 4)
    # R0 = torch.stack([ unpackone(dens_BR[i,1], 4*nHeavy, nHydro, molsize * 4)
    #     for i in range(nmol)]).view(nmol,molsize * 4, molsize * 4)
    R0 = unpackone_batch(cis_densities["transition_density"], 4 * nHeavy, nHydro, molsize * 4)

    del cis_densities

    ###############################
    # Calculate the gradient of CIS energies

    npairs = mol.rij.shape[0]
    dtype = B0.dtype
    device = B0.device
    nmol = mol.nmol
    omx_orthogonalization_grad = None

    if method in {"OM1", "OM2", "OM3"}:
        overlap_x = torch.zeros((npairs, 3, 4, 4), dtype=dtype, device=device)
        Xij = mol.xij * mol.rij.unsqueeze(1) * a0
        w_x = torch.zeros(mol.rij.shape[0], 3, 10, 10, dtype=dtype, device=device)
        ortho_density = B0 if not include_ground_state else B0 + P0
        e1b_x, e2a_x, fko_x, omx_orthogonalization_grad, _ = omx_fd(
            mol, overlap_x, w_x, Xij, mol.ni, mol.nj, mol.idxi, mol.idxj, method, ortho_density
        )

        tore = mol.const.tore
        ZAZB = tore[mol.ni] * tore[mol.nj]
        pair_grad = torch.zeros_like(Xij)
        if include_ground_state:
            pair_grad = (
                ZAZB.unsqueeze(1)
                * ev
                * (
                    fko_x / mol.rij.unsqueeze(1)
                    + gam.unsqueeze(1) * Xij / (a0 * a0 * torch.pow(mol.rij, 3)).unsqueeze(1)
                )
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

        if include_ground_state:
            pair_grad = core_core_der(mol, gam, w_x, method, parnuc)
            if mol.seqm_parameters.get("dispersion", False) and method == "AM1":
                pair_grad += dEdisp_dr(mol)
    if not include_ground_state:
        pair_grad = torch.zeros_like(Xij)

    ind = _cached_index(K_ind_4, device)
    weight = _cached_tensor(WEIGHT_10, device, dtype).reshape((-1, 10))
    idx0 = _cached_index(UPPER_IDX0_4, device)
    idx1 = _cached_index(UPPER_IDX1_4, device)
    scale_emat = _cached_tensor(EMAT_SCALE_4, device, dtype)

    contract = _rcis_grad_contract_dispatch or _rcis_grad_contract_kernel
    grad_cis = contract(
        B0,
        R0,
        P0,
        overlap_x,
        w_x,
        e1b_x,
        e2a_x,
        pair_grad,
        mol.mask,
        mol.maskd,
        mol.idxi,
        mol.idxj,
        ind,
        idx0,
        idx1,
        weight,
        scale_emat,
        int(nmol),
        int(molsize),
        bool(include_ground_state),
    )

    if omx_orthogonalization_grad is not None:
        grad_cis += omx_orthogonalization_grad

    return grad_cis


def _rcis_grad_contract_kernel(
    B0,
    R0,
    P0,
    overlap_x,
    w_x,
    e1b_x,
    e2a_x,
    pair_grad,
    mask,
    maskd,
    idxi,
    idxj,
    ind,
    idx0,
    idx1,
    weight,
    scale_emat,
    nmol: int,
    molsize: int,
    include_ground_state: bool,
):
    B = B0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    P = P0.reshape(nmol, molsize, 4, molsize, 4).transpose(2, 3).reshape(nmol * molsize * molsize, 4, 4)
    if include_ground_state:
        B = B + 0.5 * P
        pair_grad = pair_grad + 0.5 * (P[mask].unsqueeze(1) * overlap_x).sum(dim=(2, 3))

    overlap_KAB_x = overlap_x.clone()
    Pp = P[mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            overlap_KAB_x[..., i, j] -= torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad = pair_grad + (B[mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3))

    PA = (P[maskd[idxi]][..., idx0, idx1] * weight).unsqueeze(-1)
    PB = (P[maskd[idxj]][..., idx0, idx1] * weight).unsqueeze(-2)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)
    if include_ground_state:
        pair_grad = pair_grad + (
            0.5 * (P[maskd[idxj], None, :, :] * e2a_x * scale_emat).sum(dim=(2, 3))
            + 0.5 * (P[maskd[idxi], None, :, :] * e1b_x * scale_emat).sum(dim=(2, 3))
        )

    sumA = torch.zeros_like(overlap_x)
    sumA[..., idx0, idx1] = suma
    e2a_eff = e2a_x + sumA

    sumB = torch.zeros_like(overlap_x)
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)
    sumB[..., idx0, idx1] = sumb
    e1b_eff = e1b_x + sumB

    e1b_eff = e1b_eff * scale_emat
    e2a_eff = e2a_eff * scale_emat
    pair_grad = pair_grad + (
        (B[maskd[idxj], None, :, :] * e2a_eff).sum(dim=(2, 3))
        + (B[maskd[idxi], None, :, :] * e1b_eff).sum(dim=(2, 3))
    )

    R_symmetrized = 0.5 * (R0 + R0.transpose(1, 2))
    R_symm = (
        R_symmetrized.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )

    Rdiag_symmetrized = R_symm[maskd]
    PA = (Rdiag_symmetrized[idxi][..., idx0, idx1] * weight).unsqueeze(-1)
    PB = (Rdiag_symmetrized[idxj][..., idx0, idx1] * weight).unsqueeze(-2)

    suma = torch.sum(PA.unsqueeze(1) * w_x, dim=2)
    sumA = torch.zeros_like(overlap_x)
    sumA[..., idx0, idx1] = suma
    J_x_2a = sumA * scale_emat

    sumB = torch.zeros_like(overlap_x)
    sumb = torch.sum(PB.unsqueeze(1) * w_x, dim=3)
    sumB[..., idx0, idx1] = sumb
    J_x_1b = sumB * scale_emat

    pair_grad = pair_grad + (
        (2.0 * R_symm[maskd[idxj], None, :, :] * J_x_2a).sum(dim=(2, 3))
        + (2.0 * R_symm[maskd[idxi], None, :, :] * J_x_1b).sum(dim=(2, 3))
    )

    overlap_KAB_x = torch.zeros_like(overlap_x)
    Pp = R_symm[mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            overlap_KAB_x[..., i, j] = -0.5 * torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad = pair_grad + (4.0 * R_symm[mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3))

    R_antisymmetrized = 0.5 * (R0 - R0.transpose(1, 2))
    R_antisymm = (
        R_antisymmetrized.reshape(nmol, molsize, 4, molsize, 4)
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, 4, 4)
    )
    Pp = R_antisymm[mask].unsqueeze(1)
    for i in range(4):
        w_x_i = w_x[..., ind[i], :]
        for j in range(4):
            overlap_KAB_x[..., i, j] = -0.5 * torch.sum(Pp * (w_x_i[..., :, ind[j]]), dim=(2, 3))

    pair_grad = pair_grad + (4.0 * R_antisymm[mask].unsqueeze(1) * overlap_KAB_x).sum(dim=(2, 3))

    grad_cis = torch.zeros(nmol * molsize, 3, dtype=B0.dtype, device=B0.device)
    grad_cis.index_add_(0, idxi, pair_grad)
    grad_cis.index_add_(0, idxj, pair_grad, alpha=-1.0)

    grad_cis = grad_cis.view(nmol, molsize, 3)
    return grad_cis


from .constants import debye_to_AU, to_debye
from .dipole import calc_dipole_matrix
from .rcis_batch import pack_dipole_matrix


def make_cis_state_dipole(mol, difference_density, relaxed_difference_density, P0):
    dipole_mat = calc_dipole_matrix(mol)
    dipole_mat_packed = pack_dipole_matrix(mol, dipole_mat)

    mol.cis_state_unrelaxed_dipole = (
        -torch.einsum("Nnm,Ndnm->Nd", difference_density, dipole_mat_packed) * to_debye * debye_to_AU
        + mol.dipole
    )
    mol.cis_state_relaxed_dipole = (
        -torch.einsum("Nnm,Ndnm->Nd", relaxed_difference_density, dipole_mat_packed) * to_debye * debye_to_AU
        + mol.dipole
    )
