import torch

from .cal_par import dd_qq
from .constants import a0, debye_to_AU, to_debye
from .hcore import orthogonalize_operator_from_overlap, overlap_matrix_current_geometry
from .om1_overlap import _XQQ_CUTOFF
from .omx_basis import select_om1_basis_payload
from .omx_utils import OMX_METHODS, get_orbital_zetas


def calc_dipole_matrix_nddo(mol, return_diag_dipole=False):
    """Build the AO coordinate-operator matrix r for NDDO-like methods."""
    dtype = mol.rij.dtype
    device = mol.rij.device
    zetas, zetap = get_orbital_zetas(mol.parameters, mol.method)
    Z = mol.Z
    qn0 = mol.const.qn[Z]
    isX = Z > 2  # Heavy atom
    isH = Z == 1
    dd, _ = dd_qq(qn0[isX], zetas[isX], zetap[isX])
    dd = dd.to(dtype=dtype, device=device) * a0

    valid_atom = (mol.species > 0).reshape(-1)
    n_valid_atoms = mol.maskd.numel()
    coord = mol.coordinates.reshape(mol.nmol * mol.molsize, 3)[valid_atom]
    diagonal_dipole = torch.zeros((3, n_valid_atoms, 4, 4), dtype=dtype, device=device)

    diagonal_dipole[:, isX] = coord[isX].T.unsqueeze(-1).unsqueeze(-1) * torch.eye(
        4, device=device, dtype=dtype
    )

    cart_idx = torch.arange(3, device=device)[:, None]
    heavy_idx = isX.nonzero(as_tuple=False).squeeze(1)[None, :]
    diagonal_dipole[cart_idx, heavy_idx, 0, cart_idx + 1] = dd.unsqueeze(0)
    diagonal_dipole[cart_idx, heavy_idx, cart_idx + 1, 0] = dd.unsqueeze(0)

    diagonal_dipole[:, isH, 0, 0] = coord[isH].T

    if return_diag_dipole:
        diag = torch.zeros(3, mol.nmol * mol.molsize, 4, 4, dtype=dtype, device=device)
        diag[:, valid_atom] = diagonal_dipole
        return diag.reshape(3, mol.nmol, mol.molsize, 4, 4)

    return _assemble_full_dipole_matrix(mol, diagonal_dipole.permute(1, 2, 3, 0))


def calc_ground_dipole(molecule, P):
    with torch.no_grad():
        if molecule.method in OMX_METHODS:
            # Ground-state OMx dipoles are temporarily disabled since we might need to recalculate AO dipole matrix for transition dipoles.
            # Keep a tensor placeholder so downstream output paths do not fail on None.
            molecule.dipole = torch.zeros(
                (molecule.nmol, 3), dtype=molecule.coordinates.dtype, device=molecule.coordinates.device
            )
            return

        dipole_mat = calc_dipole_matrix(molecule)
        electronic_position = torch.einsum("bnm,bknm->bk", P.sum(dim=1) if P.ndim == 4 else P, dipole_mat)

        # Nuclear dipole
        nuclear_dipole = (molecule.const.tore[molecule.species].unsqueeze(-1) * molecule.coordinates).sum(
            dim=1
        )  # (b, 3)

        molecule.dipole = (nuclear_dipole - electronic_position) * to_debye * debye_to_AU
        return


def _assemble_full_dipole_matrix(molecule, diagonal_blocks, pair_blocks=None):
    nblocks = molecule.nmol * molecule.molsize * molecule.molsize
    dipole_blocks = torch.zeros(
        (nblocks, 4, 4, 3), dtype=molecule.coordinates.dtype, device=molecule.coordinates.device
    )
    dipole_blocks[molecule.maskd] = diagonal_blocks
    if pair_blocks is not None:
        dipole_blocks[molecule.mask] = pair_blocks
        dipole_blocks[molecule.mask_l] = pair_blocks.transpose(1, 2)
    return (
        dipole_blocks.reshape(molecule.nmol, molecule.molsize, molecule.molsize, 4, 4, 3)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(molecule.nmol, 3, 4 * molecule.molsize, 4 * molecule.molsize)
    )


def _omx_dipole_blocks(molecule):
    basis_data = molecule.parameters.get("_omx_basis_data")
    if basis_data is None:
        raise RuntimeError("OMx basis tables have not been cached on the molecule")

    real_atom_mask = (molecule.species > 0).reshape(-1)
    coords_real = molecule.coordinates.reshape(-1, 3)[real_atom_mask]
    zero_rij = torch.zeros(coords_real.shape[0], dtype=coords_real.dtype, device=coords_real.device)
    unit_x = torch.zeros_like(coords_real)
    unit_x[:, 0] = 1.0

    diag_dipole = omx_pair_dipole_matrix_sp(zero_rij, unit_x, coords_real, basis_data, basis_data)

    pair_i = select_om1_basis_payload(basis_data, molecule.idxi)
    pair_j = select_om1_basis_payload(basis_data, molecule.idxj)
    pair_dipole = omx_pair_dipole_matrix_sp(
        molecule.rij, molecule.xij, coords_real[molecule.idxi], pair_i, pair_j
    )
    return diag_dipole, pair_dipole


def omx_pair_dipole_matrix_sp(rij, direction, coord_i, basis_i, basis_j):
    """
    Batched <AO_i | r | AO_j> coordinate-operator blocks for an sp ECP-3G/OM1 basis.
    AO order: [s, px, py, pz]

    Inputs
    ------
    rij       : [npairs]
    direction : [npairs, 3], same direction used in diatom_overlap_matrix_OM1
    coord_i   : [npairs, 3], absolute Cartesian coordinate of atom i

    Returns
    -------
    dip : [npairs, 4, 4, 3]

    """
    dtype = rij.dtype
    device = rij.device

    shell_i, exp_i, cs_i, cp_i = (
        basis_i["shell_type"],
        basis_i["exponents"],
        basis_i["coeff_s"],
        basis_i["coeff_p"],
    )
    shell_j, exp_j, cs_j, cp_j = (
        basis_j["shell_type"],
        basis_j["exponents"],
        basis_j["coeff_s"],
        basis_j["coeff_p"],
    )

    a = exp_i.unsqueeze(2)  # [P, Ki, 1]
    b = exp_j.unsqueeze(1)  # [P, 1, Kj]
    g = a + b
    inv_g = 1.0 / g

    r = rij.view(-1, 1, 1)
    r2 = r * r

    xqq = a * b * r2 * inv_g
    active = xqq <= _XQQ_CUTOFF

    s00 = (torch.pi * inv_g) * torch.sqrt(torch.pi * inv_g) * torch.exp(-xqq)
    s00 = torch.where(active, s00, torch.zeros_like(s00))

    # Pair vector from atom i to atom j, consistent with your overlap convention.
    Rij = r.unsqueeze(-1) * direction[:, None, None, :]  # [P, Ki, Kj, 3]

    # Gaussian product center:
    # P = Ri + b/(a+b) * (Rj - Ri)
    # `rij` and the Gaussian exponents are in bohr-based units, while molecular
    # coordinates are stored in Angstrom. Build the operator in bohr and convert
    # back to Angstrom at the end so it stays consistent with the rest of the
    # dipole-matrix code path.
    Ri = (coord_i / a0)[:, None, None, :]
    P = Ri + (b * inv_g).unsqueeze(-1) * Rij

    # u = P - Ri
    # v = P - Rj
    u = (b * inv_g).unsqueeze(-1) * Rij
    v = -(a * inv_g).unsqueeze(-1) * Rij

    sigma = 0.5 * inv_g
    eye = torch.eye(3, dtype=dtype, device=device)

    # Kill p coefficients on atoms without p shells.
    cp_i = torch.where(shell_i[:, None] == 1, cp_i, torch.zeros_like(cp_i))
    cp_j = torch.where(shell_j[:, None] == 1, cp_j, torch.zeros_like(cp_j))

    # ss block:
    # <s_i | r_k | s_j> = S P_k
    w_ss = cs_i.unsqueeze(2) * cs_j.unsqueeze(1)
    R_ss = torch.sum(w_ss.unsqueeze(-1) * s00.unsqueeze(-1) * P, dim=(1, 2))

    # s-p block:
    # <s_i | r_k | p_j> = S [P_k v_j + sigma delta_jk]
    w_sp = cs_i.unsqueeze(2) * cp_j.unsqueeze(1)
    T_sp = P.unsqueeze(-2) * v.unsqueeze(-1) + sigma[..., None, None] * eye
    R_sp = torch.sum(w_sp[..., None, None] * s00[..., None, None] * T_sp, dim=(1, 2))

    # p-s block:
    # <p_i | r_k | s_j> = S [u_i P_k + sigma delta_ik]
    w_ps = cp_i.unsqueeze(2) * cs_j.unsqueeze(1)
    T_ps = u.unsqueeze(-1) * P.unsqueeze(-2) + sigma[..., None, None] * eye
    R_ps = torch.sum(w_ps[..., None, None] * s00[..., None, None] * T_ps, dim=(1, 2))

    # p-p block:
    # <p_i | r_k | p_j>
    # = S [
    #     P_k (u_i v_j + sigma delta_ij)
    #     + sigma (delta_ki v_j + delta_kj u_i)
    #   ]
    w_pp = cp_i.unsqueeze(2) * cp_j.unsqueeze(1)

    uv = u.unsqueeze(-1) * v.unsqueeze(-2) + sigma[..., None, None] * eye

    extra = eye[:, None, :] * v[..., None, :, None] + eye[None, :, :] * u[..., :, None, None]

    T_pp = uv.unsqueeze(-1) * P[..., None, None, :] + sigma[..., None, None, None] * extra

    R_pp = torch.sum(w_pp[..., None, None, None] * s00[..., None, None, None] * T_pp, dim=(1, 2))

    block = torch.zeros((rij.shape[0], 4, 4, 3), dtype=dtype, device=device)

    block[:, 0, 0, :] = R_ss
    block[:, 0, 1:, :] = R_sp
    block[:, 1:, 0, :] = R_ps
    block[:, 1:, 1:, :] = R_pp

    return block * a0


def calc_dipole_matrix(molecule, orthogonalize=None):
    """
    Return the AO coordinate-operator matrix r.

    For OMx methods, Löwdin orthogonalization should be applied before
    contracting any property in the orthogonalized AO basis, so the default is
    to return the orthogonalized operator.
    """
    if orthogonalize is None:
        orthogonalize = molecule.method in OMX_METHODS

    if molecule.method not in OMX_METHODS:
        return calc_dipole_matrix_nddo(molecule)

    diag_dipole, pair_dipole = _omx_dipole_blocks(molecule)
    dipole_mat = _assemble_full_dipole_matrix(molecule, diag_dipole, pair_dipole)
    if orthogonalize:
        overlap = overlap_matrix_current_geometry(molecule)
        dipole_mat = orthogonalize_operator_from_overlap(overlap, dipole_mat)
    return dipole_mat
