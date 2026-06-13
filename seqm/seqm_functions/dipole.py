import torch

from .cal_par import dd_qq
from .constants import a0, debye_to_AU, to_debye
from .omx_utils import get_orbital_zetas


def calc_dipole_matrix(mol, return_diag_dipole=False):
    """
    Build the block-diagonal dipole tensor for atoms.

    Each atom has 4 orbitals (sp), so the full dipole matrix is represented by a tensor
    of shape (natom*natom, 4, 4, 3) where the last dimension corresponds to the x, y, and z components.
    but here we build only the diagonal blocks of the full dipole matrix

    For non-hydrogen atoms
      - All diagonal entries (positions (i,i) for i=0,...,3) are set to -coord.
      - Additionally, for each Cartesian direction i (i = 0 for x, 1 for y, 2 for z),
        the off-diagonal element (0, i+1) is set to multip_2c_elec_params.

    For hydrogen atoms
      - Only the (0,0) element is set to -coord, with the other diagonal elements remaining zero.

    Parameters:
      mol: contains all the info on the molecules

    Returns:
      diagonal_dipole: Tensor of shape (natom, 4, 4, 3) with the dipole blocks.
    """

    dtype = mol.rij.dtype
    device = mol.rij.device
    # for non-zero atoms
    zetas, zetap = get_orbital_zetas(mol.parameters, mol.method)
    qn = mol.const.qn
    # Z is a flattened tensor of the atomic numbers of non-zero atoms across molecular batches
    Z = mol.Z
    qn0 = qn[Z]
    isX = Z > 2  # Heavy atom
    isH = Z == 1
    dd, _ = dd_qq(qn0[isX], zetas[isX], zetap[isX])
    dd *= a0

    valid_atom = (mol.species > 0).reshape(-1)
    n_valid_atoms = mol.maskd.numel()
    coord = mol.coordinates.reshape(mol.nmol * mol.molsize, 3)[valid_atom]
    diagonal_dipole = torch.zeros((3, n_valid_atoms, 4, 4), dtype=dtype, device=device)

    I_4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # shape (1,1,4,4)
    # Get -coord for non-H atoms and rearrange from (n_nonH, 3) to (3, n_nonH, 1, 1).
    nonH_coord = -coord[isX].T.unsqueeze(-1).unsqueeze(-1)  # shape (3, n_nonH, 1, 1)
    # Multiply by the identity so that only the diagonal entries are nonzero.
    diag_block_nonH = nonH_coord * I_4  # shape (3, n_nonH, 4, 4)
    diagonal_dipole[:, isX, :, :] = diag_block_nonH

    # Set the off-diagonal s-p interaction elements:
    # For each Cartesian direction i (0: x, 1: y, 2: z), set
    for i in range(3):
        diagonal_dipole[i, isX, 0, i + 1] = -dd
        diagonal_dipole[i, isX, i + 1, 0] = -dd
    # cart_idx = torch.arange(3, device=device)
    # diagonal_dipole[cart_idx, isX, 0, cart_idx + 1] = -dd
    # diagonal_dipole[cart_idx, isX, cart_idx + 1, 0] = -dd

    # --- Process hydrogen atoms ---
    # For hydrogen atoms, only the (0,0) element is set.
    diagonal_dipole[:, isH, 0, 0] = -coord[isH].T

    if return_diag_dipole:
        diag = torch.zeros(3, mol.nmol * mol.molsize, 4, 4, dtype=dtype, device=device)
        diag[:, valid_atom] = diagonal_dipole
        return diag.reshape(3, mol.nmol, mol.molsize, 4, 4)

    dipole_mat = torch.zeros(3, mol.nmol * mol.molsize * mol.molsize, 4, 4, dtype=dtype, device=device)
    dipole_mat[:, mol.maskd] = diagonal_dipole
    dipole_mat = (
        dipole_mat.reshape(3, mol.nmol, mol.molsize, mol.molsize, 4, 4)
        .permute(1, 0, 2, 4, 3, 5)
        .reshape(mol.nmol, 3, 4 * mol.molsize, 4 * mol.molsize)
    )

    return dipole_mat


def calc_ground_dipole(molecule, P):
    with torch.no_grad():
        b, n = molecule.coordinates.shape[:2]
        dipole_diag_blocks = calc_dipole_matrix(molecule, return_diag_dipole=True)  # (3, b, n, 4, 4)

        # Extract 4x4 block diagonals
        if len(P.size()) == 4:  # open-shell
            P_blocks = P[:, 0].view(b, n, 4, n, 4).diagonal(0, 1, 3) + P[:, 1].view(b, n, 4, n, 4).diagonal(
                0, 1, 3
            )
        else:
            P_blocks = P.view(b, n, 4, n, 4).diagonal(0, 1, 3)  # (b, 4, 4, n)

        # Electronic dipole
        electronic_dipole = torch.einsum("bxyn,dbnxy->bd", P_blocks, dipole_diag_blocks)  # (b, 3)

        # Nuclear dipole
        nuclear_dipole = (molecule.const.tore[molecule.species].unsqueeze(-1) * molecule.coordinates).sum(
            dim=1
        )  # (b, 3)

        molecule.dipole = (electronic_dipole + nuclear_dipole) * to_debye * debye_to_AU
        return


from .om1_overlap import _XQQ_CUTOFF


def omx_pair_dipole_matrix_sp(rij, direction, coord_i, basis_i, basis_j):
    """
    Batched <AO_i | r | AO_j> dipole blocks for an sp ECP-3G/OM1 basis.
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
    Ri = coord_i[:, None, None, :]
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

    return -block


from .om2_hcore import select_om1_basis_payload
from .two_elec_two_center_int import rotate_with_quaternion


def calc_dipole_matrix_omx_ecp3g_sp(molecule, return_pair_blocks=False):
    """
    Full batched dipole matrix using the same pair utilities as overlap.

    Returns:
        [nmol, 3, 4*molsize, 4*molsize]
    """
    orb_dim = 4
    nmol = molecule.nmol
    molsize = molecule.molsize
    nblocks = nmol * molsize * molsize

    basis_data = molecule.parameters.get("_omx_basis_data")

    basis_i = select_om1_basis_payload(basis_data, molecule.idxi)
    basis_j = select_om1_basis_payload(basis_data, molecule.idxj)

    rot = rotate_with_quaternion(molecule.xij)
    rot_t = rot.transpose(1, 2)
    direction = rot_t[:, :, 0]

    # Pair-ordered atom-i coordinates: [mol, i, j, xyz] -> [nblocks, xyz]
    coord_i = molecule.coordinates[:, :, None, :].expand(nmol, molsize, molsize, 3).reshape(nblocks, 3)

    pair_dipole = omx_pair_dipole_matrix_sp(
        molecule.rij, direction, coord_i, basis_i, basis_j
    )  # [nblocks, 4, 4, 3]

    if return_pair_blocks:
        return pair_dipole

    dipole_mat = (
        pair_dipole.reshape(nmol, molsize, molsize, orb_dim, orb_dim, 3)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(nmol, 3, orb_dim * molsize, orb_dim * molsize)
    )

    return dipole_mat
