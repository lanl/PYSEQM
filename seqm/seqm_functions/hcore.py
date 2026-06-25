import torch

from .constants import overlap_cutoff
from .diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from .diat_overlapD import diatom_overlap_matrixD
from .om1_overlap import diatom_overlap_matrix_OM1
from .om2_hcore import build_omx_hcore
from .omx_basis import select_om1_basis_payload
from .omx_utils import get_orbital_zeta_tensor
from .two_elec_two_center_int import two_elec_two_center_int as TETCI


def hcore(molecule, doTETCI=True):
    """
    Get Hcore and two electron two center integrals
    doTETCI : bool, optional
        Whether to compute two‐electron integrals via TETCI.
        This flag was added because pyseqm is also used in SEDACS,
        which only needs the Hcore matrix (no two‐electron integrals),
        so you can skip TETCI to save time/memory.
    """

    # pair type tensor: idxi, idxj, ni,nj,xij,rij, mask (batch dim is pair)
    # atom type tensor: Z, zetas,zetap, uss, upp , gss, gpp, gp2, hsp, beta(isbeta_pair=False)
    #                   (batch dim is atom)
    #
    # nmol, number of molecules in this batch
    # ntotatoms = nmol * molsize, i.e. the padding zero is also included
    # will call diat and rotate to create overlap matrix and two electron two
    # center integrals
    # and return Hcore and structured two electron two center integrals

    # molsize : number of atoms in each molecule, including the padding zero
    # mask: tell the postion of each pair, shape (npairs,)

    # idxi, idxj, index for atom i and j in the current batch, shape (nparis,)
    # in the data_loader.py, the index for for each atom is the index across whole dataset
    # should take the remainder before passing into this funcition %(batch_size*molsize)

    # ni, nj atomic number, shape (npairs,)
    # xij, unit vector from i to j (xj-xi)/|xj-xi|, shape (npairs,3)
    # rij, distance between i and j, in atomic units, shape (npairs,)
    # Z, atomic number, shape (ntotatoms,)
    # zetas,zetap: zeta parameters for s and p orbitals, shape(ntotatoms,)
    # will use it to create zeta_a and zeta_b
    # zeta_a, zeta_b: zeta for atom i and j, shape (npairs,2), for s and p orbitals
    # uss, upp: Uss Upp energy for each atom, shape (ntotatoms,)
    # gss, gpp, gp2, hsp: parameters, shape (ntotatoms,)

    # isbeta_pair : beta is for each pair in the molecule, shape (npairs, 4) or
    #              for each atom in the molecule, shape (ntotatoms, 2)
    #              check diat.py for detail

    # calpar will create dd, qq, rho0, rho1, rho2 used in rotate from zetas, zetap
    # and qn, gss, hsp, hpp (hpp = 0.5*(gpp-gp2))
    # all the hpp in the code is replaced with gpp and gp2, and thus not used

    # qn : principal quantum number for valence shell
    # tore: charge for the valence shell of each atom, will used as constants

    # rotate(ni,nj,xij,rij,tore,da,db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b) => w, e1b, e2a
    # h1elec(idxi, idxj, ni, nj, xij, rij, zeta_a, zeta_b, beta, ispair=False) =>  beta_mu_nu

    # t0 = time.time()
    is_omx = molecule.method in {"OM1", "OM2", "OM3"}
    if is_omx:
        return build_omx_hcore(molecule)

    is_pm6 = molecule.method == "PM6"
    orb_dim = 9 if is_pm6 else 4
    if is_pm6:
        overlap_fn = diatom_overlap_matrixD
        overlap_args = (molecule.const.qn_int, molecule.const.qnD_int)
    else:
        overlap_fn = diatom_overlap_matrix_PM6_SP
        overlap_args = (molecule.const.qn_int,)

    zeta = get_orbital_zeta_tensor(molecule.parameters, molecule.method, include_d=is_pm6)

    xij, rij = molecule.xij, molecule.rij
    ni, nj = molecule.ni, molecule.nj
    idxi, idxj = molecule.idxi, molecule.idxj

    npairs = xij.size(0)
    mask_ov = rij <= overlap_cutoff

    # Compute diatomic overlaps only where rij ≤ cutoff
    di = torch.zeros((npairs, orb_dim, orb_dim), dtype=xij.dtype, device=xij.device)
    di[mask_ov] = overlap_fn(
        ni[mask_ov],
        nj[mask_ov],
        xij[mask_ov],
        rij[mask_ov],
        zeta[idxi[mask_ov]],
        zeta[idxj[mask_ov]],
        *overlap_args,
    )

    # Optionally run the full TETCI kernel
    # (skip when using this in the external SEDACS program to get only Hcore)
    if doTETCI:
        w, e1b, e2a, rho0xi, rho0xj, riXH, ri = TETCI(
            molecule.const,
            idxi,
            idxj,
            ni,
            nj,
            xij,
            rij,
            molecule.Z,
            molecule.parameters["zeta_s"],
            molecule.parameters["zeta_p"],
            molecule.parameters.get("zeta_d", None),
            molecule.parameters.get("s_orb_exp_tail", None),
            molecule.parameters.get("p_orb_exp_tail", None),
            molecule.parameters.get("d_orb_exp_tail", None),
            molecule.parameters["g_ss"],
            molecule.parameters["g_pp"],
            molecule.parameters["g_p2"],
            molecule.parameters["h_sp"],
            molecule.parameters["F0SD"],
            molecule.parameters["G2SD"],
            molecule.parameters["rho_core"],
            molecule.alp,
            molecule.chi,
            molecule.method,
        )
    else:
        w = e1b = e2a = rho0xi = rho0xj = riXH = ri = None

    # Allocate final Hcore matrix (called the block-matrix M)
    Nblocks = molecule.nmol * molecule.molsize * molecule.molsize
    M = torch.zeros((Nblocks, orb_dim, orb_dim), dtype=di.dtype, device=di.device)

    # Fill one-center once electron pure-atomic (U_ss, U_pp, …) diagonal
    U_keys = ["U_ss"] + ["U_pp"] * 3
    if is_pm6:
        U_keys += ["U_dd"] * 5
    for orb, key in enumerate(U_keys):
        M[molecule.maskd, orb, orb] = molecule.parameters[key].to(dtype=M.dtype, device=M.device)

    # Scatter in core-electron TETCI terms which go into the diagonal blocks
    # V_{mu,nv,B} = -ZB*(mu^A nv^A, s^B s^B), stored in e1b, e2a
    # \sum_B V_{ss,B}
    # e1b ==> V_{,B} E1B = ELECTRON ON ATOM NI ATTRACTING NUCLEUS OF NJ.
    # e2a ==> V_{,A}
    if doTETCI:
        if is_pm6:
            # PM6: idxj gets e1b, idxi gets e2a
            M.index_add_(0, molecule.maskd[idxj], e1b)
            M.index_add_(0, molecule.maskd[idxi], e2a)
        else:
            # non-PM6: idxi gets e1b, idxj gets e2a
            M.index_add_(0, molecule.maskd[idxi], e1b)
            M.index_add_(0, molecule.maskd[idxj], e2a)

    # Build two-center Hcore term: βsum * overlap

    # First, build per-orbital beta from per-atom beta
    b_atom = molecule.parameters["beta"]  # shape (n_atoms, 3) or (n_atoms, 2)
    beta_layout = [0, 1, 1, 1, 2, 2, 2, 2, 2] if is_pm6 else [0, 1, 1, 1]
    beta_atoms = b_atom[:, beta_layout]

    # Then, make the two-center one-elecron matrix terms
    bi = beta_atoms[idxi]  # (npairs, orb_dim)
    bj = beta_atoms[idxj]
    bsum = (bi.unsqueeze(2) + bj.unsqueeze(1)) * 0.5  # (npairs,orb_dim,orb_dim)
    Kb = molecule.parameters.get("Kbeta", None)
    if torch.is_tensor(Kb):
        bsum[:, 0, 0] *= Kb[:, 0]
        bsum[:, 0, 1:4] *= Kb[:, 1].unsqueeze(-1)
        bsum[:, 1:4, 0] *= Kb[:, 2].unsqueeze(-1)
        bsum[:, 1:4, 1:4] *= Kb[:, 3:].unsqueeze(-1)

    M[molecule.mask] = di * bsum

    # caution
    # the lower triangle part of Hcore is not filled here
    # It is easier to retain Hcore as M without reshaping it

    return M, w, rho0xi, rho0xj, riXH, ri


def overlap_between_geometries(molecule, coords1, coords2):
    """
    Compute the overlap matrix between atomic orbitals centered at coords1 (rows)
    and coords2 (cols) for a batch of molecules.
    coords1, coords2: (nmol, molsize, 3)
    returns: (nmol, orb_dim * molsize, orb_dim * molsize)
    """
    if coords1.shape != coords2.shape:
        raise ValueError("coords1 and coords2 must have the same shape")

    overlap_fn, overlap_args, orb_dim, is_pm6, is_omx, basis_data = _overlap_kernel_data(molecule)

    # Parameters are stored only for real atoms; rebuild a padded view for indexing.
    zeta = None if is_omx else get_orbital_zeta_tensor(molecule.parameters, molecule.method, include_d=is_pm6)
    nmol, molsize = molecule.species.shape
    species = molecule.species
    device = coords1.device
    dtype = coords1.dtype

    flat_atom_index = torch.arange(nmol * molsize, device=device, dtype=torch.int64)
    real_atoms = flat_atom_index[(species.reshape(-1) > 0)]
    real_atom_lookup = torch.full((nmol * molsize,), -1, dtype=torch.int64, device=device)
    real_atom_lookup[real_atoms] = torch.arange(real_atoms.numel(), device=device, dtype=torch.int64)
    if not is_omx:
        zeta_flat = torch.zeros((nmol * molsize, *zeta.shape[1:]), dtype=zeta.dtype, device=zeta.device)
        zeta_flat[real_atoms] = zeta

    # Pair geometry between coords1 (row atoms) and coords2 (column atoms)
    diff = coords2.unsqueeze(1) - coords1.unsqueeze(2)  # (nmol, molsize, molsize, 3)
    dist = torch.linalg.norm(diff, dim=-1)
    rij = dist * molecule.const.length_conversion_factor
    xij = torch.zeros_like(diff)
    nonzero_dist = dist > 0
    xij[nonzero_dist] = diff[nonzero_dist] / dist[nonzero_dist].unsqueeze(-1)
    xij[..., 0] = torch.where(nonzero_dist, xij[..., 0], torch.ones_like(xij[..., 0]))

    atom_mask = (species.unsqueeze(2) > 0) & (species.unsqueeze(1) > 0)
    close_pairs = atom_mask & (rij <= overlap_cutoff)
    diag_mask = torch.eye(molsize, dtype=torch.bool, device=device).unsqueeze(0)
    diag_zero = close_pairs & diag_mask & (~nonzero_dist)

    di_blocks = torch.zeros((nmol, molsize, molsize, orb_dim, orb_dim), dtype=dtype, device=device)

    valid_pairs = close_pairs & (~diag_zero)

    if valid_pairs.any():
        pair_idx = valid_pairs.nonzero(as_tuple=False)
        batch_idx = pair_idx[:, 0]
        row_idx = pair_idx[:, 1]
        col_idx = pair_idx[:, 2]
        flat_row = batch_idx * molsize + row_idx
        flat_col = batch_idx * molsize + col_idx

        ni = species[batch_idx, row_idx]
        nj = species[batch_idx, col_idx]
        x_flat = xij[batch_idx, row_idx, col_idx]
        r_flat = rij[batch_idx, row_idx, col_idx]

        swap = ni < nj  # enforce ni >= nj as expected by overlap kernels
        x_use = torch.where(swap.unsqueeze(-1), -x_flat, x_flat)
        if is_omx:
            from .two_elec_two_center_int import rotate_with_quaternion

            direction = rotate_with_quaternion(x_use).transpose(1, 2)[:, :, 0]
            basis_i = select_om1_basis_payload(
                basis_data, real_atom_lookup[torch.where(swap, flat_col, flat_row)]
            )
            basis_j = select_om1_basis_payload(
                basis_data, real_atom_lookup[torch.where(swap, flat_row, flat_col)]
            )
            di_tmp = overlap_fn(x_use, r_flat, direction, basis_i, basis_j)
        else:
            ni_use = torch.where(swap, nj, ni)
            nj_use = torch.where(swap, ni, nj)
            zeta_i = zeta_flat[flat_row]
            zeta_j = zeta_flat[flat_col]
            if zeta.dim() == 1:
                zeta_i_use = torch.where(swap, zeta_j, zeta_i)
                zeta_j_use = torch.where(swap, zeta_i, zeta_j)
            else:
                zeta_i_use = torch.where(swap.unsqueeze(-1), zeta_j, zeta_i)
                zeta_j_use = torch.where(swap.unsqueeze(-1), zeta_i, zeta_j)
            di_tmp = overlap_fn(ni_use, nj_use, x_use, r_flat, zeta_i_use, zeta_j_use, *overlap_args)

        di_tmp[swap] = di_tmp[swap].transpose(1, 2)
        di_blocks[batch_idx, row_idx, col_idx] = di_tmp

    # When coords1 and coords2 coincide for the same atom, the overlap is identity
    if diag_zero.any():
        di_blocks[diag_zero] = torch.eye(orb_dim, dtype=dtype, device=device)

    overlap_matrix = di_blocks.transpose(2, 3).reshape(nmol, orb_dim * molsize, orb_dim * molsize)
    return overlap_matrix


def overlap_matrix_current_geometry(molecule):
    """Build the AO overlap matrix for the molecule's current geometry."""
    overlap_fn, overlap_args, orb_dim, is_pm6, is_omx, basis_data = _overlap_kernel_data(molecule)
    nmol, molsize = molecule.species.shape
    dtype = molecule.coordinates.dtype
    device = molecule.coordinates.device
    nblocks = nmol * molsize * molsize

    blocks = torch.zeros((nblocks, orb_dim, orb_dim), dtype=dtype, device=device)

    if is_omx:
        pair_i = select_om1_basis_payload(basis_data, molecule.idxi)
        pair_j = select_om1_basis_payload(basis_data, molecule.idxj)
        from .two_elec_two_center_int import rotate_with_quaternion

        direction = rotate_with_quaternion(molecule.xij).transpose(1, 2)[:, :, 0]
        pair_blocks = overlap_fn(molecule.xij, molecule.rij, direction, pair_i, pair_j)
    else:
        zeta = get_orbital_zeta_tensor(molecule.parameters, molecule.method, include_d=is_pm6)
        pair_blocks = overlap_fn(
            molecule.ni,
            molecule.nj,
            molecule.xij,
            molecule.rij,
            zeta[molecule.idxi],
            zeta[molecule.idxj],
            *overlap_args,
        )

    blocks[molecule.mask] = pair_blocks
    blocks[molecule.mask_l] = pair_blocks.transpose(-1, -2)

    if is_pm6:
        eye = torch.eye(orb_dim, dtype=dtype, device=device)
        h_self = torch.zeros_like(eye)
        h_self[0, 0] = 1.0
        blocks[molecule.maskd[molecule.Z == 1]] = h_self
        blocks[molecule.maskd[molecule.Z > 1]] = eye
    else:
        blocks[molecule.maskd] = torch.eye(orb_dim, dtype=dtype, device=device)

    return (
        blocks.reshape(nmol, molsize, molsize, orb_dim, orb_dim)
        .transpose(2, 3)
        .reshape(nmol, orb_dim * molsize, orb_dim * molsize)
    )


def _overlap_kernel_data(molecule):
    is_pm6 = molecule.method == "PM6"
    is_omx = molecule.method in {"OM1", "OM2", "OM3"}
    orb_dim = 9 if is_pm6 else 4

    if is_pm6:
        return (
            diatom_overlap_matrixD,
            (molecule.const.qn_int, molecule.const.qnD_int),
            orb_dim,
            is_pm6,
            is_omx,
            None,
        )
    if is_omx:
        basis_data = molecule.parameters.get("_omx_basis_data")
        if basis_data is None:
            raise RuntimeError("OMx basis tables have not been cached on the molecule")
        return diatom_overlap_matrix_OM1, (), orb_dim, is_pm6, is_omx, basis_data
    return diatom_overlap_matrix_PM6_SP, (molecule.const.qn_int,), orb_dim, is_pm6, is_omx, None


def _overlap_matrix_for_geometry(molecule, coords):
    if coords.shape == molecule.coordinates.shape and coords.data_ptr() == molecule.coordinates.data_ptr():
        return overlap_matrix_current_geometry(molecule)
    return overlap_between_geometries(molecule, coords, coords)


def _symmetric_inverse_square_root_eigh(S, eigval_tol=None):
    S_sym = 0.5 * (S + S.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(S_sym)

    if eigval_tol is None:
        eps = torch.finfo(S.dtype).eps
        eigval_tol = eps * S.size(-1) * eigvals.abs().amax(dim=-1, keepdim=True)

    active = eigvals > eigval_tol
    inv_sqrt_eigvals = torch.zeros_like(eigvals)
    inv_sqrt_eigvals[active] = torch.rsqrt(eigvals[active])

    return eigvals, eigvecs, inv_sqrt_eigvals


def orthogonalized_overlap_from_matrices(S1, S12, S2, eigval_tol=None):
    """Orthogonalize a precomputed overlap triple without rebuilding AO overlaps."""
    _, eigvecs1, inv_sqrt1 = _symmetric_inverse_square_root_eigh(S1, eigval_tol)
    if S1.data_ptr() == S2.data_ptr():
        eigvecs2, inv_sqrt2 = eigvecs1, inv_sqrt1
    else:
        _, eigvecs2, inv_sqrt2 = _symmetric_inverse_square_root_eigh(S2, eigval_tol)

    S12_orth = eigvecs1.transpose(-1, -2) @ S12 @ eigvecs2
    S12_orth = S12_orth * inv_sqrt1.unsqueeze(-1) * inv_sqrt2.unsqueeze(-2)
    return eigvecs1 @ S12_orth @ eigvecs2.transpose(-1, -2)


def orthogonalize_operator_from_overlap(S, operator, eigval_tol=None):
    """
    Orthogonalize an AO-basis operator with the same overlap on both sides:

        O_orth = S^(-1/2) O S^(-1/2)

    `operator` may have shape `(nmol, nao, nao)` or `(nmol, ncomp, nao, nao)`.
    """
    _, eigvecs, inv_sqrt = _symmetric_inverse_square_root_eigh(S, eigval_tol)
    xform = (eigvecs * inv_sqrt.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    if operator.dim() == 3:
        return xform @ operator @ xform
    return xform.unsqueeze(1) @ operator @ xform.unsqueeze(1)


def orthogonalized_overlap_between_geometries(molecule, coords1, coords2, eigval_tol=None, pack_fn=None):
    """
    Compute S(R1)^(-1/2)^T S(R1,R2) S(R2)^(-1/2).

    This is the efficient path when the orthogonalizing matrices are only needed
    for this product. It avoids explicitly materializing S(R1)^(-1/2) and
    S(R2)^(-1/2). If pack_fn is provided, the three overlap matrices are packed
    before the eigendecompositions.

    coords1, coords2: (nmol, molsize, 3)
    pack_fn: optional callable applied to S(R1), S(R1,R2), and S(R2)
    returns: (nmol, nao, nao), where nao is the packed size if pack_fn is used
    """
    S1 = _overlap_matrix_for_geometry(molecule, coords1)
    S12 = overlap_between_geometries(molecule, coords1, coords2)
    S2 = _overlap_matrix_for_geometry(molecule, coords2)

    if pack_fn is not None:
        S1 = pack_fn(S1)
        S12 = pack_fn(S12)
        S2 = pack_fn(S2)

    return orthogonalized_overlap_from_matrices(S1, S12, S2, eigval_tol)
