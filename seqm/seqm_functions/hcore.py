import torch

from .constants import overlap_cutoff
from .diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from .diat_overlapD import diatom_overlap_matrixD
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
    is_pm6 = molecule.method == "PM6"
    orb_dim = 9 if is_pm6 else 4
    if is_pm6:
        overlap_fn = diatom_overlap_matrixD
        overlap_args = (molecule.const.qn_int, molecule.const.qnD_int)
    else:
        overlap_fn = diatom_overlap_matrix_PM6_SP
        overlap_args = (molecule.const.qn_int,)

    # Compute the overlap matrix (called di)
    # Prepare the arguments for the overlap function
    # 1) Stack zeta for s/p(/d)
    zeta_fields = ["zeta_s", "zeta_p", "zeta_d"] if is_pm6 else ["zeta_s", "zeta_p"]
    zeta = torch.stack([molecule.parameters[f] for f in zeta_fields], dim=1)

    # 3) Compute diatomic overlaps only where rij ≤ cutoff
    xij, rij = molecule.xij, molecule.rij
    ni, nj = molecule.ni, molecule.nj
    idxi, idxj = molecule.idxi, molecule.idxj

    npairs = xij.size(0)
    di = torch.zeros((npairs, orb_dim, orb_dim), dtype=xij.dtype, device=xij.device)
    mask_ov = rij <= overlap_cutoff
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
        M[molecule.maskd, orb, orb] = molecule.parameters[key]

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
    beta_atoms = torch.empty((b_atom.shape[0], orb_dim), dtype=b_atom.dtype, device=b_atom.device)
    beta_atoms[:, 0] = b_atom[:, 0]  # s
    beta_atoms[:, 1:4] = b_atom[:, 1].unsqueeze(-1)  # p
    if is_pm6:
        beta_atoms[:, 4:9] = b_atom[:, 2].unsqueeze(-1)  # d

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

    is_pm6 = molecule.method == "PM6"
    orb_dim = 9 if is_pm6 else 4
    if is_pm6:
        overlap_fn = diatom_overlap_matrixD
        overlap_args = (molecule.const.qn_int, molecule.const.qnD_int)
        zeta_fields = ["zeta_s", "zeta_p", "zeta_d"]
    else:
        overlap_fn = diatom_overlap_matrix_PM6_SP
        overlap_args = (molecule.const.qn_int,)
        zeta_fields = ["zeta_s", "zeta_p"]

    # Parameters are stored only for real atoms; rebuild a padded view for indexing
    zeta = torch.stack([molecule.parameters[f] for f in zeta_fields], dim=1)
    nmol, molsize = molecule.species.shape
    species = molecule.species
    device = coords1.device
    dtype = coords1.dtype

    atom_index = torch.arange(nmol * molsize, device=device, dtype=torch.int64)
    real_atoms = atom_index[(species.reshape(-1) > 0)]
    zeta_full = torch.zeros((nmol * molsize, zeta.shape[1]), dtype=zeta.dtype, device=zeta.device)
    zeta_full[real_atoms] = zeta
    zeta_full = zeta_full.view(nmol, molsize, -1)

    # Pair geometry between coords1 (row atoms) and coords2 (column atoms)
    diff = coords2.unsqueeze(1) - coords1.unsqueeze(2)  # (nmol, molsize, molsize, 3)
    dist = torch.linalg.norm(diff, dim=-1)
    rij = dist * molecule.const.length_conversion_factor
    xij = torch.zeros_like(diff)
    nonzero_dist = dist > 0
    xij[nonzero_dist] = diff[nonzero_dist] / dist[nonzero_dist].unsqueeze(-1)
    xij[~nonzero_dist] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

    atom_mask = (species.unsqueeze(2) > 0) & (species.unsqueeze(1) > 0)
    close_pairs = atom_mask & (rij <= overlap_cutoff)
    diag_mask = torch.eye(molsize, dtype=torch.bool, device=device).unsqueeze(0)
    diag_zero = close_pairs & diag_mask & (~nonzero_dist)

    di_blocks = torch.zeros((nmol, molsize, molsize, orb_dim, orb_dim), dtype=dtype, device=device)

    valid_pairs = close_pairs & (~diag_zero)

    if valid_pairs.any():
        ni = species.unsqueeze(2).expand(-1, -1, molsize)[valid_pairs]
        nj = species.unsqueeze(1).expand(-1, molsize, -1)[valid_pairs]
        x_flat = xij[valid_pairs]
        r_flat = rij[valid_pairs]
        zeta_i = zeta_full.unsqueeze(2).expand(-1, -1, molsize, -1)[valid_pairs]
        zeta_j = zeta_full.unsqueeze(1).expand(-1, molsize, -1, -1)[valid_pairs]

        swap = ni < nj  # enforce ni >= nj as expected by overlap kernels
        ni_use = torch.where(swap, nj, ni)
        nj_use = torch.where(swap, ni, nj)
        x_use = torch.where(swap.unsqueeze(-1), -x_flat, x_flat)
        zeta_i_use = torch.where(swap.unsqueeze(-1), zeta_j, zeta_i)
        zeta_j_use = torch.where(swap.unsqueeze(-1), zeta_i, zeta_j)

        di_tmp = overlap_fn(ni_use, nj_use, x_use, r_flat, zeta_i_use, zeta_j_use, *overlap_args)
        di_tmp[swap] = di_tmp[swap].transpose(1, 2)
        di_blocks[valid_pairs] = di_tmp

    # When coords1 and coords2 coincide for the same atom, the overlap is identity
    if diag_zero.any():
        di_blocks[diag_zero] = torch.eye(orb_dim, dtype=dtype, device=device)

    overlap_matrix = di_blocks.transpose(2, 3).reshape(nmol, orb_dim * molsize, orb_dim * molsize)
    return overlap_matrix
