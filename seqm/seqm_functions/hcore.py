import torch
from .diat_overlap import diatom_overlap_matrix
from .diat_overlapD import diatom_overlap_matrixD
from .diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from .two_elec_two_center_int import two_elec_two_center_int as TETCI
from .constants import overlap_cutoff
import time
import sys


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
    #ntotatoms = nmol * molsize, i.e. the padding zero is also included
    #will call diat and rotate to create overlap matrix and two electron two
    #center integrals
    #and return Hcore and structured two electron two center integrals

    #molsize : number of atoms in each molecule, including the padding zero
    #mask: tell the postion of each pair, shape (npairs,)

    #idxi, idxj, index for atom i and j in the current batch, shape (nparis,)
    #in the data_loader.py, the index for for each atom is the index across whole dataset
    #should take the remainder before passing into this funcition %(batch_size*molsize)

    #ni, nj atomic number, shape (npairs,)
    #xij, unit vector from i to j (xj-xi)/|xj-xi|, shape (npairs,3)
    #rij, distance between i and j, in atomic units, shape (npairs,)
    # Z, atomic number, shape (ntotatoms,)
    #zetas,zetap: zeta parameters for s and p orbitals, shape(ntotatoms,)
    #will use it to create zeta_a and zeta_b
    #zeta_a, zeta_b: zeta for atom i and j, shape (npairs,2), for s and p orbitals
    # uss, upp: Uss Upp energy for each atom, shape (ntotatoms,)
    #gss, gpp, gp2, hsp: parameters, shape (ntotatoms,)

    #isbeta_pair : beta is for each pair in the molecule, shape (npairs, 4) or
    #              for each atom in the molecule, shape (ntotatoms, 2)
    #              check diat.py for detail

    #calpar will create dd, qq, rho0, rho1, rho2 used in rotate from zetas, zetap
    # and qn, gss, hsp, hpp (hpp = 0.5*(gpp-gp2))
    # all the hpp in the code is replaced with gpp and gp2, and thus not used

    #qn : principal quantum number for valence shell
    #tore: charge for the valence shell of each atom, will used as constants

    #rotate(ni,nj,xij,rij,tore,da,db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b) => w, e1b, e2a
    #h1elec(idxi, idxj, ni, nj, xij, rij, zeta_a, zeta_b, beta, ispair=False) =>  beta_mu_nu

    #t0 = time.time()
    is_pm6 = (molecule.method == 'PM6')
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
    zeta_fields = ['zeta_s', 'zeta_p', 'zeta_d'] if is_pm6 else ['zeta_s', 'zeta_p']
    zeta = torch.stack([molecule.parameters[f] for f in zeta_fields], dim=1)

    # 3) Compute diatomic overlaps only where rij ≤ cutoff
    xij, rij = molecule.xij, molecule.rij
    ni, nj      = molecule.ni, molecule.nj
    idxi, idxj  = molecule.idxi, molecule.idxj

    npairs = xij.size(0)
    di = torch.zeros((npairs, orb_dim, orb_dim),
                     dtype=xij.dtype, device=xij.device)
    mask_ov = (rij <= overlap_cutoff)
    di[mask_ov] = overlap_fn(
        ni[mask_ov], nj[mask_ov],
        xij[mask_ov], rij[mask_ov],
        zeta[idxi[mask_ov]], zeta[idxj[mask_ov]],
        *overlap_args
    )

    # Optionally run the full TETCI kernel
    # (skip when using this in the external SEDACS program to get only Hcore)
    if doTETCI:
        w, e1b, e2a, rho0xi,rho0xj, riXH, ri  = TETCI(
            molecule.const, idxi, idxj, ni, nj, xij, rij, molecule.Z,
            molecule.parameters['zeta_s'], molecule.parameters['zeta_p'],
            molecule.parameters.get('zeta_d', None),
            molecule.parameters.get('s_orb_exp_tail', None),
            molecule.parameters.get('p_orb_exp_tail', None),
            molecule.parameters.get('d_orb_exp_tail', None),
            molecule.parameters['g_ss'], molecule.parameters['g_pp'],
            molecule.parameters['g_p2'], molecule.parameters['h_sp'],
            molecule.parameters['F0SD'], molecule.parameters['G2SD'],
            molecule.parameters['rho_core'], molecule.alp, molecule.chi,
            molecule.method
        )
    else:
        w = e1b = e2a = rho0xi = rho0xj = riXH = ri = None

    # Allocate final Hcore matrix (called the block-matrix M)
    Nblocks = molecule.nmol * molecule.molsize * molecule.molsize
    M = torch.zeros((Nblocks, orb_dim, orb_dim),
                    dtype=di.dtype, device=di.device)

    # Fill one-center once electron pure-atomic (U_ss, U_pp, …) diagonal
    U_keys = ['U_ss'] + ['U_pp']*3
    if is_pm6:
        U_keys += ['U_dd']*5
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
    b_atom = molecule.parameters['beta']  # shape (n_atoms, 3) or (n_atoms, 2)
    beta_atoms = torch.empty((b_atom.shape[0], orb_dim),
                             dtype=b_atom.dtype, device=b_atom.device)
    beta_atoms[:, 0]   = b_atom[:, 0]            # s
    beta_atoms[:, 1:4] = b_atom[:, 1].unsqueeze(-1)  # p
    if is_pm6:
        beta_atoms[:, 4:9] = b_atom[:, 2].unsqueeze(-1)  # d

    # Then, make the two-center one-elecron matrix terms
    bi   = beta_atoms[idxi]          # (npairs, orb_dim)
    bj   = beta_atoms[idxj]
    bsum = (bi.unsqueeze(2) + bj.unsqueeze(1)) * 0.5      # (npairs,orb_dim,orb_dim)
    Kb = molecule.parameters.get('Kbeta', None)
    if torch.is_tensor(Kb):
        bsum[:, 0, 0]   *= Kb[:, 0]
        bsum[:, 0, 1:4] *= Kb[:, 1].unsqueeze(-1)
        bsum[:, 1:4, 0] *= Kb[:, 2].unsqueeze(-1)
        bsum[:, 1:4, 1:4] *= Kb[:, 3:].unsqueeze(-1)

    M[molecule.mask] = di * bsum

    #caution
    #the lower triangle part of Hcore is not filled here
    # It is easier to retain Hcore as M without reshaping it 

    return M, w, rho0xi, rho0xj, riXH, ri
