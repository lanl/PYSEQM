import torch
from .diat_overlap import diatom_overlap_matrix
from .two_elec_two_center_int import two_elec_two_center_int as TETCI
from .constants import overlap_cutoff

def overlap_full(const,nmol,molsize, maskd, mask, mask_l, idxi,idxj, ni,nj,xij,rij,
                Z, zetas,zetap):
    """
    Get Hcore and two electron and two center integrals
    """
    dtype = xij.dtype
    device = xij.device
    qn_int = const.qn_int
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

    #use uss upp to the diagonal block for hcore
    zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)),dim=1)
    overlap_pairs = rij<=overlap_cutoff
    #di=th.zeros((npairs,4,4),dtype=dtype, device=device)
    di = torch.zeros((xij.shape[0], 4, 4),dtype=dtype, device=device)
    di_full = torch.zeros((nmol*molsize*molsize, 4, 4),dtype=dtype, device=device)
    di[overlap_pairs] = diatom_overlap_matrix(ni[overlap_pairs],
                               nj[overlap_pairs],
                               xij[overlap_pairs],
                               rij[overlap_pairs],
                               zeta[idxi][overlap_pairs],
                               zeta[idxj][overlap_pairs],
                               qn_int)

    mask_H = Z==1
    mask_heavy = Z>1

    H_self_ovr = torch.zeros((4,4), dtype=dtype, device=device)
    H_self_ovr[0,0] = 1.0

    #print(mask_H, mask_heavy)

    di_full[maskd[mask_H]] = H_self_ovr
    di_full[maskd[mask_heavy]] = torch.eye(4, dtype=dtype, device=device)
    di_full[mask] = di
    di_full[mask_l] = di.transpose(1,2)

    #print(di_full)
    #print(di_full.shape)

    di_full = di_full.reshape(nmol,molsize,molsize,4,4) \
                 .transpose(2,3) \
                 .reshape(nmol, 4*molsize, 4*molsize)
    #print(di_full, di_full.shape)

    return di_full