import torch
from .constants import a0


def elec_energy_isolated_atom(const, Z, uss, upp, gss, gpp, gsp, gp2, hsp):
    """
    electrionc energy for a single atom
    #eisol in block.f or in calpar.f
    return Eiso, shape (natoms,)
    """
    Eiso = uss * const.ussc[Z] \
        + upp * const.uppc[Z] \
        + gss * const.gssc[Z] \
        + gpp * const.gppc[Z] \
        + gsp * const.gspc[Z] \
        + gp2 * const.gp2c[Z] \
        + hsp * const.hspc[Z]
    return Eiso


def elec_energy(P, F, Hcore):
    """
    Get the electronic energy
    P: density matrix, shape (nmol, molsize*4, molsize*4)
    F: fock matrix, shape same as P
    Hcore: Hcore matrix, shape (nmol, 4*molsize, 4*molsize)
    return Eelec : electronic energy, shape (nmol,)
    P, F: full, has upper and lower triangle
    Hcore : only have upper triangle as constructed from hcore.py
    """
    h = Hcore.triu() + Hcore.triu(1).transpose(1, 2)

    # Eelec = 0.5 * tr(P(Hcore+F))  # matmul
    # Eelec = 0.5 * \sum P*(H+F)    # elementwise product

    Eelec = 0.5 * torch.sum(P * (h + F), dim=(1, 2))

    return Eelec


def pair_nuclear_energy(const, nmol, ni, nj, idxi, idxj, rij, gam, method='AM1', parameters=None):
    """
    Compute Nuclear Energy
    method='MNDO', 'AM1', 'PM3'
    nmol : number of molecules
    pair_molid : molecule id for each pair, shape (npairs,)
    ni, nj: atomic number, shape (npairs,)
    rij: pair distance in atomic units, shape (npairs,)
    gam : (s^A s^A, s^B, s^B) = w[...,0,0], shape(npairs,): w ==> second return vaule of hcore
    parameters : tuple, (alpha,) or (alpha, K, L, M)
    alpha : shape (natoms,)
    K,L,M : guassian terms in PM3 or AM1, shape (natoms, 2 or 4)
    return nuclear interaction energy for each molecule, (nmol, )
    """
    rija = rij * a0
    tore = const.tore
    alpha = parameters[0]
    t1 = tore[ni] * tore[nj] * gam
    # special case for C-H and O-H
    XH = ((ni == 7) | (ni == 8)) & (nj == 1)
    t2 = torch.zeros_like(t1)
    tmp = torch.exp(-alpha[idxi] * rija)
    t2[~XH] = tmp[~XH]
    t2[XH] = tmp[XH] * rija[XH]
    t3 = torch.exp(-alpha[idxj] * rija)
    if method == 'MNDO':
        # in mopac, rij is in unit of angstrom
        # EnucAB = torch.abs(t1*(1.0+t2+t3))
        EnucAB = t1 * (1.0 + t2 + t3)
    elif method == 'PM3' or method == 'AM1':
        # two gaussian terms for PM3
        # 3~4 terms for AM1
        _, K, L, M = parameters
        # K, L , M shape (natoms,2 or 4)
        t4 = tore[ni] * tore[nj] / rija
        t5 = torch.sum(K[idxi] * torch.exp(-L[idxi] * (rija.reshape((-1, 1)) - M[idxi])**2), dim=1)
        t6 = torch.sum(K[idxj] * torch.exp(-L[idxj] * (rija.reshape((-1, 1)) - M[idxj])**2), dim=1)
        EnucAB = t1 * (1.0 + t2 + t3) + t4 * (t5 + t6)
    else:
        raise ValueError("Supported Method: MNDO, AM1, PM3")
    return EnucAB


def total_energy(nmol, pair_molid, EnucAB, Eelec):
    """
    total energy for each molecule
    total energy E_tot^mol= Eelec + sum{pair A,B,A<B} E_nuc^AB
    #nuclear energy between pair of atom A and B: E_nuc^AB

    as index_add is expensive, there is no need to do this during training
    EnucAB :computed from pair_nuclear_energy, shape (npairs,)
    pair_molid : molecule id for each pair
    Eelec : electronic energy for each molecule, computed from elec_energy, shape (nmol)

    """
    Enuc = torch.zeros((nmol,), dtype=EnucAB.dtype, device=EnucAB.device)
    Enuc.index_add_(0, pair_molid, EnucAB)
    Etot = Eelec + Enuc
    return Etot, Enuc


def heat_formation(const, nmol, atom_molid, Z, Etot, Eiso, flag=True):
    """
    get the heat of formation for each molecule
    return Hf : shape (nmol,)
    #heat of formation : delta H_f^mol
    #electronic energies of isolated atom: E_el^A
    #experimental heat of formation of isolatied atom : delta_H_f^A
    # delta H_f^mol = E_tot^mol - sum_A E_el^A + sum_A delta_H_f^A
    #flag: True, return Hf = Etot - Eiso_sum + eheat_sum
           False, return Etot - Eiso_sum
    """
    # electronic energy for isolated atom, sum for each molecule
    Eiso_sum = torch.zeros_like(Etot)
    Eiso_sum.index_add_(0, atom_molid, Eiso)
    if flag:
        # experimental heat of formation for each atom, sum for each molecule
        eheat_sum = torch.zeros_like(Etot)
        eheat_sum.index_add_(0, atom_molid, const.eheat[Z])
        # Hf = Etot - Eiso_sum + eheat_sum
        return Etot - Eiso_sum + eheat_sum, Eiso_sum
    else:
        return Etot - Eiso_sum, Eiso_sum
