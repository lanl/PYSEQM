from functools import lru_cache

import torch

from .constants import a0, ev
from .two_elec_two_center_int import GetSlaterCondonParameter

# MOPAC v_par6(1), v_par6(2): scalar and exponent of the C-C triple-bond term in the PM6 core-core repulsion
PM6_CC_SCALE = 9.278465
PM6_CC_EXPONENT = 5.983752


def elec_energy_isolated_atom(
    const, Z, uss, upp, gss, gpp, gsp, gp2, hsp, udd=None, zsn=None, zdn=None, f0sd=None, g2sd=None
):
    """
    electrionc energy for a single atom
    #eisol in block.f or in calpar.f
    return Eiso, shape (natoms,)
    """
    Eiso = (
        uss * const.ussc[Z]
        + upp * const.uppc[Z]
        + gss * const.gssc[Z]
        + gpp * const.gppc[Z]
        + gsp * const.gspc[Z]
        + gp2 * const.gp2c[Z]
        + hsp * const.hspc[Z]
    )
    if udd is not None:  # PM6 with d orbitals
        Eiso = Eiso + d_shell_isolated_atom_energy(const, Z, udd, zsn, zdn, f0sd, g2sd)
    return Eiso


# One-center terms of the free atom for partly filled d shells: coefficients of the Slater-Condon integrals
# (R016 = F0SD, R066 = F0DD, R244 = G2SD, R266 = F2DD, R466 = F4DD) in MOPAC's eiscor (integrals/mndod.F90).
_EISCOR = {
    21: (2, 0, 1, 0, 0),
    22: (4, 1, 2, 8, 1),
    23: (6, 3, 3, 15, 8),
    24: (5, 10, 5, 35, 35),
    25: (10, 10, 5, 35, 35),
    26: (12, 15, 6, 35, 35),
    27: (14, 21, 7, 43, 36),
    28: (16, 28, 8, 50, 43),
    29: (10, 45, 5, 70, 70),
    39: (2, 0, 1, 0, 0),
    40: (4, 1, 2, 8, 1),
    41: (4, 6, 4, 21, 21),
    42: (5, 10, 5, 35, 35),
    43: (10, 10, 5, 35, 35),
    44: (7, 21, 5, 43, 36),
    45: (8, 28, 5, 50, 43),
    46: (0, 45, 0, 70, 70),
    47: (10, 45, 5, 70, 70),
    57: (2, 0, 1, 0, 0),
    71: (2, 0, 1, 0, 0),
    72: (4, 1, 2, 8, 1),
    73: (6, 3, 3, 15, 8),
    74: (5, 10, 5, 35, 35),
    75: (10, 10, 5, 35, 35),
    76: (12, 15, 6, 35, 35),
    77: (14, 21, 7, 43, 36),
    78: (9, 36, 5, 56, 56),
    79: (10, 45, 5, 70, 70),
}


@lru_cache(maxsize=None)
def _eiscor(z, ns, nd, zsn, zdn, f0sd, g2sd):
    """MOPAC eiscor for one element, in eV. ns, nd: principal quantum numbers of the s and d shells."""
    sc = GetSlaterCondonParameter
    r016 = f0sd if f0sd > 0.001 else sc(0, ns, zsn, ns, zsn, nd, zdn, nd, zdn)
    r244 = g2sd if g2sd > 0.001 else sc(2, ns, zsn, nd, zdn, ns, zsn, nd, zdn)
    r066 = sc(0, nd, zdn, nd, zdn, nd, zdn, nd, zdn)
    r266 = sc(2, nd, zdn, nd, zdn, nd, zdn, nd, zdn)
    r466 = sc(4, nd, zdn, nd, zdn, nd, zdn, nd, zdn)
    a016, a066, a244, a266, a466 = _EISCOR[z]
    return a016 * r016 + a066 * r066 - a244 * r244 / 5 - a266 * r266 / 49 - a466 * r466 / 49


def d_shell_isolated_atom_energy(const, Z, udd, zsn, zdn, f0sd, g2sd):
    """
    udd*iod plus the one-center integral terms (eiscor) of the free atom's d shell, as in MOPAC's eisol. The integral
    terms are constants. zsn, zdn: the s and d exponents of the one-center integrals (s_orb_exp_tail, d_orb_exp_tail).
    """
    E = (
        udd * (const.tore - const.ussc - const.uppc)[Z]
    )  # iod, the d electrons of the free atom, is tore - ios - iop
    sel = [i for i, z in enumerate(Z.tolist()) if z in _EISCOR]
    if sel:
        idx = torch.as_tensor(sel, device=Z.device)
        vals = [
            _eiscor(z, int(const.qn_int[z]), int(const.qnD_int[z]), a, b, c, d)
            for z, a, b, c, d in zip(
                Z[idx].tolist(), zsn[idx].tolist(), zdn[idx].tolist(), f0sd[idx].tolist(), g2sd[idx].tolist()
            )
        ]
        corr = torch.zeros_like(E)
        corr[idx] = torch.as_tensor(vals, dtype=E.dtype, device=E.device)
        E = E + corr
    return E


def elec_energy(P, F, Hcore, doTriu=True):
    """
    Get the electronic energy
    P: density matrix, shape (nmol, molsize*4, molsize*4)
    F: fock matrix, shape same as P
    Hcore: Hcore matrix, shape (nmol, 4*molsize, 4*molsize)
    return Eelec : electronic energy, shape (nmol,)
    P, F: full, has upper and lower triangle
    Hcore : only have upper triangle as constructed from hcore.py
    """
    if doTriu:
        h = Hcore.triu() + Hcore.triu(1).transpose(1, 2)
    else:
        h = Hcore

    # Eelec = 0.5 * tr(P(Hcore+F))  # matmul
    # Eelec = 0.5 * \sum P*(H+F)    # elementwise product

    if len(F.size()) == 4:  # open-shell
        Eelec = 0.5 * torch.sum((P[:, 0] + P[:, 1]) * h + P[:, 0] * F[:, 0] + P[:, 1] * F[:, 1], dim=(1, 2))
    else:  # closed-shell
        Eelec = 0.5 * torch.sum(P * (h + F), dim=(1, 2))

    return Eelec


# def elec_energy_open_shell(P,F,Hcore):
#     """
#     Get the electronic energy
#     P: density matrix, shape (nmol, molsize*4, molsize*4)
#     F: fock matrix, shape same as P
#     Hcore: Hcore matrix, shape (nmol, 4*molsize, 4*molsize)
#     return Eelec : electronic energy, shape (nmol,)
#     P, F: full, has upper and lower triangle
#     Hcore : only have upper triangle as constructed from hcore.py
#     """
#     h = Hcore.triu()+Hcore.triu(1).transpose(1,2)

#     # Eelec = 0.5 * tr(P(Hcore+F))  # matmul
#     # Eelec = 0.5 * \sum P*(H+F)    # elementwise product

#     Eelec = 0.5*torch.sum(P*(h+F),dim=(1,2))
#     # print('h\n', h)
#     # print('P\n', P)
#     # print('F\n', F)
#     # print('Eelec\n', Eelec)

#     return Eelec


def elec_energy_xl(D, P, F, Hcore):
    """
    XL_BOMD
    electrionic energy is defined as:
    E(D,P) = (2*tr(Hcore*D) + tr((2D-P)*G(P)))/2.0
           = tr(D*F(P))-0.5*Tr((F(P)-Hcore)*P)
    """
    # Hcore : only has upper triangle as constructed from hcore.py
    h = Hcore.triu() + Hcore.triu(1).transpose(1, 2)

    Eelec = torch.sum(D * F - 0.5 * (F - h) * P, dim=(1, 2))

    return Eelec


# AM1 B-H, B-C and B-halogen pairs: MOPAC's ccrep drops boron's own Gaussians and uses two others instead,
# (K, L, M) for each partner (the halogens are F, Cl, Br and I)
_AM1_B_GAUSSIANS = {
    (1,): ((0.412253, 10.0, 0.832586), (-0.149917, 6.0, 1.186220)),
    (6,): ((0.261751, 8.0, 1.063995), (0.050275, 5.0, 1.936492)),
    (9, 17, 35, 53): ((0.359244, 9.0, 0.819351), (0.074729, 9.0, 1.574414)),
}


@lru_cache(maxsize=None)
def _am1_boron_table(dtype, device):
    """(K, L, M) of the two special Gaussians by partner atomic number, shape (108, 2, 3); K = 0 for other partners"""
    table = torch.zeros((108, 2, 3), dtype=dtype, device=device)
    table[..., 1] = 1.0
    for partners, gaussians in _AM1_B_GAUSSIANS.items():
        table[list(partners)] = torch.tensor(gaussians, dtype=dtype, device=device)
    return table


def pair_bond_parameters(alp, chi, ni, nj):
    """MOPAC's pair parameters (alpb, xfac) of the core-core term of MNDO/AM1/PM3, and which pairs have them."""
    xfac = chi[ni, nj]
    alpb = torch.where(alp[ni, nj] < 1e-6, torch.full_like(xfac, 1.2), alp[ni, nj])
    return xfac.abs() > 1e-5, alpb, xfac


def gaussian_core_core(method, rija, ni, nj, idxi, idxj, K, L, M, xfac, derivative=False):
    """
    Sum over the Gaussians of both atoms of K*exp(-L*(r-M)^2) for each pair, shape (npairs,); with derivative=True
    shape (2, npairs), the second row being the sum of K*L*(r-M)*exp(-L*(r-M)^2).
    As in MOPAC's ccrep, pairs with xfac > 1e-4 have no Gaussians, and for AM1/RM1 boron bonded to H, C or a halogen
    uses two special Gaussians instead of its own.
    """

    def terms(K, L, M, r):
        d = r[:, None] - M
        g = K * torch.exp(-L * d * d)
        return torch.stack((g.sum(dim=1), (g * L * d).sum(dim=1))) if derivative else g.sum(dim=1)

    wi = wj = xfac <= 1e-4
    if method in {"AM1", "RM1"}:
        table = _am1_boron_table(K.dtype, K.device)
        boron_i = (ni == 5) & (table[nj, 0, 0] != 0)
        boron_j = (nj == 5) & (table[ni, 0, 0] != 0)
        wi, wj = wi & ~boron_i, wj & ~boron_j
    total = wi * terms(K[idxi], L[idxi], M[idxi], rija) + wj * terms(K[idxj], L[idxj], M[idxj], rija)
    if method in {"AM1", "RM1"}:
        pairs = (boron_i | boron_j).nonzero().squeeze(1)
        special = table[torch.where(boron_i, nj, ni)[pairs]]
        total.index_add_(-1, pairs, terms(special[..., 0], special[..., 1], special[..., 2], rija[pairs]))
    return total


def pair_nuclear_energy(
    Z, const, nmol, ni, nj, idxi, idxj, rij, rho0xi, rho0xj, alp, chi, gam, method="AM1", parameters=None
):
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

    if method in {"OM1", "OM2", "OM3"}:
        # For OMx, gam stores fko (Klopman-Ohno)
        EnucAB = gam * tore[ni] * tore[nj] * ev / rij
        return EnucAB

    alpha = parameters[0]
    t1 = tore[ni] * tore[nj] * gam
    # special case for C-H and O-H
    XH = ((ni == 7) | (ni == 8)) & (nj == 1)
    tmp = torch.exp(-alpha[idxi] * rija)
    t2 = torch.where(XH, tmp * rija, tmp)
    t3 = torch.exp(-alpha[idxj] * rija)
    if method in {"MNDO", "PM3", "AM1", "RM1"}:
        # pairs with parameters (alpb, xfac) in MOPAC have their own scaling of the core-core term
        has_bond, alpb, xfac = pair_bond_parameters(alp, chi, ni, nj)
        scale = torch.where(has_bond, 1.0 + 2.0 * xfac * torch.exp(-alpb * rija), 1.0 + t2 + t3)
    if method == "MNDO":
        EnucAB = t1 * scale
    elif method in {"PM3", "AM1", "RM1"}:
        # two gaussian terms for PM3
        # 3~4 terms for AM1
        _, K, L, M = parameters
        # K, L , M shape (natoms,2 or 4)

        # Gaussian corrections
        t4 = tore[ni] * tore[nj] / rija
        EnucAB = t1 * scale + t4 * gaussian_core_core(method, rija, ni, nj, idxi, idxj, K, L, M, xfac)
    elif method in {"PM6", "PM6_SP", "PM6_SP_STAR"}:
        # follows MOPAC's ccrep (integrals/ccrep.F90)
        _, K, L, M = parameters
        dtype = rija.dtype
        chi_ij = chi[ni, nj]
        # a pair has bond parameters if its xfac is defined; otherwise MOPAC uses a generic core-core term
        has_bond = chi_ij.abs() > 1e-5
        abond = torch.where(alp[ni, nj] < 1e-6, torch.full_like(rija, 1.2), alp[ni, nj])

        XH = ((ni == 6) | (ni == 7) | (ni == 8)) & (nj == 1)
        XCC = (ni == 6) & (nj == 6)
        XSiO = (ni == 14) & (nj == 8)
        scale = 1.0 + 2.0 * chi_ij * torch.exp(-abond * (rija + 0.0003 * torch.pow(rija, 6)))
        scale = torch.where(XH, 1.0 + 2.0 * chi_ij * torch.exp(-abond * rija**2), scale)  # C-H, N-H, O-H
        scale = torch.where(XCC, scale + PM6_CC_SCALE * torch.exp(-PM6_CC_EXPONENT * rija), scale)
        scale = torch.where(XSiO, scale - 0.0007 * torch.exp(-((rija - 2.9) ** 2)), scale)
        lanthanide = ((ni > 56) & (ni < 72)) | ((nj > 56) & (nj < 72))
        generic = 1.0 + 10.0 * torch.exp(-torch.where(lanthanide, 3.0, 2.18) * rija)
        scale = torch.where(has_bond, scale, generic)
        EnucAB = tore[ni] * tore[nj] * ev / torch.sqrt(rij * rij + (rho0xi + rho0xj) ** 2) * scale

        # Gaussian corrections: pairs with bond parameters use only the first Gaussian of each atom; pairs without
        # use all four, and MOPAC counts the first one twice
        w_bond = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=rija.device)
        w_generic = torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=dtype, device=rija.device)
        W = torch.where(has_bond.unsqueeze(1), w_bond, w_generic)
        t4 = tore[ni] * tore[nj] / rija
        t5 = torch.sum(W * K[idxi] * torch.exp(-L[idxi] * (rija.reshape((-1, 1)) - M[idxi]) ** 2), dim=1)
        t6 = torch.sum(W * K[idxj] * torch.exp(-L[idxj] * (rija.reshape((-1, 1)) - M[idxj]) ** 2), dim=1)
        EnucAB = EnucAB + t4 * (t5 + t6)

        # unpolarizable core - unpolarizable core repulsion (a short-range wall; MOPAC uses 0.3333 for the cube root)
        ax = rija / (ni.to(dtype) ** 0.3333 + nj.to(dtype) ** 0.3333)
        EnucAB = EnucAB + torch.where(ax < 3.0, (1.0e-8 / ax**12).clamp(max=1.0e5), torch.zeros_like(ax))
    else:
        raise ValueError("Supported Method: MNDO, AM1, PM3, PM6, PM6_SP, PM6_SP_STAR, OM1, OM2, OM3")
    return EnucAB


def total_energy(nmol, pair_molid, EnucAB, Eelec):
    r"""
    total energy for each molecule
    total energy E_tot^mol= Eelec + \sum{pair A,B,A<B} E_nuc^AB
    #nuclear energy between pair of atom A and B: E_nuc^AB

    as index_add is expensive, there is no need to do this during training
    EnucAB :computed from pair_nuclear_energy, shape (npairs,)
    pair_molid : molecule id for each pair
    Eelec : electronic energy for each molecule, computed from elec_energy, shape (nmol)

    """
    Enuc = torch.zeros((nmol,), dtype=EnucAB.dtype, device=EnucAB.device)
    Enuc.index_add_(0, pair_molid, EnucAB)
    return Eelec + Enuc, Enuc


def heat_formation(const, nmol, atom_molid, Z, Etot, Eiso, flag=True):
    r"""
    get the heat of formation for each molecule
    return Hf : shape (nmol,)
    #heat of formation : delta H_f^mol
    #electronic energies of isolated atom: E_el^A
    #experimental heat of formation of isolatied atom : delta_H_f^A
    # delta H_f^mol = E_tot^mol - \sum_A E_el^A + \sum_A delta_H_f^A
    #flag: True, return Hf = Etot - Eiso_sum + eheat_sum
           False, return Etot - Eiso_sum
    """
    # electronic energy for isolated atom, sum for each molecule
    Eiso_sum = torch.zeros_like(Etot)
    Eiso_sum.index_add_(0, atom_molid, Eiso.to(dtype=Etot.dtype, device=Etot.device))
    if flag:
        # experimental heat of formation for each atom, sum for each molecule
        eheat_sum = torch.zeros_like(Etot)
        eheat_sum.index_add_(0, atom_molid, const.eheat[Z].to(dtype=Etot.dtype, device=Etot.device))
        # Hf = Etot - Eiso_sum + eheat_sum
        return Etot - Eiso_sum + eheat_sum, Eiso_sum
    else:
        return Etot - Eiso_sum, Eiso_sum
