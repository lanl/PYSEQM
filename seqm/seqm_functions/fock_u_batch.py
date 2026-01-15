import torch

from .fock import (
    DEFAULT_NBF,
    PM6_FLOCAL_MAP,
    PM6_NBF,
    TRIL_IDX_4,
    TRIL_IDX_9,
    WEIGHT_10,
    WEIGHT_45,
    K_ind_4,
    K_ind_9,
    _cached_index,
    _cached_tensor,
)


def fock_u_batch(
    nmol,
    molsize,
    P0,
    M,
    maskd,
    mask,
    idxi,
    idxj,
    w,
    W,
    gss,
    gpp,
    gsp,
    gp2,
    hsp,
    themethod,
    zetas,
    zetap,
    zetad,
    Z,
    F0SD,
    G2SD,
):
    """
    construct fock matrix

    P0 : total density matrix, P0 = Palpha + Pbeta, Palpha==Pbeta,
        shape (nmol, 4*molsize, 4*molsize)
        for closed shell molecule only, RHF is used, alpha and beta has same WF
    M : Hcore in the shape of (nmol*molsize**2,4,4)
    to construct Hcore from M, check hcore.py
    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)

    maskd : mask for diagonal block for M, shape(ntotatoms,)
    M[maskd] take out the diagonal block
    gss, gpp, gsp, shape (ntotatoms, )
    P0: shape (nmol, 4*molsize, 4*molsize)
    """
    nbf = PM6_NBF if themethod == "PM6" else DEFAULT_NBF
    P_tot = (
        (P0[:, 0] + P0[:, 1])
        .reshape((nmol, molsize, nbf, molsize, nbf))
        .transpose(2, 3)
        .reshape(nmol * molsize * molsize, nbf, nbf)
    )

    P_spin = (
        P0.transpose(0, 1)
        .reshape((2, nmol, molsize, nbf, molsize, nbf))
        .transpose(3, 4)
        .reshape(2, nmol * molsize * molsize, nbf, nbf)
    )

    F_ = M.expand(2, -1, -1, -1).clone()

    # one-center (intra-atomic) ERI contributions
    F_ = _one_center_u(F_, P_tot, P_spin, maskd, gss, gpp, gsp, gp2, hsp)

    # PM6 one-center ERI contributions from d-orbitals
    if themethod == "PM6":
        F_ = _d_contrib_one_center_u(F_, P_tot, P_spin, W, maskd)

    # 5) two-center (coulomb J & exchange K) neighbor-atom terms
    F_ = _two_center_u(F_, P_tot, P_spin, w, maskd, mask, idxi, idxj, themethod)

    # 6) reassemble full Fock matrix
    nrs = nbf * molsize
    F_full = (
        F_.view(2, nmol, molsize, molsize, nbf, nbf)
        .transpose(3, 4)
        .reshape(2, nmol, nrs, nrs)
        .transpose(0, 1)
    )
    # symmetrize upper triangle since only the lower triangle of F has been built so far
    F_full += F_full.triu(1).transpose(2, 3)

    return F_full


def _one_center_u(F, Ptot, P_spin, maskd, gss, gpp, gsp, gp2, hsp):
    """
    Adds the intra-atomic (one-center) two-electron contributions.
    """
    Ptot_d = Ptot[maskd].unsqueeze(0)  # (npairs, nbf, nbf)
    Pspin_d = P_spin[:, maskd]  # (2, npairs, nbf, nbf)
    P_opp_spin_d = Pspin_d[[1, 0]]

    # precompute p‐shell populations
    Pptot = Ptot_d[..., 1, 1] + Ptot_d[..., 2, 2] + Ptot_d[..., 3, 3]  # (npairs,)
    Pspin_ptot = Pspin_d[..., 1, 1] + Pspin_d[..., 2, 2] + Pspin_d[..., 3, 3]  # (2, npairs)

    # temporary container for alpha/beta contributions
    tmp = torch.zeros_like(Pspin_d)  # (2, npairs, nbf, nbf)

    # F(s,s)
    tmp[..., 0, 0] = P_opp_spin_d[..., 0, 0] * gss + Pptot * gsp - Pspin_ptot * hsp
    pp_fac_d = gpp - gp2
    sp_fac = hsp + gsp
    pp_fac_off = gpp + gp2

    # 2) (p,p) diagonal for i=1,2,3
    for i in (1, 2, 3):
        tmp[..., i, i] = (
            Ptot_d[..., 0, 0] * gsp
            - Pspin_d[..., 0, 0] * hsp
            + P_opp_spin_d[..., i, i] * gpp
            + (Pptot - Ptot_d[..., i, i]) * gp2
            - 0.5 * (Pspin_ptot - Pspin_d[..., i, i]) * pp_fac_d
        )

    # 3) (s,p) and (p,s)
    for i in (1, 2, 3):
        tmp[..., 0, i] = 2 * Ptot_d[..., 0, i] * hsp - Pspin_d[..., 0, i] * sp_fac

    # 4) (p,p*) off‐diagonals
    for i, j in ((1, 2), (1, 3), (2, 3)):
        tmp[..., i, j] = Ptot_d[..., i, j] * pp_fac_d - 0.5 * Pspin_d[..., i, j] * pp_fac_off

    F[:, maskd] += tmp
    return F


def _d_contrib_one_center_u(F, P_tot, P_spin, W, maskd):
    """
    Adds one-center PM6 d-orbital terms for UHF:
      F:       (2, ncenters, 9, 9)
      P_tot:   (ncenters, 9, 9)    = Palpha + Pbeta
      P_spin:  (2, ncenters, 9, 9)  P_spin[0]=Palpha, P_spin[1]=Pbeta
      W:       (ncenters, 243)      three-center integrals
      maskd:   bool[ncenters]       selects on-atom blocks
    """
    dtype, device = F.dtype, F.device

    # lower-triangle coords & scaling
    tril_idx = _cached_index(TRIL_IDX_9, device)
    i0, i1 = tril_idx
    tril_scale = _cached_tensor(WEIGHT_45, device, dtype)

    # extract only the on-atom blocks
    Ptot_d = P_tot[maskd]  # (nc, 9, 9)
    Palpha_d = P_spin[0, maskd]  # (nc, 9, 9)
    Pbeta_d = P_spin[1, maskd]

    # pack the (i1,i0) entries into length-45 vectors
    Pnew_tot = Ptot_d[:, i0, i1] * tril_scale.unsqueeze(0)  # (nc, 45)
    Pnew_alpha = Palpha_d[:, i0, i1] * tril_scale.unsqueeze(0)
    Pnew_beta = Pbeta_d[:, i0, i1] * tril_scale.unsqueeze(0)

    nc = Pnew_tot.shape[0]
    Wd = W[maskd]  # (nc, 243)

    # build local F contributions for alpha and beta
    Floc_alpha = torch.zeros(nc, 45, device=device, dtype=dtype)
    Floc_beta = torch.zeros(nc, 45, device=device, dtype=dtype)

    for col, w_idxs, p_idxs in PM6_FLOCAL_MAP:
        Jcol = (Wd[:, w_idxs] * Pnew_tot[:, p_idxs]).sum(dim=1)
        Kcol_alpha = (Wd[:, w_idxs] * Pnew_alpha[:, p_idxs]).sum(dim=1)
        Kcol_beta = (Wd[:, w_idxs] * Pnew_beta[:, p_idxs]).sum(dim=1)

        # Coulomb minus half-exchange
        Floc_alpha[:, col] = Jcol - 0.5 * Kcol_alpha
        Floc_beta[:, col] = Jcol - 0.5 * Kcol_beta

    F[0, maskd, i1, i0] += Floc_alpha
    F[1, maskd, i1, i0] += Floc_beta
    return F


def _two_center_u(F, P_tot, P_spin, w, maskd, mask, idxi, idxj, themethod):
    """
    Adds two-center (neighbor-atom) J & K contributions for UHF.

    F        : Tensor of shape (2, nPairs, nbf, nbf) – spin‐stacked Fock blocks
    P_spin   : Tensor of shape (2, nPairs, nbf, nbf) – P_spin[0]=Pα, P_spin[1]=Pβ
    P_tot    : Tensor of shape (nPairs, nbf, nbf)    – Pα + Pβ for Coulomb
    w        : Tensor of shape (nPairs, nP, nP)       – two‐center integrals
    maskd    : Boolean mask of length nPairs selecting “A” blocks
    mask     : Boolean mask of length nPairs selecting pairs for exchange
    idxi, idxj : LongTensors of length nPairs giving neighbor‐pair indices
    themethod: 'PM6' or other

    Returns: F updated in‐place with J/K for both α and β channels.
    """
    # pick basis‐size‐dependent parameters
    if themethod == "PM6":
        nbf = PM6_NBF
        tril_idx = _cached_index(TRIL_IDX_9, F.device)
        weight_tc = _cached_tensor(WEIGHT_45, F.device, F.dtype)
    else:
        nbf = DEFAULT_NBF
        tril_idx = _cached_index(TRIL_IDX_4, F.device)
        weight_tc = _cached_tensor(WEIGHT_10, F.device, F.dtype)

    i0, i1 = tril_idx

    # neighbor‐pair ordering: A and B centers
    idxA, idxB = (idxj, idxi) if themethod == "PM6" else (idxi, idxj)

    # ——— Coulomb (J) ———
    # pack P_tot for A and B blocks
    PA = (P_tot[maskd[idxA]][:, i1, i0] * weight_tc).unsqueeze(-1)  # (nPairs, nP, 1)
    PB = (P_tot[maskd[idxB]][:, i1, i0] * weight_tc).unsqueeze(-2)  # (nPairs, 1, nP)

    # contract to get J contributions
    J_A = (PA * w).sum(dim=1)  # (nPairs, nP)
    J_B = (PB * w).sum(dim=2)

    # scatter J back into each spin‐block
    B = w.shape[0]
    sumA = torch.zeros(B, nbf, nbf, device=F.device, dtype=F.dtype)
    sumB = torch.zeros_like(sumA)
    sumA[:, i1, i0] = J_A
    sumB[:, i1, i0] = J_B

    # both α and β see the same Coulomb
    F[0].index_add_(0, maskd[idxA], sumB)
    F[0].index_add_(0, maskd[idxB], sumA)
    F[1].index_add_(0, maskd[idxA], sumB)
    F[1].index_add_(0, maskd[idxB], sumA)

    # ——— Exchange (K) ———
    # prepare indirect indexing
    ind = _cached_index(K_ind_9 if themethod == "PM6" else K_ind_4, F.device)
    p_idx = ind.view(-1)  # (nbf*nbf,)
    w1 = w[:, p_idx, :].view(B, nbf, nbf, -1)

    # for each spin, build and scatter its exchange
    # same‐spin density, with the -½ factor
    Pp = -P_spin[:, mask]  # (nExPairs, nbf, nbf)
    Ksum = torch.zeros(2, B, nbf, nbf, device=F.device, dtype=F.dtype)

    if themethod == "PM6":
        Pp = Pp.transpose(2, 3)  # align for d‐orbitals
        for j in range(nbf):
            q_idx = ind[j]  # (nbf,)
            w2 = w1[..., q_idx]  # (nPairs, nbf, nbf)
            Ksum[..., j, :] = (w2 * Pp.unsqueeze(2)).sum(dim=(3, 4))
    else:
        for j in range(nbf):
            q_idx = ind[j]
            w2 = w1[..., q_idx]
            Ksum[..., :, j] = (w2 * Pp.unsqueeze(2)).sum(dim=(3, 4))

    F.index_add_(1, mask, Ksum)

    return F


# This is the original version that works
def old_fock_u_batch(
    nmol,
    molsize,
    P0,
    M,
    maskd,
    mask,
    idxi,
    idxj,
    w,
    W,
    gss,
    gpp,
    gsp,
    gp2,
    hsp,
    themethod,
    zetas,
    zetap,
    zetad,
    Z,
    F0SD,
    G2SD,
):
    """
    construct fock matrix

    P0 : total density matrix, P0 = Palpha + Pbeta, Palpha==Pbeta,
        shape (nmol, 4*molsize, 4*molsize)
        for closed shell molecule only, RHF is used, alpha and beta has same WF
    M : Hcore in the shape of (nmol*molsize**2,4,4)
    to construct Hcore from M, check hcore.py
    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)

    maskd : mask for diagonal block for M, shape(ntotatoms,)
    M[maskd] take out the diagonal block
    gss, gpp, gsp, shape (ntotatoms, )
    P0: shape (nmol, 4*molsize, 4*molsize)
    """
    dtype = M.dtype
    device = M.device

    if themethod == "PM6":
        P = (
            (P0[:, 0] + P0[:, 1])
            .reshape((nmol, molsize, 9, molsize, 9))
            .transpose(2, 3)
            .reshape(nmol * molsize * molsize, 9, 9)
        )

        PAlpha_ = (
            P0.transpose(0, 1)
            .reshape((2, nmol, molsize, 9, molsize, 9))
            .transpose(3, 4)
            .reshape(2, nmol * molsize * molsize, 9, 9)
        )
    else:
        P = (
            (P0[:, 0] + P0[:, 1])
            .reshape((nmol, molsize, 4, molsize, 4))
            .transpose(2, 3)
            .reshape(nmol * molsize * molsize, 4, 4)
        )

        PAlpha_ = (
            P0.transpose(0, 1)
            .reshape((2, nmol, molsize, 4, molsize, 4))
            .transpose(3, 4)
            .reshape(2, nmol * molsize * molsize, 4, 4)
        )

    # at this moment,  P has the same shape as M, as it is more convenient
    # to use here
    # while for diagonalization, may have to reshape

    # for the diagonal block, the summation over ortitals on the same atom in Fock matrix
    F_ = M.expand(2, -1, -1, -1).clone()
    Pptot = P[..., 1, 1] + P[..., 2, 2] + P[..., 3, 3]
    PAlpha_ptot_ = PAlpha_[..., 1, 1] + PAlpha_[..., 2, 2] + PAlpha_[..., 3, 3]

    #  F_mu_mu = Hcore + \sum_nu^A P_nu_nu (g_mu_nu - 0.5 h_mu_nu) + \sum^B
    """
    #(s,s)
    F[maskd,0,0].add_( 0.5*P[maskd,0,0]*gss + Pptot[maskd]*(gsp-0.5*hsp) )

    for i in range(1,4):
        #(p,p)
        F[maskd,i,i].add_( P[maskd,0,0]*(gsp-0.5*hsp) + 0.5*P[maskd,i,i]*gpp \
                        + (Pptot[maskd] - P[maskd,i,i]) * (1.25*gp2-0.25*gpp) )
        #(s,p) = (p,s) upper triangle
        F[maskd,0,i].add_( P[maskd,0,i]*(1.5*hsp - 0.5*gsp) )

    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        F[maskd,i,j].add_( P[maskd,i,j]* (0.75*gpp - 1.25*gp2) )
    #
    """
    ### http://openmopac.net/manual/1c2e.html
    if themethod == "PM6":
        # (s,s)
        TMP_ = torch.zeros_like(F_)
        size = nmol * molsize * molsize
        # AllIntegrals = calc_integral(zetas, zetap, zetad, Z, size, maskd, P, F0SD, G2SD)
        TMP_[:, maskd, 0, 0] = (
            PAlpha_[[1, 0]][:, maskd, 0, 0] * gss + Pptot[maskd] * gsp - PAlpha_ptot_[:, maskd] * hsp
        )
        for i in range(1, 4):
            # (p,p)
            TMP_[:, maskd, i, i] = (
                P[maskd, 0, 0] * gsp
                - PAlpha_[:, maskd, 0, 0] * hsp
                + PAlpha_[[1, 0]][:, maskd, i, i] * gpp
                + (Pptot[maskd] - P[maskd, i, i]) * gp2
                - 0.5 * (PAlpha_ptot_[:, maskd] - PAlpha_[:, maskd, i, i]) * (gpp - gp2)
            )

            # (s,p) = (p,s) upper triangle
            TMP_[:, maskd, 0, i] = 2 * P[maskd, 0, i] * hsp - PAlpha_[:, maskd, 0, i] * (hsp + gsp)

        # (p,p*)
        for i, j in [(1, 2), (1, 3), (2, 3)]:
            TMP_[:, maskd, i, j] = P[maskd, i, j] * (gpp - gp2) - 0.5 * PAlpha_[:, maskd, i, j] * (gpp + gp2)

        ##(d,d) should go here.  Do not know exact form
        ##for i in range(4,9):
        ##TMP[maskd,i,i] = P[maskd,i,i]*(AllIntegrals[maskd,i,i])
        ##print( P[maskd,i,i],AllIntegrals[maskd,i,i])
        # (d,s) should go here.  Do not know exact form
        ##TMP[maskd,0,i] = P[maskd,0,i]*(AllIntegrals[maskd,i,i])
        # (d,d*) should go here.  Do not know exact form
        ##for i,j in [(4,5), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8), (6,7), (6,8), (7,8)]:
        ##TMP[maskd,i,j] = P[maskd,i,j]*(AllIntegrals[maskd,i,j])

        # (d,p) should go here.  Do not know exact form
        ##for i,j in [(1,4), (1,5), (1,6), (1,7), (1,8), (2,4), (2,5), (2,6), (2,7), (2,8), (3,4), (3,5), (3,6), (3,7), (3,8)]:
        ##TMP[maskd,i,j] = P[maskd,i,j]*(AllIntegrals[maskd,i,j])
        ##print(TMP)
        ##print(F.add(AllIntegrals))
        F_.add_(TMP_)

        FLocal = torch.zeros(2, size, 45, device=device)
        TMP_d = torch.zeros(2, size, 9, 9, device=device)

        IntIJ = [
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
            10,
            10,
            10,
            10,
            10,
            10,
            11,
            11,
            11,
            11,
            11,
            11,
            12,
            12,
            12,
            12,
            12,
            13,
            13,
            13,
            13,
            13,
            14,
            14,
            14,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            16,
            16,
            16,
            16,
            16,
            17,
            17,
            17,
            17,
            17,
            18,
            18,
            18,
            19,
            19,
            19,
            19,
            19,
            20,
            20,
            20,
            20,
            20,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            23,
            23,
            23,
            23,
            23,
            24,
            24,
            24,
            24,
            24,
            25,
            25,
            25,
            25,
            26,
            26,
            26,
            26,
            26,
            26,
            27,
            27,
            27,
            27,
            27,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            29,
            29,
            29,
            29,
            29,
            30,
            30,
            30,
            31,
            31,
            31,
            31,
            31,
            32,
            32,
            32,
            32,
            32,
            33,
            33,
            33,
            33,
            33,
            34,
            34,
            34,
            34,
            35,
            35,
            35,
            35,
            35,
            36,
            36,
            36,
            36,
            36,
            36,
            36,
            36,
            36,
            36,
            36,
            36,
            37,
            37,
            37,
            37,
            38,
            38,
            38,
            38,
            38,
            39,
            39,
            39,
            39,
            39,
            40,
            40,
            40,
            41,
            42,
            42,
            42,
            42,
            42,
            43,
            43,
            43,
            43,
            44,
            44,
            44,
            44,
            44,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
        ]

        IntKL = [
            15,
            21,
            28,
            36,
            45,
            12,
            19,
            23,
            39,
            11,
            15,
            21,
            22,
            26,
            28,
            36,
            45,
            13,
            24,
            32,
            38,
            34,
            37,
            43,
            11,
            15,
            21,
            22,
            26,
            28,
            36,
            45,
            17,
            25,
            31,
            16,
            20,
            27,
            44,
            29,
            33,
            35,
            42,
            15,
            21,
            22,
            28,
            36,
            45,
            3,
            6,
            11,
            21,
            26,
            36,
            2,
            12,
            19,
            23,
            39,
            4,
            13,
            24,
            32,
            38,
            14,
            17,
            31,
            1,
            3,
            6,
            10,
            15,
            21,
            22,
            28,
            36,
            45,
            8,
            16,
            20,
            27,
            44,
            7,
            14,
            17,
            25,
            31,
            18,
            30,
            40,
            2,
            12,
            19,
            23,
            39,
            8,
            16,
            20,
            27,
            44,
            1,
            3,
            6,
            10,
            11,
            15,
            21,
            22,
            26,
            28,
            36,
            45,
            3,
            6,
            10,
            15,
            21,
            22,
            28,
            36,
            45,
            2,
            12,
            19,
            23,
            39,
            4,
            13,
            24,
            32,
            38,
            7,
            17,
            25,
            31,
            3,
            6,
            11,
            21,
            26,
            36,
            8,
            16,
            20,
            27,
            44,
            1,
            3,
            6,
            10,
            15,
            21,
            22,
            28,
            36,
            45,
            9,
            29,
            33,
            35,
            42,
            18,
            30,
            40,
            7,
            14,
            17,
            25,
            31,
            4,
            13,
            24,
            32,
            38,
            9,
            29,
            33,
            35,
            42,
            5,
            34,
            37,
            43,
            9,
            29,
            33,
            35,
            42,
            1,
            3,
            6,
            10,
            11,
            15,
            21,
            22,
            26,
            28,
            36,
            45,
            5,
            34,
            37,
            43,
            4,
            13,
            24,
            32,
            38,
            2,
            12,
            19,
            23,
            39,
            18,
            30,
            40,
            41,
            9,
            29,
            33,
            35,
            42,
            5,
            34,
            37,
            43,
            8,
            16,
            20,
            27,
            44,
            1,
            3,
            6,
            10,
            15,
            21,
            22,
            28,
            36,
            45,
        ]

        j = 0
        filla = [
            0,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ]
        fillb = [
            0,
            0,
            1,
            0,
            1,
            2,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            4,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        PTMP = P.clone()
        PTMP_alpha = PAlpha_.clone()

        i = 0
        while i < 9:
            j = 0
            while j < 9:
                PTMP[..., i, j] = P[..., j, i]
                PTMP_alpha[..., i, j] = PAlpha_[..., j, i]

                if i != j:
                    PTMP[..., i, j] = 2.0 * PTMP[..., i, j]
                    PTMP_alpha[..., i, j] = 2.0 * PTMP_alpha[..., i, j]
                j = j + 1
            i = i + 1

        Pnew = torch.zeros(size, 45, device=device)
        Pnew_alpha = torch.zeros(2, size, 45, device=device)
        i = 0
        while i < 45:
            Pnew[..., i] = PTMP[..., filla[i], fillb[i]]
            Pnew_alpha[..., i] = PTMP_alpha[..., filla[i], fillb[i]]
            i = i + 1
        j = 0

        while j < 243:
            ij = IntIJ[j] - 1
            kl = IntKL[j] - 1
            # FLocal[...,ij]=FLocal[...,ij] + W[0][...,j]*Pnew[...,kl] - 0.5*W[1][...,j]*Pnew_alpha[...,kl]
            # print(Pnew_alpha*2-Pnew)
            FLocal[..., ij] = (
                FLocal[..., ij] + (W[0][..., j] - W[1][..., j]) * Pnew[..., kl]
            )  # - 2*W[1][...,j]*Pnew_alpha[...,kl]

            j = j + 1
        j = 0

        while j < 45:
            k = filla[j]
            l = fillb[j]
            TMP_d[..., k, l] = FLocal[..., j]
            j = j + 1
        TMP2 = TMP_d.clone()
        i = 0
        while i < 9:
            j = 0
            while j < 9:
                TMP2[..., j, i] = TMP_d[..., i, j]
                j = j + 1
            i = i + 1

        i = 0
        j = 0
        while i < TMP2.shape[2]:
            j = 0
            while j < TMP2.shape[3]:
                if i > j:
                    TMP2[..., i, j] = 0.0
                j = j + 1
            i = i + 1

        # print(time.time()-t0)
        F_.add_(TMP2)
        # del TMP_d, TMP2, PTMP, Pnew, Pptot

    else:
        # (s,s)
        TMP_ = torch.zeros_like(F_)
        TMP_[:, maskd, 0, 0] = (
            PAlpha_[[1, 0]][:, maskd, 0, 0] * gss + Pptot[maskd] * gsp - PAlpha_ptot_[:, maskd] * hsp
        )
        for i in range(1, 4):
            # (p,p)
            TMP_[:, maskd, i, i] = (
                P[maskd, 0, 0] * gsp
                - PAlpha_[:, maskd, 0, 0] * hsp
                + PAlpha_[[1, 0]][:, maskd, i, i] * gpp
                + (Pptot[maskd] - P[maskd, i, i]) * gp2
                - 0.5 * (PAlpha_ptot_[:, maskd] - PAlpha_[:, maskd, i, i]) * (gpp - gp2)
            )

            # (s,p) = (p,s) upper triangle
            TMP_[:, maskd, 0, i] = 2 * P[maskd, 0, i] * hsp - PAlpha_[:, maskd, 0, i] * (hsp + gsp)

        # (p,p*)
        for i, j in [(1, 2), (1, 3), (2, 3)]:
            TMP_[:, maskd, i, j] = P[maskd, i, j] * (gpp - gp2) - 0.5 * PAlpha_[:, maskd, i, j] * (gpp + gp2)

        F_.add_(TMP_)

    ###############
    del TMP_, PAlpha_ptot_, Pptot

    # sumation over two electron two center integrals over the neighbor atoms

    # for the diagonal block, check JAB in fock2.f
    # F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    # as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    # weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    if themethod == "PM6":
        weight = torch.tensor(
            [
                1.0,
                2.0,
                1.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                1.0,
            ],
            dtype=dtype,
            device=device,
        ).reshape((-1, 45))

    else:
        weight = torch.tensor(
            [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=dtype, device=device
        ).reshape((-1, 10))
    #
    # P[maskd[idxi]] : P^tot_{mu,nu \in A} shape (npairs, 4,4)
    # P[maskd[idxj]] : P^tot_{mu,nu \in B} shape (npairs, 4,4)

    # take out the upper triangle part in the same order as in W
    # shape (nparis, 10)
    if themethod == "PM6":
        PA = (
            P[maskd[idxj]][
                ...,
                (
                    0,
                    0,
                    1,
                    0,
                    1,
                    2,
                    0,
                    1,
                    2,
                    3,
                    0,
                    1,
                    2,
                    3,
                    4,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ),
                (
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                ),
            ]
            * weight
        ).reshape((-1, 45, 1))
        PB = (
            P[maskd[idxi]][
                ...,
                (
                    0,
                    0,
                    1,
                    0,
                    1,
                    2,
                    0,
                    1,
                    2,
                    3,
                    0,
                    1,
                    2,
                    3,
                    4,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ),
                (
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                    8,
                ),
            ]
            * weight
        ).reshape((-1, 1, 45))

    else:
        PA = (
            P[maskd[idxi]][..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)] * weight
        ).reshape((-1, 10, 1))
        PB = (
            P[maskd[idxj]][..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)] * weight
        ).reshape((-1, 1, 10))

    # suma \sum_{mu,nu \in A} P_{mu, nu in A} (mu nu, lamda sigma) = suma_{lambda sigma \in B}
    # suma shape (npairs, 10)
    suma = torch.sum(PA * w, dim=1)
    # sumb \sum_{l,s \in B} P_{l, s inB} (mu nu, l s) = sumb_{mu nu \in A}
    # sumb shape (npairs, 10)
    sumb = torch.sum(PB * w, dim=2)
    # reshape back to (npairs 4,4)
    # as will use index add in the following part

    # torch.use_deterministic_algorithms(True)
    del PA, PB

    # $$$ Why in pm6 (0, maskd[idxi], sumA) but in pm3 (0, maskd[idxj], sumA) ??????
    if themethod == "PM6":
        sumA = torch.zeros(w.shape[0], 9, 9, dtype=dtype, device=device)
        sumB = torch.zeros_like(sumA)
        sumA[
            ...,
            (
                0,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
            (
                0,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
            ),
        ] = suma
        sumB[
            ...,
            (
                0,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
            (
                0,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
            ),
        ] = sumb

        F_[0].index_add_(0, maskd[idxi], sumA)
        F_[1].index_add_(0, maskd[idxi], sumA)

        # \sum_B
        # F_.index_add_(1, maskd[idxi], sumB)
        F_[0].index_add_(0, maskd[idxj], sumB)
        F_[1].index_add_(0, maskd[idxj], sumB)

    else:
        sumA = torch.zeros(w.shape[0], 4, 4, dtype=dtype, device=device)
        sumB = torch.zeros_like(sumA)
        sumA[..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)] = suma
        sumB[..., (0, 0, 1, 0, 1, 2, 0, 1, 2, 3), (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)] = sumb

        # F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
        # $$$ index_add_ below could be done in a more efficient way
        # \sum_A
        F_[0].index_add_(0, maskd[idxj], sumA)
        F_[1].index_add_(0, maskd[idxj], sumA)

        # \sum_B
        F_[0].index_add_(0, maskd[idxi], sumB)
        F_[1].index_add_(0, maskd[idxi], sumB)
    del suma, sumb, sumA, sumB
    ###################

    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)

    if themethod == "PM6":
        sum_ = torch.zeros(2, w.shape[0], 9, 9, dtype=dtype, device=device)
    else:
        sum_ = torch.zeros(2, w.shape[0], 4, 4, dtype=dtype, device=device)

    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    if themethod == "PM6":
        ind = torch.tensor(
            [
                [0, 1, 3, 6, 10, 15, 21, 28, 36],
                [1, 2, 4, 7, 11, 16, 22, 29, 37],
                [3, 4, 5, 8, 12, 17, 23, 30, 38],
                [6, 7, 8, 9, 13, 18, 24, 31, 39],
                [10, 11, 12, 13, 14, 19, 25, 32, 40],
                [15, 16, 17, 18, 19, 20, 26, 33, 41],
                [21, 22, 23, 24, 25, 26, 27, 34, 42],
                [28, 29, 30, 31, 32, 33, 34, 35, 43],
                [36, 37, 38, 39, 40, 41, 42, 43, 44],
            ],
            dtype=torch.int64,
            device=device,
        )
    else:
        ind = torch.tensor(
            [[0, 1, 3, 6], [1, 2, 4, 7], [3, 4, 5, 8], [6, 7, 8, 9]], dtype=torch.int64, device=device
        )

    # Pp =P[mask], P_{mu \in A, lambda \in B}
    if themethod == "PM6":
        for i in range(9):
            for j in range(9):
                # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                # sum_[...,i,j] = torch.sum(Pp*torch.transpose(w[...,ind[j],:][...,:,ind[i]],1,2),dim=(1,2))
                sum_[..., i, j] = torch.sum(
                    -PAlpha_[:, mask] * torch.transpose(w[..., ind[j], :][..., :, ind[i]], 1, 2), dim=(2, 3)
                )
    else:
        for i in range(4):
            for j in range(4):
                # \sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sum_[..., i, j] = torch.sum(-PAlpha_[:, mask] * w[..., ind[i], :][..., :, ind[j]], dim=(2, 3))

    F_.index_add_(1, mask, sum_)

    # torch.use_deterministic_algorithms(False)

    ####################

    if themethod == "PM6":
        F0_ = (
            F_.reshape(2, nmol, molsize, molsize, 9, 9)
            .transpose(3, 4)
            .reshape(2, nmol, 9 * molsize, 9 * molsize)
            .transpose(0, 1)
        )
    else:
        F0_ = (
            F_.reshape(2, nmol, molsize, molsize, 4, 4)
            .transpose(3, 4)
            .reshape(2, nmol, 4 * molsize, 4 * molsize)
            .transpose(0, 1)
        )

    F0_.add_(F0_.triu(1).transpose(2, 3))

    return F0_
