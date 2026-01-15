import torch

# ——— Constants —————————————————————————————————————————————————————————
# fmt: off
# Basis function counts
DEFAULT_NBF = 4  # s,p basis
PM6_NBF = 9  # s,p,d basis

# Precompute lower-triangle indices for packing (including diagonal)
TRIL_IDX_4 = torch.tril_indices(DEFAULT_NBF, DEFAULT_NBF, offset=0)
TRIL_IDX_9 = torch.tril_indices(PM6_NBF, PM6_NBF, offset=0)
# Weight tensor to scale the lower-triangle elements by 2
WEIGHT_10 = torch.tensor([1.0,
                          2.0, 1.0,
                          2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 1.0])

WEIGHT_45 = torch.tensor([1.0,
                          2.0, 1.0,
                          2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0])
# Mapping for PM6 one-center two-electron integral (W) -> local F contributions.
# Each tuple is (FLocal_column, [W_idxs], [Pnew_idxs])
PM6_FLOCAL_MAP = [
    (0, [0, 1, 2, 3, 4], [14, 20, 27, 35, 44]),
    (1, [5, 6, 7, 8], [11, 18, 22, 38]),
    (2, [9, 10, 11, 12, 13, 14, 15, 16], [10, 14, 20, 21, 25, 27, 35, 44]),
    (3, [17, 18, 19, 20], [12, 23, 31, 37]),
    (4, [21, 22, 23], [33, 36, 42]),
    (5, [24, 25, 26, 27, 28, 29, 30, 31], [10, 14, 20, 21, 25, 27, 35, 44]),
    (6, [32, 33, 34], [16, 24, 30]),
    (7, [35, 36, 37, 38], [15, 19, 26, 43]),
    (8, [39, 40, 41, 42], [28, 32, 34, 41]),
    (9, [43, 44, 45, 46, 47, 48], [14, 20, 21, 27, 35, 44]),
    (10, [49, 50, 51, 52, 53, 54], [2, 5, 10, 20, 25, 35]),
    (11, [55, 56, 57, 58, 59], [1, 11, 18, 22, 38]),
    (12, [60, 61, 62, 63, 64], [3, 12, 23, 31, 37]),
    (13, [65, 66, 67], [13, 16, 30]),
    (14, [68, 69, 70, 71, 72, 73, 74, 75, 76, 77], [0, 2, 5, 9, 14, 20, 21, 27, 35, 44]),
    (15, [78, 79, 80, 81, 82], [7, 15, 19, 26, 43]),
    (16, [83, 84, 85, 86, 87], [6, 13, 16, 24, 30]),
    (17, [88, 89, 90], [17, 29, 39]),
    (18, [91, 92, 93, 94, 95], [1, 11, 18, 22, 38]),
    (19, [96, 97, 98, 99, 100], [7, 15, 19, 26, 43]),
    (
        20,
        [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        [0, 2, 5, 9, 10, 14, 20, 21, 25, 27, 35, 44],
    ),
    (21, [113, 114, 115, 116, 117, 118, 119, 120, 121], [2, 5, 9, 14, 20, 21, 27, 35, 44]),
    (22, [122, 123, 124, 125, 126], [1, 11, 18, 22, 38]),
    (23, [127, 128, 129, 130, 131], [3, 12, 23, 31, 37]),
    (24, [132, 133, 134, 135], [6, 16, 24, 30]),
    (25, [136, 137, 138, 139, 140, 141], [2, 5, 10, 20, 25, 35]),
    (26, [142, 143, 144, 145, 146], [7, 15, 19, 26, 43]),
    (27, [147, 148, 149, 150, 151, 152, 153, 154, 155, 156], [0, 2, 5, 9, 14, 20, 21, 27, 35, 44]),
    (28, [157, 158, 159, 160, 161], [8, 28, 32, 34, 41]),
    (29, [162, 163, 164], [17, 29, 39]),
    (30, [165, 166, 167, 168, 169], [6, 13, 16, 24, 30]),
    (31, [170, 171, 172, 173, 174], [3, 12, 23, 31, 37]),
    (32, [175, 176, 177, 178, 179], [8, 28, 32, 34, 41]),
    (33, [180, 181, 182, 183], [4, 33, 36, 42]),
    (34, [184, 185, 186, 187, 188], [8, 28, 32, 34, 41]),
    (
        35,
        [189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200],
        [0, 2, 5, 9, 10, 14, 20, 21, 25, 27, 35, 44],
    ),
    (36, [201, 202, 203, 204], [4, 33, 36, 42]),
    (37, [205, 206, 207, 208, 209], [3, 12, 23, 31, 37]),
    (38, [210, 211, 212, 213, 214], [1, 11, 18, 22, 38]),
    (39, [215, 216, 217], [17, 29, 39]),
    (40, [218], [40]),
    (41, [219, 220, 221, 222, 223], [8, 28, 32, 34, 41]),
    (42, [224, 225, 226, 227], [4, 33, 36, 42]),
    (43, [228, 229, 230, 231, 232], [7, 15, 19, 26, 43]),
    (44, [233, 234, 235, 236, 237, 238, 239, 240, 241, 242], [0, 2, 5, 9, 14, 20, 21, 27, 35, 44]),
]

K_ind_9 = torch.tensor([
    [ 0, 1, 3, 6,10,15,21,28,36],
    [ 1, 2, 4, 7,11,16,22,29,37],
    [ 3, 4, 5, 8,12,17,23,30,38],
    [ 6, 7, 8, 9,13,18,24,31,39],
    [10,11,12,13,14,19,25,32,40],
    [15,16,17,18,19,20,26,33,41],
    [21,22,23,24,25,26,27,34,42],
    [28,29,30,31,32,33,34,35,43],
    [36,37,38,39,40,41,42,43,44]
], dtype=torch.long )
K_ind_4 = torch.tensor([
    [0,1,3,6],
    [1,2,4,7],
    [3,4,5,8],
    [6,7,8,9]
], dtype=torch.long)
# fmt: on
P_INDEX_3 = torch.tensor([1, 2, 3], dtype=torch.long)
P_OFF_I = torch.tensor([1, 1, 2], dtype=torch.long)
P_OFF_J = torch.tensor([2, 3, 3], dtype=torch.long)

_WEIGHT_CACHE = {}
_INDEX_CACHE = {}


def _cached_tensor(base, device, dtype=None):
    key = (id(base), device, dtype or base.dtype)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        cached = base.to(device=device, dtype=(dtype or base.dtype))
        _WEIGHT_CACHE[key] = cached
    return cached


def _cached_index(base, device):
    key = (id(base), device)
    cached = _INDEX_CACHE.get(key)
    if cached is None:
        cached = base.to(device=device)
        _INDEX_CACHE[key] = cached
    return cached


# ——— Main fock function ——————————————————————————————————————————————


def fock(
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
    Construct the Fock matrix for either the default (4-basis) or PM6 (9-basis) method.
    """
    # 1) reshape density into blocks of shape (nbf,nbf)
    nbf = PM6_NBF if themethod == "PM6" else DEFAULT_NBF
    P = P0.view(nmol, molsize, nbf, molsize, nbf).transpose(2, 3).reshape(-1, nbf, nbf)

    # 2) start F with Hcore part
    F = M.clone()

    # 3) one-center (intra-atomic) ERI contributions
    F = _one_center(F, P, maskd, gss, gpp, gsp, gp2, hsp)

    # 4) PM6 one-center ERI contributions from d-orbitals
    if themethod == "PM6":
        F = _d_contrib_one_center(F, P, W, maskd)

    # 5) two-center (coulomb J & exchange K) neighbor-atom terms
    F = _two_center(F, P, w, maskd, mask, idxi, idxj, themethod)

    # 6) reassemble full Fock matrix
    nrs = nbf * molsize
    F_full = F.view(nmol, molsize, molsize, nbf, nbf).transpose(2, 3).reshape(nmol, nrs, nrs)
    # symmetrize lower triangle since only the upper triangle of F has been built so far
    F_full += F_full.triu(1).transpose(1, 2)
    return F_full


# ——— Helper: one-center —————————————————————————————————————————————


def _one_center(F, P, maskd, gss, gpp, gsp, gp2, hsp):
    """
    Adds the intra-atomic (one-center) two-electron contributions.
    """
    Pdiag = P[maskd]
    Pss = Pdiag[..., 0, 0]
    Pptot = Pdiag[..., 1, 1] + Pdiag[..., 2, 2] + Pdiag[..., 3, 3]

    # precompute factors
    sp_fac_1 = gsp - 0.5 * hsp
    sp_fac_2 = 1.5 * hsp - 0.5 * gsp
    pp_fac_d = 1.25 * gp2 - 0.25 * gpp
    pp_fac_off = 0.75 * gpp - 1.25 * gp2

    tmp = torch.zeros_like(Pdiag)

    # F(s,s)
    tmp[:, 0, 0] = 0.5 * Pss * gss + Pptot * sp_fac_1

    i = _cached_index(P_INDEX_3, P.device)
    Pp_all = Pdiag[:, i, i]
    tmp[:, i, i] = (
        (Pss * sp_fac_1).unsqueeze(-1)
        + 0.5 * Pp_all * gpp.unsqueeze(-1)
        + (Pptot.unsqueeze(-1) - Pp_all) * pp_fac_d.unsqueeze(-1)
    )
    tmp[:, 0, i] = Pdiag[:, 0, i] * sp_fac_2.unsqueeze(-1)
    ij_0 = _cached_index(P_OFF_I, P.device)
    ij_1 = _cached_index(P_OFF_J, P.device)
    tmp[:, ij_0, ij_1] = Pdiag[:, ij_0, ij_1] * pp_fac_off.unsqueeze(-1)

    # # (p,p) diag + (s,p)/(p,s)
    # for i in range(1,4):
    #     # (pp|pp)
    #     Pp = Pdiag[:,i,i]
    #     tmp[:,i,i] = Pss*sp_fac_1 + 0.5*Pp*gpp + (Pptot-Pp) * pp_fac_d
    #     # (ss|pp)
    #     tmp[:,0,i] = Pdiag[:,0,i] * sp_fac_2
    #
    # # (p,p*) off-diagonals
    # for i,j in [(1,2),(1,3),(2,3)]:
    #     tmp[:,i,j] = Pdiag[:,i,j] * pp_fac_off

    F[maskd] += tmp
    return F


# ——— Helper: one-center with d-orbitals —————————————————————————————


def _d_contrib_one_center(F, P, W, maskd):
    """
    Adds one-center terms from d-orbitals via the W-integrals.
    """
    tril_idx = _cached_index(TRIL_IDX_9, P.device)  # lower-triangle coords
    i0, i1 = tril_idx
    tril_scale = _cached_tensor(WEIGHT_45, P.device, P.dtype)
    Pnew = P[:, i0, i1] * tril_scale.unsqueeze(0)

    blk = P.shape[0]
    Floc = torch.zeros(blk, 45, device=P.device, dtype=P.dtype)

    for col, w_idxs, p_idxs in PM6_FLOCAL_MAP:
        Floc[:, col] = (W[:, w_idxs] * Pnew[:, p_idxs]).sum(dim=1)

    F[:, i1, i0] += Floc
    return F


# ——— Helper: two-center (J & K) —————————————————————————————————————


def _two_center(F, P, w, maskd, mask, idxi, idxj, themethod):
    """
    Adds two-center (neighbor-atom) J and K contributions.
    """
    # Two-electron two-center weight factors

    if themethod == "PM6":
        nbf = PM6_NBF
        tril_idx = _cached_index(TRIL_IDX_9, P.device)
        weight_tc = _cached_tensor(WEIGHT_45, P.device, P.dtype)
    else:
        nbf = DEFAULT_NBF
        tril_idx = _cached_index(TRIL_IDX_4, P.device)
        weight_tc = _cached_tensor(WEIGHT_10, P.device, P.dtype)

    i0, i1 = tril_idx

    # Pack intra-atomic blocks for neighbors A and B by multiplying the lower triangle blocks by 2
    idxA, idxB = (idxj, idxi) if themethod == "PM6" else (idxi, idxj)
    PA = (P[maskd[idxA]][:, i1, i0] * weight_tc).unsqueeze(-1)  # (...,nP,1)
    PB = (P[maskd[idxB]][:, i1, i0] * weight_tc).unsqueeze(-2)  # (...,1,nP)

    # J contributions: ∑ P * (μν|λσ)
    # w has shape (nPairs, nP, nP)
    J_A = (PA * w).sum(dim=1)  # shape (nPairs, nP)
    J_B = (PB * w).sum(dim=2)

    # scatter J back
    B = w.shape[0]
    sumA = torch.zeros(B, nbf, nbf, device=P.device, dtype=P.dtype)
    sumB = torch.zeros_like(sumA)
    sumA[:, i1, i0] = J_A
    sumB[:, i1, i0] = J_B

    F.index_add_(0, maskd[idxA], sumB)
    F.index_add_(0, maskd[idxB], sumA)

    # K (exchange) contributions:  −½ ∑ P_{νσ} (μν|λσ)
    Pp = -0.5 * P[mask]  # (nPairs, nbf, nbf)

    Ksum = torch.zeros_like(sumA)

    # The commented-out code below is lower-memory version but has two python nested for-loops

    # # Contract Pp[b,ν,σ] with w[b, ind[j,i,νσ] ] over ν,σ
    # if themethod=='PM6':
    #     Pp = Pp.transpose(1,2)
    #     # ind is mapping from (i,j) → integral indices in w
    #     ind = K_ind_9.to(device=P.device)
    #     for i in range(9):
    #         for j in range(9):
    #             # extract the 9×9 (or 4×4) block of w for this (i,j)
    #             # and sum ν,σ
    #             wblk = w[..., ind[j], :][..., :, ind[i]]  # shape (nPairs,nbf,nbf)
    #             Ksum[..., i, j] = (Pp * wblk).sum(dim=(1,2))
    # else:
    #     ind = K_ind_4.to(device=P.device)
    #     for i in range(4):
    #         for j in range(4):
    #             # and sum ν,σ
    #             wblk = w[..., ind[i], :][..., :, ind[j]]  # shape (nPairs,nbf,nbf)
    #             Ksum[..., i, j] = (Pp * wblk).sum(dim=(1,2))

    # This eliminates one of the for-loops but uses more memory
    ind = _cached_index(K_ind_9 if themethod == "PM6" else K_ind_4, P.device)

    p_idx = ind.view(-1)  # (i*ν,)
    w1 = w[:, p_idx, :].view(B, nbf, nbf, -1)  # last dim = nP, which is the packed index that packs nbf*nbf

    if themethod == "PM6":
        # With d-orbitals, the w tensor seems to align with the
        # upper triangle blocks of P, but Pp=P[mask] gets the
        # blocks of the lower triangle of P. So we transpose it here
        Pp = Pp.transpose(1, 2)
        for j in range(9):
            q_idx = ind[j]  # (nbf,)
            # gather w2[b, i, ν, σ] = w1[b, i, ν, q_idx[σ]]
            # i.e. pick out the “λσ” slot for each σ
            w2 = w1[..., q_idx]
            Ksum[:, j, :] = (w2 * Pp.unsqueeze(1)).sum(dim=(2, 3))
    else:
        for j in range(4):
            q_idx = ind[j]  # (nbf,)
            w2 = w1[..., q_idx]
            Ksum[:, :, j] = (w2 * Pp.unsqueeze(1)).sum(dim=(2, 3))

    F.index_add_(0, mask, Ksum)

    return F
