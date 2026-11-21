import math

import numpy as np
import torch

# d-orbital quadratic forms Y = c * r^T M r, in the local order (dz2, dxz, dyz, dx2-y2, dxy) and in the
# global order (dx2-y2, dxz, dz2, dyz, dxy)
_SQ15_16PI = math.sqrt(15.0 / (16.0 * math.pi))
_SQ15_4PI = math.sqrt(15.0 / (4.0 * math.pi))
_SQ5_16PI = math.sqrt(5.0 / (16.0 * math.pi))


def _quadratic_forms(order):
    mats = {
        "z2": (np.diag([-1.0, -1.0, 2.0]), _SQ5_16PI),
        "xz": (np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]]), _SQ15_4PI),
        "yz": (np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]), _SQ15_4PI),
        "x2y2": (np.diag([1.0, -1.0, 0.0]), _SQ15_16PI),
        "xy": (np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]), _SQ15_4PI),
    }
    return np.stack([mats[o][0] for o in order]), np.array([mats[o][1] for o in order])


_M_LOCAL, _C_LOCAL = _quadratic_forms(["z2", "xz", "yz", "x2y2", "xy"])
M_D_GLOBAL, C_D_GLOBAL = _quadratic_forms(["x2y2", "xz", "z2", "yz", "xy"])
NORM_D_GLOBAL = np.einsum("kab,kba->k", M_D_GLOBAL, M_D_GLOBAL)


def GenerateRotationMatrix(P):
    """
    Rotation of the two-center integrals from the local frame to the molecular frame, shape (npairs, 15, 45).

    P (npairs, 3, 3) is the rotation matrix from ``rotate_with_quaternion(-xij)``: its rows are the bond direction
    and two perpendicular axes, i.e. the local sigma, pi1, pi2 p orbitals. The 5x5 d-orbital rotation is derived
    from P, so no Euler angles (and no singularity along z) are needed.
    """
    dtype = P.dtype
    device = P.device
    npairs = P.shape[0]
    INDX = [0, 1, 3, 6, 10, 15, 21, 28, 36]

    # local d orbital K, as a function of the global coordinates, expanded in the global d orbitals L:
    # Y_K(Q r) = sum_L D[K, L] Y_L(r), with Q the rows (pi1, pi2, sigma) of P
    Q = P[:, [1, 2, 0], :]
    m_loc = torch.as_tensor(_M_LOCAL, dtype=dtype, device=device)
    m_glob = torch.as_tensor(M_D_GLOBAL, dtype=dtype, device=device)
    c_loc = torch.as_tensor(_C_LOCAL, dtype=dtype, device=device)
    c_glob = torch.as_tensor(C_D_GLOBAL, dtype=dtype, device=device)
    norm = torch.as_tensor(NORM_D_GLOBAL, dtype=dtype, device=device)
    t = torch.einsum("pba,kbc,pcd->pkad", Q, m_loc, Q)
    D = torch.einsum("pkad,lad->pkl", t, m_glob) / norm * (c_loc[:, None] / c_glob[None, :])

    K = 0
    matrix = torch.zeros(npairs, 15, 45, dtype=dtype, device=device)

    ### S-S ###
    matrix[..., 0, 0] = 1.00000
    ### P-S ###
    K = 0
    while K < 3:
        KL = INDX[K + 1] + 1 - 1
        matrix[..., 0, KL] = P[..., K, 0]
        matrix[..., 1, KL] = P[..., K, 1]
        matrix[..., 2, KL] = P[..., K, 2]
        K = K + 1
    ### P-P ###
    K = 0
    while K < 3:
        KL = INDX[K + 1] + K + 1
        matrix[..., 0, KL] = P[..., K, 0] * P[..., K, 0]
        matrix[..., 1, KL] = P[..., K, 0] * P[..., K, 1]
        matrix[..., 2, KL] = P[..., K, 1] * P[..., K, 1]
        matrix[..., 3, KL] = P[..., K, 0] * P[..., K, 2]
        matrix[..., 4, KL] = P[..., K, 1] * P[..., K, 2]
        matrix[..., 5, KL] = P[..., K, 2] * P[..., K, 2]
        K = K + 1
    K = 1
    while K < 3:
        L = 0
        while L <= K - 1:
            KL = INDX[K + 1] + L + 1
            matrix[..., 0, KL] = P[..., K, 0] * P[..., L, 0] * 2.000
            matrix[..., 1, KL] = P[..., K, 0] * P[..., L, 1] + P[..., K, 1] * P[..., L, 0]
            matrix[..., 2, KL] = P[..., K, 1] * P[..., L, 1] * 2.000
            matrix[..., 3, KL] = P[..., K, 0] * P[..., L, 2] + P[..., K, 2] * P[..., L, 0]
            matrix[..., 4, KL] = P[..., K, 1] * P[..., L, 2] + P[..., K, 2] * P[..., L, 1]
            matrix[..., 5, KL] = P[..., K, 2] * P[..., L, 2] * 2.000
            L = L + 1
        K = K + 1
    K = 0
    ### D-S ###
    while K < 5:
        KL = INDX[K + 3 + 1]
        matrix[..., 0, KL] = D[..., K, 0]
        matrix[..., 1, KL] = D[..., K, 1]
        matrix[..., 2, KL] = D[..., K, 2]
        matrix[..., 3, KL] = D[..., K, 3]
        matrix[..., 4, KL] = D[..., K, 4]
        K = K + 1
    K = 0
    L = 0
    while K < 5:
        L = 0
        while L < 3:
            KL = INDX[K + 4] + L + 1
            matrix[..., 1 - 1, KL] = D[..., K, 1 - 1] * P[..., L, 1 - 1]
            matrix[..., 2 - 1, KL] = D[..., K, 1 - 1] * P[..., L, 2 - 1]
            matrix[..., 3 - 1, KL] = D[..., K, 1 - 1] * P[..., L, 3 - 1]
            matrix[..., 4 - 1, KL] = D[..., K, 2 - 1] * P[..., L, 1 - 1]
            matrix[..., 5 - 1, KL] = D[..., K, 2 - 1] * P[..., L, 2 - 1]
            matrix[..., 6 - 1, KL] = D[..., K, 2 - 1] * P[..., L, 3 - 1]
            matrix[..., 7 - 1, KL] = D[..., K, 3 - 1] * P[..., L, 1 - 1]
            matrix[..., 8 - 1, KL] = D[..., K, 3 - 1] * P[..., L, 2 - 1]
            matrix[..., 9 - 1, KL] = D[..., K, 3 - 1] * P[..., L, 3 - 1]
            matrix[..., 10 - 1, KL] = D[..., K, 4 - 1] * P[..., L, 1 - 1]
            matrix[..., 11 - 1, KL] = D[..., K, 4 - 1] * P[..., L, 2 - 1]
            matrix[..., 12 - 1, KL] = D[..., K, 4 - 1] * P[..., L, 3 - 1]
            matrix[..., 13 - 1, KL] = D[..., K, 5 - 1] * P[..., L, 1 - 1]
            matrix[..., 14 - 1, KL] = D[..., K, 5 - 1] * P[..., L, 2 - 1]
            matrix[..., 15 - 1, KL] = D[..., K, 5 - 1] * P[..., L, 3 - 1]
            L = L + 1
        K = K + 1
    K = 0
    ### DD  ###
    while K < 5:
        KL = INDX[K + 4] + K + 4
        matrix[..., 1 - 1, KL] = D[..., K, 1 - 1] * D[..., K, 1 - 1]
        matrix[..., 2 - 1, KL] = D[..., K, 1 - 1] * D[..., K, 2 - 1]
        matrix[..., 3 - 1, KL] = D[..., K, 2 - 1] * D[..., K, 2 - 1]
        matrix[..., 4 - 1, KL] = D[..., K, 1 - 1] * D[..., K, 3 - 1]
        matrix[..., 5 - 1, KL] = D[..., K, 2 - 1] * D[..., K, 3 - 1]
        matrix[..., 6 - 1, KL] = D[..., K, 3 - 1] * D[..., K, 3 - 1]
        matrix[..., 7 - 1, KL] = D[..., K, 1 - 1] * D[..., K, 4 - 1]
        matrix[..., 8 - 1, KL] = D[..., K, 2 - 1] * D[..., K, 4 - 1]
        matrix[..., 9 - 1, KL] = D[..., K, 3 - 1] * D[..., K, 4 - 1]
        matrix[..., 10 - 1, KL] = D[..., K, 4 - 1] * D[..., K, 4 - 1]
        matrix[..., 11 - 1, KL] = D[..., K, 1 - 1] * D[..., K, 5 - 1]
        matrix[..., 12 - 1, KL] = D[..., K, 2 - 1] * D[..., K, 5 - 1]
        matrix[..., 13 - 1, KL] = D[..., K, 3 - 1] * D[..., K, 5 - 1]
        matrix[..., 14 - 1, KL] = D[..., K, 4 - 1] * D[..., K, 5 - 1]
        matrix[..., 15 - 1, KL] = D[..., K, 5 - 1] * D[..., K, 5 - 1]
        K = K + 1
    K = 0
    L = 0
    while K < 5:
        L = 0
        while L < K:
            KL = INDX[K + 4] + L + 4
            matrix[..., 1 - 1, KL] = D[..., K, 1 - 1] * D[..., L, 1 - 1] * 2.000
            matrix[..., 2 - 1, KL] = D[..., K, 1 - 1] * D[..., L, 2 - 1] + D[..., K, 2 - 1] * D[..., L, 1 - 1]
            matrix[..., 3 - 1, KL] = D[..., K, 2 - 1] * D[..., L, 2 - 1] * 2.000
            matrix[..., 4 - 1, KL] = D[..., K, 1 - 1] * D[..., L, 3 - 1] + D[..., K, 3 - 1] * D[..., L, 1 - 1]
            matrix[..., 5 - 1, KL] = D[..., K, 2 - 1] * D[..., L, 3 - 1] + D[..., K, 3 - 1] * D[..., L, 2 - 1]
            matrix[..., 6 - 1, KL] = D[..., K, 3 - 1] * D[..., L, 3 - 1] * 2.000
            matrix[..., 7 - 1, KL] = D[..., K, 1 - 1] * D[..., L, 4 - 1] + D[..., K, 4 - 1] * D[..., L, 1 - 1]
            matrix[..., 8 - 1, KL] = D[..., K, 2 - 1] * D[..., L, 4 - 1] + D[..., K, 4 - 1] * D[..., L, 2 - 1]
            matrix[..., 9 - 1, KL] = D[..., K, 3 - 1] * D[..., L, 4 - 1] + D[..., K, 4 - 1] * D[..., L, 3 - 1]
            matrix[..., 10 - 1, KL] = D[..., K, 4 - 1] * D[..., L, 4 - 1] * 2.000
            matrix[..., 11 - 1, KL] = (
                D[..., K, 1 - 1] * D[..., L, 5 - 1] + D[..., K, 5 - 1] * D[..., L, 1 - 1]
            )
            matrix[..., 12 - 1, KL] = (
                D[..., K, 2 - 1] * D[..., L, 5 - 1] + D[..., K, 5 - 1] * D[..., L, 2 - 1]
            )
            matrix[..., 13 - 1, KL] = (
                D[..., K, 3 - 1] * D[..., L, 5 - 1] + D[..., K, 5 - 1] * D[..., L, 3 - 1]
            )
            matrix[..., 14 - 1, KL] = (
                D[..., K, 4 - 1] * D[..., L, 5 - 1] + D[..., K, 5 - 1] * D[..., L, 4 - 1]
            )
            matrix[..., 15 - 1, KL] = D[..., K, 5 - 1] * D[..., L, 5 - 1] * 2.000
            L = L + 1
        K = K + 1
    # print("BUILDING ROATION MATRIX: ", time.time() -t," (RotationMatrixD.py,GenerateRotationMatrix)")
    return matrix


def Rotate2Center2Electron(WW, rotationMatrix):
    MET = [
        1,
        2,
        3,
        2,
        3,
        3,
        2,
        3,
        3,
        3,
        4,
        5,
        5,
        5,
        6,
        4,
        5,
        5,
        5,
        6,
        6,
        4,
        5,
        5,
        5,
        6,
        6,
        6,
        4,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        4,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
    ]
    META = [1, 2, 5, 11, 16, 31]
    METB = [1, 4, 10, 15, 30, 45]
    METI = np.array(
        [
            [1, 2, 3, 11, 12, 15],
            [0, 4, 5, 16, 13, 20],
            [0, 7, 6, 22, 14, 21],
            [0, 0, 8, 29, 17, 26],
            [0, 0, 9, 37, 18, 27],
            [0, 0, 10, 0, 19, 28],
            [0, 0, 0, 0, 23, 33],
            [0, 0, 0, 0, 24, 34],
            [0, 0, 0, 0, 25, 35],
            [0, 0, 0, 0, 30, 36],
            [0, 0, 0, 0, 31, 41],
            [0, 0, 0, 0, 32, 42],
            [0, 0, 0, 0, 38, 43],
            [0, 0, 0, 0, 39, 44],
            [0, 0, 0, 0, 40, 45],
        ]
    )
    i = 0

    YM = torch.zeros_like(WW)

    YM[..., 0, 0] = 1.000
    KL = 1
    i = 0
    while KL < 45:
        mkl = MET[KL] - 1
        NKL = METB[mkl] - META[mkl] + 1
        I = 0
        while I < NKL:
            YM[..., METI[I, mkl] - 1, KL] = rotationMatrix[..., I, KL]
            #             print (  I, mkl, KL,  METI[I,mkl]-1, rotationMatrix[...,I,KL])
            I = I + 1
            i = i + 1
        KL = KL + 1
    FINAL = WW.clone()
    #      while ( I < WW.shape[0]):
    STEP1 = torch.bmm(YM[:, ..., ...], torch.transpose(WW, 1, 2)[:, ..., ...])
    FINAL[:, :, :] = torch.bmm(STEP1, torch.transpose(YM, 1, 2)[:, ..., ...])
    FINAL[..., :10, :10] = WW[..., :10, :10]
    FINAL = torch.transpose(FINAL, 1, 2)
    return FINAL


def RotateCore(core, matrix, index):
    ### SS ###
    ##    t = time.time()
    rotCore = torch.zeros_like(core)
    if core.shape[0] == 0:
        return rotCore
    rotCore[..., 0] = core[..., 0]
    ### most likely correct ? ###
    pp = [2, 4, 5, 7, 8, 9]
    I = 0
    if index > 1:
        ### PS ###  6 1 3
        ### Probably correct ####
        rotCore[..., 1] = core[..., 7 - 1] * matrix[..., 1 - 1, 2 - 1]
        rotCore[..., 3] = core[..., 7 - 1] * matrix[..., 2 - 1, 2 - 1]
        rotCore[..., 6] = core[..., 7 - 1] * matrix[..., 3 - 1, 2 - 1]
        ### PP ###
        while I < 6:
            rotCore[..., pp[I]] = core[..., 9] * matrix[..., I, 3 - 1] + core[..., 2] * (
                matrix[..., I, 6 - 1] + matrix[..., I, 10 - 1]
            )
            I = I + 1
    if index > 2:
        ### DD and DP ###
        I = 0
        dp = [11, 12, 13, 16, 17, 18, 22, 23, 24, 29, 30, 31, 37, 38, 39]
        dd = [14, 19, 20, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 44]  ## probably correct
        while I < 15:
            rotCore[..., dp[I]] = core[..., 24] * matrix[..., I, 12 - 1] + core[..., 16] * (
                matrix[..., I, 18 - 1] + matrix[..., I, 25 - 1]
            )
            rotCore[..., dd[I]] = (
                core[..., 27] * matrix[..., I, 15 - 1]
                + core[..., 20] * (matrix[..., I, 21 - 1] + matrix[..., I, 28 - 1])
                + core[..., 14] * (matrix[..., I, 36 - 1] + matrix[..., I, 45 - 1])
            )
            I = I + 1

        I = 4
        ds = [10, 15, 21, 28, 36]
        while I < 9:
            rotCore[..., ds[I - 4]] = core[..., 21] * matrix[..., I - 4, 11 - 1]
            I = I + 1
    return rotCore
