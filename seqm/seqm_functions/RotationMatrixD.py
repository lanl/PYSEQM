import numpy as np
import torch


def GenerateRotationMatrix(xij):
    ##     ROTATION MATRIX FOR A GIVEN ATOM PAIR I-J (I.GT.J).
    ##     R         INTERATOMIC DISTANCE, IN ATOMIC UNITS (O).
    ##     matrix()      PRECOMBINED ELEMENTS OF THE ROTATION MATRIX (O).

    ##INDX = torch.as_tensor( [ 0,6,1,3,36,21,28,15,10 ])
    INDX = [0, 1, 3, 6, 10, 15, 21, 28, 36]
    PT5SQ3 = 0.8660254037841
    PT5 = 0.5

    dtype = xij.dtype
    device = xij.device
    xij = -xij
    # xij unit vector points from i to j: xij = (xj-xi)/|xi-xj|, shape (npairs, 3)
    # rij : distance between atom i and j in atomic units, shape (npairs,)
    # ni, nj, atomic number of atom i and j, shape (npairs,)
    # x = x2-x1
    # y = y2-y1
    # z = z2-z1
    # zeta_a, zeta_b: zeta_s and zeta_p for atom pair a-b, shape(npairs, 2)
    # output di: overlap matrix between AOs from atom i an j, shape (npairs, 4,4)
    # 0,1,2,3: sigma, px, py, pz

    xy = torch.norm(xij[..., :2], dim=1)

    # ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.

    tmp = torch.where(
        xij[..., 2] < 0.0,
        torch.tensor(-1.0, dtype=dtype, device=device),
        torch.where(
            xij[..., 2] > 0.0,
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(0.0, dtype=dtype, device=device),
        ),
    )

    # ca = th.where(xy>=1.0e-10, xij[...,0]/xy, tmp)
    cond_xy = xy >= 1.0e-10
    CA = tmp.clone()
    CA[cond_xy] = xij[cond_xy, 0] / xy[cond_xy]

    CB = torch.where(xy >= 1.0e-10, xij[..., 2], tmp)  # xij is a unti vector already
    SA = torch.zeros_like(xy)

    SA[cond_xy] = xij[cond_xy, 1] / xy[cond_xy]
    SB = torch.where(xy >= 1.0e-10, xy, torch.tensor(0.0, dtype=dtype, device=device))

    C2A = 2.000 * CA * CA - 1.000
    C2B = 2.000 * CB * CB - 1.000
    S2A = 2.000 * SA * CA
    S2B = 2.000 * SB * CB

    P = torch.zeros(xij.shape[0], 3, 3)
    P[..., 1 - 1, 1 - 1] = CA * SB
    P[..., 2 - 1, 1 - 1] = CA * CB
    P[..., 3 - 1, 1 - 1] = -SA
    P[..., 1 - 1, 2 - 1] = SA * SB
    P[..., 2 - 1, 2 - 1] = SA * CB
    P[..., 3 - 1, 2 - 1] = CA
    P[..., 1 - 1, 3 - 1] = CB
    P[..., 2 - 1, 3 - 1] = -SB

    D = torch.zeros(xij.shape[0], 5, 5)
    D[..., 1 - 1, 1 - 1] = PT5SQ3 * C2A * SB * SB
    D[..., 2 - 1, 1 - 1] = PT5 * C2A * S2B
    D[..., 3 - 1, 1 - 1] = -S2A * SB
    D[..., 4 - 1, 1 - 1] = C2A * (CB * CB + PT5 * SB * SB)
    D[..., 5 - 1, 1 - 1] = -S2A * CB
    D[..., 1 - 1, 2 - 1] = PT5SQ3 * CA * S2B
    D[..., 2 - 1, 2 - 1] = CA * C2B
    D[..., 3 - 1, 2 - 1] = -SA * CB
    D[..., 4 - 1, 2 - 1] = -PT5 * CA * S2B
    D[..., 5 - 1, 2 - 1] = SA * SB
    D[..., 1 - 1, 3 - 1] = CB * CB - PT5 * SB * SB
    D[..., 2 - 1, 3 - 1] = -PT5SQ3 * S2B
    D[..., 4 - 1, 3 - 1] = PT5SQ3 * SB * SB
    D[..., 1 - 1, 4 - 1] = PT5SQ3 * SA * S2B
    D[..., 2 - 1, 4 - 1] = SA * C2B
    D[..., 3 - 1, 4 - 1] = CA * CB
    D[..., 4 - 1, 4 - 1] = -PT5 * SA * S2B
    D[..., 5 - 1, 4 - 1] = -CA * SB
    D[..., 1 - 1, 5 - 1] = PT5SQ3 * S2A * SB * SB
    D[..., 2 - 1, 5 - 1] = PT5 * S2A * S2B
    D[..., 3 - 1, 5 - 1] = C2A * SB
    D[..., 4 - 1, 5 - 1] = S2A * (CB * CB + PT5 * SB * SB)
    D[..., 5 - 1, 5 - 1] = C2A * CB
    K = 0
    matrix = torch.zeros(xij.shape[0], 15, 45, device=device)

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
