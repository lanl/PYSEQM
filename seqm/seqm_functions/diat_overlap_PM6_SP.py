import torch as th

from .two_elec_two_center_int import rotate_with_quaternion


def diatom_overlap_matrix_PM6_SP(ni, nj, xij, rij, zeta_a, zeta_b, qn_int):
    """
    compute the overlap matrix for each pair
    """
    # t0 =time.time()
    dtype = xij.dtype
    device = xij.device
    # xij unit vector points from i to j: xij = (xj-xi)/|xi-xj|, shape (npairs, 3)
    # rij : distance between atom i and j in atomic units, shape (npairs,)
    # ni, nj, atomic number of atom i and j, shape (npairs,)
    # x = x2-x1
    # y = y2-y1
    # z = z2-z1
    # for first atom index == second atom index, (same atom)
    # x = x1, y = y1, z = z1
    npairs = xij.shape[0]
    # zeta_a, zeta_b: zeta_s and zeta_p for atom pair a-b, shape(npairs, 2)
    # output di: overlap matrix between AOs from atom i an j, shape (npairs, 4,4)
    # 0,1,2,3: sigma, px, py, pz

    # I'm doing rotations with quaternions so don't need the below
    # xy = th.norm(xij[...,:2],dim=1)
    #
    #
    # tmp = th.where(xij[...,2]<0.0,th.tensor(-1.0,dtype=dtype, device=device), \
    #       th.where(xij[...,2]>0.0,th.tensor(1.0,dtype=dtype, device=device), \
    #       th.tensor(0.0,dtype=dtype, device=device)))
    #
    # cond_xy = xy>=1.0e-10
    # ca = tmp.clone()
    # ca[cond_xy] = xij[cond_xy,0]/xy[cond_xy]
    #
    # cb = th.where(xy>=1.0e-10, xij[...,2], tmp)  #xij is a unti vector already
    # sa = th.zeros_like(xy)
    #
    # sa[cond_xy] = xij[cond_xy,1]/xy[cond_xy]
    # #sb = th.where(xy>=1.0e-10, xy/rij, th.tensor(0.0,dtype=dtype))
    # sb = th.where(xy>=1.0e-10, xy, th.tensor(0.0,dtype=dtype, device=device))
    ################################
    # ok to use th.where here as postion doesn't require grad
    # if update to do MD, ca, cb, sa, sb should be chaneged to the indexing version
    ################################

    # overlap matrix in the local frame
    # first row  - first row  : ii = 1, jcall = 2
    # first row  - second row : ii = 2, jcall = 3
    # second row - secpmd row : ii = 4, jcall = 4

    # only first and second row are included here
    # one-element slice to keep dim
    # jcall shape (npairs, )

    jcall = th.zeros_like(ni)
    qni = qn_int[ni]
    qnj = qn_int[nj]
    jcall[(qni == 1) & (qnj == 1)] = 2
    jcall[(qni == 2) & (qnj == 1)] = 3
    jcall[(qni == 2) & (qnj == 2)] = 4

    jcall[(qni == 3) & (qnj == 1)] = 431
    jcall[(qni == 3) & (qnj == 2)] = 5
    jcall[(qni == 3) & (qnj == 3)] = 6

    if th.any(jcall == 0):
        raise ValueError("\nError from diat.py, overlap matrix\nSome elements are not supported yet")

    # na>=nb
    # setc.isp=2
    # setc.ips=1
    # setc.sa = s2 = zeta_B
    # setc.sb = s1 = zeta_A

    # change to atomic units, a0 here is taken from mopac, different from the standard value
    # r is already in unit of bohr radius, AU

    # parameter_set = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
    #                 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha']

    A111, B111 = SET(rij, jcall, zeta_a[..., 0], zeta_b[..., 0])
    S111 = th.zeros((npairs,), dtype=dtype, device=device)

    S211 = th.zeros_like(S111)
    A211, B211 = SET(rij, jcall, zeta_a[..., 1], zeta_b[..., 0])

    S121 = th.zeros_like(S111)
    A121, B121 = SET(rij, jcall, zeta_a[..., 0], zeta_b[..., 1])

    S221 = th.zeros_like(S111)
    A22, B22 = SET(rij, jcall, zeta_a[..., 1], zeta_b[..., 1])

    S222 = th.zeros_like(S221)

    # print(id(S111))
    # s-s
    jcall2 = jcall == 2  # ii=1
    if jcall2.sum() != 0:
        S111[jcall2] = (
            th.pow(zeta_a[jcall2, 0] * zeta_b[jcall2, 0] * rij[jcall2] ** 2, 1.5)
            * (A111[jcall2, 2] * B111[jcall2, 0] - B111[jcall2, 2] * A111[jcall2, 0])
            / 4.0
        )
    jcall3 = jcall == 3  # ii=2
    if jcall3.sum() != 0:
        S111[jcall3] = (
            th.pow(zeta_b[jcall3, 0], 1.5)
            * th.pow(zeta_a[jcall3, 0], 2.5)
            * rij[jcall3] ** 4
            * (
                A111[jcall3, 3] * B111[jcall3, 0]
                - B111[jcall3, 3] * A111[jcall3, 0]
                + A111[jcall3, 2] * B111[jcall3, 1]
                - B111[jcall3, 2] * A111[jcall3, 1]
            )
            / (th.sqrt(th.tensor(3.0)) * 8.0)
        )
        S211[jcall3] = (
            th.pow(zeta_b[jcall3, 0], 1.5)
            * th.pow(zeta_a[jcall3, 1], 2.5)
            * rij[jcall3] ** 4
            * (
                A211[jcall3, 2] * B211[jcall3, 0]
                - B211[jcall3, 2] * A211[jcall3, 0]
                + A211[jcall3, 3] * B211[jcall3, 1]
                - B211[jcall3, 3] * A211[jcall3, 1]
            )
            / 8.0
        )

    jcall4 = jcall == 4  # ii=4
    if jcall4.sum() != 0:
        S111[jcall4] = (
            th.pow(zeta_b[jcall4, 0] * zeta_a[jcall4, 0], 2.5)
            * rij[jcall4] ** 5
            * (
                A111[jcall4, 4] * B111[jcall4, 0]
                + B111[jcall4, 4] * A111[jcall4, 0]
                - 2.0 * A111[jcall4, 2] * B111[jcall4, 2]
            )
            / 48.0
        )
        S211[jcall4] = (
            th.pow(zeta_b[jcall4, 0] * zeta_a[jcall4, 1], 2.5)
            * rij[jcall4] ** 5
            * (
                A211[jcall4, 3] * (B211[jcall4, 0] - B211[jcall4, 2])
                - A211[jcall4, 1] * (B211[jcall4, 2] - B211[jcall4, 4])
                + B211[jcall4, 3] * (A211[jcall4, 0] - A211[jcall4, 2])
                - B211[jcall4, 1] * (A211[jcall4, 2] - A211[jcall4, 4])
            )
            / (16.0 * th.sqrt(th.tensor(3.0)))
        )
        S121[jcall4] = (
            th.pow(zeta_b[jcall4, 1] * zeta_a[jcall4, 0], 2.5)
            * rij[jcall4] ** 5
            * (
                A121[jcall4, 3] * (B121[jcall4, 0] - B121[jcall4, 2])
                - A121[jcall4, 1] * (B121[jcall4, 2] - B121[jcall4, 4])
                - B121[jcall4, 3] * (A121[jcall4, 0] - A121[jcall4, 2])
                + B121[jcall4, 1] * (A121[jcall4, 2] - A121[jcall4, 4])
            )
            / (16.0 * th.sqrt(th.tensor(3.0)))
        )
        w = th.pow(zeta_b[jcall4, 1] * zeta_a[jcall4, 1], 2.5) * rij[jcall4] ** 5 / 16.0
        S221[jcall4] = -w * (
            B22[jcall4, 2] * (A22[jcall4, 4] + A22[jcall4, 0])
            - A22[jcall4, 2] * (B22[jcall4, 4] + B22[jcall4, 0])
        )
        S222[jcall4] = (
            0.5
            * w
            * (
                A22[jcall4, 4] * (B22[jcall4, 0] - B22[jcall4, 2])
                - B22[jcall4, 4] * (A22[jcall4, 0] - A22[jcall4, 2])
                - A22[jcall4, 2] * B22[jcall4, 0]
                + B22[jcall4, 2] * A22[jcall4, 0]
            )
        )

    jcall431 = jcall == 431  # ii=4
    if jcall431.sum() != 0:
        S111[jcall431] = (
            th.pow(zeta_b[jcall431, 0], 1.5)
            * th.pow(zeta_a[jcall431, 0], 3.5)
            * rij[jcall431] ** 5
            * (
                A111[jcall431, 4] * B111[jcall431, 0]
                + 2 * B111[jcall431, 1] * A111[jcall431, 3]
                - 2.0 * A111[jcall431, 1] * B111[jcall431, 3]
                - B111[jcall431, 4] * A111[jcall431, 0]
            )
            / (th.sqrt(th.tensor(10.0)) * 24.0)
        )
        S211[jcall431] = (
            th.pow(zeta_b[jcall431, 0], 1.5)
            * th.pow(zeta_a[jcall431, 1], 3.5)
            * rij[jcall431] ** 5
            * (
                A211[jcall431, 3] * (B211[jcall431, 0] + B211[jcall431, 2])
                - A211[jcall431, 1] * (B211[jcall431, 4] + B211[jcall431, 2])
                + B211[jcall431, 1] * (A211[jcall431, 2] + A211[jcall431, 4])
                - B211[jcall431, 3] * (A211[jcall431, 2] + A211[jcall431, 0])
            )
            / (8.0 * th.sqrt(th.tensor(30.0)))
        )

    jcall5 = jcall == 5  # ii=4
    if jcall5.sum() != 0:
        S111[jcall5] = (
            th.pow(zeta_b[jcall5, 0], 2.5)
            * th.pow(zeta_a[jcall5, 0], 3.5)
            * rij[jcall5] ** 6
            * (
                A111[jcall5, 5] * B111[jcall5, 0]
                + B111[jcall5, 1] * A111[jcall5, 4]
                - 2 * B111[jcall5, 2] * A111[jcall5, 3]
                - 2.0 * A111[jcall5, 2] * B111[jcall5, 3]
                + B111[jcall5, 4] * A111[jcall5, 1]
                + B111[jcall5, 5] * A111[jcall5, 0]
            )
            / (th.sqrt(th.tensor(30.0)) * 48.0)
        )
        S211[jcall5] = (
            th.pow(zeta_b[jcall5, 0], 2.5)
            * th.pow(zeta_a[jcall5, 1], 3.5)
            * rij[jcall5] ** 6
            * (
                A211[jcall5, 4] * B211[jcall5, 0]
                + B211[jcall5, 1] * A211[jcall5, 5]
                - 2.0 * B211[jcall5, 3] * A211[jcall5, 3]
                - 2.0 * A211[jcall5, 2] * B211[jcall5, 2]
                + A211[jcall5, 1] * B211[jcall5, 5]
                + A211[jcall5, 0] * B211[jcall5, 4]
            )
            / (48.0 * th.sqrt(th.tensor(10.0)))
        )
        S121[jcall5] = (
            th.pow(zeta_b[jcall5, 1], 2.5)
            * th.pow(zeta_a[jcall5, 0], 3.5)
            * rij[jcall5] ** 6
            * (
                (A121[jcall5, 4] * B121[jcall5, 0] - A121[jcall5, 5] * B121[jcall5, 1])
                + 2.0 * (A121[jcall5, 3] * B121[jcall5, 1] - A121[jcall5, 4] * B121[jcall5, 2])
                - 2.0 * (A121[jcall5, 1] * B121[jcall5, 3] - A121[jcall5, 2] * B121[jcall5, 4])
                - (A121[jcall5, 0] * B121[jcall5, 4] - A121[jcall5, 1] * B121[jcall5, 5])
            )
            / (48.0 * th.sqrt(th.tensor(10.0)))
        )
        S221[jcall5] = (
            th.pow(zeta_b[jcall5, 1], 2.5)
            * th.pow(zeta_a[jcall5, 1], 3.5)
            * rij[jcall5] ** 6
            * (
                (A22[jcall5, 3] * B22[jcall5, 0] - A22[jcall5, 5] * B22[jcall5, 2])
                + (A22[jcall5, 2] * B22[jcall5, 1] - A22[jcall5, 4] * B22[jcall5, 3])
                - (A22[jcall5, 1] * B22[jcall5, 2] - A22[jcall5, 3] * B22[jcall5, 4])
                - (A22[jcall5, 0] * B22[jcall5, 3] - A22[jcall5, 2] * B22[jcall5, 5])
            )
            / (16.0 * th.sqrt(th.tensor(30.0)))
        )
        S222[jcall5] = (
            th.pow(zeta_b[jcall5, 1], 2.5)
            * th.pow(zeta_a[jcall5, 1], 3.5)
            * rij[jcall5] ** 6
            * (
                (A22[jcall5, 5] - A22[jcall5, 3]) * (B22[jcall5, 0] - B22[jcall5, 2])
                + (A22[jcall5, 4] - A22[jcall5, 2]) * (B22[jcall5, 1] - B22[jcall5, 3])
                - (A22[jcall5, 3] - A22[jcall5, 1]) * (B22[jcall5, 2] - B22[jcall5, 4])
                - (A22[jcall5, 2] - A22[jcall5, 0]) * (B22[jcall5, 3] - B22[jcall5, 5])
            )
            / (32.0 * th.sqrt(th.tensor(30.0)))
        )

    jcall6 = jcall == 6
    if jcall6.sum() != 0:
        S111[jcall6] = (
            th.pow(zeta_b[jcall6, 0] * zeta_a[jcall6, 0], 3.5)
            * rij[jcall6] ** 7
            * (
                A111[jcall6, 6] * B111[jcall6, 0]
                - 3.0 * B111[jcall6, 2] * A111[jcall6, 4]
                + 3.0 * A111[jcall6, 2] * B111[jcall6, 4]
                - A111[jcall6, 0] * B111[jcall6, 6]
            )
            / 1440.0
        )
        S211[jcall6] = (
            th.pow(zeta_b[jcall6, 0] * zeta_a[jcall6, 1], 3.5)
            * rij[jcall6] ** 7
            * (
                (A211[jcall6, 5] * B211[jcall6, 0] + A211[jcall6, 6] * B211[jcall6, 1])
                + (-A211[jcall6, 4] * B211[jcall6, 1] - A211[jcall6, 5] * B211[jcall6, 2])
                - 2.0 * (A211[jcall6, 3] * B211[jcall6, 2] + A211[jcall6, 4] * B211[jcall6, 3])
                - 2.0 * (-A211[jcall6, 2] * B211[jcall6, 3] - A211[jcall6, 3] * B211[jcall6, 4])
                + (A211[jcall6, 1] * B211[jcall6, 4] + A211[jcall6, 2] * B211[jcall6, 5])
                + (-A211[jcall6, 0] * B211[jcall6, 5] - A211[jcall6, 1] * B211[jcall6, 6])
            )
            / (480.0 * th.sqrt(th.tensor(3.0)))
        )
        S121[jcall6] = (
            th.pow(zeta_b[jcall6, 1] * zeta_a[jcall6, 0], 3.5)
            * rij[jcall6] ** 7
            * (
                (A121[jcall6, 5] * B121[jcall6, 0] - A121[jcall6, 6] * B121[jcall6, 1])
                + (A121[jcall6, 4] * B121[jcall6, 1] - A121[jcall6, 5] * B121[jcall6, 2])
                - 2.0 * (A121[jcall6, 3] * B121[jcall6, 2] - A121[jcall6, 4] * B121[jcall6, 3])
                - 2.0 * (A121[jcall6, 2] * B121[jcall6, 3] - A121[jcall6, 3] * B121[jcall6, 4])
                + (A121[jcall6, 1] * B121[jcall6, 4] - A121[jcall6, 2] * B121[jcall6, 5])
                + (A121[jcall6, 0] * B121[jcall6, 5] - A121[jcall6, 1] * B121[jcall6, 6])
            )
            / (480.0 * th.sqrt(th.tensor(3.0)))
        )
        S221[jcall6] = (
            th.pow(zeta_b[jcall6, 1], 3.5)
            * th.pow(zeta_a[jcall6, 1], 3.5)
            * rij[jcall6] ** 7
            * (
                (A22[jcall6, 4] * B22[jcall6, 0] - A22[jcall6, 6] * B22[jcall6, 2])
                - 2.0 * (A22[jcall6, 2] * B22[jcall6, 2] - A22[jcall6, 4] * B22[jcall6, 4])
                + (A22[jcall6, 0] * B22[jcall6, 4] - A22[jcall6, 2] * B22[jcall6, 6])
            )
            / (480.0)
        )
        S222[jcall6] = (
            th.pow(zeta_b[jcall6, 1], 3.5)
            * th.pow(zeta_a[jcall6, 1], 3.5)
            * rij[jcall6] ** 7
            * (
                (A22[jcall6, 6] - A22[jcall6, 4]) * (B22[jcall6, 0] - B22[jcall6, 2])
                - 2.0 * (A22[jcall6, 4] - A22[jcall6, 2]) * (B22[jcall6, 2] - B22[jcall6, 4])
                + (A22[jcall6, 2] - A22[jcall6, 0]) * (B22[jcall6, 4] - B22[jcall6, 6])
            )
            / (960.0)
        )

    ###up to here

    #
    # form di
    # check di_index.txt
    di = th.zeros((npairs, 4, 4), dtype=dtype, device=device)

    # diat coe
    # c : computed in coe, with shape (75)
    # c0 : used in diat, shape (3,5,5)
    # index relation
    # c with index 56, 41, 26, 53, 38, 50, 37, 20
    #  c, c0
    # 37, 1 3 3  1.0
    # 20, 2 2 2  ca
    # 35, 2 2 3  sa*sb
    # 50, 2 2 4  sa*cb
    # 23, 2 3 2  0.0
    # 38, 2 3 3  cb
    # 53, 2 3 4  -sb
    # 26, 2 4 2  -sa
    # 41, 2 4 3  ca*sb
    # 56, 2 4 4  ca*cb

    ## c[37] = 1.0
    ## c[23] = 0.0
    # all not listed are 0.0

    # check di_index.txt for details
    del A111, B111, A211, B211, A121, B121, A22, B22

    # sasb = sa*sb
    # sacb = sa*cb
    # casb = ca*sb
    # cacb = ca*cb

    rot = rotate_with_quaternion(xij)
    rot = rot.transpose(1, 2)

    di[..., 0, 0] = S111
    c0 = rot[:, :, 0]
    di[:, 1:, 0] = S211.unsqueeze(1) * c0
    di[:, 0, 1:] = -S121.unsqueeze(1) * c0
    M = th.stack([-S221, S222, S222], dim=1)  # → (N,3)
    B = rot * M.unsqueeze(1)  # → (N,3,3)
    di[:, 1:, 1:] = th.bmm(B, rot.transpose(1, 2))

    # #jcallg2 = (jcall>2)
    # #di[jcallg2,1,0] = S211[jcallg2,0]*ca[jcallg2,0]*sb[jcallg2,0]
    # #the fraction of H-H is low, may not necessary to use indexing
    # di[...,1,0] = S211*ca*sb
    # di[...,2,0] = S211*sa*sb
    # di[...,3,0] = S211*cb
    # #jcall==4, pq1=pq2=2, ii=4, second - second row
    # #di[jcall4,0,1] = -S121[jcall4]*ca[jcall4]*sb[jcall4]
    # di[...,0,1] = -S121*casb
    # #di[jcall4,0,2] = -S121[jcall4]*sa[jcall4]*sb[jcall4]
    # di[...,0,2] = -S121*sasb
    # di[...,0,3] = -S121*cb
    # #di[jcall4,1,1] = -S221[jcall4]*ca[jcall4]**2*sb[jcall4]**2
    # #                 +S222[jcall4]*(ca[jcall4]**2*cb[jcall4]**2+sa[jcall4]**2)
    # di[...,1,1] = -S221*casb**2 \
    #                  +S222*(cacb**2+sa**2)
    # di[...,1,2] = -S221*casb*sasb \
    #                  +S222*(cacb*sacb-sa*ca)
    # di[...,1,3] = -S221*casb*cb \
    #                  -S222*cacb*sb
    # di[...,2,1] = -S221*sasb*casb \
    #                  +S222*(sacb*cacb-ca*sa)
    # di[...,2,2] = -S221*sasb**2 \
    #                  +S222*(sacb**2+ca**2)
    # di[...,2,3] = -S221*sasb*cb \
    #                  -S222*sacb*sb
    # di[...,3,1] = -S221*cb*casb \
    #                  -S222*sb*cacb
    # di[...,3,2] = -S221*cb*sasb \
    #                  -S222*sb*sacb
    # di[...,3,3] = -S221*cb**2 \
    #                  +S222*sb**2
    del S111, S121, S211  # , sasb, sacb, casb, cacb, ca, sa

    # on pairs with same atom, diagonal part
    # di[jcall==0,:,:] = th.diag(th.ones(4,dtype=dtype)).reshape((-1,4,4))

    """
    return S111, S211, S121, S221, S222


    """
    """
    #  c, c0
    # 37, 1 3 3  1.0
    # 20, 2 2 2  ca
    # 35, 2 2 3  sa*sb
    # 50, 2 2 4  sa*cb
    # 23, 2 3 2  0.0
    # 38, 2 3 3  cb
    # 53, 2 3 4  -sb
    # 26, 2 4 2  -sa
    # 41, 2 4 3  ca*sb
    # 56, 2 4 4  ca*cb
    """
    # print("CALCULATION OF DIATOMIC OVERLAP: ", time.time()-t0, " (diat_overlapD.py, diatom_overlap_matrixD)")

    return di


# parameter_set = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
#                 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha']


def SET(rij, jcall, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    # alpha, beta below is used in aintgs and bintgs, not the parameters for AM1/MNDO/PM3
    # rij: distance between atom i and j in atomic unit
    alpha = 0.5 * rij * (z1[..., :] + z2[..., :])
    beta = 0.5 * rij * (z1[..., :] - z2[..., :])
    A = aintgs(alpha, jcall)
    B = bintgs(beta, jcall)
    return A, B


def aintgs(x0, jcall):
    """
    A integrals for diatom_overlap_matrix
    """
    dtype = x0.dtype
    device = x0.device
    # indxa, indxb, index to show which one to choose
    # idxa = parameter_set.index('zeta_X') X=s,p for atom a
    # idxb = parameter_set.index('zeta_X') for b
    # jcall : k in aintgs
    # alpha = 0.5*rab*(zeta_a + zeta_b)
    # c = exp(-alpha)
    # jcall >=2, and = 2,3,4 for first and second row elements

    # in same case x will be zero, which causes an issue when backpropagating
    # like zete_p from Hydrogen, H -  H Pair
    # or pairs for same atom, then rab = 0
    t = 1.0 / th.tensor(0.0, dtype=dtype, device=device)
    x = th.where(x0 != 0, x0, t).reshape(-1, 1)
    a1 = th.exp(-x) / x

    a2 = a1 + a1 / x
    # jcall >= 2
    a3 = a1 + 2.0 * a2 / x
    # jcall >= 3
    jcallp3 = (jcall >= 3).reshape((-1, 1))
    a4 = th.where(jcallp3, a1 + 3.0 * a3 / x, th.tensor(0.0, dtype=dtype, device=device))
    # jcall >=4
    jcallp4 = (jcall >= 4).reshape((-1, 1))
    a5 = th.where(jcallp4, a1 + 4.0 * a4 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp5 = (jcall >= 5).reshape((-1, 1))
    a6 = th.where(jcallp5, a1 + 5.0 * a5 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp6 = (jcall >= 6).reshape((-1, 1))
    a7 = th.where(jcallp6, a1 + 6.0 * a6 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp7 = (jcall >= 7).reshape((-1, 1))
    a8 = th.where(jcallp7, a1 + 7.0 * a7 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp8 = (jcall >= 8).reshape((-1, 1))
    a9 = th.where(jcallp8, a1 + 8.0 * a8 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp9 = (jcall >= 9).reshape((-1, 1))
    a10 = th.where(jcallp9, a1 + 9.0 * a9 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp10 = (jcall >= 10).reshape((-1, 1))
    a11 = th.where(jcallp10, a1 + 10.0 * a10 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp11 = (jcall >= 11).reshape((-1, 1))
    a12 = th.where(jcallp11, a1 + 11.0 * a11 / x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp12 = (jcall >= 12).reshape((-1, 1))
    a13 = th.where(jcallp12, a1 + 12.0 * a12 / x, th.tensor(0.0, dtype=dtype, device=device))

    return th.cat((a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13), dim=1)


def bintgs(x0, jcall):
    """
    B integrals for diatom_overlap_matrix
    """
    # jcall not used here, but may used later for more element types
    # may ignore jcall, and just compute all the b1, b2, ..., b5 will be used
    # as na>=nb, and in the set in diat2.f
    # setc.sa = s2 =  zeta_b, setc.sb = s1 = zeta_a

    # beta = 0.5*rab*(setc.sb-setc.sa)
    # beta = 0.5*rab*(zeta_a - zeta_b)

    ## |x|<=0.5, last = 6, goto 60
    # else goto 40
    # for convenience, may use goto90 for x<1.0e-6
    x = x0.reshape(-1, 1)
    absx = th.abs(x)

    cond1 = absx > 0.5

    # tx = th.exp(x)/x    # exp(x)/x  #not working with backward
    # tx = th.where(cond1, th.exp(x)/x, th.tensor(0.0,dtype=dtype)) # not working with backward
    x_cond1 = x[cond1]
    tx_cond1 = th.exp(x_cond1) / x_cond1

    # tmx = -th.exp(-x)/x # -exp(-x)/x #not working with backward
    # tmx = th.where(cond1, -th.exp(-x)/x,  th.tensor(0.0,dtype=dtype)) #not working with backward
    tmx_cond1 = -th.exp(-x_cond1) / x_cond1
    # b1(x=0)=2, b2(x=0)=0,b3(x=0)=2/3,b4(x=0)=0,b5(x=0)=2/5

    # do some test to choose which one is faster
    # b1 = th.where(cond1,  tx+tmx         , th.tensor(2.0))
    # b2 = th.where(cond1, -tx+tmx+b1/x    , th.tensor(0.0))
    # b3 = th.where(cond1,  tx+tmx+2.0*b2/x, th.tensor(2.0/3.0))
    # b4 = th.where(cond1, -tx+tmx+3.0*b3/x, th.tensor(0.0))
    # b5 = th.where(cond1,  tx+tmx+4.0*b4/x, th.tensor(2.0/5.0))
    # b4 = th.where(cond1 & (jcall>=3), -tx+tmx+3.0*b3/x, th.tensor(0.0))
    # b5 = th.where(cond1 & (jcall>=4),  tx+tmx+4.0*b4/x, th.tensor(2.0/5.0))

    # do some test to choose which one is faster

    # can't use this way th.where to do backpropagating
    b1 = th.ones_like(x) * 2.0
    b2 = th.zeros_like(x)
    b3 = th.ones_like(x) * (2.0 / 3.0)
    b4 = th.zeros_like(x)
    b5 = th.ones_like(x) * (2.0 / 5.0)
    b6 = th.zeros_like(x)
    b7 = th.ones_like(x) * (2.0 / 7.0)
    b8 = th.zeros_like(x)
    b9 = th.ones_like(x) * (2.0 / 9.0)
    b10 = th.zeros_like(x)
    b11 = th.ones_like(x) * (2.0 / 11.0)
    b12 = th.zeros_like(x)
    b13 = th.ones_like(x) * (2.0 / 13.0)

    b1_cond1 = tx_cond1 + tmx_cond1
    b1[cond1] = b1_cond1

    b2_cond1 = -tx_cond1 + tmx_cond1 + b1_cond1 / x_cond1
    b2[cond1] = b2_cond1

    b3_cond1 = tx_cond1 + tmx_cond1 + 2.0 * b2_cond1 / x_cond1
    b3[cond1] = b3_cond1

    b4_cond1 = -tx_cond1 + tmx_cond1 + 3.0 * b3_cond1 / x_cond1
    b4[cond1] = b4_cond1

    b5_cond1 = tx_cond1 + tmx_cond1 + 4.0 * b4_cond1 / x_cond1
    b5[cond1] = b5_cond1

    b6_cond1 = -tx_cond1 + tmx_cond1 + 5.0 * b5_cond1 / x_cond1
    b6[cond1] = b6_cond1

    b7_cond1 = tx_cond1 + tmx_cond1 + 6.0 * b6_cond1 / x_cond1
    b7[cond1] = b7_cond1

    b8_cond1 = -tx_cond1 + tmx_cond1 + 7.0 * b7_cond1 / x_cond1
    b8[cond1] = b8_cond1

    b9_cond1 = tx_cond1 + tmx_cond1 + 8.0 * b8_cond1 / x_cond1
    b9[cond1] = b9_cond1

    b10_cond1 = -tx_cond1 + tmx_cond1 + 9.0 * b9_cond1 / x_cond1
    b10[cond1] = b10_cond1

    b11_cond1 = tx_cond1 + tmx_cond1 + 10.0 * b10_cond1 / x_cond1
    b11[cond1] = b11_cond1

    b12_cond1 = -tx_cond1 + tmx_cond1 + 11.0 * b11_cond1 / x_cond1
    b12[cond1] = b12_cond1

    b13_cond1 = tx_cond1 + tmx_cond1 + 12.0 * b12_cond1 / x_cond1
    b13[cond1] = b13_cond1

    # b1 = th.where(cond1,  tx + tmx           , th.tensor(2.0, dtype=dtype))
    # b2 = th.where(cond1, -tx + tmx +     b1/x, th.tensor(0.0, dtype=dtype))
    # b3 = th.where(cond1,  tx + tmx + 2.0*b2/x, th.tensor(2.0/3.0, dtype=dtype))
    # b4 = th.where(cond1, -tx + tmx + 3.0*b3/x, th.tensor(0.0, dtype=dtype))
    # b5 = th.where(cond1,  tx + tmx + 4.0*b4/x, th.tensor(2.0/5.0, dtype=dtype))

    # |x|<=0.5
    cond2 = (absx <= 0.5) & (absx > 1.0e-6)
    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is even, i=0,2,4, m = 0,2,4,6
    # b_{i+1} (x) = \sum_{m=0,2,4,6} x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =   0    2      4       6
    # i=0        2  1/3   1/60  1/2520     2/(m!*(m+1))
    # i=2      2/3  1/5   1/84  1/3240     2/(m!*(m+3))
    # i=4      2/5  1/7   1/108  1/3960     2/(m!*(m+5))
    # i=6      2/7  1/9   1/132  1/4680     2/(m!*(m+7))
    # i=8      2/9  1/11  1/156  1/5400     2/(m!*(m+9))
    # i=10     2/11 1/13  1/180  1/6120     2/(m!*(m+11))
    # i=12     2/13 1/15  1/204  1/6840     2/(m!*(m+13))
    x = x[cond2]
    b1[cond2] = 2.0 + x**2 / 3.0 + x**4 / 60.0 + x**6 / 2520.0
    b3[cond2] = 2.0 / 3.0 + x**2 / 5.0 + x**4 / 84.0 + x**6 / 3240.0
    b5[cond2] = 2.0 / 5.0 + x**2 / 7.0 + x**4 / 108.0 + x**6 / 3960.0
    b7[cond2] = 2.0 / 7.0 + x**2 / 9.0 + x**4 / 132.0 + x**6 / 4680.0
    b9[cond2] = 2.0 / 9.0 + x**2 / 11.0 + x**4 / 156.0 + x**6 / 5400.0
    b11[cond2] = 2.0 / 11.0 + x**2 / 13.0 + x**4 / 180.0 + x**6 / 6120.0
    b13[cond2] = 2.0 / 13.0 + x**2 / 15.0 + x**4 / 204.0 + x**6 / 6840.0

    # b5[cond2 & (jcall>=4)] = -x*2.0/7.0 - x**3/27.0 - x**5/660.0

    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is odd, i = 1,3, m= 1,3,5
    # b_{i+1} (x) = \sum_{m=1,3,5} -x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =    1     3      5
    # i=1       2/3  1/15  1/420  2/(m!*(m+2))
    # i=3       2/5  1/21  1/540  2/(m!*(m+4))
    # i=5       2/7  1/27  1/660  2/(m!*(m+6))
    # i=7       2/9  1/33  1/780  2/(m!*(m+8))
    # i=9       2/11 1/39  1/900  2/(m!*(m+10))
    # i=11      2/13 1/45  1/1020  2/(m!*(m+12))

    b2[cond2] = -2.0 / 3.0 * x - x**3 / 15.0 - x**5 / 420.0
    b4[cond2] = -2.0 / 5.0 * x - x**3 / 21.0 - x**5 / 540.0
    b6[cond2] = -2.0 / 7.0 * x - x**3 / 27.0 - x**5 / 660.0
    b8[cond2] = -2.0 / 9.0 * x - x**3 / 33.0 - x**5 / 780.0
    b10[cond2] = -2.0 / 11.0 * x - x**3 / 39.0 - x**5 / 900.0
    b12[cond2] = -2.0 / 13.0 * x - x**3 / 45.0 - x**5 / 1020.0

    # b4[cond2 & (jcall>=3)] = -2.0/5.0*x[cond2] - x[cond2]**3/21.0 - x[cond2]**5/540.0
    # print(b1)
    # print(b3)
    return th.cat((b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13), dim=1)
