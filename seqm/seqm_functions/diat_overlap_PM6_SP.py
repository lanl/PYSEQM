import math

import torch as th

from seqm.utils.torch_compile import optional_compile_function

from .two_elec_two_center_int import rotate_with_quaternion

_set_dispatch = None


def enable_overlap_compile(mode=None, **options):
    """Compile the A and B overlap integrals; the mask-driven overlap assembly around them stays eager."""
    global _set_dispatch

    compile_options = dict(options)
    compile_options.setdefault("dynamic", True)  # the number of pairs changes between calls
    if mode is not None:
        compile_options["mode"] = mode
    _set_dispatch = optional_compile_function(_SET_kernel, compile_options=compile_options, label="overlap.SET")


def diatom_overlap_matrix_PM6_SP(ni, nj, xij, rij, zeta_a, zeta_b, qn_int):
    """
    compute the overlap matrix for each pair
    """
    return diatom_overlap_rotate(diatom_overlap_local_terms(ni, nj, rij, zeta_a, zeta_b, qn_int), xij)


def diatom_overlap_local_terms(ni, nj, rij, zeta_a, zeta_b, qn_int):
    """
    Local-frame overlaps (S111, S211, S121, S221, S222), shape (npairs, 5). They depend on the pair
    distance only; ``diatom_overlap_rotate`` turns them into the 4x4 overlap block for a pair direction.
    """
    dtype = rij.dtype
    device = rij.device
    # xij unit vector points from i to j: xij = (xj-xi)/|xi-xj|, shape (npairs, 3)
    # rij : distance between atom i and j in atomic units, shape (npairs,)
    # ni, nj, atomic number of atom i and j, shape (npairs,)
    # x = x2-x1
    # y = y2-y1
    # z = z2-z1
    # for first atom index == second atom index, (same atom)
    # x = x1, y = y1, z = z1
    npairs = rij.shape[0]
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
            / (math.sqrt(3.0) * 8.0)
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
            / (16.0 * math.sqrt(3.0))
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
            / (16.0 * math.sqrt(3.0))
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
            / (math.sqrt(10.0) * 24.0)
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
            / (8.0 * math.sqrt(30.0))
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
            / (math.sqrt(30.0) * 48.0)
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
            / (48.0 * math.sqrt(10.0))
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
            / (48.0 * math.sqrt(10.0))
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
            / (16.0 * math.sqrt(30.0))
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
            / (32.0 * math.sqrt(30.0))
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
            / (480.0 * math.sqrt(3.0))
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
            / (480.0 * math.sqrt(3.0))
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

    del A111, B111, A211, B211, A121, B121, A22, B22
    return th.stack([S111, S211, S121, S221, S222], dim=1)


def diatom_overlap_rotate(terms, xij):
    """4x4 overlap block (s, px, py, pz) of every pair from its local terms and unit direction ``xij``."""
    S111, S211, S121, S221, S222 = terms.unbind(1)
    npairs = xij.shape[0]
    di = th.zeros((npairs, 4, 4), dtype=xij.dtype, device=xij.device)

    rot = rotate_with_quaternion(xij)
    rot = rot.transpose(1, 2)

    di[..., 0, 0] = S111
    c0 = rot[:, :, 0]
    di[:, 1:, 0] = S211.unsqueeze(1) * c0
    di[:, 0, 1:] = -S121.unsqueeze(1) * c0
    M = th.stack([-S221, S222, S222], dim=1)  # (N, 3)
    B = rot * M.unsqueeze(1)  # (N, 3, 3)
    di[:, 1:, 1:] = th.bmm(B, rot.transpose(1, 2))
    return di


# parameter_set = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
#                 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha']


def SET(rij, jcall, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    return (_set_dispatch or _SET_kernel)(rij, jcall, z1, z2)


def _SET_kernel(rij, jcall, z1, z2):
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
    inv_x = x.reciprocal()
    a1 = th.exp(-x) * inv_x

    a2 = a1 + a1 * inv_x
    # jcall >= 2
    a3 = a1 + 2.0 * a2 * inv_x
    # jcall >= 3
    jcallp3 = (jcall >= 3).reshape((-1, 1))
    a4 = th.where(jcallp3, a1 + 3.0 * a3 * inv_x, th.tensor(0.0, dtype=dtype, device=device))
    # jcall >=4
    jcallp4 = (jcall >= 4).reshape((-1, 1))
    a5 = th.where(jcallp4, a1 + 4.0 * a4 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp5 = (jcall >= 5).reshape((-1, 1))
    a6 = th.where(jcallp5, a1 + 5.0 * a5 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp6 = (jcall >= 6).reshape((-1, 1))
    a7 = th.where(jcallp6, a1 + 6.0 * a6 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp7 = (jcall >= 7).reshape((-1, 1))
    a8 = th.where(jcallp7, a1 + 7.0 * a7 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp8 = (jcall >= 8).reshape((-1, 1))
    a9 = th.where(jcallp8, a1 + 8.0 * a8 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp9 = (jcall >= 9).reshape((-1, 1))
    a10 = th.where(jcallp9, a1 + 9.0 * a9 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp10 = (jcall >= 10).reshape((-1, 1))
    a11 = th.where(jcallp10, a1 + 10.0 * a10 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp11 = (jcall >= 11).reshape((-1, 1))
    a12 = th.where(jcallp11, a1 + 11.0 * a11 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    jcallp12 = (jcall >= 12).reshape((-1, 1))
    a13 = th.where(jcallp12, a1 + 12.0 * a12 * inv_x, th.tensor(0.0, dtype=dtype, device=device))

    return th.cat((a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13), dim=1)


_B_SERIES_CUTOFF = 1.8  # the Taylor series is used for |x| up to this, the upward recurrence beyond it
_B_SERIES_TERMS = 8  # terms in x^2: ~1e-10 accurate at the cutoff
_B_NORDER = 13
_b_series_cache = {}


def _b_series_tables(dtype, device):
    """Taylor coefficients of B_k in u = x^2 and the odd-k mask.

    B_k(-x) = (-1)^k B_k(x), so B_k = sum_j C[j, k] u^j, times x for odd k. With m = 2j (even k) or 2j + 1 (odd k),
    C[j, k] = 2 (-1)^m / (m! (k + m + 1)).
    """
    key = (dtype, device)
    if key not in _b_series_cache:
        C = th.zeros(_B_SERIES_TERMS, _B_NORDER, dtype=th.float64)
        for k in range(_B_NORDER):
            for j in range(_B_SERIES_TERMS):
                m = 2 * j + k % 2
                C[j, k] = 2.0 * (-1) ** m / (math.factorial(m) * (k + m + 1))
        odd = th.arange(_B_NORDER, dtype=th.float64) % 2
        _b_series_cache[key] = (C.to(dtype=dtype, device=device), odd.to(dtype=dtype, device=device))
    return _b_series_cache[key]


def bintgs(x0, jcall):
    """Evaluate the 13 B overlap integrals, B_k(x) = int_{-1}^{1} t^k exp(-x t) dt, k = 0..12.

    The upward recurrence in k magnifies rounding error by about (k-1)/|x| at each order, which is too much for small
    |x|, so there B_k comes from its Taylor series instead. The result is accurate to ~1e-10 for every k, and the
    columns the overlap routines read (k <= 6) are mostly better.
    """
    del jcall  # Kept in the public signature for compatibility with SET.
    x = x0.reshape(-1, 1)
    large = th.abs(x) > _B_SERIES_CUTOFF

    # A safe placeholder prevents singular values and gradients in the branch
    # that torch.where discards.
    x_large = th.where(large, x, th.ones_like(x))
    inv_x = x_large.reciprocal()
    tx = th.exp(x_large) * inv_x
    tmx = -th.exp(-x_large) * inv_x

    large_values = []
    previous = tx + tmx
    large_values.append(previous)
    for order in range(2, 14):
        leading = (-tx if order % 2 == 0 else tx) + tmx
        previous = leading + (order - 1) * previous * inv_x
        large_values.append(previous)

    x_small = th.where(large, th.zeros_like(x), x)
    u = x_small * x_small
    coeff, odd = _b_series_tables(x.dtype, x.device)
    series = coeff[-1].expand(x.shape[0], _B_NORDER)
    for j in range(_B_SERIES_TERMS - 2, -1, -1):
        series = series * u + coeff[j]
    series = series * (1.0 + odd * (x_small - 1.0))

    return th.where(large, th.cat(large_values, dim=1), series)
